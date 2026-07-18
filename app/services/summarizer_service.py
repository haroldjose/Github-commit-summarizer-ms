"""Carga del modelo seq2seq (Prompt 1) e inferencia con beam search en CPU.

Las clases Encoder/Decoder replican EXACTAMENTE la arquitectura y los nombres
de atributos del notebook de entrenamiento — de lo contrario el state_dict no
carga. Los hiperparámetros viajan dentro del checkpoint.
"""
import json
import logging
import re
from pathlib import Path

import torch
import torch.nn as nn

from app.services.retrieval import RecuperadorNNGen

log = logging.getLogger(__name__)

PAD, SOS, EOS, UNK = 0, 1, 2, 3

# Normalización del corpus NNGen (Jiang et al. 2017): el modelo se entrenó
# sobre diffs con este formato, no sobre `git diff` crudo. Sin esta etapa,
# la mayoría de los tokens reales caen fuera del vocabulario (<unk>).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^A-Za-z0-9_\s]")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@ ?")


def normalizar_diff(diff: str) -> str:
    """Convierte un `git diff` crudo al formato del corpus de entrenamiento.

    - `--- a/x` → `mmm a / x`, `+++ b/x` → `ppp b / x`
    - la línea `diff --git ...` se elimina; del hunk `@@ ... @@` se conserva
      solo el contexto que lo sigue
    - saltos de línea → token `<nl>`; puntuación separada por espacios
      (el `_` se mantiene dentro de los identificadores)
    """
    lineas = []
    for linea in diff.splitlines():
        if not linea.strip() or linea.startswith("diff --git"):
            continue
        if linea.startswith("--- "):
            linea = "mmm " + linea[4:]
        elif linea.startswith("+++ "):
            linea = "ppp " + linea[4:]
        else:
            recorte = _HUNK_RE.match(linea)
            if recorte:
                linea = linea[recorte.end():]
                if not linea.strip():
                    continue
        lineas.append(" ".join(_TOKEN_RE.findall(linea)))
    return " <nl> ".join(lineas)


# Líneas que definen el cambio: cabeceras de archivo, metadatos del diff y
# líneas añadidas/eliminadas. El contexto sin modificar queda fuera.
_PREFIJOS_CAMBIO = ("mmm", "ppp", "+", "-", "new file", "deleted file",
                    "Binary", "index", "rename", "old mode", "new mode")


def podar_diff(normalizado: str, presupuesto: int = 100) -> str:
    """Poda al estilo CODISUM: gasta el presupuesto de tokens del modelo en
    las líneas que definen el commit (+/-, cabeceras, metadatos) en vez de en
    imports y contexto. Sin esto, en un diff grande el modelo solo "ve" el
    principio del archivo.

    El presupuesto se reparte **round-robin entre bloques de cambio** (los
    hunks/archivos, delimitados por líneas de contexto): primero la primera
    línea de cada bloque, luego la segunda, etc. Así un diff enorme
    multi-archivo queda representado por el inicio de *todos* sus cambios en
    lugar de agotar el presupuesto en el primero. Si el diff no tiene líneas
    de cambio (raro), se devuelve intacto y el truncado normal decide.
    """
    lineas = normalizado.split(" <nl> ")

    # Bloques: rachas consecutivas de líneas de cambio; el contexto separa.
    bloques: list[list[int]] = []
    actual: list[int] = []
    for i, linea in enumerate(lineas):
        if linea.startswith(_PREFIJOS_CAMBIO):
            actual.append(i)
        elif actual:
            bloques.append(actual)
            actual = []
    if actual:
        bloques.append(actual)
    if not bloques:
        return normalizado

    elegidos: set[int] = set()
    usado = 0
    for ronda in range(max(len(b) for b in bloques)):
        if usado >= presupuesto:
            break
        for b in bloques:
            if ronda >= len(b):
                continue
            i = b[ronda]
            costo = len(lineas[i].split()) + 1  # +1 por el <nl> separador
            if usado + costo > presupuesto:
                continue  # esta no cabe; otra más corta quizá sí
            elegidos.add(i)
            usado += costo
    if not elegidos:  # presupuesto menor que cualquier línea: tomar la primera
        elegidos.add(bloques[0][0])
    return " <nl> ".join(lineas[i] for i in sorted(elegidos))


def detokenizar(texto: str) -> str:
    """Une la salida pre-tokenizada del corpus/modelo en texto natural:
    `don ' t use rc version .` → `don't use rc version.`"""
    texto = re.sub(r"(\w) ' (\w)", r"\1'\2", texto)
    texto = re.sub(r"(?<=\d) ?\. ?(?=\d)", ".", texto)  # versiones: 0 . 6 . 2
    texto = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", texto)
    texto = re.sub(r"([(\[{$#])\s+", r"\1", texto)
    texto = re.sub(r"(\w)\. (\w)", r"\1.\2", texto)  # engine. io → engine.io
    return re.sub(r"\s+", " ", texto).strip()


# Patrones heurísticos (línea ChangeScribe / reglas SE, sin red neuronal).
# El normalizador separa puntuación: `!=` → `! =`, `deleteUser(` → `deleteUser (`.
_VER_RE = re.compile(
    r"- < version > (?P<old>[^<]+?) < / version > .*?"
    r"\+ < version > (?P<new>[^<]+?) < / version >",
    re.DOTALL,
)
_METODO_JAVA_RE = re.compile(
    r"\+ (?:public|private|protected) (?:static )?(?:final )?(?:\w+ )+(\w+) \(",
)
_METODO_PY_RE = re.compile(r"\+ def (\w+) \(")
_FUNC_JS_RE = re.compile(r"\+ (?:export )?(?:async )?function (\w+) \(")
_ARROW_JS_RE = re.compile(r"\+ const (\w+) = (?:async )?\(")
_IMPORT_LINEA_RE = re.compile(
    r"(?:^| <nl> )[+-] (?:import |from |const \w+ = require \()")
_CTX_PY_RE = re.compile(r"(?:^| <nl> )def (\w+) \(")
_CTX_JAVA_RE = re.compile(
    r"(?:^| <nl> )(?:public|private|protected) (?:static )?(?:final )?(?:\w+ )+(\w+) \(",
)
_NULL_CHECK_RE = re.compile(r"\+ if \( \w+ ! = null \)")


_PPP_ARCHIVO_RE = re.compile(r"ppp b / (.+?)(?: <nl> |$)")
_MMM_ARCHIVO_RE = re.compile(r"mmm a / (.+?)(?: <nl> |$)")
_ES_NUEVO_RE = re.compile(r"(?:^| <nl> )(?:new file mode|mmm / dev / null)")
_ES_BORRADO_RE = re.compile(r"(?:^| <nl> )(?:deleted file mode|ppp / dev / null)")


def _nombres(regex: re.Pattern, normalizado: str) -> list[str]:
    vistos = []
    for ruta in regex.findall(normalizado):
        nombre = detokenizar(ruta).split("/")[-1].strip()
        if nombre and nombre not in vistos:
            vistos.append(nombre)
    return vistos


def heuristica_diff(normalizado: str) -> str:
    """Mensaje plantilla a partir de patrones claros del diff normalizado."""
    # Archivo nuevo / borrado: la señal más fuerte, gana a las demás.
    if _ES_NUEVO_RE.search(normalizado):
        nombres = _nombres(_PPP_ARCHIVO_RE, normalizado)
        if len(nombres) == 1:
            return f"Add {nombres[0]}"
        if nombres:
            return f"Add {nombres[0]} and {len(nombres) - 1} more file(s)"
    if _ES_BORRADO_RE.search(normalizado):
        nombres = _nombres(_MMM_ARCHIVO_RE, normalizado)
        if len(nombres) == 1:
            return f"Delete {nombres[0]}"
        if nombres:
            return f"Delete {nombres[0]} and {len(nombres) - 1} more file(s)"

    # Composición: se acumulan TODOS los patrones detectados (cada parte es
    # verificable contra el diff — más descriptivo sin inventar nada).
    partes = []

    m = _VER_RE.search(normalizado)
    if m:
        nueva = detokenizar(m.group("new").strip())
        partes.append(f"bump version to {nueva}" if nueva else "bump version")

    metodos = []
    for regex in (_METODO_JAVA_RE, _METODO_PY_RE, _FUNC_JS_RE, _ARROW_JS_RE):
        for nombre in regex.findall(normalizado):
            if nombre not in metodos:
                metodos.append(nombre)
    if len(metodos) == 1:
        partes.append(f"Add {metodos[0]} method")
    elif len(metodos) == 2:
        partes.append(f"Add {metodos[0]} and {metodos[1]} methods")
    elif metodos:
        partes.append(f"Add {len(metodos)} methods")

    if _NULL_CHECK_RE.search(normalizado):
        partes.append("Add null check")

    if len(_IMPORT_LINEA_RE.findall(normalizado)) >= 2:
        partes.append("update imports")

    if partes:
        # El corte no es un número mágico: se acumulan partes mientras quepan
        # en el contrato de salida (25 tokens, el MAX_MSG del entrenamiento).
        elegidas, usado = [], 0
        for parte in partes:
            costo = len(parte.split()) + (1 if elegidas else 0)  # +1 por el ';'
            if usado + costo > 25:
                break
            elegidas.append(parte)
            usado += costo
        return "; ".join(elegidas)

    # Función/método tocado (contexto del hunk, sin declaración nueva con +).
    if " <nl> + " in normalizado or normalizado.startswith("+ ") or " <nl> - " in normalizado:
        ctx = _CTX_PY_RE.findall(normalizado) + _CTX_JAVA_RE.findall(normalizado)
        if len(set(ctx)) == 1:
            return f"Update {ctx[0]}"

    return ""


def _salida_degenerada(texto: str) -> bool:
    """True si el generativo produjo basura: 'Fix #', 'Fix. (#)', o muletillas
    de ruido del corpus (mensajes de merge/revert) que nunca describen el diff."""
    palabras = re.findall(r"[A-Za-z]{2,}", texto)
    if len(palabras) < 2:
        return True
    bajo = texto.lower()
    return bajo.startswith(("merge pull request", "merge branch", "merge remote",
                            "revert ", "rollback "))


_ARCHIVO_RE = re.compile(r"mmm a / (.+?) <nl>")


def respaldo_archivos(normalizado: str) -> str:
    """Último recurso: mensaje honesto a partir de los archivos modificados.

    Para diffs fuera de distribución (otros lenguajes, cambios inusuales)
    donde recuperación, heurísticas y generativo no dan nada confiable,
    'Update UserProfile.js' es preferible a una respuesta vacía o inventada.
    """
    nombres = []
    for ruta in _ARCHIVO_RE.findall(normalizado):
        nombre = detokenizar(ruta).split("/")[-1].strip()
        if nombre and nombre not in nombres:
            nombres.append(nombre)
    if not nombres:
        return ""

    # Si los cambios son mayoritariamente imports, decirlo es más informativo.
    cambios = [l for l in normalizado.split(" <nl> ")
               if l.startswith("+ ") or l.startswith("- ")]
    imports = [l for l in cambios
               if l.startswith(("+ import ", "- import ", "+ from ", "- from "))]
    if cambios and len(imports) / len(cambios) >= 0.8:
        return f"Update imports in {nombres[0]}"

    if len(nombres) == 1:
        return f"Update {nombres[0]}"
    return f"Update {nombres[0]} and {len(nombres) - 1} more file(s)"

class Encoder(nn.Module):
    def __init__(self, vocab: int, emb: int, hid: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.proy = nn.Linear(hid * 2, hid)

    def forward(self, x):
        salidas, h = self.gru(self.emb(x))
        h0 = torch.tanh(self.proy(torch.cat([h[0], h[1]], dim=1)))
        return salidas, h0.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, vocab: int, emb: int, hid: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.atn_proj = nn.Linear(hid, hid * 2)
        self.salida = nn.Linear(hid * 3, vocab)

    def forward(self, y_prev, h, enc_out, mascara):
        e = self.emb(y_prev)
        o, h = self.gru(e, h)
        puntajes = torch.bmm(self.atn_proj(o), enc_out.transpose(1, 2))
        puntajes = puntajes.masked_fill(mascara.unsqueeze(1), -1e9)
        pesos = torch.softmax(puntajes, dim=-1)
        contexto = torch.bmm(pesos, enc_out)
        return self.salida(torch.cat([o, contexto], dim=-1)), h


class DecoderPtrGen(nn.Module):
    """Decoder con pointer-generator (See et al. 2017) — Fase 3.

    P(w) = p_gen · P_vocab(w) + (1 − p_gen) · Σ atención sobre posiciones del
    diff donde aparece w. Permite copiar identificadores OOV (deleteUser,
    nombres de clases) directamente del diff al mensaje. Los nombres de
    atributos replican el script de entrenamiento (training/train_ptrgen.py).
    """

    def __init__(self, vocab: int, emb: int, hid: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.atn_proj = nn.Linear(hid, hid * 2)
        self.salida = nn.Linear(hid * 3, vocab)
        self.p_gen = nn.Linear(hid * 3 + emb, 1)
        self.vocab = vocab

    def forward(self, y_prev, h, enc_out, mascara, src_map, n_oov):
        e = self.emb(y_prev)
        o, h = self.gru(e, h)
        puntajes = torch.bmm(self.atn_proj(o), enc_out.transpose(1, 2))
        puntajes = puntajes.masked_fill(mascara.unsqueeze(1), -1e9)
        atencion = torch.softmax(puntajes, dim=-1)
        contexto = torch.bmm(atencion, enc_out)
        rasgos = torch.cat([o, contexto], dim=-1)
        p_vocab = torch.softmax(self.salida(rasgos), dim=-1)
        pg = torch.sigmoid(self.p_gen(torch.cat([rasgos, e], dim=-1)))

        B, T, _ = p_vocab.shape
        dist = p_vocab.new_zeros(B, T, self.vocab + n_oov)
        dist[..., :self.vocab] = pg * p_vocab
        indices = src_map.unsqueeze(1).expand(-1, T, -1)
        dist.scatter_add_(2, indices, (1.0 - pg) * atencion)
        return dist, h


class SummarizerService:
    # Umbrales del híbrido NNGen+seq2seq. El coseno solo no basta: se infla
    # con tokens estructurales (mmm/ppp/<nl>). El BLEU-4 entre diffs es la
    # señal de calidad del paper; exige solapamiento real del cambio.
    UMBRAL_COS = 0.5
    UMBRAL_BLEU = 0.5
    MIN_TOKENS = 3  # longitud mínima del seq2seq: evita salidas tipo "Fix"

    def __init__(self, models_dir: str):
        base = Path(models_dir)
        self.itos_d = json.loads((base / "vocab_diff.json").read_text())
        self.itos_m = json.loads((base / "vocab_msg.json").read_text())
        self.stoi_d = {w: i for i, w in enumerate(self.itos_d)}
        self.stoi_m = {w: i for i, w in enumerate(self.itos_m)}

        # Recuperación NNGen (opcional: solo si el corpus viaja con los artefactos)
        corpus_diff = base / "corpus_train.diff"
        corpus_msg = base / "corpus_train.msg"
        self.recuperador = None
        if corpus_diff.exists() and corpus_msg.exists():
            self.recuperador = RecuperadorNNGen(corpus_diff, corpus_msg)
        else:
            log.warning("Sin corpus de recuperación en %s: solo seq2seq", base)

        # Fase 3: preferir el pointer-generator si sus artefactos existen.
        ruta_ptrgen = base / "modelo_resumidor_ptrgen.pt"
        ruta_base = base / "modelo_resumidor.pt"
        ruta = ruta_ptrgen if ruta_ptrgen.exists() else ruta_base
        ckpt = torch.load(ruta, map_location="cpu")
        hp = ckpt["hiperparametros"]
        self.max_diff, self.max_msg = hp["MAX_DIFF"], hp["MAX_MSG"]
        self.es_ptrgen = ckpt.get("arquitectura") == "ptrgen"
        # v2: el checkpoint declara si fue entrenado con poda estricta; si no,
        # la poda solo se aplica a diffs que exceden el presupuesto.
        self.poda_estricta = ckpt.get("poda") == "estricta"

        self.encoder = Encoder(len(self.itos_d), hp["EMB"], hp["HID"])
        if self.es_ptrgen:
            self.decoder = DecoderPtrGen(len(self.itos_m), hp["EMB"], hp["HID"])
        else:
            self.decoder = Decoder(len(self.itos_m), hp["EMB"], hp["HID"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.encoder.eval()
        self.decoder.eval()
        log.info("Modelo resumidor cargado: %s (%d/%d tokens de vocabulario)",
                 "pointer-generator" if self.es_ptrgen else "seq2seq base",
                 len(self.itos_d), len(self.itos_m))

    def resumir(self, diff: str, beam: int = 5) -> str:
        normalizado = normalizar_diff(diff)

        # 1) Heurística SE (ChangeScribe-like): patrones inequívocos con
        #    valores extraídos del PROPIO diff — siempre verificables. Van
        #    antes que la recuperación porque un vecino casi idéntico en
        #    estructura puede traer detalles ajenos (otra versión, otro
        #    artefacto) y "mentir" con confianza.
        plantilla = heuristica_diff(normalizado)
        if plantilla:
            return plantilla

        # 2) Recuperación NNGen: solo si el vecino es realmente parecido
        #    (coseno + BLEU entre diffs). Si no, el mensaje recuperado miente.
        vecino_msg, vecino_cos, vecino_bleu = "", 0.0, 0.0
        if self.recuperador is not None:
            vecino_msg, vecino_cos, vecino_bleu = self.recuperador.consultar(
                normalizado
            )
            if (vecino_cos >= self.UMBRAL_COS
                    and vecino_bleu >= self.UMBRAL_BLEU
                    and vecino_msg):
                return detokenizar(vecino_msg)

        # 3) Fallback generativo: el modelo entrenado desde cero.
        texto = self._generar(normalizado, beam)
        if texto and not _salida_degenerada(texto):
            return texto

        # 4) Último recurso: plantilla con los archivos tocados. Nunca
        #    devolver basura tipo "Fix #" ni un mensaje recuperado engañoso.
        return respaldo_archivos(normalizado)
    @torch.no_grad()
    def _generar(self, normalizado: str, beam: int = 5) -> str:
        tokens = normalizado.split()
        if self.poda_estricta or len(tokens) > self.max_diff:
            tokens = podar_diff(normalizado, self.max_diff).split()
        tokens = tokens[: self.max_diff]
        ids = [self.stoi_d.get(t, UNK) for t in tokens] or [UNK]
        X = torch.tensor([ids])
        enc_out, h = self.encoder(X)
        mascara = X == PAD

        # Vocabulario extendido para el pointer-generator: cada token del diff
        # se mapea al vocab de mensajes, o a un id de copia Vm+j si es OOV.
        Vm = len(self.itos_m)
        oovs: list[str] = []
        if self.es_ptrgen:
            src_ids = []
            for t in (tokens or ["<unk>"]):
                i = self.stoi_m.get(t)
                if i is None:
                    if t not in oovs:
                        oovs.append(t)
                    i = Vm + oovs.index(t)
                src_ids.append(i)
            src_map = torch.tensor([src_ids])
            n_oov = max(1, len(oovs))

        haces = [(0.0, [], h, False)]
        for _ in range(self.max_msg):
            candidatos = []
            for lp, toks, hh, fin in haces:
                if fin:
                    candidatos.append((lp, toks, hh, True))
                    continue
                ultimo = SOS if not toks else toks[-1]
                if ultimo >= Vm:  # token copiado: el embedding no lo conoce
                    ultimo = UNK
                entrada = torch.full((1, 1), ultimo, dtype=torch.long)
                if self.es_ptrgen:
                    dist, h2 = self.decoder(entrada, hh, enc_out, mascara,
                                            src_map, n_oov)
                    logprobs = torch.log(dist.view(-1) + 1e-9)
                else:
                    logits, h2 = self.decoder(entrada, hh, enc_out, mascara)
                    logprobs = torch.log_softmax(logits.view(-1), dim=-1)
                # no-repeat bigram: bloquea los bucles degenerativos tipicos
                # de los seq2seq compactos ("reviewed by ; reviewed by ; ...").
                # Restriccion estandar de decodificacion, no altera el modelo.
                bigramas = {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}
                top_lp, top_ix = logprobs.topk(min(beam * 3, logprobs.numel()))
                agregados = 0
                for k in range(top_ix.numel()):
                    if agregados >= beam:
                        break
                    tok = int(top_ix[k])
                    if tok == EOS and len(toks) < self.MIN_TOKENS:
                        continue  # longitud mínima: no cerrar en 1-2 tokens
                    if toks and (toks[-1], tok) in bigramas:
                        continue
                    candidatos.append((lp + float(top_lp[k]), toks + [tok], h2, tok == EOS))
                    agregados += 1
            nuevos = sorted(candidatos, key=lambda c: c[0] / max(len(c[1]), 1),
                            reverse=True)[:beam]
            if not nuevos:
                break  # todos los candidatos bloqueados: conservar haces previos
            haces = nuevos
            if all(c[3] for c in haces):
                break

        # Post-proceso: filtrar especiales; resolver ids de copia con el diff
        palabras = []
        for t in haces[0][1]:
            if t in (PAD, SOS, EOS, UNK):
                continue
            palabras.append(self.itos_m[t] if t < Vm else oovs[t - Vm])
        texto = detokenizar(re.sub(r"\(\s*\)", " ", " ".join(palabras)))
        # puntuacion colgante al final, sin tocar cierres legitimos como ')'
        return texto.rstrip(" .,;:-_")
