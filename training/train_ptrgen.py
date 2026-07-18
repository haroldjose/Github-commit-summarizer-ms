"""Fase 3: pointer-generator (See et al., ACL 2017) entrenado DESDE CERO.

Reentrena el resumidor de commits con mecanismo de copia: en cada paso el
decoder decide entre generar una palabra de su vocabulario o copiar un token
del diff de entrada (usando los pesos de atencion como distribucion de copia).
Esto ataca la limitacion principal del seq2seq base: no podia mencionar
identificadores del codigo (deleteUser, nombres de clases) porque no existen
en el vocabulario de mensajes.

Cumple los limites del proyecto: sin modelos pre-entrenados, mismo corpus
publico NNGen, misma semilla (42), mismos vocabularios que la Fase 1.

Uso:  python training/train_ptrgen.py
Artefactos: training/artefactos/modelo_resumidor_ptrgen.pt + metricas_ptrgen.json
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.summarizer_service import podar_diff  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
ART = BASE / "artefactos"

PAD, SOS, EOS, UNK = 0, 1, 2, 3
MAX_DIFF, MAX_MSG = 100, 25
EMB, HID = 256, 256
EPOCAS, PACIENCIA = 25, 3
BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Datos ────────────────────────────────────────────────────────────────────
def cargar(split):
    with open(DATA / f"cleaned.{split}.diff", encoding="utf-8", errors="replace") as f:
        diffs = [l.strip() for l in f]
    with open(DATA / f"cleaned.{split}.msg", encoding="utf-8", errors="replace") as f:
        msgs = [l.strip() for l in f]
    assert len(diffs) == len(msgs)
    return diffs, msgs


# Vocabularios de la Fase 1 (mismos artefactos): garantiza comparabilidad.
itos_d = json.loads((ART / "vocab_diff.json").read_text())
itos_m = json.loads((ART / "vocab_msg.json").read_text())
stoi_d = {w: i for i, w in enumerate(itos_d)}
stoi_m = {w: i for i, w in enumerate(itos_m)}
Vd, Vm = len(itos_d), len(itos_m)


def preparar_ejemplo(diff: str, msg: str):
    """Codifica un par con vocabulario extendido por copia.

    - x: ids del diff en vocab_diff (entrada del encoder)
    - src_map: cada token del diff mapeado al espacio del vocab de mensajes,
      o a un id extendido Vm+j si es OOV (candidato a copia)
    - oovs: lista de tokens OOV del diff, en orden de aparicion
    - y_in: mensaje en vocab_msg (entrada teacher forcing; OOV -> UNK)
    - y_out: mensaje en vocabulario extendido (objetivo de la perdida;
      un OOV del mensaje que aparece en el diff recibe su id extendido)
    """
    # v2: poda estricta (misma que el servicio) — el presupuesto de tokens se
    # gasta en cabeceras y líneas +/- del diff, no en contexto sin cambiar.
    toks_x = podar_diff(diff, MAX_DIFF).split()[:MAX_DIFF] or ["<unk>"]
    x = [stoi_d.get(t, UNK) for t in toks_x]
    oovs, src_map = [], []
    for t in toks_x:
        i = stoi_m.get(t)
        if i is not None:
            src_map.append(i)
        else:
            if t not in oovs:
                oovs.append(t)
            src_map.append(Vm + oovs.index(t))
    toks_y = msg.split()[:MAX_MSG]
    y_in = [SOS] + [stoi_m.get(t, UNK) for t in toks_y]
    y_out = []
    for t in toks_y:
        i = stoi_m.get(t)
        if i is None and t in oovs:
            i = Vm + oovs.index(t)
        y_out.append(i if i is not None else UNK)
    y_out.append(EOS)
    return x, src_map, oovs, y_in, y_out


class ParesDataset(Dataset):
    def __init__(self, diffs, msgs):
        self.ej = [preparar_ejemplo(d, m) for d, m in zip(diffs, msgs)]

    def __len__(self):
        return len(self.ej)

    def __getitem__(self, i):
        return self.ej[i]


def colacionar(lote):
    xs, maps, oovs, yins, youts = zip(*lote)
    lx = max(len(x) for x in xs)
    ly = max(len(y) for y in yins)
    n_oov = max(1, max(len(o) for o in oovs))
    X = torch.full((len(xs), lx), PAD, dtype=torch.long)
    M = torch.zeros((len(xs), lx), dtype=torch.long)  # PAD -> id 0, atencion ya lo anula
    Yi = torch.full((len(xs), ly), PAD, dtype=torch.long)
    Yo = torch.full((len(xs), ly), PAD, dtype=torch.long)
    for i, (x, m, _, yi, yo) in enumerate(lote):
        X[i, :len(x)] = torch.tensor(x)
        M[i, :len(m)] = torch.tensor(m)
        Yi[i, :len(yi)] = torch.tensor(yi)
        Yo[i, :len(yo)] = torch.tensor(yo)
    return X, M, Yi, Yo, n_oov


# ── Modelo ───────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """Identico al de la Fase 1 (biGRU + proyeccion del estado)."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Vd, EMB, padding_idx=PAD)
        self.gru = nn.GRU(EMB, HID, batch_first=True, bidirectional=True)
        self.proy = nn.Linear(HID * 2, HID)

    def forward(self, x):
        salidas, h = self.gru(self.emb(x))
        h0 = torch.tanh(self.proy(torch.cat([h[0], h[1]], dim=1)))
        return salidas, h0.unsqueeze(0)


class DecoderPtrGen(nn.Module):
    """GRU + atencion Luong + pointer-generator (See et al. 2017).

    P(w) = p_gen * P_vocab(w) + (1 - p_gen) * sum_{i: x_i = w} atencion_i
    con p_gen = sigmoide(W [estado; contexto; embedding_entrada]).
    Acepta secuencias completas (teacher forcing) o de longitud 1 (beam).
    """

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Vm, EMB, padding_idx=PAD)
        self.gru = nn.GRU(EMB, HID, batch_first=True)
        self.atn_proj = nn.Linear(HID, HID * 2)
        self.salida = nn.Linear(HID * 3, Vm)
        self.p_gen = nn.Linear(HID * 3 + EMB, 1)

    def forward(self, y_prev, h, enc_out, mascara, src_map, n_oov):
        e = self.emb(y_prev)                                   # B,T,E
        o, h = self.gru(e, h)                                  # B,T,H
        puntajes = torch.bmm(self.atn_proj(o), enc_out.transpose(1, 2))
        puntajes = puntajes.masked_fill(mascara.unsqueeze(1), -1e9)
        atencion = torch.softmax(puntajes, dim=-1)             # B,T,L
        contexto = torch.bmm(atencion, enc_out)                # B,T,2H
        rasgos = torch.cat([o, contexto], dim=-1)              # B,T,3H
        p_vocab = torch.softmax(self.salida(rasgos), dim=-1)   # B,T,Vm
        pg = torch.sigmoid(self.p_gen(torch.cat([rasgos, e], dim=-1)))  # B,T,1

        B, T, _ = p_vocab.shape
        dist = p_vocab.new_zeros(B, T, Vm + n_oov)
        dist[..., :Vm] = pg * p_vocab
        indices = src_map.unsqueeze(1).expand(-1, T, -1)       # B,T,L
        dist.scatter_add_(2, indices, (1.0 - pg) * atencion)
        return dist, h


def perdida_lote(encoder, decoder, X, M, Yi, Yo, n_oov):
    X, M, Yi, Yo = (t.to(DEVICE) for t in (X, M, Yi, Yo))
    enc_out, h = encoder(X)
    mascara = X == PAD
    dist, _ = decoder(Yi, h, enc_out, mascara, M, n_oov)
    probs = dist.gather(2, Yo.unsqueeze(-1)).squeeze(-1)       # B,T
    nll = -torch.log(probs + 1e-9)
    activo = Yo != PAD
    return (nll * activo).sum() / activo.sum()


# ── Entrenamiento ────────────────────────────────────────────────────────────
def main():
    print(f"Dispositivo: {DEVICE} | torch {torch.__version__}", flush=True)
    train_diffs, train_msgs = cargar("train")
    val_diffs, val_msgs = cargar("valid")
    print(f"train {len(train_diffs)} | valid {len(val_diffs)} | "
          f"vocab diff {Vd} | vocab msg {Vm}", flush=True)

    dl_train = DataLoader(ParesDataset(train_diffs, train_msgs), batch_size=BATCH,
                          shuffle=True, collate_fn=colacionar)
    dl_val = DataLoader(ParesDataset(val_diffs, val_msgs), batch_size=BATCH,
                        shuffle=False, collate_fn=colacionar)

    encoder, decoder = Encoder().to(DEVICE), DecoderPtrGen().to(DEVICE)
    params = sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
    print(f"Parametros: {params / 1e6:.1f}M (pointer-generator, desde cero)", flush=True)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    mejor_val, sin_mejora, epoca_final = float("inf"), 0, 0

    for ep in range(1, EPOCAS + 1):
        t0 = time.time()
        encoder.train()
        decoder.train()
        perdidas = []
        for X, M, Yi, Yo, n_oov in dl_train:
            perdida = perdida_lote(encoder, decoder, X, M, Yi, Yo, n_oov)
            opt.zero_grad()
            perdida.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 5.0)
            opt.step()
            perdidas.append(perdida.item())

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val = np.mean([perdida_lote(encoder, decoder, *lote).item()
                           for lote in dl_val])
        marca = ""
        if val < mejor_val:
            mejor_val, sin_mejora = val, 0
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "hiperparametros": {"EMB": EMB, "HID": HID,
                                    "MAX_DIFF": MAX_DIFF, "MAX_MSG": MAX_MSG},
                "arquitectura": "ptrgen",
                "poda": "estricta",
            }, ART / "modelo_resumidor_ptrgen_v2.pt")
            marca = " *guardado*"
        else:
            sin_mejora += 1
        epoca_final = ep
        print(f"epoca {ep:02d} | train {np.mean(perdidas):.3f} | val {val:.3f} "
              f"| {time.time() - t0:.0f}s{marca}", flush=True)
        if sin_mejora >= PACIENCIA:
            print(f"Parada temprana tras {PACIENCIA} epocas sin mejora.", flush=True)
            break

    # ── Evaluacion: BLEU-4 y ROUGE-L en test (mismo protocolo que Fase 1) ────
    print("Evaluando en test con beam search (esto tarda)...", flush=True)
    ckpt = torch.load(ART / "modelo_resumidor_ptrgen_v2.pt", map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    test_diffs, test_msgs = cargar("test")

    @torch.no_grad()
    def resumir(diff_texto, beam=5):
        x, src_map, oovs, _, _ = preparar_ejemplo(diff_texto, "")
        X = torch.tensor([x], device=DEVICE)
        M = torch.tensor([src_map], device=DEVICE)
        n_oov = max(1, len(oovs))
        enc_out, h = encoder(X)
        mascara = X == PAD
        haces = [(0.0, [], h, False)]
        for _ in range(MAX_MSG):
            candidatos = []
            for lp, toks, hh, fin in haces:
                if fin:
                    candidatos.append((lp, toks, hh, True))
                    continue
                ultimo = SOS if not toks else toks[-1]
                if ultimo >= Vm:      # token copiado: el embedding no lo conoce
                    ultimo = UNK
                entrada = torch.full((1, 1), ultimo, dtype=torch.long, device=DEVICE)
                dist, h2 = decoder(entrada, hh, enc_out, mascara, M, n_oov)
                logprobs = torch.log(dist.view(-1) + 1e-9)
                top_lp, top_ix = logprobs.topk(beam)
                for k in range(beam):
                    tok = int(top_ix[k])
                    candidatos.append((lp + float(top_lp[k]), toks + [tok], h2,
                                       tok == EOS))
            haces = sorted(candidatos, key=lambda c: c[0] / max(len(c[1]), 1),
                           reverse=True)[:beam]
            if all(c[3] for c in haces):
                break
        palabras = []
        for t in haces[0][1]:
            if t in (PAD, SOS, EOS, UNK):
                continue
            palabras.append(itos_m[t] if t < Vm else oovs[t - Vm])
        return " ".join(palabras)

    import sacrebleu
    from rouge_score import rouge_scorer

    generados = []
    t0 = time.time()
    for i, d in enumerate(test_diffs):
        generados.append(resumir(d))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(test_diffs)} ({time.time() - t0:.0f}s)", flush=True)

    bleu = sacrebleu.corpus_bleu(generados, [test_msgs], tokenize="none", force=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = float(np.mean([scorer.score(r, g)["rougeL"].fmeasure
                            for r, g in zip(test_msgs, generados)]))
    print(f"BLEU-4: {bleu.score:.2f} | ROUGE-L: {rougeL:.4f}", flush=True)

    (ART / "metricas_ptrgen_v2.json").write_text(json.dumps({
        "arquitectura": "pointer-generator (See et al. 2017), biGRU + Luong, desde cero",
        "poda": "estricta (cabeceras + lineas +/-, sin contexto; igual que el servicio)",
        "bleu4_test": round(float(bleu.score), 2),
        "rougeL_test": round(rougeL, 4),
        "val_loss": round(float(mejor_val), 4),
        "epocas": epoca_final,
        "torch_version": torch.__version__,
        "seed": SEED,
        "baseline_fase1": {"bleu4_test": 6.68, "rougeL_test": 0.2092},
        "baseline_ptrgen_v1": {"bleu4_test": 11.62, "rougeL_test": 0.2927},
    }, indent=2))
    print("Listo: modelo_resumidor_ptrgen_v2.pt + metricas_ptrgen_v2.json", flush=True)


if __name__ == "__main__":
    main()
