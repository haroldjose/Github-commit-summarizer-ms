"""Servicio de inferencia v2: Transformer + SentencePiece (CPU).

Las clases replican EXACTAMENTE la arquitectura y nombres de atributos del
notebook 02_resumidor_v2_transformer.ipynb — de lo contrario el state_dict no
carga. La limpieza del diff tambien debe ser identica a la del entrenamiento.
"""
import logging
import math
import re
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

PAD, UNK, SOS, EOS = 0, 1, 2, 3

# ── Limpieza: identica a la celda 4 del notebook v2 ──────────────────────────
RE_DIFF_HEADER = re.compile(
    r"\b(diff --git|index [0-9a-f]+\.\.[0-9a-f]+|new file mode \d+|"
    r"deleted file mode \d+|@@[^@]*@@)")
RE_FILEPATH = re.compile(
    r"[ab]/([\w./-]+\.(java|py|js|go|rb|php|xml|md|txt|json|yml|yaml|gradle|properties))")
RE_ESPACIOS = re.compile(r"\s+")


def limpiar_diff(d: str) -> str:
    archivos = list(dict.fromkeys(m.group(1) for m in RE_FILEPATH.finditer(d)))[:3]
    d = RE_DIFF_HEADER.sub(" ", d)
    d = RE_ESPACIOS.sub(" ", d).strip()
    prefijo = " ".join(f"<file> {a} </file>" for a in archivos)
    return (prefijo + " " + d).strip()


class PosEnc(nn.Module):
    def __init__(self, d: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ResumidorV2(nn.Module):
    def __init__(self, vocab: int, d_model: int, nhead: int, layers: int, ff: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = PosEnc(d_model)
        self.tf = nn.Transformer(d_model=d_model, nhead=nhead,
                                 num_encoder_layers=layers, num_decoder_layers=layers,
                                 dim_feedforward=ff, dropout=0.1, batch_first=True)
        self.salida = nn.Linear(d_model, vocab)
        self.salida.weight = self.emb.weight
        self.escala = math.sqrt(d_model)

    def forward(self, X, Y_in):
        mask_tgt = nn.Transformer.generate_square_subsequent_mask(Y_in.size(1)).to(X.device)
        out = self.tf(self.pos(self.emb(X) * self.escala),
                      self.pos(self.emb(Y_in) * self.escala),
                      tgt_mask=mask_tgt,
                      src_key_padding_mask=(X == PAD),
                      tgt_key_padding_mask=(Y_in == PAD),
                      memory_key_padding_mask=(X == PAD))
        return self.salida(out)


class SummarizerV2Service:
    def __init__(self, models_dir: str):
        base = Path(models_dir)
        self.sp = spm.SentencePieceProcessor(model_file=str(base / "spm_v2.model"))

        ckpt = torch.load(base / "modelo_resumidor_v2.pt", map_location="cpu")
        hp = ckpt["hiperparametros"]
        self.max_diff, self.max_msg = hp["MAX_DIFF"], hp["MAX_MSG"]

        self.modelo = ResumidorV2(hp["VOCAB"], hp["D_MODEL"], hp["NHEAD"],
                                  hp["LAYERS"], hp["FF"])
        self.modelo.load_state_dict(ckpt["modelo"])
        self.modelo.eval()
        log.info("Modelo resumidor V2 cargado (Transformer %d+%d, vocab %d)",
                 hp["LAYERS"], hp["LAYERS"], hp["VOCAB"])

    @torch.no_grad()
    def resumir(self, diff: str, beam: int = 5, alpha: float = 0.7) -> str:
        ids = self.sp.encode(limpiar_diff(diff))[: self.max_diff] or [UNK]
        X = torch.tensor([ids])
        haces = [(0.0, [SOS], False)]
        for _ in range(self.max_msg):
            candidatos = []
            for lp, toks, fin in haces:
                if fin:
                    candidatos.append((lp, toks, True))
                    continue
                Y_in = torch.tensor([toks])
                logits = self.modelo(X, Y_in)[0, -1]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                bigramas = {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}
                top_lp, top_ix = logprobs.topk(beam * 3)
                agregados = 0
                for k in range(top_ix.numel()):
                    if agregados >= beam:
                        break
                    t = int(top_ix[k])
                    if len(toks) >= 1 and (toks[-1], t) in bigramas:
                        continue
                    candidatos.append((lp + float(top_lp[k]), toks + [t], t == EOS))
                    agregados += 1
            haces = sorted(candidatos,
                           key=lambda c: c[0] / (max(len(c[1]) - 1, 1) ** alpha),
                           reverse=True)[:beam]
            if all(c[2] for c in haces):
                break
        mejores = [t for t in haces[0][1] if t not in (PAD, SOS, EOS, UNK)]
        texto = self.sp.decode(mejores)
        texto = re.sub(r"\s+", " ", texto).strip()
        while texto and not texto[-1].isalnum():
            texto = texto[:-1].rstrip()
        return texto
