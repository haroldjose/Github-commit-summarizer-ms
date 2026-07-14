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

log = logging.getLogger(__name__)

PAD, SOS, EOS, UNK = 0, 1, 2, 3


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


class SummarizerService:
    def __init__(self, models_dir: str):
        base = Path(models_dir)
        self.itos_d = json.loads((base / "vocab_diff.json").read_text())
        self.itos_m = json.loads((base / "vocab_msg.json").read_text())
        self.stoi_d = {w: i for i, w in enumerate(self.itos_d)}

        ckpt = torch.load(base / "modelo_resumidor.pt", map_location="cpu")
        hp = ckpt["hiperparametros"]
        self.max_diff, self.max_msg = hp["MAX_DIFF"], hp["MAX_MSG"]

        self.encoder = Encoder(len(self.itos_d), hp["EMB"], hp["HID"])
        self.decoder = Decoder(len(self.itos_m), hp["EMB"], hp["HID"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.encoder.eval()
        self.decoder.eval()
        log.info("Modelo resumidor cargado (%d/%d tokens de vocabulario)",
                 len(self.itos_d), len(self.itos_m))

    @torch.no_grad()
    def resumir(self, diff: str, beam: int = 5) -> str:
        ids = [self.stoi_d.get(t, UNK) for t in diff.split()[: self.max_diff]] or [UNK]
        X = torch.tensor([ids])
        enc_out, h = self.encoder(X)
        mascara = X == PAD

        haces = [(0.0, [], h, False)]
        for _ in range(self.max_msg):
            candidatos = []
            for lp, toks, hh, fin in haces:
                if fin:
                    candidatos.append((lp, toks, hh, True))
                    continue
                ultimo = SOS if not toks else toks[-1]
                entrada = torch.full((1, 1), ultimo, dtype=torch.long)
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
                    if toks and (toks[-1], tok) in bigramas:
                        continue
                    candidatos.append((lp + float(top_lp[k]), toks + [tok], h2, tok == EOS))
                    agregados += 1
            haces = sorted(candidatos, key=lambda c: c[0] / max(len(c[1]), 1),
                           reverse=True)[:beam]
            if all(c[3] for c in haces):
                break

        # Post-proceso (contrato del Prompt 1): filtrar tokens especiales y <unk>
        palabras = [self.itos_m[t] for t in haces[0][1]
                    if t not in (PAD, SOS, EOS, UNK)]
        texto = " ".join(palabras)
        # limpieza cosmetica: parentesis vacios y puntuacion colgante al final
        texto = re.sub(r"\(\s*\)", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        while texto and not texto[-1].isalnum():
            texto = texto[:-1].rstrip()
        return texto
