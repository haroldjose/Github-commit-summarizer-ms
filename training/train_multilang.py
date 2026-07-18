"""Fase 4: pointer-generator multilenguaje entrenado DESDE CERO.

Corpus mixto (training/data/multilang, ver prepara_multilang.py):
Java (NNGen) + Python/JavaScript/C++/C# (MCMD via RACE). Vocabularios
reconstruidos sobre la mezcla. Misma arquitectura de la Fase 3 (las clases
se importan del servicio: una sola definición).

Artefactos → training/artefactos_v3/ (directorio drop-in completo para el
servicio: vocabs + modelo + corpus de recuperación). NO toca los artefactos
en producción; la promoción se decide tras evaluar por lenguaje.

Uso:  python training/train_multilang.py
"""
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from app.services.summarizer_service import (  # noqa: E402
    Encoder, DecoderPtrGen, PAD, SOS, EOS, UNK)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA = BASE / "data" / "multilang"
ART = BASE / "artefactos_v3"
ART.mkdir(exist_ok=True)

MAX_DIFF, MAX_MSG = 100, 25
EMB, HID = 256, 256
EPOCAS, PACIENCIA = 25, 3
BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESPECIALES = ["<pad>", "<sos>", "<eos>", "<unk>"]


def cargar(split):
    with open(DATA / f"{split}.diff", encoding="utf-8") as f:
        diffs = [l.rstrip("\n") for l in f]
    with open(DATA / f"{split}.msg", encoding="utf-8") as f:
        msgs = [l.strip() for l in f]
    with open(DATA / f"{split}.lang", encoding="utf-8") as f:
        langs = [l.strip() for l in f]
    assert len(diffs) == len(msgs) == len(langs)
    return diffs, msgs, langs


train_diffs, train_msgs, _ = cargar("train")
val_diffs, val_msgs, _ = cargar("valid")


def construir_vocab(textos, max_tam, min_frec=2):
    c = Counter(tok for t in textos for tok in t.split())
    palabras = [w for w, n in c.most_common(max_tam - len(ESPECIALES))
                if n >= min_frec]
    itos = ESPECIALES + palabras
    return {w: i for i, w in enumerate(itos)}, itos


stoi_d, itos_d = construir_vocab(train_diffs, 50_000)
stoi_m, itos_m = construir_vocab(train_msgs, 20_000)
Vd, Vm = len(itos_d), len(itos_m)


def preparar_ejemplo(diff: str, msg: str):
    toks_x = diff.split()[:MAX_DIFF] or ["<unk>"]
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
    M = torch.zeros((len(xs), lx), dtype=torch.long)
    Yi = torch.full((len(xs), ly), PAD, dtype=torch.long)
    Yo = torch.full((len(xs), ly), PAD, dtype=torch.long)
    for i, (x, m, _, yi, yo) in enumerate(lote):
        X[i, :len(x)] = torch.tensor(x)
        M[i, :len(m)] = torch.tensor(m)
        Yi[i, :len(yi)] = torch.tensor(yi)
        Yo[i, :len(yo)] = torch.tensor(yo)
    return X, M, Yi, Yo, n_oov


def perdida_lote(encoder, decoder, X, M, Yi, Yo, n_oov):
    X, M, Yi, Yo = (t.to(DEVICE) for t in (X, M, Yi, Yo))
    enc_out, h = encoder(X)
    mascara = X == PAD
    dist, _ = decoder(Yi, h, enc_out, mascara, M, n_oov)
    probs = dist.gather(2, Yo.unsqueeze(-1)).squeeze(-1)
    nll = -torch.log(probs + 1e-9)
    activo = Yo != PAD
    return (nll * activo).sum() / activo.sum()


def main():
    print(f"Dispositivo: {DEVICE} | torch {torch.__version__}", flush=True)
    print(f"train {len(train_diffs)} | valid {len(val_diffs)} | "
          f"vocab diff {Vd} | vocab msg {Vm}", flush=True)

    dl_train = DataLoader(ParesDataset(train_diffs, train_msgs), batch_size=BATCH,
                          shuffle=True, collate_fn=colacionar)
    dl_val = DataLoader(ParesDataset(val_diffs, val_msgs), batch_size=BATCH,
                        shuffle=False, collate_fn=colacionar)

    encoder = Encoder(Vd, EMB, HID).to(DEVICE)
    decoder = DecoderPtrGen(Vm, EMB, HID).to(DEVICE)
    params = sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
    print(f"Parametros: {params / 1e6:.1f}M (multilenguaje, desde cero)", flush=True)

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
            }, ART / "modelo_resumidor_ptrgen.pt")
            marca = " *guardado*"
        else:
            sin_mejora += 1
        epoca_final = ep
        print(f"epoca {ep:02d} | train {np.mean(perdidas):.3f} | val {val:.3f} "
              f"| {time.time() - t0:.0f}s{marca}", flush=True)
        if sin_mejora >= PACIENCIA:
            print(f"Parada temprana tras {PACIENCIA} epocas sin mejora.", flush=True)
            break

    # ── Artefactos drop-in ───────────────────────────────────────────────────
    (ART / "vocab_diff.json").write_text(json.dumps(itos_d))
    (ART / "vocab_msg.json").write_text(json.dumps(itos_m))
    shutil.copy(DATA / "train.diff", ART / "corpus_train.diff")
    shutil.copy(DATA / "train.msg", ART / "corpus_train.msg")

    # ── Evaluación por lenguaje ──────────────────────────────────────────────
    print("Evaluando en test por lenguaje (esto tarda)...", flush=True)
    ckpt = torch.load(ART / "modelo_resumidor_ptrgen.pt", map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    test_diffs, test_msgs, test_langs = cargar("test")

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
                if ultimo >= Vm:
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

    generados = []
    t0 = time.time()
    for i, d in enumerate(test_diffs):
        generados.append(resumir(d))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(test_diffs)} ({time.time() - t0:.0f}s)", flush=True)

    por_lengua = {}
    for lang in sorted(set(test_langs)):
        idx = [i for i, l in enumerate(test_langs) if l == lang]
        b = sacrebleu.corpus_bleu([generados[i] for i in idx],
                                  [[test_msgs[i] for i in idx]],
                                  tokenize="none", force=True)
        por_lengua[lang] = round(float(b.score), 2)
        print(f"  BLEU-4 {lang}: {b.score:.2f} ({len(idx)} pares)", flush=True)
    global_bleu = sacrebleu.corpus_bleu(generados, [test_msgs],
                                        tokenize="none", force=True)
    print(f"BLEU-4 global: {global_bleu.score:.2f}", flush=True)

    (ART / "metricas_multilang.json").write_text(json.dumps({
        "arquitectura": "pointer-generator multilenguaje (desde cero)",
        "corpus": "Java: NNGen 22k | Python/JS/C++/C#: MCMD (RACE) 30k c/u, semilla 42",
        "bleu4_test_global": round(float(global_bleu.score), 2),
        "bleu4_test_por_lenguaje": por_lengua,
        "val_loss": round(float(mejor_val), 4),
        "epocas": epoca_final,
        "torch_version": torch.__version__,
        "seed": SEED,
        "baseline_java_ptrgen_v1": {"bleu4_test": 11.62},
    }, indent=2))
    print("Listo: training/artefactos_v3/", flush=True)


if __name__ == "__main__":
    main()
