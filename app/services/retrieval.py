"""Recuperación de mensajes al estilo NNGen (Liu et al., ASE 2018).

Sin red neuronal ni modelos pre-entrenados: ante un diff nuevo, busca los k
diffs más parecidos del corpus de entrenamiento (bolsa de palabras + coseno),
re-rankea por BLEU-4 entre diffs y devuelve el mensaje del vecino ganador.
Réplica del algoritmo de referencia (github.com/Tbabm/nngen): CountVectorizer
con defaults (tf crudo, minúsculas, tokens \\w\\w+), coseno, top-5, re-rank
con sentence-BLEU. En este corpus NNGen reporta BLEU ~38 frente a ~16 del
NMT grande — la recuperación es el estado del arte sin pre-entrenamiento.
"""
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# mismo token_pattern por defecto de CountVectorizer: palabras de 2+ chars
_PALABRA_RE = re.compile(r"\b\w\w+\b")


def _bleu4(referencia: list[str], hipotesis: list[str]) -> float:
    """BLEU-4 de oración con suavizado epsilon (para re-rank de candidatos)."""
    if not hipotesis or not referencia:
        return 0.0
    log_p = 0.0
    for n in range(1, 5):
        ref_ngr = Counter(tuple(referencia[i:i + n])
                          for i in range(len(referencia) - n + 1))
        hip_ngr = Counter(tuple(hipotesis[i:i + n])
                          for i in range(len(hipotesis) - n + 1))
        total = sum(hip_ngr.values())
        if total == 0:
            return 0.0
        aciertos = sum(min(c, ref_ngr[g]) for g, c in hip_ngr.items())
        log_p += math.log((aciertos + 1e-9) / total)
    bp = min(1.0, math.exp(1 - len(referencia) / len(hipotesis)))
    return bp * math.exp(log_p / 4)


class RecuperadorNNGen:
    def __init__(self, diffs_path: Path, msgs_path: Path, k: int = 5):
        self.k = k
        with open(diffs_path, encoding="utf-8", errors="replace") as f:
            self.diffs = [linea.rstrip("\n") for linea in f]
        with open(msgs_path, encoding="utf-8", errors="replace") as f:
            self.msgs = [linea.strip() for linea in f]
        if len(self.diffs) != len(self.msgs):
            raise ValueError("corpus de recuperación no paralelo")

        # Índice invertido: token -> (ids de doc, frecuencias) como arrays.
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        normas = np.zeros(len(self.diffs), dtype=np.float32)
        for i, d in enumerate(self.diffs):
            tf = Counter(_PALABRA_RE.findall(d.lower()))
            for tok, n in tf.items():
                postings[tok].append((i, n))
            normas[i] = math.sqrt(sum(n * n for n in tf.values()))
        self._postings = {
            tok: (np.fromiter((i for i, _ in pares), dtype=np.int32, count=len(pares)),
                  np.fromiter((n for _, n in pares), dtype=np.float32, count=len(pares)))
            for tok, pares in postings.items()
        }
        self._normas = np.maximum(normas, 1e-9)
        log.info("Índice NNGen: %d pares, %d tokens", len(self.diffs), len(self._postings))

    def consultar(self, diff_normalizado: str) -> tuple[str, float, float]:
        """Devuelve (mensaje, coseno top-1, BLEU-4 del vecino re-rankeado).

        El BLEU entre diffs es la señal de calidad real de NNGen: el coseno
        se infla con tokens estructurales (mmm/ppp/<nl>) en diffs fuera de
        distribución, así que el servicio debe gatear por ambos.
        """
        tf_q = Counter(_PALABRA_RE.findall(diff_normalizado.lower()))
        if not tf_q:
            return "", 0.0, 0.0
        puntajes = np.zeros(len(self.diffs), dtype=np.float32)
        for tok, n in tf_q.items():
            entrada = self._postings.get(tok)
            if entrada is not None:
                ids, tfs = entrada
                puntajes[ids] += n * tfs
        cos = puntajes / (self._normas * math.sqrt(sum(n * n for n in tf_q.values())))

        k = min(self.k, len(cos))
        top = np.argpartition(cos, -k)[-k:]
        top = top[np.argsort(cos[top])[::-1]]

        # Re-rank NNGen: BLEU-4 entre el diff de consulta y cada candidato.
        toks_q = diff_normalizado.split()
        mejor, mejor_bleu = int(top[0]), -1.0
        for i in top:
            b = _bleu4(self.diffs[int(i)].split(), toks_q)
            if b > mejor_bleu:
                mejor, mejor_bleu = int(i), b
        return self.msgs[mejor], float(cos[int(top[0])]), float(mejor_bleu)
