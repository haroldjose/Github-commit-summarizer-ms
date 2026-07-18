"""Prepara el corpus multilenguaje (Fase 4): Java NNGen + 4 lenguajes de MCMD.

Fuente MCMD: splits procesados distribuidos con RACE (Shi et al., EMNLP 2022),
zenodo.org/record/7196966 — mismo formato pre-tokenizado que NNGen
(mmm/ppp, <nl>, puntuación separada). Muestreo con semilla 42.

Filtros por par: mensaje de 3-25 tokens, diff no vacío, sin duplicados.
A los diffs que exceden el presupuesto del modelo (100 tokens) se les aplica
la MISMA poda condicional del servicio (podar_diff), de modo que el modelo
entrena con la distribución exacta que verá en producción.

Salida: training/data/multilang/{split}.{diff,msg,lang}
"""
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from app.services.summarizer_service import podar_diff  # noqa: E402

SEED = 42
MAX_DIFF = 100
POR_LENGUAJE_TRAIN = 30_000
POR_LENGUAJE_EVAL = 2_500
LENGUAJES_MCMD = ["python", "javascript", "cpp", "csharp"]

DATA = BASE / "data"
MCMD = DATA / "mcmd" / "dataset"
SALIDA = DATA / "multilang"
SALIDA.mkdir(exist_ok=True)

rng = random.Random(SEED)


RUIDO_MSG = ("merge ", "merge!", "revert ", "rollback ", "update from ",
             "auto commit", "automatic commit")


def filtrar(diff: str, msg: str) -> str | None:
    """Devuelve el diff (podado si hace falta) o None si el par no sirve."""
    n_msg = len(msg.split())
    if not (3 <= n_msg <= 25) or not diff.strip():
        return None
    # ruido tipo merge/revert/bot (criterio de limpieza de Liu et al. 2018,
    # que NNGen aplica pero los splits de MCMD/RACE conservan)
    if msg.lower().startswith(RUIDO_MSG):
        return None
    if len(diff.split()) > MAX_DIFF:
        diff = podar_diff(diff, MAX_DIFF)
    return diff


def cargar_mcmd(lang: str, split: str, cupo: int) -> list[tuple[str, str]]:
    pares, vistos = [], set()
    with open(MCMD / lang / f"{split}.jsonl", encoding="utf-8") as f:
        for linea in f:
            r = json.loads(linea)
            diff, msg = r["diff"].strip(), r["msg"].strip()
            clave = hash((diff[:500], msg))
            if clave in vistos:
                continue
            diff = filtrar(diff, msg)
            if diff is None:
                continue
            vistos.add(clave)
            pares.append((diff, msg))
    rng.shuffle(pares)
    return pares[:cupo]


def cargar_nngen(split: str) -> list[tuple[str, str]]:
    with open(DATA / f"cleaned.{split}.diff", encoding="utf-8", errors="replace") as f:
        diffs = [l.rstrip("\n") for l in f]
    with open(DATA / f"cleaned.{split}.msg", encoding="utf-8", errors="replace") as f:
        msgs = [l.strip() for l in f]
    # el corpus NNGen ya está limpio y cabe en el presupuesto: se usa entero
    return list(zip(diffs, msgs))


def escribir(split: str, por_lengua: dict[str, list[tuple[str, str]]]):
    filas = []
    for lang, pares in por_lengua.items():
        filas.extend((lang, d, m) for d, m in pares)
    rng.shuffle(filas)
    with open(SALIDA / f"{split}.diff", "w", encoding="utf-8") as fd, \
         open(SALIDA / f"{split}.msg", "w", encoding="utf-8") as fm, \
         open(SALIDA / f"{split}.lang", "w", encoding="utf-8") as fl:
        for lang, d, m in filas:
            fd.write(d.replace("\n", " ") + "\n")
            fm.write(m.replace("\n", " ") + "\n")
            fl.write(lang + "\n")
    conteo = {lang: len(p) for lang, p in por_lengua.items()}
    print(f"{split}: {sum(conteo.values())} pares {conteo}", flush=True)


for split, cupo in (("train", POR_LENGUAJE_TRAIN), ("valid", POR_LENGUAJE_EVAL),
                    ("test", POR_LENGUAJE_EVAL)):
    por_lengua = {"java": cargar_nngen(split)}
    for lang in LENGUAJES_MCMD:
        por_lengua[lang] = cargar_mcmd(lang, split, cupo)
    escribir(split, por_lengua)

print("Listo: training/data/multilang/", flush=True)
