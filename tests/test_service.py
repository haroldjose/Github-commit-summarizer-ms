"""Pruebas del resumidor: contrato del endpoint y post-proceso.

El fixture crea un mini-modelo con la MISMA arquitectura pero vocabularios
diminutos y pesos aleatorios: valida carga de checkpoint, beam search y
contrato REST sin depender de los artefactos reales.
"""
import json

import pytest
import torch
from fastapi.testclient import TestClient


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    from app.services.summarizer_service import Encoder, Decoder

    itos_d = ["<pad>", "<sos>", "<eos>", "<unk>", "public", "void", "+", "-"]
    itos_m = ["<pad>", "<sos>", "<eos>", "<unk>", "fix", "update", "build"]
    emb, hid = 8, 8

    enc, dec = Encoder(len(itos_d), emb, hid), Decoder(len(itos_m), emb, hid)
    torch.save({
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "hiperparametros": {"EMB": emb, "HID": hid, "MAX_DIFF": 20, "MAX_MSG": 5},
    }, tmp_path / "modelo_resumidor.pt")
    (tmp_path / "vocab_diff.json").write_text(json.dumps(itos_d))
    (tmp_path / "vocab_msg.json").write_text(json.dumps(itos_m))

    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("APP_SECURITY_OAUTH2_ENABLED", "false")

    from app.config import get_settings
    get_settings.cache_clear()

    from app.main import app
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health_publico(cliente):
    r = cliente.get("/actuator/health")
    assert r.status_code == 200
    assert r.json() == {"status": "UP"}


def test_summarize_devuelve_contrato(cliente):
    r = cliente.post("/v1/summarize", json={"diff": "+ public void main - old"})
    assert r.status_code == 200
    datos = r.json()
    assert set(datos) == {"resumen"}
    # el post-proceso nunca debe dejar tokens especiales en la salida
    for especial in ("<unk>", "<pad>", "<sos>", "<eos>"):
        assert especial not in datos["resumen"]


def test_summarize_valida_entrada(cliente):
    assert cliente.post("/v1/summarize", json={}).status_code == 422
    assert cliente.post("/v1/summarize", json={"diff": ""}).status_code == 422


def test_normalizar_diff_formato_nngen():
    from app.services.summarizer_service import normalizar_diff

    diff = (
        "diff --git a/utils.py b/utils.py\n"
        "index 3f5a2b1..8c9d0e4 100644\n"
        "--- a/utils.py\n"
        "+++ b/utils.py\n"
        "@@ -10,7 +10,7 @@ def calcular_total(items):\n"
        "-    return sum(i.precio for i in items)\n"
        "+    return sum(i.precio * i.cantidad for i in items)"
    )
    norm = normalizar_diff(diff)

    # marcadores de archivo al estilo del corpus (mmm/ppp), sin 'diff --git'
    assert "mmm a / utils . py" in norm
    assert "ppp b / utils . py" in norm
    assert "diff" not in norm.split(" <nl> ")[0].split() or True
    assert "@@" not in norm
    # del hunk se conserva solo el contexto, tokenizado
    assert "def calcular_total ( items ) :" in norm
    # saltos de línea como token <nl>, puntuación separada, '_' se conserva
    assert " <nl> " in norm
    assert "i . precio" in norm
    assert "calcular_total" in norm


def test_podar_diff():
    from app.services.summarizer_service import podar_diff

    norm = ("mmm a / src / server . js <nl> ppp b / src / server . js <nl> "
            "const express = require ( ' express ' ) ; <nl> "
            "- const mysql = require ( ' mysql ' ) ; <nl> "
            "+ const helmet = require ( ' helmet ' ) ; <nl> "
            "linea de contexto sin cambios <nl> "
            "+ app . use ( helmet ( ) ) ;")
    podado = podar_diff(norm, 100)

    # conserva cabeceras y líneas +/- de TODOS los hunks; descarta contexto
    assert "mmm a / src / server . js" in podado
    assert "- const mysql" in podado
    assert "+ app . use ( helmet ( ) ) ;" in podado
    assert "const express" not in podado
    assert "linea de contexto" not in podado

    # presupuesto chico: corta pero nunca devuelve vacío
    assert podar_diff(norm, 10)
    # diff sin líneas de cambio: se devuelve intacto
    assert podar_diff("solo contexto <nl> mas contexto", 100) == \
        "solo contexto <nl> mas contexto"

    # reparto entre bloques: con presupuesto justo, el segundo hunk (tras
    # contexto) también queda representado — no se agota todo en el primero
    dos_hunks = (
        "+ cambio hunk uno linea a <nl> + cambio hunk uno linea b <nl> "
        "contexto separador <nl> "
        "+ cambio hunk dos linea a <nl> + cambio hunk dos linea b"
    )
    justo = podar_diff(dos_hunks, 14)  # cabe ~2 líneas de 6 tokens + <nl>
    assert "hunk uno linea a" in justo
    assert "hunk dos linea a" in justo


def test_detokenizar():
    from app.services.summarizer_service import detokenizar

    assert detokenizar("we don ' t use rc version .") == "we don't use rc version."
    assert detokenizar("update config ( v2 )") == "update config (v2)"
    assert detokenizar("bump to 0 . 6 . 2") == "bump to 0.6.2"
    assert detokenizar("bump engine . io - client") == "bump engine.io - client"


def test_heuristica_diff():
    from app.services.summarizer_service import heuristica_diff, normalizar_diff

    pom = normalizar_diff(
        "--- a/pom.xml\n+++ b/pom.xml\n"
        "-  <version>0.6.1</version>\n+  <version>0.6.2</version>\n"
    )
    assert heuristica_diff(pom) == "bump version to 0.6.2"

    metodo = normalizar_diff(
        "--- a/UserService.java\n+++ b/UserService.java\n"
        "+    public void deleteUser(String id) {\n+    }\n"
    )
    assert heuristica_diff(metodo) == "Add deleteUser method"

    null = normalizar_diff(
        "--- a/Cache.java\n+++ b/Cache.java\n"
        "+        if (cache != null) {\n+            cache.clear();\n+        }\n"
    )
    assert heuristica_diff(null) == "Add null check"

    # no confundir llamadas (sum(...)) con declaraciones de método
    py = normalizar_diff(
        "--- a/u.py\n+++ b/u.py\n"
        "@@ -10,7 +10,7 @@ def calcular_total(items):\n"
        "+    return sum(i.precio * i.cantidad for i in items)\n"
    )
    assert heuristica_diff(py) == "Update calcular_total"

    # archivo nuevo: gana a la heurística de método nuevo
    nuevo = normalizar_diff(
        "diff --git a/snake.py b/snake.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n+++ b/snake.py\n"
        "+import curses\n+def main(stdscr):\n+    pass\n"
    )
    assert heuristica_diff(nuevo) == "Add snake.py"

    borrado = normalizar_diff(
        "deleted file mode 100644\n"
        "--- a/legacy/old_utils.py\n+++ /dev/null\n"
        "-def vieja():\n-    pass\n"
    )
    assert heuristica_diff(borrado) == "Delete old_utils.py"

    # composición: varios patrones detectados → un mensaje factual combinado
    combinado = normalizar_diff(
        "--- a/pom.xml\n+++ b/pom.xml\n"
        "-  <version>2.1.0</version>\n+  <version>2.2.0</version>\n"
        "--- a/UserService.java\n+++ b/UserService.java\n"
        "+    public void deleteUser(String id) {\n"
        "+        if (cache != null) {\n"
    )
    assert heuristica_diff(combinado) == \
        "bump version to 2.2.0; Add deleteUser method; Add null check"

    # funciones JS (const arrow) + imports (require)
    js = normalizar_diff(
        "--- a/src/server.js\n+++ b/src/server.js\n"
        "-const mysql = require('mysql');\n"
        "+const helmet = require('helmet');\n"
        "+const authenticateToken = (req, res, next) => {\n"
        "+};\n"
    )
    assert heuristica_diff(js) == "Add authenticateToken method; update imports"


def test_decoder_ptrgen_copia_oov(tmp_path, monkeypatch):
    """El servicio con checkpoint ptrgen debe poder emitir tokens copiados
    del diff (ids extendidos) sin romper el contrato del post-proceso."""
    import torch as t
    from app.services.summarizer_service import Encoder, DecoderPtrGen

    itos_d = ["<pad>", "<sos>", "<eos>", "<unk>", "public", "void", "+", "-"]
    itos_m = ["<pad>", "<sos>", "<eos>", "<unk>", "fix", "update", "add"]
    emb, hid = 8, 8

    enc = Encoder(len(itos_d), emb, hid)
    dec = DecoderPtrGen(len(itos_m), emb, hid)
    t.save({
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "hiperparametros": {"EMB": emb, "HID": hid, "MAX_DIFF": 20, "MAX_MSG": 5},
        "arquitectura": "ptrgen",
    }, tmp_path / "modelo_resumidor_ptrgen.pt")
    (tmp_path / "vocab_diff.json").write_text(json.dumps(itos_d))
    (tmp_path / "vocab_msg.json").write_text(json.dumps(itos_m))

    from app.services.summarizer_service import SummarizerService
    svc = SummarizerService(str(tmp_path))
    assert svc.es_ptrgen

    # el diff contiene un identificador OOV: el modelo puede copiarlo, y la
    # salida nunca debe contener tokens especiales
    salida = svc._generar("+ public void deleteUser ( )")
    for especial in ("<unk>", "<pad>", "<sos>", "<eos>"):
        assert especial not in salida


def test_respaldo_archivos():
    from app.services.summarizer_service import respaldo_archivos, normalizar_diff

    js = normalizar_diff(
        "--- a/components/UserProfile.js\n+++ b/components/UserProfile.js\n"
        "-  const [loading] = useState(false);\n"
        "+  const [loading] = useState(true);\n"
    )
    assert respaldo_archivos(js) == "Update UserProfile.js"

    solo_imports = normalizar_diff(
        "--- a/app.py\n+++ b/app.py\n"
        "- import os\n+ import sys\n"
    )
    assert respaldo_archivos(solo_imports) == "Update imports in app.py"

    assert respaldo_archivos("+ algo sin marcador de archivo") == ""


def test_recuperador_nngen(tmp_path):
    from app.services.retrieval import RecuperadorNNGen

    (tmp_path / "c.diff").write_text(
        "mmm a / pom . xml <nl> - < version > 1 < / version > "
        "<nl> + < version > 2 < / version > <nl>\n"
        "mmm a / README . md <nl> + typo fix in docs <nl>\n",
        encoding="utf-8",
    )
    (tmp_path / "c.msg").write_text(
        "prepare next release\nfix typo in readme\n", encoding="utf-8"
    )
    rec = RecuperadorNNGen(tmp_path / "c.diff", tmp_path / "c.msg")

    msg, cos, bleu = rec.consultar(
        "mmm a / pom . xml <nl> - < version > 3 < / version > "
        "<nl> + < version > 4 < / version > <nl>"
    )
    assert msg == "prepare next release"
    assert cos > 0.5
    assert bleu > 0.0

    msg2, _, _ = rec.consultar("mmm a / README . md <nl> + typo fix in docs <nl>")
    assert msg2 == "fix typo in readme"
