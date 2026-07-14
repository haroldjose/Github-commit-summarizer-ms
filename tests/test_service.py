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
