from pathlib import Path
from types import SimpleNamespace

from app.flows import stt, tts
from app.main import app
from tests.asgi_client import ASGITestClient


client = ASGITestClient(app)


def test_vosk_model_scan_requires_expected_structure(tmp_path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "am").mkdir()
    assert stt.find_vosk_model(tmp_path) is None

    valid = tmp_path / "vosk-model-en-in-test"
    valid.mkdir()
    for directory in ("am", "conf", "graph"):
        (valid / directory).mkdir()
    assert stt.find_vosk_model(tmp_path) == valid


def test_vosk_missing_model_returns_documented_503(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        stt,
        "settings",
        SimpleNamespace(stt_provider="vosk", stt_mode="backend", vosk_models_path=tmp_path),
    )
    monkeypatch.setattr(stt, "_model", None)
    monkeypatch.setattr(stt, "_model_path", None)
    monkeypatch.setattr(stt, "_load_error", None)

    status = client.get("/api/flows/stt/status")
    response = client.post("/api/flows/stt/transcribe", content=b"not a wav")

    assert status.status_code == 200
    assert status.json()["model"] == "not_found"
    assert response.status_code == 503
    assert response.json() == {"error": stt.NO_MODEL_ERROR}


def test_piper_voice_scan_needs_model_and_config(tmp_path) -> None:
    model = tmp_path / "voice.onnx"
    model.touch()
    assert tts.find_piper_voice(tmp_path) is None

    (tmp_path / "voice.onnx.json").touch()
    assert tts.find_piper_voice(tmp_path) == model


def test_piper_voice_scan_prefers_general_us_voice(tmp_path) -> None:
    fallback = tmp_path / "aaa-fallback.onnx"
    fallback.touch()
    (tmp_path / "aaa-fallback.onnx.json").touch()
    preferred = tmp_path / tts.DEFAULT_PIPER_VOICE
    preferred.touch()
    Path(f"{preferred}.json").touch()

    assert tts.find_piper_voice(tmp_path) == preferred


def test_piper_missing_voice_returns_clear_503(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        tts,
        "settings",
        SimpleNamespace(
            tts_provider="piper",
            tts_mode="backend",
            tts_length_scale=1.2,
            piper_models_path=tmp_path,
        ),
    )
    monkeypatch.setattr(tts, "_voice", None)
    monkeypatch.setattr(tts, "_voice_path", None)
    monkeypatch.setattr(tts, "_load_error", None)

    status = client.get("/api/flows/tts/status")
    response = client.post("/api/flows/tts/speak", json={"text": "Test question"})

    assert status.status_code == 200
    assert status.json()["model"] == "not_found"
    assert response.status_code == 503
    assert response.json() == {"error": tts.NO_MODEL_ERROR}
