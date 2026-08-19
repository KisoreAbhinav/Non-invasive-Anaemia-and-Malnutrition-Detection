"""Offline speech-to-text flow backed by one cached Vosk model."""

from __future__ import annotations

import io
import json
import logging
import threading
import wave
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.settings import settings

router = APIRouter(prefix="/api/flows/stt", tags=["stt"])
LOGGER = logging.getLogger(__name__)

NO_MODEL_ERROR = (
    "no STT model found in backend/models/vosk — "
    "drop a Vosk model folder there and restart"
)
WAV_REQUIREMENT = "audio must be a 16kHz mono 16-bit PCM WAV file"

_model: Any = None
_model_path: Path | None = None
_load_error: str | None = None
_load_lock = threading.Lock()


class STTUnavailable(RuntimeError):
    pass


class InvalidAudio(ValueError):
    pass


def find_vosk_model(root: Path) -> Path | None:
    """Find the first extracted folder with the standard Vosk structure."""

    if not root.is_dir():
        return None
    for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
        if all((candidate / required).is_dir() for required in ("am", "conf", "graph")):
            return candidate
    return None


def initialize_stt() -> None:
    """Scan and load once at application startup when Vosk is selected."""

    global _load_error, _model, _model_path

    if settings.stt_provider != "vosk" or _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        discovered = find_vosk_model(settings.vosk_models_path)
        if discovered is None:
            _model_path = None
            _load_error = NO_MODEL_ERROR
            LOGGER.warning(NO_MODEL_ERROR)
            return
        try:
            from vosk import Model

            LOGGER.info("Loading Vosk STT model from %s", discovered)
            _model = Model(str(discovered))
            _model_path = discovered
            _load_error = None
        except Exception as exc:  # pragma: no cover - requires a real Vosk model
            _model_path = discovered
            _load_error = f"failed to load Vosk model at {discovered}: {exc}"
            LOGGER.exception("Unable to load Vosk STT model")


def _ensure_vosk_model() -> Any:
    if settings.stt_provider != "vosk":
        raise STTUnavailable(f"unsupported STT provider: {settings.stt_provider}")
    if _model is None:
        initialize_stt()
    if _model is None:
        raise STTUnavailable(_load_error or NO_MODEL_ERROR)
    return _model


def _json_result(raw: str) -> dict[str, Any]:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def transcribe_wav(audio: bytes) -> dict[str, Any]:
    """Transcribe a complete 16 kHz mono PCM WAV payload with Vosk."""

    model = _ensure_vosk_model()
    try:
        wav = wave.open(io.BytesIO(audio), "rb")
    except (EOFError, wave.Error) as exc:
        raise InvalidAudio(WAV_REQUIREMENT) from exc

    with wav:
        if (
            wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getframerate() != 16000
            or wav.getcomptype() != "NONE"
        ):
            raise InvalidAudio(WAV_REQUIREMENT)

        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)
        complete_parts: list[str] = []
        words: list[dict[str, Any]] = []
        last_partial = ""

        while chunk := wav.readframes(4000):
            if recognizer.AcceptWaveform(chunk):
                result = _json_result(recognizer.Result())
                if result.get("text"):
                    complete_parts.append(str(result["text"]))
                if isinstance(result.get("result"), list):
                    words.extend(result["result"])
            else:
                partial = _json_result(recognizer.PartialResult())
                last_partial = str(partial.get("partial", last_partial))

        final = _json_result(recognizer.FinalResult())
        if final.get("text"):
            complete_parts.append(str(final["text"]))
        if isinstance(final.get("result"), list):
            words.extend(final["result"])

    confidences = [float(word["conf"]) for word in words if "conf" in word]
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
    return {
        "text": " ".join(part for part in complete_parts if part).strip(),
        "confidence": confidence,
        "partial": last_partial,
        "result": words,
        "provider": "vosk",
    }


def mock_transcription(text: str) -> dict[str, Any]:
    return {
        "text": text.strip(),
        "confidence": 1.0,
        "partial": "",
        "result": [],
        "provider": "mock",
    }


@router.get("/status")
async def stt_status() -> dict[str, object]:
    if settings.stt_provider == "vosk" and _model is None:
        initialize_stt()
    if settings.stt_provider == "mock":
        status, model = "ready", "not_required"
    elif _model is not None:
        status, model = "ready", str(_model_path)
    else:
        status, model = "unavailable", "not_found" if _model_path is None else "load_failed"
    return {
        "flow": "stt",
        "status": status,
        "mode": settings.stt_mode,
        "provider": settings.stt_provider,
        "model": model,
        "requirement": WAV_REQUIREMENT,
        "error": _load_error,
    }


@router.post("/transcribe")
async def transcribe(request: Request) -> JSONResponse:
    body = await request.body()
    try:
        if settings.stt_provider == "mock":
            if "application/json" in request.headers.get("content-type", ""):
                payload = json.loads(body or b"{}")
                text = payload.get("text", "") if isinstance(payload, dict) else ""
            else:
                text = body.decode("utf-8")
            result = mock_transcription(str(text))
        else:
            result = transcribe_wav(body)
        from app.flows.answer_matcher import parse_spoken_number

        result["number"] = parse_spoken_number(str(result.get("text", "")))
        return JSONResponse(result)
    except STTUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except (InvalidAudio, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
