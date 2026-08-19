"""Offline text-to-speech flow backed by one cached Piper voice."""

from __future__ import annotations

import io
import logging
import threading
import wave
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from app.settings import settings

router = APIRouter(prefix="/api/flows/tts", tags=["tts"])
LOGGER = logging.getLogger(__name__)

NO_MODEL_ERROR = (
    "no Piper voice found in backend/models/piper — "
    "run backend/scripts/setup_piper_voice.py and restart"
)
DEFAULT_PIPER_VOICE = "en_US-lessac-medium.onnx"

_voice: Any = None
_voice_path: Path | None = None
_load_error: str | None = None
_load_lock = threading.Lock()
_synthesis_lock = threading.Lock()


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class TTSUnavailable(RuntimeError):
    pass


def find_piper_voice(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    preferred = root / DEFAULT_PIPER_VOICE
    if preferred.is_file() and Path(f"{preferred}.json").is_file():
        return preferred
    for model_path in sorted(root.rglob("*.onnx")):
        if Path(f"{model_path}.json").is_file():
            return model_path
    return None


def initialize_tts() -> None:
    """Load a configured Piper voice once at application startup."""

    global _load_error, _voice, _voice_path

    if settings.tts_provider != "piper" or _voice is not None:
        return
    with _load_lock:
        if _voice is not None:
            return
        discovered = find_piper_voice(settings.piper_models_path)
        if discovered is None:
            _voice_path = None
            _load_error = NO_MODEL_ERROR
            LOGGER.warning(NO_MODEL_ERROR)
            return
        try:
            from piper import PiperVoice

            LOGGER.info("Loading Piper TTS voice from %s", discovered)
            _voice = PiperVoice.load(str(discovered))
            _voice_path = discovered
            _load_error = None
        except Exception as exc:  # pragma: no cover - requires a real Piper voice
            _voice_path = discovered
            _load_error = f"failed to load Piper voice at {discovered}: {exc}"
            LOGGER.exception("Unable to load Piper TTS voice")


def _ensure_piper_voice() -> Any:
    if settings.tts_provider != "piper":
        raise TTSUnavailable(f"unsupported TTS provider: {settings.tts_provider}")
    if _voice is None:
        initialize_tts()
    if _voice is None:
        raise TTSUnavailable(_load_error or NO_MODEL_ERROR)
    return _voice


def synthesize_text(text: str) -> bytes:
    """Synthesize slowed, mono WAV audio using the cached Piper voice."""

    voice = _ensure_piper_voice()
    from piper import SynthesisConfig

    synthesis_config = SynthesisConfig(
        length_scale=settings.tts_length_scale,
    )
    output = io.BytesIO()
    with _synthesis_lock:
        with wave.open(output, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=synthesis_config)
    return output.getvalue()


@router.get("/status")
async def tts_status() -> dict[str, object]:
    if settings.tts_provider == "piper" and _voice is None:
        initialize_tts()
    if settings.tts_provider == "mock":
        status, model = "ready", "not_required"
    elif _voice is not None:
        status, model = "ready", str(_voice_path)
    else:
        status, model = "unavailable", "not_found" if _voice_path is None else "load_failed"
    return {
        "flow": "tts",
        "status": status,
        "mode": settings.tts_mode,
        "provider": settings.tts_provider,
        "model": model,
        "length_scale": settings.tts_length_scale,
        "error": _load_error,
    }


@router.post("/speak")
async def speak(payload: SpeakRequest) -> Response:
    if settings.tts_provider == "mock":
        return JSONResponse(
            {
                "provider": "mock",
                "text": payload.text,
                "length_scale": settings.tts_length_scale,
            }
        )
    try:
        return Response(
            content=synthesize_text(payload.text),
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="question.wav"'},
        )
    except TTSUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
