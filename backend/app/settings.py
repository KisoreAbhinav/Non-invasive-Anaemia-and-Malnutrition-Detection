from dataclasses import dataclass
from os import getenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_list(name: str, default: str) -> list[str]:
    return [value.strip() for value in getenv(name, default).split(",") if value.strip()]


def _env_path(name: str, default_relative_path: str) -> Path:
    configured_path = Path(getenv(name, default_relative_path))
    return configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path


def _env_float(name: str, default: str) -> float:
    return float(getenv(name, default))


@dataclass(frozen=True)
class Settings:
    cors_origins: list[str]
    app_runtime_mode: str
    questionnaire_path: Path
    prediction_models_path: Path
    visual_cues_path: Path
    clinical_fields_path: Path
    vision_models_path: Path
    stt_mode: str
    stt_provider: str
    vosk_models_path: Path
    tts_mode: str
    tts_provider: str
    piper_models_path: Path
    tts_length_scale: float
    prediction_default_model: str


settings = Settings(
    cors_origins=_env_list("BACKEND_CORS_ORIGINS", "http://localhost:5173,http://localhost:8080"),
    app_runtime_mode=getenv("APP_RUNTIME_MODE", "raspi-local"),
    questionnaire_path=_env_path("QUESTIONNAIRE_PATH", "config/questionnaire.default.json"),
    prediction_models_path=_env_path("PREDICTION_MODELS_PATH", "config/prediction-models.default.json"),
    visual_cues_path=_env_path("VISUAL_CUES_PATH", "config/visual-cues.default.json"),
    clinical_fields_path=_env_path("CLINICAL_FIELDS_PATH", "config/clinical-fields.default.json"),
    vision_models_path=PROJECT_ROOT / "models" / "vision",
    stt_mode=getenv("STT_MODE", "backend"),
    stt_provider=getenv("STT_PROVIDER", "mock"),
    vosk_models_path=PROJECT_ROOT / "models" / "vosk",
    tts_mode=getenv("TTS_MODE", "backend"),
    tts_provider=getenv("TTS_PROVIDER", "mock"),
    piper_models_path=PROJECT_ROOT / "models" / "piper",
    tts_length_scale=_env_float("TTS_LENGTH_SCALE", "1.2"),
    prediction_default_model=getenv("PREDICTION_DEFAULT_MODEL", "baseline-rule-v1"),
)
