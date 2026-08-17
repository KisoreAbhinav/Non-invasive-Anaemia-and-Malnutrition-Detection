from dataclasses import dataclass
from os import getenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_list(name: str, default: str) -> list[str]:
    return [value.strip() for value in getenv(name, default).split(",") if value.strip()]


def _env_path(name: str, default_relative_path: str) -> Path:
    configured_path = Path(getenv(name, default_relative_path))
    return configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path


@dataclass(frozen=True)
class Settings:
    cors_origins: list[str]
    app_runtime_mode: str
    questionnaire_path: Path
    prediction_models_path: Path
    stt_mode: str
    stt_provider: str
    tts_mode: str
    tts_provider: str
    prediction_default_model: str


settings = Settings(
    cors_origins=_env_list("BACKEND_CORS_ORIGINS", "http://localhost:5173,http://localhost:8080"),
    app_runtime_mode=getenv("APP_RUNTIME_MODE", "raspi-local"),
    questionnaire_path=_env_path("QUESTIONNAIRE_PATH", "config/questionnaire.default.json"),
    prediction_models_path=_env_path("PREDICTION_MODELS_PATH", "config/prediction-models.default.json"),
    stt_mode=getenv("STT_MODE", "backend"),
    stt_provider=getenv("STT_PROVIDER", "mock"),
    tts_mode=getenv("TTS_MODE", "backend"),
    tts_provider=getenv("TTS_PROVIDER", "mock"),
    prediction_default_model=getenv("PREDICTION_DEFAULT_MODEL", "baseline-rule-v1"),
)
