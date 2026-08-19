from fastapi import APIRouter

from app.settings import settings

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


@router.get("/config")
async def runtime_config() -> dict[str, object]:
    return {
        "runtime_mode": settings.app_runtime_mode,
        "audio_execution": {
            "stt_mode": settings.stt_mode,
            "stt_provider": settings.stt_provider,
            "tts_mode": settings.tts_mode,
            "tts_provider": settings.tts_provider,
            "tts_length_scale": settings.tts_length_scale,
        },
        "questionnaire_path": str(settings.questionnaire_path),
        "prediction_models_path": str(settings.prediction_models_path),
        "prediction_default_model": settings.prediction_default_model,
        "visual_cues_path": str(settings.visual_cues_path),
        "clinical_fields_path": str(settings.clinical_fields_path),
        "vision_models_path": str(settings.vision_models_path),
    }
