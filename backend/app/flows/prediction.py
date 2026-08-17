from fastapi import APIRouter

from app.config_loader import read_json_object
from app.settings import settings

router = APIRouter(prefix="/api/flows/prediction", tags=["prediction"])


@router.get("/status")
def prediction_status() -> dict[str, str]:
    return {
        "flow": "prediction",
        "status": "scaffolded",
        "default_model": settings.prediction_default_model,
    }


@router.get("/models")
def prediction_models() -> dict[str, object]:
    return read_json_object(settings.prediction_models_path)
