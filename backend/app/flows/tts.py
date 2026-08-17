from fastapi import APIRouter

from app.settings import settings

router = APIRouter(prefix="/api/flows/tts", tags=["tts"])


@router.get("/status")
def tts_status() -> dict[str, str]:
    return {
        "flow": "tts",
        "status": "scaffolded",
        "mode": settings.tts_mode,
        "provider": settings.tts_provider,
    }
