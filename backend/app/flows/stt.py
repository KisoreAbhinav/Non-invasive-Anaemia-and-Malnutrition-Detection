from fastapi import APIRouter

from app.settings import settings

router = APIRouter(prefix="/api/flows/stt", tags=["stt"])


@router.get("/status")
def stt_status() -> dict[str, str]:
    return {
        "flow": "stt",
        "status": "scaffolded",
        "mode": settings.stt_mode,
        "provider": settings.stt_provider,
    }
