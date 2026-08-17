from fastapi import APIRouter

from app.config_loader import read_json_object
from app.settings import settings

router = APIRouter(prefix="/api/flows/questionnaire", tags=["questionnaire"])


@router.get("/status")
def questionnaire_status() -> dict[str, object]:
    schema = read_json_object(settings.questionnaire_path)
    questions = schema.get("questions", [])
    return {
        "flow": "questionnaire",
        "status": "scaffolded",
        "source": str(settings.questionnaire_path),
        "question_count": len(questions) if isinstance(questions, list) else 0,
    }


@router.get("/schema")
def questionnaire_schema() -> dict[str, object]:
    return read_json_object(settings.questionnaire_path)
