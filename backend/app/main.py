from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.flows.prediction import router as prediction_router
from app.flows.questionnaire import router as questionnaire_router
from app.flows.runtime import router as runtime_router
from app.flows.stt import router as stt_router
from app.flows.tts import router as tts_router
from app.settings import settings

app = FastAPI(title="Non-invasive Screening Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "backend"}


app.include_router(questionnaire_router)
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(prediction_router)
app.include_router(runtime_router)
