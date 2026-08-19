from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.flows.prediction import router as prediction_router
from app.flows.questionnaire import router as questionnaire_router
from app.flows.runtime import router as runtime_router
from app.flows.screening import router as screening_router
from app.flows.model_registry import initialize_registry
from app.flows.stt import initialize_stt, router as stt_router
from app.flows.tts import initialize_tts, router as tts_router
from app.settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Real audio providers are expensive to load, so each is initialized once
    # and then shared by its per-request recognizer/synthesizer operations.
    initialize_stt()
    initialize_tts()
    initialize_registry()
    yield


app = FastAPI(
    title="Non-invasive Screening Backend",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "backend"}


app.include_router(questionnaire_router)
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(prediction_router)
app.include_router(runtime_router)
app.include_router(screening_router)
