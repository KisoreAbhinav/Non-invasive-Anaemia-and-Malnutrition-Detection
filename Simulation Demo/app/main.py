from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .classifier import predict, train
from .physics import TissueParameters, evaluate_pit, simulate_press
from .synthetic_data import make_record

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "synthetic_patients.jsonl"
MODEL_PATH = DATA_DIR / "classifier.json"

app = FastAPI(title="Pitting Edema Simulation Demo", version="0.1.0")


class SimulationRequest(BaseModel):
    mode: Literal["normal", "edema"] = "normal"
    force: float = Field(7.0, ge=0.2, le=16)
    duration: float = Field(3.0, ge=0.2, le=12)
    threshold_fraction: float = Field(0.10, ge=0.01, le=0.75)
    window_seconds: float = Field(15.0, ge=1, le=21)
    # Supplied only after a synthetic patient is generated. This makes the live
    # run use that patient's same parameter draw rather than a fresh preset.
    parameters: dict[str, float] | None = None


class GenerateRequest(BaseModel):
    # yes/no generates a class-specific batch; mixed keeps a balanced split.
    label: Literal["yes", "no", "mixed"] = "mixed"
    count: int = Field(20, ge=1, le=1000)
    force: float | None = Field(None, ge=0.2, le=16)
    duration: float | None = Field(None, ge=0.2, le=12)


def read_records() -> list[dict]:
    if not DATASET_PATH.exists():
        return []
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def classifier_model() -> dict | None:
    return json.loads(MODEL_PATH.read_text()) if MODEL_PATH.exists() else None


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "dataset_records": len(read_records()), "classifier_trained": MODEL_PATH.exists()}


@app.post("/api/simulate")
async def run_simulation(request: SimulationRequest) -> dict:
    try:
        parameters = TissueParameters(**request.parameters) if request.parameters else None
    except TypeError as error:
        raise HTTPException(422, "Invalid synthetic tissue parameters.") from error
    simulation = simulate_press(request.mode, request.force, request.duration, params=parameters)
    deterministic = evaluate_pit(simulation, request.threshold_fraction, request.window_seconds)
    model = classifier_model()
    image_classifier = (
        predict(model, simulation["series"], request.duration)
        if model
        else {"available": False, "message": "Train the image classifier with generated synthetic records first."}
    )
    return {"simulation": simulation, "deterministic": deterministic, "image_classifier": image_classifier}


@app.post("/api/dataset/generate")
async def generate_dataset(request: GenerateRequest) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    rng = Random()
    labels: list[bool]
    if request.label == "yes":
        labels = [True] * request.count
    elif request.label == "no":
        labels = [False] * request.count
    else:
        labels = [index % 2 == 0 for index in range(request.count)]
        rng.shuffle(labels)
    new_records = [make_record(label, rng, force=request.force, duration=request.duration) for label in labels]
    with DATASET_PATH.open("a", encoding="utf-8") as handle:
        for record in new_records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"generated": len(new_records), "total_records": len(read_records()), "records": new_records[-10:]}


@app.get("/api/dataset")
async def dataset(limit: int = Query(30, ge=1, le=100)) -> dict:
    records = read_records()
    return {"total": len(records), "records": list(reversed(records[-limit:]))}


@app.get("/api/dataset/export")
async def export_dataset() -> Response:
    if not DATASET_PATH.exists():
        raise HTTPException(404, "No synthetic dataset exists yet. Generate records first.")
    # A normal byte response avoids thread-backed file streaming and is suitable
    # for this intentionally small, locally generated demo dataset.
    return Response(
        DATASET_PATH.read_bytes(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="synthetic_patients.jsonl"'},
    )


@app.post("/api/classifier/train")
async def train_classifier() -> dict:
    records = read_records()
    try:
        model = train(records)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return {"trained": True, **{key: model[key] for key in ("training_samples", "training_accuracy")}}


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
