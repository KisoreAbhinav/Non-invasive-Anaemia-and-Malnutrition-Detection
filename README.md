# Non-invasive-Anaemia-and-Malnutrition-Detection

Non-invasive anaemia and malnutrition detection in children and pregnant women on local edge devices.

## Clean hackathon scaffold

- `backend/`: FastAPI with independent flow modules
  - `questionnaire`
  - `stt`
  - `tts`
  - `prediction`
  - `runtime` (central runtime/audio config surface)
- `frontend/`: Vite React UI scaffold (questionnaire + placeholder criticality score)
- `docker-compose.yml`: local run for both services

No full product logic is baked in yet; this is intentionally modular so each flow can evolve independently for experiments and paper comparisons.

## Python setup (UV + 3.13.5)

- Python version is pinned with `backend/.python-version` to **3.13.5**
- Backend dependency management uses **uv** via `backend/pyproject.toml`

Local backend run:

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Config-first architecture

- Questionnaire is external JSON: `backend/config/questionnaire.default.json`
- Prediction model catalog is external JSON: `backend/config/prediction-models.default.json`
- Runtime/env config template: `backend/.env.example`

Key env vars:

- `APP_RUNTIME_MODE` (default: `raspi-local`)
- `STT_MODE` / `TTS_MODE` (use `backend` for local Pi audio execution)
- `STT_PROVIDER` / `TTS_PROVIDER`
- `QUESTIONNAIRE_PATH`
- `PREDICTION_MODELS_PATH`
- `PREDICTION_DEFAULT_MODEL`

API surfaces for harness integration:

- `GET /api/runtime/config`
- `GET /api/flows/questionnaire/schema`
- `GET /api/flows/questionnaire/status`
- `GET /api/flows/stt/status`
- `GET /api/flows/tts/status`
- `GET /api/flows/prediction/status`
- `GET /api/flows/prediction/models`

## Docker run (recommended for Pi/dev parity)

```bash
docker compose up --build
```

- Frontend: `http://localhost:8080`
- Backend health: `http://localhost:8000/api/health`

Stop:

```bash
docker compose down
```

## Raspberry Pi 5

The stack uses multi-arch base images (`python:3.13.5-slim`, `node:20-alpine`, `nginx:alpine`) and is suitable for local ARM64 deployment on Raspberry Pi 5.
