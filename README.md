# Non-invasive-Anaemia-and-Malnutrition-Detection

Non-invasive anaemia and malnutrition risk screening on local edge devices.

## Stage 1 + Stage 2 screening flow

- `backend/`: FastAPI with independent flow modules
  - `questionnaire`
  - `stt`
  - `tts`
  - `prediction`
  - `runtime` (central runtime/audio config surface)
- `frontend/`: Vite React session UI with microphone recording, local WAV
  conversion, answer confirmation, and score results
- `docker-compose.yml`: local run for both services

Stage 2 adds an 800×480 kiosk UI, continuous silence-detected listening,
adaptive non-invasive test planning, risk-aware clinical fields, explicit skip
state, result fusion, and spoken verdicts. MUAC is entered as a physical tape
measurement, while child weight, height, age, and sex are converted offline to
WHO 2006/2007 growth Z-scores. Camera/vision test execution remains an honest
placeholder until its models are installed.

The questionnaire is config-first and derives one of four populations before
walking only the eligible questions: `child_under5`, `child_5_12`,
`pregnant_woman`, or `adult_nonpregnant`. Every population continues through
all four stages unless the operator explicitly skips a section. Anemia and
malnutrition are scored independently using the weights and cutoffs in the
repository's `References.md` section 10.

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
- Visual test ordering/mappings: `backend/config/visual-cues.default.json`
- Risk-aware clinical fields: `backend/config/clinical-fields.default.json`
- Runtime/env config template: `backend/.env.example`

Key env vars:

- `APP_RUNTIME_MODE` (default: `raspi-local`)
- `STT_MODE` / `TTS_MODE` (use `backend` for local Pi audio execution)
- `STT_PROVIDER=mock|vosk`
- `TTS_PROVIDER=mock|piper`
- `TTS_LENGTH_SCALE` (default `1.2`; values above 1 slow Piper speech)
- `QUESTIONNAIRE_PATH`
- `PREDICTION_MODELS_PATH`
- `VISUAL_CUES_PATH`
- `CLINICAL_FIELDS_PATH`
- `PREDICTION_DEFAULT_MODEL`

API surfaces for harness integration:

- `GET /api/runtime/config`
- `GET /api/flows/questionnaire/schema`
- `GET /api/flows/questionnaire/status`
- `GET /api/flows/stt/status`
- `GET /api/flows/tts/status`
- `GET /api/flows/prediction/status`
- `GET /api/flows/prediction/models`
- `GET /api/flows/screening/status`
- `POST /api/flows/screening/plan`
- `POST /api/flows/screening/result`

Questionnaire session API:

- `POST /api/flows/questionnaire/session`
- `POST /api/flows/questionnaire/session/{id}/ask`
- `POST /api/flows/questionnaire/session/{id}/answer`
- `POST /api/flows/questionnaire/session/{id}/retry`
- `GET /api/flows/questionnaire/session/{id}`
- `POST /api/flows/questionnaire/session/{id}/skip`

## Vision models (drop-in)

Each model-backed non-invasive test has a folder under
`backend/models/vision/`. Install a CPU TorchScript model by adding `model.pt`
and a matching `model.json`, then restart. The registry reports incomplete or
missing pairs as unavailable and the flow skips them without failing. Physical
MUAC and WHO growth measurements bypass the registry. The full metadata
contract and example are in `backend/models/vision/README.md`.

## Speech models

### Vosk STT (manual, plug-and-play)

Download and extract a Vosk Indian-English model so its directory is directly
under `backend/models/vosk/`, for example:

```text
backend/models/vosk/vosk-model-en-in-0.5/
├── am/
├── conf/
└── graph/
```

Then set `STT_PROVIDER=vosk` and restart. The model is scanned and loaded once
at startup. If it is absent, `/api/flows/stt/status` reports
`"model": "not_found"` and transcription returns a clear 503 instead of
crashing. Vosk input must be a **16 kHz, mono, 16-bit PCM WAV**. The browser UI
decodes microphone recordings, mixes them to mono, resamples to 16 kHz, and
encodes that WAV format before upload.

### Piper TTS (automatic)

Piper setup is automatic during the backend image build and can also be run
manually:

```bash
cd backend
uv run python scripts/setup_piper_voice.py
```

The setup uses Piper's official `en_US-lessac-medium` single-speaker voice for
clear, general US English and downloads both the `.onnx` file and its
`.onnx.json` configuration. Unlike the much larger, deployment-specific Vosk
model, this compact fixed TTS asset is safe to provision automatically. Set
`TTS_PROVIDER=piper` to enable it.

### Answer matching

Select answers use high-confidence RapidFuzz matching over labels and
Indian-English/Hinglish synonyms first. Inconclusive text falls back to the
small `all-MiniLM-L6-v2` sentence-transformer, loaded lazily only when needed.
Docker caches this model during the backend image build. For a local offline
run, cache and verify it once with:

```bash
cd backend
uv run python scripts/setup_semantic_model.py
```

Number questions use a local spoken-number parser. If no match is confident,
the current question is repeated rather than guessed.

## Docker run (recommended for Pi/dev parity)

```bash
docker compose up --build
```

Compose is configured for the installed real offline providers:
`STT_PROVIDER: vosk` and `TTS_PROVIDER: piper`. To run the adaptive flow
without speech inference, temporarily change both values to `mock`.

Runtime model binaries are intentionally excluded from Git. A clean Docker
build downloads the lightweight Indian-English Vosk model, the Piper voice,
and the semantic matcher into the container image. Vision models remain
optional drop-ins and unavailable tests are reported honestly.

- Frontend: `http://localhost:8080`
- Backend health: `http://localhost:8000/api/health`

Stop:

```bash
docker compose down
```

To verify the complete real-audio Stage 1 path locally (Piper questions and
answers, browser-equivalent 16 kHz conversion, Vosk recognition, adaptive
branching, and final scoring), run:

```bash
cd backend
uv run python scripts/verify_stage1_workflow.py
```

## Raspberry Pi 5

The stack uses multi-arch base images (`python:3.13.5-slim`, `node:20-alpine`,
`nginx:alpine`) and CPU-only PyTorch packages for local ARM64 deployment on a
Raspberry Pi 5. Vosk and Piper models are cached in memory rather than loaded
per request; the semantic matcher is lazy to preserve memory on ordinary
high-confidence answers.
