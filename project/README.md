# Non-invasive-Anaemia-and-Malnutrition-Detection

Non-invasive anaemia and malnutrition risk screening on local edge devices.

## Complete screening flow

- `backend/`: FastAPI with independent flow modules
  - `questionnaire`
  - `stt`
  - `tts`
  - `prediction`
  - `runtime` (central runtime/audio config surface)
- `frontend/`: Vite React kiosk UI with microphone recording, camera/file
  capture, answer confirmation, and model/fused-score results
- `docker-compose.yml`: local run for both services

The application includes an 800×480 kiosk UI, continuous silence-detected listening,
adaptive non-invasive test planning, risk-aware clinical fields, explicit skip
state, result fusion, and spoken verdicts. MUAC is entered as a physical tape
measurement, while child weight, height, age, and sex are converted offline to
WHO 2006/2007 growth Z-scores. Anaemia screening captures conjunctiva, nail-bed,
and palm images, runs three local TorchScript models, and displays per-site plus
combined probabilities. Untrained edema and hair/skin tests remain unavailable
instead of returning placeholder predictions.

The questionnaire is config-first and derives one of four populations before
walking only the eligible questions: `child_under5`, `child_5_12`,
`pregnant_woman`, or `adult_nonpregnant`. Every population continues through
all four stages unless the operator explicitly skips a section. Anemia and
malnutrition are scored independently using the weights and cutoffs in the
repository's `References.md` section 10.

## Python setup (UV + 3.13.5)

- Python version is pinned with `backend/.python-version` to **3.13.5**
- Backend dependency management uses **uv** via `backend/pyproject.toml`
- The three vision weights use **Git LFS**; after installing Git LFS, run
  `git lfs install && git lfs pull` once from the repository root

Local backend run:

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Local frontend run in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

The Vite development server is available at `http://localhost:5173` and proxies
API requests to the backend on port 8000.

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
- `POST /api/flows/screening/vision/{test_id}?population={population}`
- `POST /api/flows/screening/result`

Questionnaire session API:

- `POST /api/flows/questionnaire/session`
- `POST /api/flows/questionnaire/session/{id}/ask`
- `POST /api/flows/questionnaire/session/{id}/answer`
- `POST /api/flows/questionnaire/session/{id}/retry`
- `GET /api/flows/questionnaire/session/{id}`
- `POST /api/flows/questionnaire/session/{id}/skip`

## Vision inference

The repository includes three trained CPU TorchScript exports:

- `backend/models/vision/pallor_eye/`
- `backend/models/vision/pallor_nail/`
- `backend/models/vision/pallor_palm/`

Each image is resized with shorter-side semantics, center-cropped to 224×224,
converted from sRGB to standards-based D65 CIE Lab, and normalized using its
`model.json` contract. The pallor endpoint requires all three multipart fields,
runs the site-specific models, and reports the unweighted mean risk probability.
The UI supports sequential camera capture, retake, and file upload fallback.

Run a pipeline smoke test after setup:

```bash
cd backend
uv run python scripts/verify_vision_inference.py
```

For an operator check with labeled images:

```bash
uv run python scripts/verify_vision_inference.py \
  --eye /path/to/eye.jpg \
  --nail /path/to/nail.jpg \
  --palm /path/to/palm.jpg
```

The generated-image default verifies decoding, preprocessing, model loading,
multipart routing, aggregation, and response contracts. It does not validate
clinical accuracy.

### Drop-in model contract

Each model-backed non-invasive test has a folder under
`backend/models/vision/`. Install a CPU TorchScript model by adding `model.pt`
and a matching `model.json`, then restart. The registry reports incomplete or
missing pairs as unavailable and the flow skips them without failing. Physical
MUAC and WHO growth measurements bypass the registry. The full metadata
contract and example are in `backend/models/vision/README.md`.

### Clinical limitations

These outputs are screening signals, not diagnoses or hemoglobin measurements.
The eye model's recorded validation accuracy is 0.626. Nail and palm validation
scores are higher, but their image-level split may contain multiple images from
the same subject across train and validation sets. Independent subject-level,
device-specific, and population-specific validation is still required before
clinical use. The UI and final API retain the screening-only disclaimer.

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

Large speech assets remain excluded from Git. A clean Docker build downloads
the Indian-English Vosk model, Piper voice, and semantic matcher into the
container image. The three trained pallor artifacts are versioned through Git
LFS; other vision tests remain optional drop-ins and unavailable tests are
reported honestly.

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

Run the complete automated backend regression suite and frontend build with:

```bash
cd backend
uv run pytest

cd ../frontend
npm run build
```

## Raspberry Pi 5

The stack uses multi-arch base images (`python:3.13.5-slim`, `node:20-alpine`,
`nginx:alpine`) and CPU-only PyTorch packages for local ARM64 deployment on a
Raspberry Pi 5. Vosk and Piper models are cached in memory rather than loaded
per request; the semantic matcher is lazy to preserve memory on ordinary
high-confidence answers.
