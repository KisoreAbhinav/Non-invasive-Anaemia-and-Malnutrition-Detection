FROM python:3.13.5-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY scripts ./scripts
COPY models ./models

# TTS is auto-provisioned because Piper voices are compact and directly
# downloadable. Vosk remains a manual, plug-and-play model under models/vosk.
RUN .venv/bin/python scripts/setup_piper_voice.py --model-dir models/piper
RUN .venv/bin/python scripts/setup_semantic_model.py --model-dir models/sentence-transformers

COPY app ./app
COPY tests ./tests
COPY config ./config

EXPOSE 8000

CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
