FROM python:3.13.5-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.4.28 /uv /usr/local/bin/uv

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY app ./app
COPY tests ./tests
COPY config ./config

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
