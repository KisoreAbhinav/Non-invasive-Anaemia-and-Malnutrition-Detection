#!/usr/bin/env python3
"""Run a real Piper -> Vosk questionnaire session through the ASGI API."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import wave
from math import gcd
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from scipy.signal import resample_poly

# Providers must be selected before importing the application settings.
os.environ["STT_PROVIDER"] = "vosk"
os.environ["TTS_PROVIDER"] = "piper"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.flows.stt import initialize_stt  # noqa: E402
from app.flows.tts import initialize_tts, synthesize_text  # noqa: E402
from app.main import app  # noqa: E402


def to_vosk_wav(source: bytes) -> bytes:
    """Convert Piper's mono PCM WAV to Vosk's required 16 kHz PCM WAV."""

    with wave.open(io.BytesIO(source), "rb") as input_wav:
        if input_wav.getnchannels() != 1 or input_wav.getsampwidth() != 2:
            raise ValueError("Piper produced an unexpected WAV format")
        source_rate = input_wav.getframerate()
        samples = np.frombuffer(input_wav.readframes(input_wav.getnframes()), dtype="<i2")

    divisor = gcd(source_rate, 16000)
    converted = resample_poly(
        samples.astype(np.float32),
        16000 // divisor,
        source_rate // divisor,
    )
    # Browser microphone recordings naturally include a little room silence.
    # Padding synthetic answers gives Vosk equivalent word boundaries.
    silence = np.zeros(8000, dtype=np.float32)
    padded = np.concatenate((silence, converted, silence))
    pcm = np.clip(np.rint(padded), -32768, 32767).astype("<i2").tobytes()

    output = io.BytesIO()
    with wave.open(output, "wb") as output_wav:
        output_wav.setnchannels(1)
        output_wav.setsampwidth(2)
        output_wav.setframerate(16000)
        output_wav.writeframes(pcm)
    return output.getvalue()


def answer_phrases(question_id: str) -> list[str]:
    if question_id == "age_years":
        return ["four years old", "the age is four years", "four"]
    if question_id == "age_months":
        return ["forty eight months", "the age is forty eight months", "forty eight"]
    return ["no", "no not at all", "the answer is no"]


async def verify() -> dict[str, Any]:
    initialize_stt()
    initialize_tts()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health = await client.get("/api/health")
        health.raise_for_status()

        stt_status = await client.get("/api/flows/stt/status")
        tts_status = await client.get("/api/flows/tts/status")
        stt_status.raise_for_status()
        tts_status.raise_for_status()
        if stt_status.json()["status"] != "ready" or tts_status.json()["status"] != "ready":
            raise RuntimeError("real audio providers are not ready")

        session_response = await client.post("/api/flows/questionnaire/session")
        session_response.raise_for_status()
        session = session_response.json()
        session_id = session["session_id"]
        transcript_log: list[dict[str, str]] = []

        while session["status"] != "complete":
            question = session["question"]
            question_id = question["id"]

            for phrase in answer_phrases(question_id):
                spoken_question = await client.post(
                    f"/api/flows/questionnaire/session/{session_id}/ask"
                )
                spoken_question.raise_for_status()
                if spoken_question.headers.get("content-type") != "audio/wav":
                    raise RuntimeError(f"question {question_id} did not produce WAV audio")

                answer_wav = to_vosk_wav(synthesize_text(phrase))
                answer_response = await client.post(
                    f"/api/flows/questionnaire/session/{session_id}/answer",
                    content=answer_wav,
                    headers={
                        "Content-Type": "audio/wav",
                        "X-Question-Id": question_id,
                    },
                )
                answer_response.raise_for_status()
                session = answer_response.json()
                matched = session.get("matched") or session.get("last_match") or {}
                transcript = matched.get("stt", {}).get("text", "")
                transcript_log.append(
                    {
                        "question_id": question_id,
                        "spoken": phrase,
                        "transcribed": transcript,
                        "matched": str(matched.get("value", "")),
                    }
                )
                if session["status"] != "repeat":
                    break
            else:
                raise RuntimeError(
                    f"Vosk could not confidently match {question_id} after three attempts"
                )

        return {
            "health": health.json(),
            "stt": stt_status.json()["status"],
            "tts": tts_status.json()["status"],
            "questions_answered": len(session["result"]["answers"]),
            "audio_attempts": len(transcript_log),
            "transcripts": transcript_log,
            "result": session["result"],
        }


def main() -> int:
    try:
        print(json.dumps(asyncio.run(verify()), indent=2))
    except Exception as exc:
        print(f"Stage 1 workflow verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
