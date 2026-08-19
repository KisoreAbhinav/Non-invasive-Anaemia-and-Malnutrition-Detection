"""HTTP session orchestration for the Stage 1 voice questionnaire."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from app.config_loader import read_json_object
from app.flows.answer_matcher import match_answer
from app.flows.questionnaire_engine import derive_population, next_question, score
from app.flows.screening import build_plan
from app.flows.stt import InvalidAudio, STTUnavailable, mock_transcription, transcribe_wav
from app.flows.tts import TTSUnavailable, synthesize_text
from app.settings import settings

router = APIRouter(prefix="/api/flows/questionnaire", tags=["questionnaire"])

_sessions: dict[str, dict[str, Any]] = {}
_sessions_lock = threading.RLock()


def _schema() -> dict[str, Any]:
    return read_json_object(settings.questionnaire_path)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _public_question(question: dict[str, Any] | None) -> dict[str, Any] | None:
    if question is None:
        return None
    public_keys = ("id", "type", "label", "required", "min", "max", "category")
    public = {key: question[key] for key in public_keys if key in question}
    if isinstance(question.get("options"), list):
        public["options"] = [
            {key: option[key] for key in ("label", "value") if key in option}
            for option in question["options"]
            if isinstance(option, dict)
        ]
    return public


def _get_session(session_id: str) -> dict[str, Any] | None:
    return _sessions.get(session_id)


def _question_for_session(
    schema: dict[str, Any], session: dict[str, Any]
) -> dict[str, Any] | None:
    current_id = session.get("current_question_id")
    if current_id is None:
        return None
    return next(
        (question for question in schema.get("questions", []) if question.get("id") == current_id),
        None,
    )


def _result(schema: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    population = session["population"]
    result: dict[str, Any] = {
        "population": population,
        "answers": dict(session["answers"]),
        "scores": score(schema, session["answers"], population),
        "next_stage": "non_invasive_tests",
    }
    exit_message = schema.get("exit_messages", {}).get(population)
    if exit_message:
        result["message"] = exit_message
        result["next_stage"] = "out_of_scope"
    elif population:
        plan = build_plan(population, result["answers"], result["scores"])
        result["visual_cues"] = plan["visual_cues"]
        result["clinical_fields"] = plan["clinical_fields"]
        result["dominant_category"] = plan["dominant_category"]
    return result


def _advance(schema: dict[str, Any], session: dict[str, Any]) -> None:
    session["population"] = derive_population(session["answers"])
    question = next_question(schema, session["answers"], session["population"])
    session["current_question_id"] = question["id"] if question else None
    session["repeat_requested"] = False
    session["updated_at"] = _now()
    if question is None:
        if session["population"] is None:
            raise ValueError("questionnaire ended before population could be derived")
        session["status"] = "complete"
        session["result"] = _result(schema, session)
    else:
        session["status"] = "active"
        session["result"] = None


def _session_view(schema: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    question = _question_for_session(schema, session)
    return {
        "session_id": session["id"],
        "status": session["status"],
        "population": session["population"],
        "answers": dict(session["answers"]),
        "question": _public_question(question),
        "repeat_requested": session["repeat_requested"],
        "last_match": session["last_match"],
        "result": session["result"],
        "sections": session["sections"],
        "audio": {
            "stt_provider": settings.stt_provider,
            "tts_provider": settings.tts_provider,
        },
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
    }


def _not_found(session_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": f"questionnaire session '{session_id}' was not found"},
    )


@router.get("/status")
async def questionnaire_status() -> dict[str, object]:
    schema = _schema()
    questions = schema.get("questions", [])
    return {
        "flow": "questionnaire",
        "status": "ready",
        "source": str(settings.questionnaire_path),
        "question_count": len(questions) if isinstance(questions, list) else 0,
        "active_sessions": sum(
            session["status"] == "active" for session in _sessions.values()
        ),
    }


@router.get("/schema")
async def questionnaire_schema() -> dict[str, object]:
    return _schema()


@router.post("/session")
async def start_session() -> dict[str, Any]:
    schema = _schema()
    created_at = _now()
    session: dict[str, Any] = {
        "id": str(uuid4()),
        "status": "active",
        "answers": {},
        "population": None,
        "current_question_id": None,
        "repeat_requested": False,
        "last_match": None,
        "history": [],
        "result": None,
        "sections": {"questionnaire": {"invalidated": False}},
        "created_at": created_at,
        "updated_at": created_at,
    }
    _advance(schema, session)
    with _sessions_lock:
        _sessions[session["id"]] = session
    return _session_view(schema, session)


@router.get("/session/{session_id}", response_model=None)
async def get_session(session_id: str) -> Response | dict[str, Any]:
    schema = _schema()
    with _sessions_lock:
        session = _get_session(session_id)
        if session is None:
            return _not_found(session_id)
        return _session_view(schema, session)


@router.post("/session/{session_id}/ask")
async def ask_current_question(session_id: str) -> Response:
    schema = _schema()
    with _sessions_lock:
        session = _get_session(session_id)
        if session is None:
            return _not_found(session_id)
        question = _question_for_session(schema, session)
        if question is None:
            return JSONResponse(
                status_code=409,
                content={"error": "the questionnaire session is already complete"},
            )
        question_id = question["id"]
        label = question["label"]

    if settings.tts_provider == "mock":
        return JSONResponse(
            {
                "provider": "mock",
                "question_id": question_id,
                "text": label,
            }
        )
    try:
        return Response(
            content=synthesize_text(label),
            media_type="audio/wav",
            headers={"X-Question-Id": question_id},
        )
    except TTSUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})


async def _transcribe_answer(request: Request) -> tuple[dict[str, Any], str | None]:
    body = await request.body()
    supplied_question_id = request.headers.get("x-question-id")
    # Physical input is already text and must bypass Vosk regardless of which
    # STT provider is active. Previously, JSON submitted by the touchscreen was
    # sent to transcribe_wav when STT_PROVIDER=vosk, causing the misleading
    # "audio must be a 16kHz... WAV" response.
    if "application/json" in request.headers.get("content-type", ""):
        payload = json.loads(body or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("physical answer body must be a JSON object")
        text = str(payload.get("text", ""))
        supplied_question_id = str(payload.get("question_id") or supplied_question_id or "") or None
        return mock_transcription(text), supplied_question_id
    if settings.stt_provider == "mock":
        return mock_transcription(body.decode("utf-8")), supplied_question_id
    return transcribe_wav(body), supplied_question_id


@router.post("/session/{session_id}/answer")
async def answer_current_question(session_id: str, request: Request) -> Response:
    schema = _schema()
    with _sessions_lock:
        session = _get_session(session_id)
        if session is None:
            return _not_found(session_id)
        question = _question_for_session(schema, session)
        if question is None:
            return JSONResponse(
                status_code=409,
                content={"error": "the questionnaire session is already complete"},
            )
        expected_question_id = question["id"]

    try:
        transcription, supplied_question_id = await _transcribe_answer(request)
    except STTUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except (InvalidAudio, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    if supplied_question_id and supplied_question_id != expected_question_id:
        return JSONResponse(
            status_code=409,
            content={
                "error": "the submitted answer does not match the current question",
                "expected_question_id": expected_question_id,
            },
        )

    matched = match_answer(question, transcription.get("text", ""))
    match_payload = matched.as_dict()
    match_payload["stt"] = transcription

    with _sessions_lock:
        session = _get_session(session_id)
        if session is None:
            return _not_found(session_id)
        if session["current_question_id"] != expected_question_id:
            return JSONResponse(
                status_code=409,
                content={"error": "the session advanced while this answer was processed"},
            )

        session["last_match"] = match_payload
        session["updated_at"] = _now()
        if matched.status == "unclear" or (
            matched.status == "command" and matched.value == "repeat"
        ):
            session["repeat_requested"] = True
            response = _session_view(schema, session)
            response["status"] = "repeat"
            response["message"] = matched.message or "Please repeat the answer."
            return JSONResponse(response)

        if matched.status == "command" and matched.value == "skip_section":
            population = derive_population(session["answers"])
            if population is None:
                return JSONResponse(
                    status_code=409,
                    content={"error": "Answer age and pregnancy routing before skipping."},
                )
            session["population"] = population
            session["current_question_id"] = None
            session["status"] = "complete"
            session["sections"]["questionnaire"]["invalidated"] = True
            session["result"] = _result(schema, session)
            session["result"]["invalidated"] = True
            response = _session_view(schema, session)
            response["matched"] = match_payload
            return JSONResponse(response)

        if matched.status != "matched":
            return JSONResponse(
                status_code=400,
                content={"error": "unsupported questionnaire command"},
            )

        session["answers"][expected_question_id] = matched.value
        session["history"].append(expected_question_id)
        _advance(schema, session)
        response = _session_view(schema, session)
        response["answered_question_id"] = expected_question_id
        response["matched"] = match_payload
        return JSONResponse(response)


@router.post("/session/{session_id}/retry")
async def retry_last_answer(session_id: str) -> Response:
    """Re-open the last answer after the UI displays its matched value."""

    schema = _schema()
    with _sessions_lock:
        session = _get_session(session_id)
        if session is None:
            return _not_found(session_id)
        if not session["history"]:
            return JSONResponse(
                status_code=409,
                content={"error": "there is no previous answer to retry"},
            )
        question_id = session["history"].pop()
        session["answers"].pop(question_id, None)
        session["population"] = derive_population(session["answers"])
        session["current_question_id"] = question_id
        session["status"] = "active"
        session["repeat_requested"] = True
        session["last_match"] = None
        session["result"] = None
        session["updated_at"] = _now()
        return JSONResponse(_session_view(schema, session))


@router.post("/session/{session_id}/skip")
async def skip_questionnaire(session_id: str) -> Response:
    """Deliberately invalidate the questionnaire while preserving partial answers."""

    schema = _schema()
    with _sessions_lock:
        session = _get_session(session_id)
        if session is None:
            return _not_found(session_id)
        population = derive_population(session["answers"])
        if population is None:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "age and pregnancy routing must be answered before the questionnaire can be skipped"
                },
            )
        session["population"] = population
        session["current_question_id"] = None
        session["status"] = "complete"
        session["sections"]["questionnaire"]["invalidated"] = True
        session["updated_at"] = _now()
        session["result"] = _result(schema, session)
        session["result"]["invalidated"] = True
        return JSONResponse(_session_view(schema, session))
