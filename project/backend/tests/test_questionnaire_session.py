from app.flows import answer_matcher
from app.main import app
from tests.asgi_client import ASGITestClient


client = ASGITestClient(app)


def start() -> dict:
    response = client.post("/api/flows/questionnaire/session")
    assert response.status_code == 200
    return response.json()


def answer(session_id: str, text: str, question_id: str) -> dict:
    response = client.post(
        f"/api/flows/questionnaire/session/{session_id}/answer",
        json={"text": text, "question_id": question_id},
    )
    assert response.status_code == 200
    return response.json()


def test_age_ten_skips_pregnancy_and_routes_to_child_track() -> None:
    session = start()
    updated = answer(session["session_id"], "ten", "age_years")

    assert updated["population"] == "child_5_12"
    assert updated["question"]["id"] == "child_anemia_pica"
    assert "is_pregnant" not in updated["answers"]


def test_age_twenty_four_then_haan_routes_to_pregnancy_track() -> None:
    session = start()
    age_answer = answer(session["session_id"], "twenty four", "age_years")
    assert age_answer["question"]["id"] == "is_pregnant"

    pregnant = answer(session["session_id"], "haan", "is_pregnant")
    assert pregnant["population"] == "pregnant_woman"
    assert pregnant["question"]["id"] == "pregnant_anemia_previous"


def test_nonpregnant_adult_enters_full_questionnaire_and_screening_flow() -> None:
    session = start()
    age_answer = answer(session["session_id"], "24", "age_years")
    state = answer(session["session_id"], "nahi", age_answer["question"]["id"])

    assert state["question"]["id"] == "adult_sex"
    while state["status"] != "complete":
        spoken = "female" if state["question"]["id"] == "adult_sex" else "no"
        state = answer(state["session_id"], spoken, state["question"]["id"])

    assert state["result"]["population"] == "adult_nonpregnant"
    assert state["result"]["next_stage"] == "non_invasive_tests"
    assert state["result"]["visual_cues"]
    assert state["result"]["clinical_fields"]
    assert state["result"]["scores"]["anemia"]["band"] == "low"


def test_unclear_answer_reasks_instead_of_advancing(monkeypatch) -> None:
    monkeypatch.setattr(answer_matcher._SEMANTIC_MATCHER, "rank", lambda *_: [])
    session = start()
    child = answer(session["session_id"], "ten", "age_years")
    repeated = answer(session["session_id"], "perhaps maybe", child["question"]["id"])

    assert repeated["status"] == "repeat"
    assert repeated["repeat_requested"] is True
    assert repeated["question"]["id"] == "child_anemia_pica"
    assert "child_anemia_pica" not in repeated["answers"]


def test_spoken_skip_section_finishes_questionnaire_without_skipping_later_stages() -> None:
    session = start()
    child = answer(session["session_id"], "ten", "age_years")
    skipped = answer(session["session_id"], "skip section", child["question"]["id"])

    assert skipped["status"] == "complete"
    assert skipped["result"]["invalidated"] is True
    assert skipped["result"]["next_stage"] == "non_invasive_tests"
    assert skipped["result"]["visual_cues"]
    assert skipped["result"]["clinical_fields"]


def test_retry_reopens_last_answer() -> None:
    session = start()
    child = answer(session["session_id"], "ten", "age_years")
    matched = answer(session["session_id"], "no", child["question"]["id"])
    assert matched["question"]["id"] != "child_anemia_pica"

    response = client.post(f"/api/flows/questionnaire/session/{session['session_id']}/retry")
    assert response.status_code == 200
    retried = response.json()
    assert retried["question"]["id"] == "child_anemia_pica"
    assert "child_anemia_pica" not in retried["answers"]


def test_mock_ask_and_stt_endpoints() -> None:
    session = start()
    ask_response = client.post(
        f"/api/flows/questionnaire/session/{session['session_id']}/ask"
    )
    assert ask_response.status_code == 200
    assert ask_response.json()["text"] == session["question"]["label"]

    stt_response = client.post("/api/flows/stt/transcribe", json={"text": "haan ji"})
    assert stt_response.status_code == 200
    assert stt_response.json()["text"] == "haan ji"


def test_complete_child_flow_returns_both_low_risk_scores() -> None:
    state = start()
    state = answer(state["session_id"], "ten", state["question"]["id"])
    while state["status"] != "complete":
        state = answer(state["session_id"], "no", state["question"]["id"])

    assert state["result"]["population"] == "child_5_12"
    assert state["result"]["scores"] == {
        "anemia": {"raw": 0.0, "band": "low"},
        "malnutrition": {"raw": 0.0, "band": "low"},
    }
    assert state["result"]["next_stage"] == "non_invasive_tests"


def test_physical_json_answer_bypasses_real_audio_decoder(monkeypatch) -> None:
    """Touch/keyboard input remains text even when the configured STT is Vosk."""

    from dataclasses import replace

    from app.flows import questionnaire

    session = start()
    monkeypatch.setattr(
        questionnaire,
        "settings",
        replace(questionnaire.settings, stt_provider="vosk"),
    )
    response = client.post(
        f"/api/flows/questionnaire/session/{session['session_id']}/answer",
        json={"text": "10", "question_id": "age_years"},
    )
    assert response.status_code == 200
    assert response.json()["population"] == "child_5_12"
