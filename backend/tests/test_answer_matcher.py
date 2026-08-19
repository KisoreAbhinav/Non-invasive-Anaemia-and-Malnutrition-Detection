import json
from pathlib import Path
from typing import Any

import pytest

from app.flows import answer_matcher
from app.flows.answer_matcher import match_answer, parse_spoken_number


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "config" / "questionnaire.default.json"
    return json.loads(path.read_text(encoding="utf-8"))


def question(schema: dict[str, Any], question_id: str) -> dict[str, Any]:
    return next(item for item in schema["questions"] if item["id"] == question_id)


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("twenty four", 24),
        ("I am ten years old", 10),
        ("my age is 32", 32),
        ("one hundred and five", 105),
        ("four years ago", 4),
        ("for years old", 4),
    ],
)
def test_spoken_number_parser(spoken: str, expected: int) -> None:
    assert parse_spoken_number(spoken) == expected


def test_number_question_skips_lexical_matcher(schema: dict[str, Any]) -> None:
    result = match_answer(question(schema, "age_years"), "twenty four")
    assert result.status == "matched"
    assert result.value == 24
    assert result.method == "number"


@pytest.mark.parametrize("spoken", ["haan", "haan ji", "ji haan", "yeah"])
def test_indian_english_affirmatives(schema: dict[str, Any], spoken: str) -> None:
    result = match_answer(question(schema, "is_pregnant"), spoken)
    assert result.status == "matched"
    assert result.value == "yes"
    assert result.method == "fuzzy"


@pytest.mark.parametrize("spoken", ["nahi ji", "no, not really", "nah", "noooo"])
def test_indian_english_negatives(schema: dict[str, Any], spoken: str) -> None:
    result = match_answer(question(schema, "is_pregnant"), spoken)
    assert result.status == "matched"
    assert result.value == "no"


def test_question_specific_phrasing(schema: dict[str, Any]) -> None:
    result = match_answer(
        question(schema, "child_anemia_recent_infection"),
        "haan ji, there were loose motions",
    )
    assert result.status == "matched"
    assert result.value == "yes"


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [("he is not tired at all", "no"), ("the child has no stamina", "yes")],
)
def test_fatigue_negation_and_idiom(
    schema: dict[str, Any], spoken: str, expected: str
) -> None:
    result = match_answer(question(schema, "child_anemia_fatigue"), spoken)
    assert result.status == "matched"
    assert result.value == expected


def test_repeat_fixed_intent(schema: dict[str, Any]) -> None:
    result = match_answer(question(schema, "child_anemia_pica"), "please repeat")
    assert result.status == "command"
    assert result.value == "repeat"


def test_skip_section_fixed_intent(schema: dict[str, Any]) -> None:
    result = match_answer(question(schema, "child_anemia_pica"), "skip section")
    assert result.status == "command"
    assert result.value == "skip_section"


def test_unclear_does_not_guess(
    schema: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(answer_matcher._SEMANTIC_MATCHER, "rank", lambda *_: [])
    result = match_answer(question(schema, "child_anemia_pica"), "maybe perhaps")
    assert result.status == "unclear"
    assert result.value is None


def test_semantic_fallback_can_select_an_option(
    schema: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        answer_matcher._SEMANTIC_MATCHER,
        "rank",
        lambda *_: [(0.78, "yes"), (0.31, "no")],
    )
    result = match_answer(
        question(schema, "pregnant_anemia_symptoms"),
        "I become winded whenever I climb the stairs",
    )
    assert result.status == "matched"
    assert result.value == "yes"
    assert result.method == "semantic"
