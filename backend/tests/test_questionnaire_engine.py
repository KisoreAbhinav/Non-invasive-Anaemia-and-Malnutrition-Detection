import json
from pathlib import Path
from typing import Any

import pytest

from app.flows.questionnaire_engine import (
    condition_matches,
    derive_population,
    next_question,
    question_applies,
    score,
)


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "config" / "questionnaire.default.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ({}, None),
        ({"age_years": 4}, "child_under5"),
        ({"age_years": 5}, "child_5_12"),
        ({"age_years": 11}, "child_5_12"),
        ({"age_years": 12}, None),
        ({"age_years": 24, "is_pregnant": "yes"}, "pregnant_woman"),
        ({"age_years": 24, "is_pregnant": "no"}, "adult_nonpregnant"),
    ],
)
def test_population_derivation(answers: dict[str, Any], expected: str | None) -> None:
    assert derive_population(answers) == expected


def test_entry_and_population_skip_logic(schema: dict[str, Any]) -> None:
    assert next_question(schema, {}, None)["id"] == "age_years"
    assert next_question(schema, {"age_years": 10}, "child_5_12")["id"] == "child_anemia_pica"
    assert next_question(schema, {"age_years": 24}, None)["id"] == "is_pregnant"
    assert next_question(
        schema, {"age_years": 24, "is_pregnant": "yes"}, "pregnant_woman"
    )["id"] == "pregnant_anemia_previous"
    assert next_question(
        schema, {"age_years": 24, "is_pregnant": "no"}, "adult_nonpregnant"
    )["id"] == "adult_sex"


def test_breastfeeding_question_only_applies_through_six_months(schema: dict[str, Any]) -> None:
    question = next(
        question
        for question in schema["questions"]
        if question["id"] == "child_malnutrition_not_exclusive_breastfeeding"
    )
    assert question_applies(question, {"age_months": 6}, "child_under5")
    assert not question_applies(question, {"age_months": 7}, "child_under5")
    assert not question_applies(question, {"age_months": 4}, "child_5_12")


def test_safe_condition_operators() -> None:
    answers = {"age": 6, "answer": "yes"}
    assert condition_matches({"question": "age", "op": ">=", "value": 6}, answers)
    assert condition_matches({"question": "answer", "op": "in", "value": ["yes", "maybe"]}, answers)
    assert not condition_matches({"question": "age", "op": "<", "value": 5}, answers)
    with pytest.raises(ValueError, match="Unsupported"):
        condition_matches({"question": "age", "op": "exec", "value": 6}, answers)


def test_pregnancy_score_and_bands(schema: dict[str, Any]) -> None:
    answers: dict[str, Any] = {"age_years": 24, "is_pregnant": "yes"}
    for question in schema["questions"]:
        if "pregnant_woman" in question.get("applies_to", []) and question.get("category"):
            answers[question["id"]] = "yes"

    result = score(schema, answers, "pregnant_woman")
    assert result == {
        "anemia": {"raw": 1.0, "band": "high"},
        "malnutrition": {"raw": 1.0, "band": "high"},
    }


def test_cutoff_boundaries_are_moderate(schema: dict[str, Any]) -> None:
    answers = {
        "age_years": 10,
        "child_anemia_pica": "yes",
        "child_anemia_fatigue": "yes",
        "child_anemia_recent_infection": "no",
        "child_anemia_low_iron_food": "yes",
        "child_malnutrition_low_diversity": "yes",
        "child_malnutrition_few_meals": "yes",
        "child_malnutrition_recent_illness": "yes",
        "child_malnutrition_food_insecurity": "yes",
    }
    result = score(schema, answers, "child_5_12")
    assert result["anemia"] == {"raw": 0.6, "band": "moderate"}
    assert result["malnutrition"] == {"raw": 0.7, "band": "high"}


def test_adult_nonpregnant_weights_and_negative_classification(schema: dict[str, Any]) -> None:
    answers: dict[str, Any] = {"age_years": 40, "is_pregnant": "no"}
    for question in schema["questions"]:
        if "adult_nonpregnant" in question.get("applies_to", []):
            answers[question["id"]] = (
                "no" if any(option.get("value") == "no" for option in question.get("options", []))
                else question.get("options", [{"value": "female"}])[0]["value"]
            )

    result = score(schema, answers, "adult_nonpregnant")
    assert result["anemia"] == {"raw": 0.0, "band": "low"}
    assert result["malnutrition"] == {"raw": 0.0, "band": "low"}


@pytest.mark.parametrize(
    ("entry_answers", "population"),
    [
        ({"age_years": 2, "age_months": 24}, "child_under5"),
        ({"age_years": 8}, "child_5_12"),
        ({"age_years": 21, "is_pregnant": "yes"}, "pregnant_woman"),
        ({"age_years": 21, "is_pregnant": "no"}, "adult_nonpregnant"),
    ],
)
def test_each_branch_has_unique_ordered_questions(
    schema: dict[str, Any], entry_answers: dict[str, Any], population: str
) -> None:
    answers = dict(entry_answers)
    seen: list[str] = []
    while question := next_question(schema, answers, population):
        assert question["id"] not in seen
        seen.append(question["id"])
        answers[question["id"]] = (
            question.get("min", 0) if question["type"] == "number"
            else question.get("options", [{"value": "no"}])[0]["value"]
        )

    assert seen
    assert len(seen) == len(set(seen))
