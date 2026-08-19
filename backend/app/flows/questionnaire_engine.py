"""Pure adaptive-questionnaire routing and scoring helpers.

This module intentionally has no FastAPI imports.  It can be exercised on its
own and later reused by the sensor/prediction stages without pulling in HTTP
state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SUPPORTED_OPERATORS = {"==", "!=", "<", "<=", ">", ">=", "in"}


def derive_population(answers: Mapping[str, Any]) -> str | None:
    """Derive the screening population once enough entry answers exist.

    Children do not need the pregnancy question, so their population is known
    immediately after age.  People aged 12 and over need ``is_pregnant`` before
    routing can finish.
    """

    age_value = answers.get("age_years")
    if age_value in (None, ""):
        return None
    if isinstance(age_value, bool):
        raise ValueError("age_years must be a number")

    try:
        age = float(age_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("age_years must be a number") from exc

    if age < 0:
        raise ValueError("age_years cannot be negative")
    if age < 5:
        return "child_under5"
    if age < 12:
        return "child_5_12"

    pregnancy_value = answers.get("is_pregnant")
    if pregnancy_value in (None, ""):
        return None
    normalized = str(pregnancy_value).strip().lower()
    if normalized == "yes":
        return "pregnant_woman"
    if normalized == "no":
        return "adult_nonpregnant"
    raise ValueError("is_pregnant must be 'yes' or 'no'")


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported questionnaire operator: {operator}")

    if operator == "==":
        return actual == expected
    if operator == "!=":
        return actual != expected
    if operator == "in":
        if not isinstance(expected, (list, tuple, set, frozenset)):
            raise ValueError("The 'in' operator requires a list-like condition value")
        return actual in expected

    try:
        if operator == "<":
            return actual < expected
        if operator == "<=":
            return actual <= expected
        if operator == ">":
            return actual > expected
        return actual >= expected
    except TypeError:
        # JSON entry values may arrive as numeric strings from a debug client.
        try:
            numeric_actual = float(actual)
            numeric_expected = float(expected)
        except (TypeError, ValueError):
            return False
        if operator == "<":
            return numeric_actual < numeric_expected
        if operator == "<=":
            return numeric_actual <= numeric_expected
        if operator == ">":
            return numeric_actual > numeric_expected
        return numeric_actual >= numeric_expected


def condition_matches(condition: Mapping[str, Any], answers: Mapping[str, Any]) -> bool:
    """Evaluate one data-only condition without executing arbitrary code."""

    question_id = condition.get("question")
    operator = condition.get("op")
    if not isinstance(question_id, str) or not isinstance(operator, str):
        raise ValueError("applies_if requires string 'question' and 'op' fields")
    if question_id not in answers:
        return False
    return _compare(answers[question_id], operator, condition.get("value"))


def question_applies(
    question: Mapping[str, Any], answers: Mapping[str, Any], population: str | None
) -> bool:
    applies_to = question.get("applies_to")
    if applies_to is not None:
        if not isinstance(applies_to, list):
            raise ValueError("applies_to must be a list")
        if population is None or population not in applies_to:
            return False

    applies_if = question.get("applies_if")
    if applies_if is not None:
        if not isinstance(applies_if, Mapping):
            raise ValueError("applies_if must be an object")
        if not condition_matches(applies_if, answers):
            return False
    return True


def next_question(
    schema: Mapping[str, Any], answers: Mapping[str, Any], population: str | None
) -> dict[str, Any] | None:
    """Return the first unanswered eligible question in configured order."""

    questions = schema.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("questionnaire schema questions must be a list")

    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("each questionnaire question must be an object")
        question_id = question.get("id")
        if not isinstance(question_id, str):
            raise ValueError("each questionnaire question requires a string id")
        if question_id in answers:
            continue
        if question_applies(question, answers, population):
            return question
    return None


def _answer_score(question: Mapping[str, Any], answer: Any) -> float:
    question_type = question.get("type")
    if question_type == "select":
        for option in question.get("options", []):
            if isinstance(option, Mapping) and option.get("value") == answer:
                return float(option.get("score", 0.0))
        return 0.0
    if question_type == "number":
        return float(answer)
    return 0.0


def _band_for(raw_score: float, cutoffs: Mapping[str, Any] | None) -> str:
    if not cutoffs:
        return "not_applicable"
    low_below = float(cutoffs["low_below"])
    high_above = float(cutoffs["high_above"])
    if raw_score < low_below:
        return "low"
    if raw_score > high_above:
        return "high"
    return "moderate"


def score(
    schema: Mapping[str, Any], answers: Mapping[str, Any], population: str
) -> dict[str, dict[str, float | str]]:
    """Calculate independent anemia and malnutrition weighted scores."""

    totals = {"anemia": 0.0, "malnutrition": 0.0}
    questions = schema.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("questionnaire schema questions must be a list")

    for question in questions:
        if not isinstance(question, Mapping):
            continue
        question_id = question.get("id")
        category = question.get("category")
        weight = question.get("weight")
        if category not in totals or weight is None or question_id not in answers:
            continue
        if not question_applies(question, answers, population):
            continue
        totals[category] += float(weight) * _answer_score(question, answers[question_id])

    population_cutoffs = schema.get("score_cutoffs", {}).get(population, {})
    return {
        category: {
            "raw": round(raw_score, 4),
            "band": _band_for(raw_score, population_cutoffs.get(category)),
        }
        for category, raw_score in totals.items()
    }
