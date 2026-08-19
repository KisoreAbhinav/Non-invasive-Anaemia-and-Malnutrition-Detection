"""Stage 2 planning and result fusion around the Stage 1 questionnaire output.

The questionnaire engine remains the sole owner of its scoring. This module
only consumes `{population, answers, scores}` and combines non-invalidated
measurements. Missing data lowers completeness; it is never treated as zero.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config_loader import read_json_object
from app.flows import model_registry
from app.flows.vision_inference import predict_batch, predict_image
from app.flows.growth_standards import assess_child_growth
from app.settings import settings

router = APIRouter(prefix="/api/flows/screening", tags=["screening"])
ELIGIBLE_POPULATIONS = {
    "child_under5", "child_5_12", "pregnant_woman", "adult_nonpregnant"
}


class PlanRequest(BaseModel):
    population: str
    answers: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, dict[str, Any]] = Field(default_factory=dict)


class VisualResult(BaseModel):
    test_id: str
    value: Any = None
    score: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    invalidated: bool = False
    status: str | None = None


class ResultRequest(PlanRequest):
    invalidated_sections: list[str] = Field(default_factory=list)
    visual_results: list[VisualResult] = Field(default_factory=list)
    clinical_values: dict[str, Any] = Field(default_factory=dict)
    invalidated_fields: list[str] = Field(default_factory=list)


class TestClassificationRequest(BaseModel):
    population: str
    test_id: str
    value: Any


def _visual_schema() -> dict[str, Any]:
    return read_json_object(settings.visual_cues_path)


def _clinical_schema() -> dict[str, Any]:
    return read_json_object(settings.clinical_fields_path)


def _raw_score(scores: dict[str, dict[str, Any]], category: str) -> float:
    try:
        return max(0.0, min(1.0, float(scores.get(category, {}).get("raw", 0))))
    except (TypeError, ValueError):
        return 0.0


def _answer_is_risk(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "true", "positive", "1"}


def applicable_clinical_fields(
    population: str, scores: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    schema = _clinical_schema()
    population_rules = schema.get("populations", {}).get(population, {})
    field_catalog = schema.get("fields", {})
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Reports are a stage in the flow, not a consequence of questionnaire
    # suspicion. Always offer every population-appropriate field and let the
    # operator explicitly skip values they do not have.
    for category in ("anemia", "malnutrition"):
        category_rules = population_rules.get(category, {})
        field_ids = list(dict.fromkeys([
            *category_rules.get("moderate", []),
            *category_rules.get("high", []),
        ]))
        for field_id in field_ids:
            if field_id in seen or field_id not in field_catalog:
                continue
            field = dict(field_catalog[field_id])
            field.update(
                {
                    "id": field_id,
                    "category": category,
                    "risk_band": str(scores.get(category, {}).get("band", "not_entered")),
                    "if_available": population.startswith("child_") and category == "malnutrition",
                }
            )
            chosen.append(field)
            seen.add(field_id)
    return chosen


def build_plan(
    population: str,
    answers: dict[str, Any],
    scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    schema = _visual_schema()
    tests = [
        dict(test)
        for test in schema.get("tests", [])
        if population in test.get("populations", [])
    ]
    test_by_id = {test["id"]: test for test in tests}
    priorities = {test["id"]: float(test.get("base_priority", 0)) for test in tests}
    for question_id, answer in answers.items():
        if not _answer_is_risk(answer):
            continue
        mapping = schema.get("question_test_mapping", {}).get(question_id, {})
        for test_id in mapping.get("tests", []):
            if test_id in priorities:
                priorities[test_id] += float(mapping.get("boost", 0))

    anemia_score = _raw_score(scores, "anemia")
    malnutrition_score = _raw_score(scores, "malnutrition")
    category_order = (
        ["anemia", "malnutrition"]
        if anemia_score >= malnutrition_score
        else ["malnutrition", "anemia"]
    )
    category_rank = {category: index for index, category in enumerate(category_order)}
    ordered_ids = sorted(
        test_by_id,
        key=lambda test_id: (
            category_rank.get(test_by_id[test_id].get("category"), 99),
            -priorities[test_id],
            test_id,
        ),
    )
    model_test_ids = [
        test_id
        for test_id in ordered_ids
        if test_by_id[test_id].get("execution_type") != "physical"
    ]
    availability = model_registry.status(model_test_ids)
    ordered_tests: list[dict[str, Any]] = []
    for order, test_id in enumerate(ordered_ids, start=1):
        test = test_by_id[test_id]
        test["order"] = order
        test["priority"] = round(priorities[test_id], 2)
        test["threshold"] = test.get("thresholds", {}).get(
            population, test.get("threshold", "Clinical reference")
        )
        test["availability"] = (
            {
                "available": True,
                "status": "physical_input",
                "message": "physical measurement — no model required",
                "model": None,
                "metadata": None,
                "error": None,
            }
            if test.get("execution_type") == "physical"
            else availability[test_id]
        )
        ordered_tests.append(test)

    return {
        "population": population,
        "dominant_category": category_order[0],
        "category_order": category_order,
        "visual_cues": ordered_tests,
        "clinical_fields": applicable_clinical_fields(population, scores),
        "fusion": dict(schema.get("fusion", {})),
    }


def _display(value: Any, unit: str = "") -> str:
    if value in (None, ""):
        return "not provided"
    text = str(value)
    return f"{text} {unit}".strip()


def _positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return str(value).strip().lower() in {
        "yes", "positive", "present", "pitting", "true", "1+", "2+", "3+", "4+"
    }


def _visual_classification(
    test_id: str, value: Any, population: str
) -> tuple[str, float | None, str, str | None]:
    if test_id == "edema":
        classification = (
            "Severe" if _positive(value) and population.startswith("child_") else
            "Positive — refer" if _positive(value) else "Normal"
        )
        return classification, (1.0 if _positive(value) else 0.0), _display(value), None
    if test_id == "muac":
        try:
            measurement = float(value.get("muac_cm") if isinstance(value, dict) else value)
        except (TypeError, ValueError):
            return "Invalid measurement", None, "invalid", None
        if population == "child_under5":
            if measurement < 11.5:
                return "Severe acute malnutrition", 1.0, f"{measurement:g} cm", None
            if measurement < 12.5:
                return "Moderate acute malnutrition", 0.5, f"{measurement:g} cm", None
            return "Normal", 0.0, f"{measurement:g} cm", None
        if population == "pregnant_woman":
            classification, score = (
                ("At risk", 0.7) if measurement < 23 else ("Normal", 0.0)
            )
            return classification, score, f"{measurement:g} cm", None
        return "Recorded — no supplied age cutoff", None, f"{measurement:g} cm", None
    if test_id == "weight_height_z" and isinstance(value, dict):
        try:
            assessment = assess_child_growth(value)
        except ValueError as exc:
            return "Invalid measurement", None, str(exc), None
        return (
            assessment["classification"],
            assessment["risk_score"],
            assessment["display"],
            assessment["reference"],
        )
    if test_id in {"adult_bmi", "prepregnancy_bmi"} and isinstance(value, dict):
        try:
            weight = float(value["weight_kg"])
            height = float(value["height_cm"])
            bmi = weight / ((height / 100) ** 2)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return "Invalid measurement", None, "invalid", None
        if bmi < 16:
            classification, risk = "Severe thinness", 1.0
        elif bmi < 17:
            classification, risk = "Moderate thinness", 0.7
        elif bmi < 18.5:
            classification, risk = "Underweight", 0.5
        elif bmi >= 30:
            classification, risk = "Obesity", 0.7
        elif bmi >= 25:
            classification, risk = "Overweight", 0.5
        else:
            classification, risk = "Normal range", 0.0
        return classification, risk, f"BMI {bmi:.1f} kg/m²", None
    if test_id == "gestational_weight_gain" and isinstance(value, dict):
        try:
            gain = float(value["current_weight_kg"]) - float(value["pre_weight_kg"])
            weeks = float(value["gestation_weeks"])
        except (KeyError, TypeError, ValueError):
            return "Invalid measurement", None, "invalid", None
        return (
            "Recorded — IOM chart review required",
            None,
            f"{gain:+.1f} kg by {weeks:g} weeks",
            None,
        )
    if test_id == "fundal_height" and isinstance(value, dict):
        try:
            fundal = float(value["fundal_height_cm"])
            weeks = float(value["gestation_weeks"])
        except (KeyError, TypeError, ValueError):
            return "Invalid measurement", None, "invalid", None
        difference = abs(fundal - weeks)
        classification = "Review measurement" if difference > 3 else "Within rough expected range"
        return classification, (0.5 if difference > 3 else 0.0), f"{fundal:g} cm at {weeks:g} weeks", None
    return "Risk estimate", None, _display(value), None


def _lab_classification(
    field_id: str, value: Any, population: str, all_values: dict[str, Any]
) -> tuple[str, str, bool]:
    """Return threshold, classification, abnormal.

    Only direct cutoffs printed in References.md are applied. In particular,
    child Hb needs the external WHO age table, so it is not guessed here.
    These deterministic lab checks stay separate from questionnaire/CV fusion.
    """

    if field_id == "hemoglobin" and population == "pregnant_woman":
        try:
            trimester = int(float(all_values.get("trimester")))
            measurement = float(value)
        except (TypeError, ValueError):
            return "<11 g/dL (T1/T3); <10.5 g/dL (T2)", "Needs trimester/value", False
        cutoff = 10.5 if trimester == 2 else 11.0
        abnormal = measurement < cutoff
        return f"{cutoff:g} g/dL (trimester {trimester})", (
            "Below cutoff — anemia" if abnormal else "At/above cutoff"
        ), abnormal
    if field_id == "hemoglobin":
        if population == "adult_nonpregnant":
            try:
                measurement = float(value)
            except (TypeError, ValueError):
                return "<12 g/dL female; <13 g/dL male", "Invalid value", False
            sex = str(all_values.get("adult_sex", "")).lower()
            cutoff = 13.0 if sex == "male" else 12.0 if sex == "female" else None
            if cutoff is None:
                return "<12 g/dL female; <13 g/dL male", "Needs sex-specific comparison", False
            abnormal = measurement < cutoff
            return f"<{cutoff:g} g/dL ({sex})", (
                "Below cutoff — anemia" if abnormal else "At/above cutoff"
            ), abnormal
        return "WHO age-band Hb cutoff", "Needs age-band clinical comparison", False
    if field_id == "mcv":
        try:
            measurement = float(value)
        except (TypeError, ValueError):
            return "<80 microcytic; 80–100 normal; >100 macrocytic", "Invalid value", False
        if measurement < 80:
            return "<80 / 80–100 / >100 fL", "Microcytic — review", True
        if measurement > 100:
            return "<80 / 80–100 / >100 fL", "Macrocytic — review", True
        return "<80 / 80–100 / >100 fL", "Normocytic", False
    return "No direct cutoff in supplied theory", "Provided", False


def calculate_result(payload: ResultRequest) -> dict[str, Any]:
    plan = build_plan(payload.population, payload.answers, payload.scores)
    invalid_sections = set(payload.invalidated_sections)
    questionnaire_invalid = "questionnaire" in invalid_sections
    visual_invalid = "visual_cues" in invalid_sections
    clinical_invalid = "clinical_results" in invalid_sections
    rows: list[dict[str, Any]] = []
    usable_inputs = 0
    expected_inputs = 2 + len(plan["visual_cues"]) + len(plan["clinical_fields"])

    for category in ("anemia", "malnutrition"):
        score = payload.scores.get(category, {})
        skipped = questionnaire_invalid
        rows.append(
            {
                "metric": f"Questionnaire — {category.title()}",
                "value": "skipped" if skipped else f"{round(_raw_score(payload.scores, category) * 100)}%",
                "threshold": "Stage 1 population risk band",
                "classification": "Skipped" if skipped else str(score.get("band", "not available")).title(),
                "status": "skipped" if skipped else "complete",
            }
        )
        if not skipped:
            usable_inputs += 1

    supplied_visual = {item.test_id: item for item in payload.visual_results}
    sensor_scores: dict[str, list[float]] = {"anemia": [], "malnutrition": []}
    edema_override = False
    for test in plan["visual_cues"]:
        item = supplied_visual.get(test["id"])
        unavailable = not test["availability"]["available"]
        skipped = visual_invalid or bool(item and item.invalidated)
        if skipped:
            value, classification, row_status = "skipped", "Skipped", "skipped"
        elif unavailable and (item is None or item.value in (None, "")):
            value, classification, row_status = "no model installed", "Not run", "unavailable"
        elif item is None or item.value in (None, ""):
            value, classification, row_status = "not completed", "No result", "missing"
        else:
            classification, derived_score, rendered_value, threshold_override = _visual_classification(
                test["id"], item.value, payload.population
            )
            value, row_status = rendered_value, "complete"
            if threshold_override:
                test["threshold"] = threshold_override
            score_value = item.score if item.score is not None else item.confidence if item.confidence is not None else derived_score
            if score_value is not None:
                sensor_scores[test["category"]].append(float(score_value))
            usable_inputs += 1
            edema_override = edema_override or (
                test.get("overrides_verdict")
                and payload.population in test.get("override_populations", [])
                and _positive(item.value)
            )
        rows.append(
            {
                "metric": test["name"],
                "value": value,
                "threshold": test["threshold"],
                "classification": classification,
                "status": row_status,
            }
        )

    invalid_fields = set(payload.invalidated_fields)
    deterministic_lab_abnormal = False
    clinical_provided = 0
    for field in plan["clinical_fields"]:
        field_id = field["id"]
        field_skipped = clinical_invalid or field_id in invalid_fields
        value = payload.clinical_values.get(field_id)
        if field_skipped:
            threshold, classification, rendered, row_status = "—", "Skipped", "skipped", "skipped"
        elif value in (None, ""):
            threshold, classification, rendered, row_status = "—", "Not provided", "not provided", "missing"
        else:
            threshold, classification, abnormal = _lab_classification(
                field_id, value, payload.population,
                {**payload.answers, **payload.clinical_values},
            )
            deterministic_lab_abnormal = deterministic_lab_abnormal or abnormal
            rendered, row_status = _display(value, field.get("unit", "")), "complete"
            usable_inputs += 1
            clinical_provided += 1
        rows.append(
            {
                "metric": field["label"],
                "value": rendered,
                "threshold": threshold,
                "classification": classification,
                "status": row_status,
            }
        )

    fusion = plan["fusion"]
    alpha = float(fusion.get("questionnaire_weight", 0.3))
    beta = float(fusion.get("sensor_weight", 0.7))
    fused: dict[str, float | None] = {}
    category_classes: dict[str, str] = {}
    for category in ("anemia", "malnutrition"):
        questionnaire_score = None if questionnaire_invalid else _raw_score(payload.scores, category)
        sensor_score = (
            sum(sensor_scores[category]) / len(sensor_scores[category])
            if sensor_scores[category]
            else None
        )
        if questionnaire_score is not None and sensor_score is not None:
            combined = alpha * questionnaire_score + beta * sensor_score
        elif sensor_score is not None:
            combined = sensor_score
        else:
            combined = questionnaire_score
        fused[category] = round(combined, 4) if combined is not None else None
        category_classes[category] = (
            "Unavailable" if combined is None else
            "Severe" if combined > 0.6 else
            "Moderate" if combined >= 0.3 else "Normal"
        )

    if edema_override:
        verdict = "Severe"
        reason = "Positive child bilateral pitting edema triggered the WHO automatic severe override."
    elif deterministic_lab_abnormal:
        verdict = "Refer"
        reason = "A provided clinical value crossed a direct deterministic cutoff and needs clinician review."
    elif all(value is None for value in fused.values()):
        verdict = "Refer"
        reason = "No valid screening inputs remain; obtain a clinical assessment."
    elif "Severe" in category_classes.values():
        verdict = "Severe"
        reason = "At least one valid category score is in the severe risk range."
    elif "Moderate" in category_classes.values():
        verdict = "Moderate"
        reason = "At least one valid category score is in the moderate risk range."
    else:
        verdict = "Normal"
        reason = "All available, non-skipped screening inputs are below moderate risk thresholds."

    if questionnaire_invalid:
        questionnaire_verdict = "Skipped"
        questionnaire_sentence = "The questionnaire section was skipped."
    else:
        bands = {
            category: str(payload.scores.get(category, {}).get("band", "not entered"))
            for category in ("anemia", "malnutrition")
        }
        band_rank = {"not_applicable": -1, "low": 0, "moderate": 1, "high": 2}
        leading_band = max(bands.values(), key=lambda item: band_rank.get(item, -1))
        questionnaire_verdict = {
            "high": "High risk", "moderate": "Moderate risk", "low": "Low risk"
        }.get(leading_band, "Not entered")
        questionnaire_sentence = (
            f"The questionnaire indicates {bands['anemia'].replace('_', ' ')} anemia risk "
            f"and {bands['malnutrition'].replace('_', ' ')} malnutrition risk."
        )

    available_sensor_scores = [score for values in sensor_scores.values() for score in values]
    if visual_invalid:
        visual_verdict = "Skipped"
        visual_sentence = "The visual cues section was skipped."
    elif not available_sensor_scores:
        visual_verdict = "Not entered"
        visual_sentence = "No classifiable visual or physical measurement was entered."
    else:
        visual_score = max(available_sensor_scores)
        visual_verdict = (
            "Severe" if visual_score > 0.6 else
            "Moderate" if visual_score >= 0.3 else "Normal"
        )
        visual_sentence = f"The available visual and physical measurements classify as {visual_verdict.lower()}."

    if clinical_invalid:
        clinical_verdict = "Skipped"
        clinical_sentence = "The clinical reports section was skipped."
    elif clinical_provided == 0:
        clinical_verdict = "Not entered"
        clinical_sentence = "No clinical report values were entered."
    elif deterministic_lab_abnormal:
        clinical_verdict = "Needs review"
        clinical_sentence = "At least one entered clinical report value crosses a direct review threshold."
    else:
        clinical_verdict = "Entered"
        clinical_sentence = "Clinical report values were entered; values without a supplied direct cutoff remain for clinician review."

    final_sentence = f"The combined screening verdict is {verdict.lower()}. {reason}"
    section_verdicts = [
        {"id": "questionnaire", "label": "Questionnaire", "verdict": questionnaire_verdict, "sentence": questionnaire_sentence},
        {"id": "visual", "label": "Visual cues", "verdict": visual_verdict, "sentence": visual_sentence},
        {"id": "clinical", "label": "Clinical reports", "verdict": clinical_verdict, "sentence": clinical_sentence},
        {"id": "combined", "label": "Combined verdict", "verdict": verdict, "sentence": final_sentence},
    ]

    return {
        "population": payload.population,
        "rows": rows,
        "fused_scores": fused,
        "category_classifications": category_classes,
        "verdict": verdict,
        "reason": reason,
        "spoken_text": " ".join(item["sentence"] for item in section_verdicts),
        "section_verdicts": section_verdicts,
        "override": "child_edema" if edema_override else None,
        "completeness": {
            "used": usable_inputs,
            "expected": expected_inputs,
            "ratio": round(usable_inputs / expected_inputs, 4) if expected_inputs else 0,
        },
        "disclaimer": "Screening result only; it is not a diagnosis.",
    }


@router.get("/status")
async def screening_status() -> dict[str, Any]:
    schema = _visual_schema()
    tests = schema.get("tests", [])
    statuses = model_registry.status(
        [test["id"] for test in tests if test.get("execution_type") != "physical"]
    )
    for test in tests:
        if test.get("execution_type") == "physical":
            statuses[test["id"]] = {
                "available": True,
                "status": "physical_input",
                "message": "physical measurement — no model required",
                "model": None,
                "metadata": None,
                "error": None,
            }
    return {"flow": "screening", "status": "ready", "tests": statuses}


@router.post("/plan")
async def screening_plan(payload: PlanRequest) -> dict[str, Any]:
    return build_plan(payload.population, payload.answers, payload.scores)


@router.post("/result")
async def screening_result(payload: ResultRequest) -> dict[str, Any]:
    return calculate_result(payload)


@router.post("/vision/{test_id}")
async def screening_vision(test_id: str, population: str, request: Request) -> dict[str, Any]:
    if population not in ELIGIBLE_POPULATIONS:
        raise HTTPException(status_code=422, detail="Unsupported population")
    plan = build_plan(population, {}, {})
    test_entry = next((t for t in plan["visual_cues"] if t["id"] == test_id), None)
    if test_entry is None:
        raise HTTPException(status_code=404, detail="Test not applicable")

    content_type = request.headers.get("content-type", "")
    sub_captures = test_entry.get("sub_captures")

    # Multi-image batch: when the test has sub_captures, the client MUST send multipart.
    # We do NOT gate on content-type because dev proxies (Vite) may lose the boundary
    # portion of the header.  If form parsing itself fails we still surface a useful error.
    if sub_captures:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=415, detail=f"Expected multipart/form-data for multi-site test: {exc}") from exc
        images: list[bytes] = []
        for capture in sub_captures:
            upload = form.get(capture["id"])
            if upload is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing image for {capture.get('label', capture['id'])}",
                )
            try:
                images.append(await upload.read())
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return predict_batch(sub_captures, images)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Single image
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image/jpeg or image/png body")
    try:
        return predict_image(test_id, await request.body())
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc



@router.post("/test-result")
async def screening_test_result(payload: TestClassificationRequest) -> dict[str, Any]:
    if payload.population not in ELIGIBLE_POPULATIONS:
        return {"test_id": payload.test_id, "classification": "Unsupported population"}
    plan = build_plan(payload.population, {}, {})
    test = next((item for item in plan["visual_cues"] if item["id"] == payload.test_id), None)
    if test is None:
        return {"test_id": payload.test_id, "classification": "Test not applicable"}
    classification, score, display, threshold_override = _visual_classification(
        payload.test_id, payload.value, payload.population
    )
    threshold = threshold_override or test["threshold"]
    return {
        "test_id": payload.test_id,
        "name": test["name"],
        "display": display,
        "classification": classification,
        "risk_score": score,
        "threshold": threshold,
        "spoken_text": f"{test['name']}. {display}. Classification: {classification}.",
    }
