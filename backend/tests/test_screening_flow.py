import json
from pathlib import Path

from app.flows import model_registry
from app.flows.screening import ResultRequest, build_plan, calculate_result


def scores(anemia: float, malnutrition: float) -> dict:
    def band(value: float) -> str:
        return "high" if value > 0.6 else "moderate" if value >= 0.3 else "low"

    return {
        "anemia": {"raw": anemia, "band": band(anemia)},
        "malnutrition": {"raw": malnutrition, "band": band(malnutrition)},
    }


def test_category_dominance_changes_visual_test_group_order() -> None:
    anemia_first = build_plan("child_under5", {}, scores(0.8, 0.1))
    malnutrition_first = build_plan("child_under5", {}, scores(0.1, 0.8))

    assert anemia_first["visual_cues"][0]["category"] == "anemia"
    assert malnutrition_first["visual_cues"][0]["category"] == "malnutrition"
    assert [test["id"] for test in anemia_first["visual_cues"]] != [
        test["id"] for test in malnutrition_first["visual_cues"]
    ]


def test_question_mapping_fine_tunes_order_within_category() -> None:
    plan = build_plan(
        "child_under5",
        {"child_malnutrition_low_diversity": "yes"},
        scores(0.1, 0.8),
    )
    malnutrition_ids = [
        test["id"] for test in plan["visual_cues"] if test["category"] == "malnutrition"
    ]
    assert malnutrition_ids.index("weight_height_z") < malnutrition_ids.index("hair_skin")


def test_growth_measurements_are_physical_and_need_no_model() -> None:
    plan = build_plan("child_under5", {}, scores(0.1, 0.8))
    tests = {test["id"]: test for test in plan["visual_cues"]}

    assert tests["muac"]["execution_type"] == "physical"
    assert tests["muac"]["availability"]["status"] == "physical_input"
    assert tests["weight_height_z"]["execution_type"] == "physical"
    assert tests["weight_height_z"]["availability"]["status"] == "physical_input"


def test_physical_growth_measurements_feed_who_result() -> None:
    result = calculate_result(
        ResultRequest(
            population="child_5_12",
            answers={"age_years": 8},
            scores=scores(0.1, 0.4),
            visual_results=[
                {
                    "test_id": "weight_height_z",
                    "value": {
                        "sex": "male",
                        "age_months": 100,
                        "weight_kg": 30,
                        "height_cm": 100,
                    },
                }
            ],
        )
    )
    row = next(row for row in result["rows"] if row["metric"] == "WHO weight and height indicators")

    assert "Height-for-age -5.04" in row["value"]
    assert row["threshold"] == "WHO Growth Reference 2007"
    assert row["classification"] == "Severely stunted"
    assert [item["id"] for item in result["section_verdicts"]] == [
        "questionnaire", "visual", "clinical", "combined"
    ]


def test_adult_bmi_is_classified_without_a_model() -> None:
    result = calculate_result(
        ResultRequest(
            population="adult_nonpregnant",
            answers={"age_years": 21, "is_pregnant": "no"},
            scores=scores(0.1, 0.1),
            visual_results=[
                {"test_id": "adult_bmi", "value": {"weight_kg": 45, "height_cm": 170}}
            ],
        )
    )
    row = next(row for row in result["rows"] if row["metric"] == "Adult body mass index (BMI)")

    assert row["classification"] == "Severe thinness"
    assert row["value"] == "BMI 15.6 kg/m²"


def test_adult_report_verdict_uses_sex_specific_hemoglobin_cutoff() -> None:
    result = calculate_result(
        ResultRequest(
            population="adult_nonpregnant",
            answers={"age_years": 21, "is_pregnant": "no", "adult_sex": "male"},
            scores=scores(0.1, 0.1),
            clinical_values={"hemoglobin": 12.5},
        )
    )
    report = next(item for item in result["section_verdicts"] if item["id"] == "clinical")

    assert report["verdict"] == "Needs review"


def test_low_risk_still_offers_all_population_clinical_fields() -> None:
    plan = build_plan("pregnant_woman", {}, scores(0.7, 0.1))
    field_ids = {field["id"] for field in plan["clinical_fields"]}
    assert "hemoglobin" in field_ids
    assert "albumin" in field_ids


def test_skipped_visual_section_is_explicit_and_questionnaire_only() -> None:
    result = calculate_result(
        ResultRequest(
            population="child_under5",
            answers={},
            scores=scores(0.4, 0.2),
            invalidated_sections=["visual_cues"],
        )
    )
    visual_names = {
        test["name"] for test in build_plan("child_under5", {}, scores(0.4, 0.2))["visual_cues"]
    }
    visual_rows = [row for row in result["rows"] if row["metric"] in visual_names]
    assert all(row["value"] == "skipped" for row in visual_rows)
    assert result["fused_scores"] == {"anemia": 0.4, "malnutrition": 0.2}
    assert result["verdict"] == "Moderate"


def test_positive_child_edema_overrides_adaptive_position() -> None:
    result = calculate_result(
        ResultRequest(
            population="child_5_12",
            answers={},
            scores=scores(0.8, 0.1),
            visual_results=[{"test_id": "edema", "value": "positive", "confidence": 1}],
        )
    )
    assert result["verdict"] == "Severe"
    assert result["override"] == "child_edema"


def test_registry_detects_valid_contract_without_loading_torch(tmp_path: Path) -> None:
    test_dir = tmp_path / "edema"
    test_dir.mkdir()
    (test_dir / "model.pt").write_bytes(b"placeholder")
    (test_dir / "model.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "test_id": "edema",
                "input_shape": [1, 3, 224, 224],
                "preprocessing": {"color_space": "RGB"},
                "output_class_labels": ["normal", "pitting"],
            }
        ),
        encoding="utf-8",
    )
    try:
        model_registry.scan_models(tmp_path)
        assert model_registry.is_available("edema")
        assert model_registry.status(["edema"])["edema"]["status"] == "will_run"
    finally:
        model_registry.scan_models()


def test_model_primary_score_is_used_for_sensor_fusion() -> None:
    result = calculate_result(
        ResultRequest(
            population="child_under5",
            answers={},
            scores=scores(0.2, 0.1),
            visual_results=[
                {
                    "test_id": "pallor",
                    "value": {"scores": {"pallor_signal": 0.8}},
                    "score": 0.8,
                    "confidence": 0.95,
                }
            ],
        )
    )

    assert result["fused_scores"]["anemia"] == 0.62
