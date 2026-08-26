import pytest

from app.flows.growth_standards import assess_child_growth


def test_who_2007_reference_example_matches_published_anthroplus_values() -> None:
    result = assess_child_growth(
        {"sex": "male", "age_months": 100, "weight_kg": 30, "height_cm": 100}
    )
    metrics = {item["name"]: item for item in result["metrics"]}

    assert metrics["Height-for-age"]["z_score"] == -5.04
    assert metrics["BMI-for-age"]["z_score"] == 5.03
    assert metrics["Weight-for-age"]["z_score"] == 0.87
    assert result["reference"] == "WHO Growth Reference 2007"


def test_under_five_uses_physical_wfa_hfa_and_weight_for_height() -> None:
    result = assess_child_growth(
        {"sex": "female", "age_months": 24, "weight_kg": 12, "height_cm": 85}
    )

    assert [item["name"] for item in result["metrics"]] == [
        "Weight-for-age",
        "Height-for-age",
        "Weight-for-height",
    ]
    assert result["classification"] == "Normal"
    assert result["reference"] == "WHO Child Growth Standards 2006"


def test_five_year_old_switches_to_who_2007_bmi_for_age() -> None:
    result = assess_child_growth(
        {"sex": "male", "age_months": 60, "weight_kg": 18, "height_cm": 110}
    )

    assert "BMI-for-age" in {item["name"] for item in result["metrics"]}
    assert result["reference"] == "WHO Growth Reference 2007"


def test_invalid_physical_measurements_are_rejected() -> None:
    with pytest.raises(ValueError):
        assess_child_growth(
            {"sex": "female", "age_months": 150, "weight_kg": 20, "height_cm": 120}
        )
