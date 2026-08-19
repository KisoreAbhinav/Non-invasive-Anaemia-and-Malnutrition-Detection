"""Offline WHO 2006/2007 LMS growth-reference calculations.

Reference tables are vendored verbatim from the WHO-maintained `anthro` and
`anthroplus` repositories. Under-five measurements use WHO 2006 WFA, HFA and
weight-for-length/height. Ages 5–12 use WHO 2007 HFA and BMI-for-age, with WFA
only through 10 years because WHO does not define that indicator after age 10.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.settings import PROJECT_ROOT

REFERENCE_ROOT = PROJECT_ROOT / "config" / "who"


@lru_cache(maxsize=None)
def _rows(filename: str) -> list[dict[str, str]]:
    with (REFERENCE_ROOT / filename).open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def _sex_code(value: Any) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "male", "m", "boy"}:
        return 1
    if normalized in {"2", "female", "f", "girl"}:
        return 2
    raise ValueError("sex must be male or female")


def _nearest_lms(
    filename: str,
    sex: int,
    key: str,
    target: float,
    measurement_type: str | None = None,
) -> tuple[float, float, float]:
    candidates = [row for row in _rows(filename) if int(row["sex"]) == sex]
    if measurement_type is not None:
        type_key = "loh" if "loh" in candidates[0] else "lorh"
        candidates = [row for row in candidates if row[type_key] == measurement_type]
    if not candidates:
        raise ValueError("WHO reference row is unavailable for this measurement")
    row = min(candidates, key=lambda item: abs(float(item[key]) - target))
    return float(row["l"]), float(row["m"]), float(row["s"])


def _measurement_at_z(l_value: float, median: float, spread: float, z_score: float) -> float:
    if l_value == 0:
        return median * math.exp(spread * z_score)
    return median * ((1 + l_value * spread * z_score) ** (1 / l_value))


def _z_score(measurement: float, l_value: float, median: float, spread: float) -> float:
    if l_value == 0:
        score = math.log(measurement / median) / spread
    else:
        score = (((measurement / median) ** l_value) - 1) / (l_value * spread)
    if score > 3:
        sd3 = _measurement_at_z(l_value, median, spread, 3)
        sd2 = _measurement_at_z(l_value, median, spread, 2)
        score = 3 + (measurement - sd3) / (sd3 - sd2)
    elif score < -3:
        sd3 = _measurement_at_z(l_value, median, spread, -3)
        sd2 = _measurement_at_z(l_value, median, spread, -2)
        score = -3 + (measurement - sd3) / (sd2 - sd3)
    return round(score, 2)


def _classify_low(score: float, moderate: str, severe: str) -> tuple[str, float]:
    if score < -3:
        return severe, 1.0
    if score < -2:
        return moderate, 0.5
    return "Normal", 0.0


def assess_child_growth(values: dict[str, Any]) -> dict[str, Any]:
    sex = _sex_code(values.get("sex"))
    try:
        age_months = float(values["age_months"])
        weight = float(values["weight_kg"])
        height = float(values["height_cm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sex, age in months, weight in kg and length/height in cm are required") from exc
    if not 0 <= age_months < 144 or weight <= 0 or height <= 0:
        raise ValueError("growth measurements are outside the supported child range")

    metrics: list[dict[str, Any]] = []
    risk_scores: list[float] = []
    if age_months < 60:
        age_days = round(age_months * 30.4375)
        posture = "L" if age_months < 24 else "H"
        wfa = _z_score(weight, *_nearest_lms("weianthro.txt", sex, "age", age_days))
        hfa = _z_score(height, *_nearest_lms("lenanthro.txt", sex, "age", age_days, posture))
        proportional_file = "wflanthro.txt" if posture == "L" else "wfhanthro.txt"
        proportional_key = "length" if posture == "L" else "height"
        whz = _z_score(
            weight,
            *_nearest_lms(proportional_file, sex, proportional_key, height, posture),
        )
        definitions = [
            ("Weight-for-age", wfa, "Underweight", "Severe underweight"),
            ("Height-for-age", hfa, "Stunted", "Severely stunted"),
            ("Weight-for-length" if posture == "L" else "Weight-for-height", whz,
             "Moderate wasting", "Severe wasting"),
        ]
        reference = "WHO Child Growth Standards 2006"
    else:
        bmi = weight / ((height / 100) ** 2)
        hfa = _z_score(height, *_nearest_lms("hfawho2007.txt", sex, "age", age_months))
        bfa = _z_score(bmi, *_nearest_lms("bfawho2007.txt", sex, "age", age_months))
        definitions = [
            ("Height-for-age", hfa, "Stunted", "Severely stunted"),
            ("BMI-for-age", bfa, "Thinness", "Severe thinness"),
        ]
        if age_months <= 120:
            wfa = _z_score(weight, *_nearest_lms("wfawho2007.txt", sex, "age", age_months))
            definitions.append(("Weight-for-age", wfa, "Underweight", "Severe underweight"))
        reference = "WHO Growth Reference 2007"

    for name, score, moderate, severe in definitions:
        classification, risk = _classify_low(score, moderate, severe)
        if name == "BMI-for-age" and classification == "Normal":
            if score > 2:
                classification, risk = "Obesity", 0.7
            elif score > 1:
                classification, risk = "Overweight", 0.5
        risk_scores.append(risk)
        metrics.append({"name": name, "z_score": score, "classification": classification})

    worst_index = max(range(len(metrics)), key=lambda index: risk_scores[index])
    summary = " · ".join(f"{item['name']} {item['z_score']:+.2f}" for item in metrics)
    return {
        "metrics": metrics,
        "classification": metrics[worst_index]["classification"],
        "risk_score": max(risk_scores, default=0.0),
        "display": summary,
        "reference": reference,
    }
