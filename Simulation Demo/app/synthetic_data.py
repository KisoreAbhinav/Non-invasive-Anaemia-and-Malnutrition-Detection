"""Synthetic patient records tied to the common simulation model.

The numeric intervals are deliberately illustrative, with random overlap. They
are *not* clinical reference intervals and must never be used for care.
"""
from __future__ import annotations

from datetime import UTC, datetime
from random import Random
from uuid import uuid4

from .physics import evaluate_pit, simulate_press, varied_parameters


def _sample(rng: Random, interval: tuple[float, float], digits: int = 1) -> float:
    return round(rng.uniform(*interval), digits)


def clinical_values(has_condition: bool, rng: Random) -> dict[str, float | str]:
    """Generate varied illustrative panels with class-correlated distributions.

    Values use conventional units from the reference project field schema.  The
    abnormal direction follows the project notes (low Hb/ferritin/albumin/etc.);
    TIBC rises in a typical iron-deficiency pattern.  Overlap is intentional so
    label recovery cannot be a one-field threshold rule.
    """
    if has_condition:
        return {
            "hemoglobin_g_dL": _sample(rng, (6.7, 11.2)),
            "hematocrit_percent": _sample(rng, (22, 35)),
            "mcv_fL": _sample(rng, (64, 82)),
            "ferritin_ng_mL": _sample(rng, (4, 29)),
            "serum_iron_ug_dL": _sample(rng, (18, 66)),
            "tibc_ug_dL": _sample(rng, (385, 505)),
            # Edema-mode albumin stays below the demo's 3.0 g/dL explanatory
            # threshold; other markers retain deliberately overlapping variance.
            "albumin_g_dL": _sample(rng, (1.7, 2.9)),
            "blood_glucose_mg_dL": _sample(rng, (54, 82)),
            "prealbumin_mg_dL": _sample(rng, (6, 18)),
            "folate_ng_mL": _sample(rng, (1.7, 5.5)),
            "vitamin_b12_pg_mL": _sample(rng, (115, 280)),
            "calcium_mg_dL": _sample(rng, (7.0, 8.8)),
            "vitamin_d_ng_mL": _sample(rng, (7, 23)),
            "zinc_ug_dL": _sample(rng, (38, 68)),
            "electrolytes": "illustrative low/variable",
        }
    return {
        "hemoglobin_g_dL": _sample(rng, (11.0, 14.9)),
        "hematocrit_percent": _sample(rng, (34, 45)),
        "mcv_fL": _sample(rng, (79, 97)),
        "ferritin_ng_mL": _sample(rng, (24, 135)),
        "serum_iron_ug_dL": _sample(rng, (57, 155)),
        "tibc_ug_dL": _sample(rng, (265, 405)),
        "albumin_g_dL": _sample(rng, (3.5, 5.0)),
        "blood_glucose_mg_dL": _sample(rng, (70, 108)),
        "prealbumin_mg_dL": _sample(rng, (18, 37)),
        "folate_ng_mL": _sample(rng, (5.0, 15)),
        "vitamin_b12_pg_mL": _sample(rng, (245, 730)),
        "calcium_mg_dL": _sample(rng, (8.5, 10.3)),
        "vitamin_d_ng_mL": _sample(rng, (22, 52)),
        "zinc_ug_dL": _sample(rng, (70, 120)),
        "electrolytes": "illustrative within expected range",
    }


def make_record(
    label: bool,
    rng: Random | None = None,
    *,
    force: float | None = None,
    duration: float | None = None,
) -> dict:
    rng = rng or Random()
    mode = "edema" if label else "normal"
    # The UI can set a press protocol; otherwise records retain variation
    # around the documented three-second educational press.
    force = float(force) if force is not None else _sample(rng, (5.1, 10.5))
    duration = float(duration) if duration is not None else _sample(rng, (2.4, 4.2))
    simulation = simulate_press(mode, force, duration, params=varied_parameters(mode, rng))
    detector = evaluate_pit(simulation)
    return {
        "id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "synthetic": True,
        "label": {"has_malnutrition_or_severe_anemia": label, "tissue_mode": mode},
        "blood_values": clinical_values(label, rng),
        "press": {
            "force": force,
            "duration": duration,
            "time_series": simulation["series"],
            "parameters": simulation["parameters"],
            "summary": {
                "peak_depth": round(simulation["peak_depth"], 4),
                "residual_depth": round(simulation["residual_depth"], 4),
                "deterministic_pit": detector,
            },
        },
    }
