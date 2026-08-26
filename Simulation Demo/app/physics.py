"""Shared, deterministic press-and-recovery model used by every demo layer.

It is intentionally an educational analogue, not a biomechanical tissue model or
clinical diagnostic algorithm.  The front end turns its centre-point displacement
into a coupled-looking canvas mesh; dataset and classifier samples come from this
same source of truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import exp
from random import Random
from typing import Literal

Mode = Literal["normal", "edema"]


@dataclass(frozen=True)
class TissueParameters:
    # compliance determines the initial, force-dependent displacement.
    compliance: float
    # accumulation is the time constant while pressure is held.
    accumulation_tau: float
    # relaxation controls rebound after release.
    recovery_tau: float
    # only edema retains an offset after the transient recovery has plateaued.
    residual_fraction: float


TISSUE_PRESETS: dict[Mode, TissueParameters] = {
    "normal": TissueParameters(0.075, 0.34, 0.85, 0.0),
    "edema": TissueParameters(0.082, 0.44, 7.8, 0.23),
}


def varied_parameters(mode: Mode, rng: Random, variation: float = 0.12) -> TissueParameters:
    """Return a patient-specific but class-faithful parameter draw."""
    base = TISSUE_PRESETS[mode]

    def jitter(value: float) -> float:
        return value * (1 + rng.uniform(-variation, variation))

    return replace(
        base,
        compliance=jitter(base.compliance),
        accumulation_tau=jitter(base.accumulation_tau),
        recovery_tau=jitter(base.recovery_tau),
        residual_fraction=max(0.0, jitter(base.residual_fraction)) if mode == "edema" else 0.0,
    )


def simulate_press(
    mode: Mode,
    force: float = 7.0,
    duration: float = 3.0,
    *,
    params: TissueParameters | None = None,
    total_time: float | None = None,
    step: float = 0.1,
) -> dict:
    """Generate a press → hold → release curve.

    Positive ``depth`` means a downward dent in the visualisation.  A first-order
    viscoelastic response is sufficient here: pressure approaches a target depth
    during contact; after release it decays towards a residual offset.  The two
    modes differ only by values in ``TISSUE_PRESETS``.
    """
    if mode not in TISSUE_PRESETS:
        raise ValueError("mode must be 'normal' or 'edema'")
    force = max(0.2, min(float(force), 16.0))
    duration = max(0.2, min(float(duration), 12.0))
    params = params or TISSUE_PRESETS[mode]
    # Always retain enough post-release samples for the largest UI window.
    total_time = max(float(total_time or 0), duration + 21.0)
    peak_target = force * params.compliance
    hold_peak = peak_target * (1 - exp(-duration / params.accumulation_tau))
    residual = hold_peak * params.residual_fraction

    values: list[dict[str, float | str]] = []
    t = 0.0
    while t <= total_time + 1e-9:
        if t <= duration:
            depth = peak_target * (1 - exp(-t / params.accumulation_tau))
            phase = "press" if t < 0.4 else "hold"
        else:
            after_release = t - duration
            depth = residual + (hold_peak - residual) * exp(-after_release / params.recovery_tau)
            phase = "recovery"
        values.append({"t": round(t, 3), "depth": round(max(depth, 0), 5), "phase": phase})
        t += step

    return {
        "mode": mode,
        "force": force,
        "duration": duration,
        "parameters": asdict(params),
        "series": values,
        "peak_depth": hold_peak,
        "residual_depth": residual,
    }


def evaluate_pit(simulation: dict, threshold_fraction: float = 0.10, window_seconds: float = 15.0) -> dict:
    """Apply the configurable deterministic detector to the generated curve."""
    peak = float(simulation["peak_depth"])
    duration = float(simulation["duration"])
    threshold = max(0.001, peak * max(0.0, threshold_fraction))
    target_time = duration + max(0.1, window_seconds)
    series = simulation["series"]
    at_window = min(series, key=lambda point: abs(float(point["t"]) - target_time))
    window_depth = float(at_window["depth"])
    recovered_at = next(
        (float(point["t"]) - duration for point in series if float(point["t"]) >= duration and float(point["depth"]) <= threshold),
        None,
    )
    # 90% recovery means depth has fallen to <= 10% of peak.
    time_to_90 = recovered_at
    positive = window_depth > threshold
    return {
        "pit_detected": positive,
        "threshold_depth": round(threshold, 4),
        "depth_at_window": round(window_depth, 4),
        "window_seconds": window_seconds,
        "time_to_90_recovery": round(time_to_90, 2) if time_to_90 is not None else None,
        "reason": (
            f"Depth remains above {threshold:.3f} at {window_seconds:.1f}s after release"
            if positive
            else f"Depth returned below {threshold:.3f} within {window_seconds:.1f}s"
        ),
    }
