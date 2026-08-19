"""Tiny vision-only classifier for rendered synthetic camera frame sequences.

The feature extractor receives only frame-like intensity maps generated from the
surface at fixed times; it is deliberately separate from the deterministic
detector.  It is an explainable laptop-fast demonstration, not a clinical model.
"""
from __future__ import annotations

from math import exp, sqrt
from statistics import mean


FRAME_OFFSETS = (0.0, 0.8, 3.0, 8.0, 15.0)


def _depth_at(series: list[dict], t: float) -> float:
    return float(min(series, key=lambda point: abs(float(point["t"]) - t))["depth"])


def render_camera_frame(depth: float, size: int = 13) -> list[float]:
    """Produce a tiny greyscale pseudo-frame of a circular dent, no label/state."""
    pixels: list[float] = []
    middle = (size - 1) / 2
    for y in range(size):
        for x in range(size):
            radius2 = ((x - middle) / middle) ** 2 + ((y - middle) / middle) ** 2
            dent = depth * exp(-radius2 * 3.3)
            # Lighting-like radial variation makes this resemble a camera view,
            # while the model only observes pixels, never hidden parameters.
            pixel = max(0.0, min(1.0, 0.92 - dent * 0.72 + (x - middle) * 0.006))
            pixels.append(pixel)
    return pixels


def camera_features(series: list[dict], duration: float) -> list[float]:
    """Extract compact visual features from synthetic frames after release."""
    centre = (13 * 13) // 2
    features: list[float] = []
    for offset in FRAME_OFFSETS:
        frame = render_camera_frame(_depth_at(series, duration + offset))
        # Centre darkness and frame-wide mean are simple image measurements.
        features.extend([1 - frame[centre], 1 - mean(frame)])
    return features


def _sigmoid(value: float) -> float:
    if value < -30:
        return 0.0
    if value > 30:
        return 1.0
    return 1 / (1 + exp(-value))


def train(records: list[dict], epochs: int = 420, learning_rate: float = 0.35) -> dict:
    if len(records) < 8:
        raise ValueError("Generate at least 8 synthetic records before training.")
    samples = [
        (camera_features(record["press"]["time_series"], record["press"]["duration"]), int(record["label"]["has_malnutrition_or_severe_anemia"]))
        for record in records
    ]
    dimensions = len(samples[0][0])
    means = [mean(row[0][i] for row in samples) for i in range(dimensions)]
    scales = [sqrt(mean((row[0][i] - means[i]) ** 2 for row in samples)) or 1.0 for i in range(dimensions)]
    normalized = [([(value - means[i]) / scales[i] for i, value in enumerate(values)], label) for values, label in samples]
    weights = [0.0] * dimensions
    bias = 0.0
    for _ in range(epochs):
        for values, label in normalized:
            probability = _sigmoid(bias + sum(weight * value for weight, value in zip(weights, values)))
            error = probability - label
            bias -= learning_rate * error
            for i, value in enumerate(values):
                weights[i] -= learning_rate * error * value
    correct = sum(
        int((_sigmoid(bias + sum(weight * value for weight, value in zip(weights, values))) >= 0.5) == bool(label))
        for values, label in normalized
    )
    return {"weights": weights, "bias": bias, "means": means, "scales": scales, "training_samples": len(records), "training_accuracy": round(correct / len(records), 3)}


def predict(model: dict, series: list[dict], duration: float) -> dict:
    raw = camera_features(series, duration)
    values = [(value - model["means"][i]) / model["scales"][i] for i, value in enumerate(raw)]
    probability = _sigmoid(model["bias"] + sum(weight * value for weight, value in zip(model["weights"], values)))
    return {"available": True, "pit_detected": probability >= 0.5, "confidence": round(probability if probability >= 0.5 else 1 - probability, 3), "positive_probability": round(probability, 3)}
