#!/usr/bin/env python3
"""Build edge-optimized TorchScript vision models for the drop-in contract.

Creates model.pt + model.json for pallor, edema, and hair_skin using
torchvision pre-trained backbones with replaced 2-class heads.

Run once from the backend directory:

    python scripts/setup_vision_models.py

The resulting models use ImageNet pre-trained weights.  Replace with
domain-fine-tuned checkpoints when clinical training data is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = PROJECT_ROOT / "models" / "vision"

INPUT_SHAPE = [1, 3, 224, 224]
PREPROCESSING = {
    "color_space": "RGB",
    "resize": [224, 224],
    "scale": [0.0, 1.0],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


# ── Model specifications ──────────────────────────────────────────────

MODELS: list[dict] = [
    {
        "test_id": "pallor_eye",
        "backbone": "mobilenet_v3_small",
        "description": "Conjunctival pallor classifier (anemia risk from inner eyelid)",
        "class_labels": ["normal", "risk"],
        "primary_score_key": "risk",
    },
    {
        "test_id": "pallor_nail",
        "backbone": "mobilenet_v3_small",
        "description": "Nail bed pallor classifier (anemia risk from fingernail colour)",
        "class_labels": ["normal", "risk"],
        "primary_score_key": "risk",
    },
    {
        "test_id": "pallor_palm",
        "backbone": "mobilenet_v3_small",
        "description": "Palm pallor classifier (anemia risk from palm colour)",
        "class_labels": ["normal", "risk"],
        "primary_score_key": "risk",
    },
    {
        "test_id": "edema",
        "backbone": "mobilenet_v3_small",
        "description": "Bilateral pitting edema detector from foot/shin images",
        "class_labels": ["normal", "pitting_edema"],
        "primary_score_key": "pitting_edema",
    },
    {
        "test_id": "hair_skin",
        "backbone": "efficientnet_b0",
        "description": "Hair texture, skin lesions & visible wasting sign classifier",
        "class_labels": ["normal", "signs_detected"],
        "primary_score_key": "signs_detected",
    },
]


def _build_model(spec: dict) -> torch.nn.Module:
    """Load a pre-trained backbone and replace the classifier for 2-class output."""
    backbone = spec["backbone"]
    num_classes = len(spec["class_labels"])

    if backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # MobileNetV3-Small classifier: Sequential(Linear(576,1024), Hardswish, Dropout, Linear(1024,num_classes))
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # EfficientNet-B0 classifier: Sequential(Dropout, Linear(1280,num_classes))
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.eval()
    return model


def build_and_save(spec: dict) -> None:
    test_id: str = spec["test_id"]
    model_dir = VISION_DIR / test_id
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Building {test_id} ({spec['backbone']})...")
    model = _build_model(spec)

    # Trace to TorchScript with the declared input shape
    example = torch.randn(*INPUT_SHAPE)
    traced = torch.jit.trace(model, example)

    # Save TorchScript module
    traced.save(str(model_dir / "model.pt"))

    # Save contract JSON
    contract = {
        "contract_version": 1,
        "test_id": test_id,
        "input_shape": INPUT_SHAPE,
        "preprocessing": PREPROCESSING,
        "output_class_labels": spec["class_labels"],
        "output_activation": "softmax",
        "primary_score_key": spec["primary_score_key"],
        "provenance": {
            "architecture": spec["backbone"],
            "training_data": "ImageNet pre-trained (replace with domain-fine-tuned checkpoint)",
        },
    }
    (model_dir / "model.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    size_mb = (model_dir / "model.pt").stat().st_size / 1_048_576
    print(f"  ✓ {test_id}: model.pt ({size_mb:.1f} MB) + model.json")


def main() -> None:
    print(f"Building TorchScript vision models → {VISION_DIR}\n")
    for spec in MODELS:
        build_and_save(spec)
    print("\nDone. Restart the backend to pick up the new models.")


if __name__ == "__main__":
    main()
