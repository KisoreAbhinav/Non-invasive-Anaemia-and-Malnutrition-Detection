#!/usr/bin/env python3
"""Export a trained model checkpoint to TorchScript + model.json contract.

Usage:
    python export.py --checkpoint checkpoints/pallor/best.pt --test-id pallor
    python export.py --checkpoint checkpoints/pallor/best.pt --test-id pallor --output ../backend/models/vision/pallor
    python export.py --checkpoint checkpoints/hair_skin/best.pt --test-id hair_skin --backbone efficientnet_b0 --num-classes 2

This produces two files in the output directory:
  - model.pt   (CPU-compatible TorchScript module)
  - model.json (backend drop-in contract)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

# Allow importing from training/ and backend/
sys_path = str(Path(__file__).resolve().parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from train import build_model  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "backend" / "models" / "vision"

PREPROCESSING = {
    "color_space": "RGB",
    "resize": [224, 224],
    "scale": [0.0, 1.0],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def export(
    checkpoint_path: Path,
    test_id: str,
    backbone: str,
    num_classes: int,
    class_labels: list[str],
    primary_score_key: str,
    output_activation: str,
    output_dir: Path,
    image_size: int = 224,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rebuild model architecture
    model = build_model(backbone, num_classes, pretrained=False)

    # 2. Load trained weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # 3. Trace to TorchScript
    example_input = torch.randn(1, 3, image_size, image_size)
    traced = torch.jit.trace(model, example_input)

    # 4. Save TorchScript module
    pt_path = output_dir / "model.pt"
    traced.save(str(pt_path))
    size_mb = pt_path.stat().st_size / 1_048_576
    print(f"Saved TorchScript module: {pt_path} ({size_mb:.1f} MB)")

    # 5. Save model.json contract
    contract = {
        "contract_version": 1,
        "test_id": test_id,
        "input_shape": [1, 3, image_size, image_size],
        "preprocessing": {
            **PREPROCESSING,
            "resize": [image_size, image_size],
        },
        "output_class_labels": class_labels,
        "output_activation": output_activation,
        "primary_score_key": primary_score_key,
        "provenance": {
            "architecture": backbone,
            "training_data": f"Fine-tuned from ImageNet pre-trained {backbone}",
            "checkpoint": str(checkpoint_path.name),
        },
    }
    json_path = output_dir / "model.json"
    json_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"Saved contract: {json_path}")

    # 6. Quick sanity check — run inference on dummy input
    with torch.inference_mode():
        output = model(example_input)
        probs = torch.softmax(output, dim=1)
        print(f"Sanity check — output shape: {output.shape}, probs: {probs.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model to TorchScript + model.json")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test-id", required=True, help="Vision test ID (pallor, edema, hair_skin)")
    parser.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--class-labels", nargs="+", default=None, help="Class label names (default: auto-detect)")
    parser.add_argument("--primary-score-key", default=None, help="Primary risk class label (default: last label)")
    parser.add_argument("--output-activation", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default=None, help="Output directory (default: backend/models/vision/{test_id})")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Auto-detect class labels from checkpoint directory structure if not given
    if args.class_labels:
        class_labels = args.class_labels
    else:
        # Try to find a data directory in the corresponding config
        config_path = SCRIPT_DIR / "configs" / f"{args.test_id}.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            class_labels = cfg.get("class_labels", [f"class_{i}" for i in range(args.num_classes)])
        else:
            class_labels = [f"class_{i}" for i in range(args.num_classes)]
            print(f"Warning: No class labels specified. Using: {class_labels}")

    primary_key = args.primary_score_key or class_labels[-1]
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / args.test_id

    export(
        checkpoint_path=checkpoint,
        test_id=args.test_id,
        backbone=args.backbone,
        num_classes=args.num_classes,
        class_labels=class_labels,
        primary_score_key=primary_key,
        output_activation=args.output_activation,
        output_dir=output_dir,
        image_size=args.image_size,
    )

    print(f"\nDone. Model deployed to {output_dir}")
    print("Restart the backend to pick up the new model.")


if __name__ == "__main__":
    main()
