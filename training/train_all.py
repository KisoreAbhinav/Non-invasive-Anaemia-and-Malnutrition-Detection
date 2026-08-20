#!/usr/bin/env python3
"""One-run training script for all 3 pallor models.

Trains pallor_palm, pallor_eye, and pallor_nail using MobileNetV3-Small
in CIELAB color space, then exports all 3 to backend/models/vision/.

Usage:
    python train_all.py

Hyperparameters are configurable at the top of the file.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.models as models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "backend" / "models" / "vision"

# ─── HYPERPARAMETERS (easy to modify) ──────────────────────────────────

BACKBONE = "mobilenet_v3_small"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_SPLIT = 0.15
EPOCHS_PALM_NAIL = 30
EPOCHS_EYE = 25
LR_PALM_NAIL = 3e-3
LR_EYE = 1e-3
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE_EPOCHS = 5
CLASS_WEIGHTS = "balanced"
COLOR_SPACE = "Lab"  # CIELAB for pallor color separation
PRETRAINED = True     # Start from ImageNet RGB weights (first layer adapts)
SEED = 42

# ─── LAB NORMALIZATION (after ToTensor: PIL Lab 0-255 → tensor 0-1) ──
# L channel: 0-100 → [0, 0.392], a/b: -128..127 → [0, 1] (centered ~0.5)
LAB_MEAN = [0.5, 0.5, 0.5]
LAB_STD = [0.5, 0.25, 0.25]


def to_lab(img):
    return img.convert("LAB")

# ─── MODEL DEFINITIONS ─────────────────────────────────────────────────

@dataclass
class ModelSpec:
    test_id: str
    data_dir: str
    class_labels: list[str]
    primary_score_key: str
    epochs: int
    learning_rate: float


MODELS: list[ModelSpec] = [
    ModelSpec(
        test_id="pallor_palm",
        data_dir="data/pallor_palm",
        class_labels=["normal", "risk"],
        primary_score_key="risk",
        epochs=EPOCHS_PALM_NAIL,
        learning_rate=LR_PALM_NAIL,
    ),
    ModelSpec(
        test_id="pallor_eye",
        data_dir="data/pallor_eye",
        class_labels=["normal", "risk"],
        primary_score_key="risk",
        epochs=EPOCHS_EYE,
        learning_rate=LR_EYE,
    ),
    ModelSpec(
        test_id="pallor_nail",
        data_dir="data/pallor_fingernail",
        class_labels=["normal", "risk"],
        primary_score_key="risk",
        epochs=EPOCHS_PALM_NAIL,
        learning_rate=LR_PALM_NAIL,
    ),
]

# ─── DATA PIPELINE ─────────────────────────────────────────────────────

def build_transforms(image_size: int, train: bool = True) -> transforms.Compose:
    steps = []
    if train:
        steps += [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.Lambda(to_lab),
        ]
    else:
        steps += [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(to_lab),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=LAB_MEAN, std=LAB_STD),
    ]
    return transforms.Compose(steps)


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, tf):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.tf = tf
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        img = Image.open(img).convert("RGB")
        return self.tf(img), label


def build_dataloaders(spec: ModelSpec) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tf = build_transforms(IMAGE_SIZE, train=True)
    val_tf = build_transforms(IMAGE_SIZE, train=False)

    data_dir = SCRIPT_DIR / spec.data_dir
    full_dataset = datasets.ImageFolder(str(data_dir), transform=train_tf)
    class_names = full_dataset.classes

    total = len(full_dataset)
    val_count = max(1, int(total * VAL_SPLIT))
    train_count = total - val_count
    train_ds, val_ds = random_split(
        full_dataset, [train_count, val_count],
        generator=torch.Generator().manual_seed(SEED),
    )

    val_ds.dataset = datasets.ImageFolder(str(data_dir), transform=val_tf)
    val_ds = TransformSubset(val_ds, val_tf)

    persistent = NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=persistent,
    )

    LOGGER.info("Dataset %s: %d train, %d val, classes=%s", spec.test_id, train_count, val_count, class_names)
    return train_loader, val_loader, class_names


# ─── MODEL ─────────────────────────────────────────────────────────────

def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ─── TRAINING ──────────────────────────────────────────────────────────

def compute_class_weights(loader: DataLoader, num_classes: int) -> torch.Tensor | None:
    counts = [0] * num_classes
    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def train_model(spec: ModelSpec) -> Path:
    torch.manual_seed(SEED)
    data_dir = SCRIPT_DIR / spec.data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    train_loader, val_loader, class_names = build_dataloaders(spec)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(BACKBONE, len(spec.class_labels), pretrained=PRETRAINED)
    model = model.to(device)
    LOGGER.info("Using device: %s", device)

    weight_tensor = None
    if CLASS_WEIGHTS == "balanced":
        weight_tensor = compute_class_weights(train_loader, len(spec.class_labels))
        weight_tensor = weight_tensor.to(device)
        LOGGER.info("Class weights: %s", weight_tensor.tolist())
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    ckpt_dir = CHECKPOINTS_DIR / spec.test_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    best_val_acc = 0.0

    LOGGER.info("Training %s (%s, %s, %d epochs)", spec.test_id, BACKBONE, COLOR_SPACE, spec.epochs)

    for epoch in range(1, spec.epochs + 1):
        import time
        t0 = time.time()

        if epoch <= FREEZE_BACKBONE_EPOCHS:
            for name, param in model.named_parameters():
                if "classifier" not in name and "fc" not in name:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if "classifier" not in name and "fc" not in name:
                    param.requires_grad = True

        lr = spec.learning_rate if epoch <= FREEZE_BACKBONE_EPOCHS else spec.learning_rate * 0.1
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, spec.epochs - FREEZE_BACKBONE_EPOCHS),
        )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        LOGGER.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f  lr=%.1e  %.1fs",
            epoch, spec.epochs, train_loss, train_acc, val_loss, val_acc, lr, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            LOGGER.info("  Saved best model (val_acc=%.3f)", val_acc)

        if epoch > FREEZE_BACKBONE_EPOCHS:
            scheduler.step()

    LOGGER.info("Training complete. Best val_acc=%.3f  Saved to %s", best_val_acc, best_path)
    return best_path


# ─── EXPORT ────────────────────────────────────────────────────────────

def export_model(spec: ModelSpec, checkpoint: Path) -> None:
    output_dir = DEFAULT_OUTPUT_ROOT / spec.test_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild model
    model = build_model(BACKBONE, len(spec.class_labels), pretrained=False)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Trace to TorchScript
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(model, example_input)

    pt_path = output_dir / "model.pt"
    traced.save(str(pt_path))
    size_mb = pt_path.stat().st_size / 1_048_576
    LOGGER.info("Exported %s: model.pt (%.1f MB)", spec.test_id, size_mb)

    # Save model.json contract
    contract = {
        "contract_version": 1,
        "test_id": spec.test_id,
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "preprocessing": {
            "color_space": COLOR_SPACE,
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "scale": [0.0, 1.0],
            "mean": LAB_MEAN,
            "std": LAB_STD,
        },
        "output_class_labels": spec.class_labels,
        "output_activation": "softmax",
        "primary_score_key": spec.primary_score_key,
        "provenance": {
            "architecture": BACKBONE,
            "training_data": f"Fine-tuned from ImageNet pre-trained {BACKBONE} (CIELAB color space)",
            "checkpoint": str(checkpoint.name),
        },
    }
    json_path = output_dir / "model.json"
    json_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    LOGGER.info("  contract: %s", json_path)

    # Sanity check
    with torch.inference_mode():
        output = model(example_input)
        probs = torch.softmax(output, dim=1)
        LOGGER.info("  sanity check — probs: %s", [round(p, 4) for p in probs.tolist()[0]])


# ─── MAIN ──────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)
    LOGGER.info("Color space: %s", COLOR_SPACE)
    LOGGER.info("Backbone: %s, pretrained=%s", BACKBONE, PRETRAINED)

    results: list[tuple[str, float, Path]] = []
    for spec in MODELS:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("TRAINING: %s", spec.test_id)
        LOGGER.info("=" * 60)
        ckpt = train_model(spec)

        # Read best val_acc from log by re-evaluating
        LOGGER.info("Exporting %s...", spec.test_id)
        export_model(spec, ckpt)

        results.append((spec.test_id, 0.0, ckpt))  # val_acc logged above

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("ALL MODELS COMPLETE")
    LOGGER.info("=" * 60)
    for test_id, _, ckpt in results:
        LOGGER.info("  %s → %s", test_id, ckpt)
        LOGGER.info("  deployed → %s", DEFAULT_OUTPUT_ROOT / test_id)
    LOGGER.info("\nRestart the backend to pick up the new models.")


if __name__ == "__main__":
    main()
