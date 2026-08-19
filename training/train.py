#!/usr/bin/env python3
"""Fine-tune edge-optimized vision models for the Nourish screening pipeline.

Usage:
    python train.py --config configs/pallor.json
    python train.py --config configs/pallor.json --data-dir /path/to/custom/data
    python train.py --config configs/pallor.json --epochs 30 --lr 1e-4

Expects ImageFolder layout (one subfolder per class).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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


# ── Defaults ──────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "image_size": 224,
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
    "freeze_backbone_epochs": 5,
    "val_split": 0.15,
    "num_workers": 0,
    "class_weights": "none",
    "preprocessing": {
        "color_space": "RGB",
        "resize": [224, 224],
        "scale": [0.0, 1.0],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


class TransformSubset(torch.utils.data.Dataset):
    """Wrap a Subset with a different transform for validation."""

    def __init__(self, subset: torch.utils.data.Subset, tf: transforms.Compose) -> None:
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.tf = tf

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        img = Image.open(img).convert("RGB")
        return self.tf(img), label


# ── Data ──────────────────────────────────────────────────────────────

def build_transforms(image_size: int, mean: list[float], std: list[float], train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    mean: list[float],
    std: list[float],
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tf = build_transforms(image_size, mean, std, train=True)
    val_tf = build_transforms(image_size, mean, std, train=False)

    full_dataset = datasets.ImageFolder(str(data_dir), transform=train_tf)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    total = len(full_dataset)
    val_count = max(1, int(total * val_split))
    train_count = total - val_count
    train_ds, val_ds = random_split(full_dataset, [train_count, val_count])

    # Apply validation transforms to val subset
    val_ds.dataset = datasets.ImageFolder(str(data_dir), transform=val_tf)
    # random_split uses indices, so val_ds still uses train transforms on the
    # underlying dataset.  Override with a wrapper:
    val_ds = TransformSubset(val_ds, val_tf)  # type: ignore[assignment]

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
    )

    LOGGER.info("Dataset: %d train, %d val, %d classes: %s", train_count, val_count, num_classes, class_names)
    return train_loader, val_loader, class_names


# ── Model ─────────────────────────────────────────────────────────────

def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Use mobilenet_v3_small or efficientnet_b0.")
    return model


def freeze_backbone(model: nn.Module, freeze: bool) -> None:
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = not freeze


# ── Training ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    test_id: str = "pallor"
    backbone: str = "mobilenet_v3_small"
    data_dir: str = ""
    num_classes: int = 2
    class_labels: list[str] = field(default_factory=lambda: ["normal", "risk"])
    primary_score_key: str = "risk"
    image_size: int = 224
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    freeze_backbone_epochs: int = 5
    val_split: float = 0.15
    num_workers: int = 0
    class_weights: str = "none"
    output_activation: str = "softmax"
    preprocessing: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["preprocessing"]))

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Merge with defaults
        merged = {**DEFAULTS, **raw}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in known}
        return cls(**filtered)


def compute_class_weights(loader: DataLoader, num_classes: int) -> torch.Tensor | None:
    counts = [0] * num_classes
    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
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
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def train(config: TrainConfig) -> Path:
    data_dir = Path(config.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    mean = config.preprocessing.get("mean", [0.485, 0.456, 0.406])
    std = config.preprocessing.get("std", [0.229, 0.224, 0.225])

    train_loader, val_loader, class_names = build_dataloaders(
        data_dir, config.image_size, mean, std,
        config.batch_size, config.val_split, config.num_workers,
    )

    model = build_model(config.backbone, config.num_classes, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    LOGGER.info("Using device: %s", device)

    # Class weights for imbalanced datasets
    weight_tensor = None
    if config.class_weights == "balanced":
        weight_tensor = compute_class_weights(train_loader, config.num_classes)
        weight_tensor = weight_tensor.to(device)
        LOGGER.info("Class weights: %s", weight_tensor.tolist())
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Output directory
    ckpt_dir = CHECKPOINTS_DIR / config.test_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_path = ckpt_dir / "best.pt"

    LOGGER.info("Starting training: %s (%s) for %d epochs", config.test_id, config.backbone, config.epochs)

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        # Freeze/unfreeze backbone
        if epoch <= config.freeze_backbone_epochs:
            freeze_backbone(model, freeze=True)
            if epoch == 1:
                LOGGER.info("Backbone frozen for first %d epochs", config.freeze_backbone_epochs)
        else:
            freeze_backbone(model, freeze=False)
            if epoch == config.freeze_backbone_epochs + 1:
                LOGGER.info("Backbone unfrozen at epoch %d", epoch)

        # Lower LR after unfreezing
        lr = config.learning_rate if epoch <= config.freeze_backbone_epochs else config.learning_rate * 0.1
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs - config.freeze_backbone_epochs)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        LOGGER.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f  lr=%.1e  %.1fs",
            epoch, config.epochs, train_loss, train_acc, val_loss, val_acc, lr, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            LOGGER.info("  → Saved best model (val_acc=%.3f)", val_acc)

        if epoch > config.freeze_backbone_epochs:
            scheduler.step()

    LOGGER.info("Training complete. Best val_acc=%.3f  Saved to %s", best_val_acc, best_path)
    return best_path


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a vision model for Nourish screening")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--data-dir", help="Override dataset directory from config")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--class-weights", choices=["none", "balanced"], help="Class weight strategy")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.class_weights:
        config.class_weights = args.class_weights

    train(config)


if __name__ == "__main__":
    main()
