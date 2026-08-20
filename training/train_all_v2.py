#!/usr/bin/env python3
"""One-run training script for all 3 pallor models.

Trains pallor_palm, pallor_eye, and pallor_nail using a MobileNetV3
backbone in CIELAB color space, then exports all 3 as TorchScript to
backend/models/vision/<test_id>/, alongside a model.json contract.

Usage (inside your venv, with a CUDA GPU such as an RTX 3050):
    python train_all.py

Each model is configured by its matching file in ``training/configs``.  The
exported contract is generated from that same configuration, so the model we
train, the model we export, and the preprocessing we advertise cannot drift.

Fixes vs. the previous version of this script (see chat for full detail):
  1. Each config's ``backbone`` is loaded and passed to ``build_model()`` --
     the architecture written in a config is now the architecture trained.
  2. Replaced PIL's Image.convert("LAB") with a correct, continuous
     sRGB -> CIE Lab conversion. PIL's built-in LAB mode encodes negative
     a/b values by wrapping them to the top of the byte range instead of
     offsetting them, so colors that are nearly identical (e.g. a = -1.8
     vs a = +1.8, right around the neutral point where pallor lives) were
     landing at opposite extremes of the input tensor. This alone is the
     most likely cause of "healthy called sick, sick at 50/50" behavior,
     since it corrupts exactly the color signal the model needs.
     >>> IMPORTANT: if your backend does its own preprocessing before
     >>> calling the model, it must use this same conversion (see the
     >>> "preprocessing" block written into each model.json) or you'll
     >>> get train/serve skew. Happy to help match that code too.
  3. Optimizer + LR scheduler are no longer rebuilt from scratch every
     epoch. Previously Adam's momentum state was wiped every epoch and
     the cosine schedule was created-then-discarded before it ever did
     anything, so training was effectively a flat learning rate the
     whole time. Now there's a real cosine decay during fine-tuning.
  4. Dataset class order (["normal", "risk"]) is asserted against what's
     actually on disk instead of assumed -- a folder-naming mismatch
     used to silently swap the meaning of the classes.
  5. Single deterministic dataset scan + explicit index-based split,
     instead of two separate ImageFolder scans that had to happen to
     agree with each other.
  6. Mixed precision (AMP) + cudnn.benchmark for speed/memory on a
     4GB-class GPU like the RTX 3050.
  7. Per-model try/except in main() -- one missing/broken dataset won't
     kill the other two models' training+export.
  8. Export step now re-loads the actual saved TorchScript file and runs
     it against real validation images (not random noise), logging a
     confusion matrix and a handful of true/pred/confidence samples --
     so you get a direct read on the exact failure mode you described.
  9. Stratified train/val split (per-class, seeded): with imbalanced
     datasets a plain random split can silently starve the val fold of
     the minority class; now each class is split proportionally.
  10. Corrupt/unreadable images are detected up front (logged and
      skipped, never crashing a DataLoader worker), plus a defensive
      fallback in the dataset __getitem__.
  11. Previous model.pt / model.json in the backend folder are backed up
      (timestamped; last 5 kept) before being overwritten by a new export.
  12. Logs are written to training/logs/train_all_v2_<timestamp>.log in
      addition to stdout. Run `python train_all_v2.py --smoke` for a fast
      end-to-end check (2 epochs on the smallest dataset + export sanity).
"""

from __future__ import annotations

import datetime
import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "backend" / "models" / "vision"
SMOKE_OUTPUT_ROOT = SCRIPT_DIR / "smoke_exports"

LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
# Only the main process writes the run log; DataLoader worker processes
# re-import this module on Windows (spawn) and would each open their own
# spurious log file otherwise.
if __name__ == "__main__":
    _LOG_FILE = LOG_DIR / f"train_all_v2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _FILE_HANDLER = logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="w")
    _FILE_HANDLER.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_FILE_HANDLER)
    LOGGER.info("Log file: %s", _LOG_FILE)

# ─── HYPERPARAMETERS (easy to modify) ──────────────────────────────────

SEED = 42
USE_AMP = True                       # mixed precision -- big win on a 3050's 4GB VRAM

# ─── LAB NORMALIZATION ──────────────────────────────────────────────────
# After rgb_to_lab_array(): L in [0,100], a/b roughly in [-128, 127].
# We map L -> [0,1] via /100, and a/b -> [0,1] via (val+128)/255 (a proper,
# continuous, monotonic mapping -- see rgb_to_lab_array / LabTensor below).
# mean=0.5 then centers all three channels; std=0.25 on a/b gives them a
# bit more spread since real skin-tone a/b variation is a fairly narrow
# slice of the full theoretical range.
LAB_MEAN = [0.5, 0.5, 0.5]
LAB_STD = [0.5, 0.25, 0.25]

# ─── MODEL DEFINITIONS ─────────────────────────────────────────────────


@dataclass
class ModelSpec:
    test_id: str
    backbone: str
    data_dir: str
    class_labels: list[str]
    primary_score_key: str
    image_size: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    fine_tune_lr_factor: float
    weight_decay: float
    freeze_backbone_epochs: int
    val_split: float
    class_weights: str
    color_space: str
    pretrained: bool
    output_activation: str
    resize: list[int]
    center_crop: int
    scale: list[float]
    mean: list[float]
    std: list[float]
    capture_zoom: float
    capture_saturation: float
    capture_contrast: float


def load_model_spec(config_name: str) -> ModelSpec:
    """Load and validate one config instead of silently ignoring it.

    ``train_all_v2.py`` currently supports the Lab pipeline only.  Failing
    early is safer than accepting RGB metadata while training Lab tensors.
    """
    config_path = SCRIPT_DIR / "configs" / config_name
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    preprocessing = raw["preprocessing"]
    capture = preprocessing.get("capture", {})
    image_size = int(raw["image_size"])
    resize = [int(v) for v in preprocessing["resize"]]
    center_crop = int(preprocessing.get("center_crop", image_size))
    if raw.get("num_classes") != len(raw["class_labels"]):
        raise ValueError(f"{config_path}: num_classes must equal class_labels length")
    if raw.get("color_space", preprocessing.get("color_space", "")).upper() != "LAB":
        raise ValueError(f"{config_path}: only Lab color_space is supported")
    if len(resize) != 2 or resize[0] != resize[1] or center_crop != image_size:
        raise ValueError(
            f"{config_path}: resize must be square [width, height] and center_crop must equal image_size"
        )
    return ModelSpec(
        test_id=raw["test_id"],
        backbone=raw["backbone"],
        data_dir=raw["data_dir"],
        class_labels=list(raw["class_labels"]),
        primary_score_key=raw["primary_score_key"],
        image_size=image_size,
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        num_workers=int(raw.get("num_workers", 4)),
        learning_rate=float(raw["learning_rate"]),
        fine_tune_lr_factor=float(raw.get("fine_tune_lr_factor", 0.03)),
        weight_decay=float(raw["weight_decay"]),
        freeze_backbone_epochs=int(raw["freeze_backbone_epochs"]),
        val_split=float(raw["val_split"]),
        class_weights=raw["class_weights"],
        color_space="Lab",
        pretrained=bool(raw["pretrained"]),
        output_activation=raw["output_activation"],
        resize=resize,
        center_crop=center_crop,
        scale=list(preprocessing["scale"]),
        mean=list(preprocessing["mean"]),
        std=list(preprocessing["std"]),
        capture_zoom=float(capture.get("center_zoom", 1.0)),
        capture_saturation=float(capture.get("saturation", 1.0)),
        capture_contrast=float(capture.get("contrast", 1.0)),
    )


MODELS: list[ModelSpec] = [
    load_model_spec("pallor_palm.json"),
    load_model_spec("pallor_eye.json"),
    load_model_spec("pallor_nail.json"),
]

# ─── COLOR CONVERSION (correct sRGB -> CIE Lab) ────────────────────────
# PIL's Image.convert("LAB") is NOT used here -- see module docstring for
# why. This is a standard D65 sRGB -> linear RGB -> XYZ -> Lab pipeline,
# vectorized with numpy, applied to one image at a time in the transform.

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883  # D65 reference white
_DELTA = 6.0 / 29.0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA**3, np.cbrt(t), t / (3 * _DELTA**2) + 4.0 / 29.0)


def rgb_to_lab_array(rgb_uint8: np.ndarray) -> np.ndarray:
    """rgb_uint8: (H, W, 3) uint8 array -> (H, W, 3) float32 Lab array.

    L in [0, 100]; a, b roughly in [-128, 127] for real-world colors.
    """
    rgb = rgb_uint8.astype(np.float64) / 255.0
    linear = _srgb_to_linear(rgb)
    xyz = linear @ _SRGB_TO_XYZ.T
    fx = _f(xyz[..., 0] / _XN)
    fy = _f(xyz[..., 1] / _YN)
    fz = _f(xyz[..., 2] / _ZN)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


class LabTensor:
    """PIL RGB image -> normalized (0..1 per-channel) Lab tensor, C,H,W."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        lab = rgb_to_lab_array(rgb)
        lab[..., 0] = lab[..., 0] / 100.0
        lab[..., 1:] = np.clip((lab[..., 1:] + 128.0) / 255.0, 0.0, 1.0)
        return torch.from_numpy(lab.transpose(2, 0, 1)).float()


class CaptureTransform:
    """Match the camera transform declared in the exported model contract."""

    def __init__(self, zoom: float, saturation: float, contrast: float) -> None:
        self.zoom = zoom
        self.saturation = saturation
        self.contrast = contrast

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.zoom > 1:
            width, height = img.size
            side = max(1, int(min(width, height) / self.zoom))
            left = (width - side) // 2
            top = (height - side) // 2
            img = img.crop((left, top, left + side, top + side))
        if self.saturation != 1:
            img = ImageEnhance.Color(img).enhance(self.saturation)
        if self.contrast != 1:
            img = ImageEnhance.Contrast(img).enhance(self.contrast)
        return img


# ─── DATA PIPELINE ─────────────────────────────────────────────────────


def build_transforms(spec: ModelSpec, train: bool = True) -> transforms.Compose:
    steps: list = [CaptureTransform(spec.capture_zoom, spec.capture_saturation, spec.capture_contrast)]
    if train:
        steps += [
            transforms.Resize(spec.resize[0]),  # shorter side; preserves aspect ratio
            transforms.RandomCrop(spec.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # Kept mild on purpose: this task's signal IS a subtle color
            # shift, so aggressive color jitter can wash out exactly what
            # the model needs to learn.
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        ]
    else:
        steps += [
            transforms.Resize(spec.resize[0]),  # must match backend's shorter-side resize
            transforms.CenterCrop(spec.center_crop),
        ]
    steps += [LabTensor(), transforms.Normalize(mean=spec.mean, std=spec.std)]
    return transforms.Compose(steps)


class PallorDataset(Dataset):
    """Wraps a fixed (path, label) sample list + explicit index subset.

    Using one shared samples list (from a single ImageFolder scan) for
    both train and val avoids ever having to trust that two separate
    directory scans landed in the same order.
    """

    def __init__(self, samples: list[tuple[str, int]], indices: list[int], transform, spec: ModelSpec):
        self.samples = samples
        self.indices = indices
        self.transform = transform
        # Defensive fallback -- should never be reached because
        # build_dataloaders() pre-filters corrupt files; this only guards
        # against a file going bad between the scan and an epoch.
        self._fallback_img = Image.new("RGB", tuple(spec.resize), (128, 128, 128))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[self.indices[idx]]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception as exc:  # noqa: BLE001 - skip, don't crash the worker
            LOGGER.error("Skipping unreadable image %s: %s", path, exc)
            return self.transform(self._fallback_img), label


def verify_samples(samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Drop files that cannot be opened/decoded; log each one."""
    valid: list[tuple[str, int]] = []
    for path, label in samples:
        try:
            with Image.open(path) as im:
                im.convert("RGB")
            valid.append((path, label))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Corrupt image skipped: %s (%s)", path, exc)
    if len(valid) != len(samples):
        LOGGER.warning("Dropped %d unreadable image(s) for this dataset.", len(samples) - len(valid))
    return valid


def stratified_split_indices(
    samples: list[tuple[str, int]], val_split: float, seed: int
) -> tuple[list[int], list[int]]:
    """Per-class (stratified) train/val index split, seeded for re-runs.

    Both classes always contribute proportionally to the val fold, so a
    small or imbalanced dataset can't silently get a val set that is
    missing the minority class.
    """
    generator = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(samples):
        by_class.setdefault(label, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(by_class):
        idxs = by_class[label]
        perm = torch.randperm(len(idxs), generator=generator).tolist()
        shuffled = [idxs[p] for p in perm]
        # Keep at least 1 sample per class in train; val is proportional.
        if len(shuffled) > 1:
            n_val = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * val_split))))
            val_idx += shuffled[:n_val]
            train_idx += shuffled[n_val:]
        else:
            train_idx += shuffled
    return train_idx, val_idx


def compute_class_weights(samples: list[tuple[str, int]], indices: list[int], num_classes: int) -> torch.Tensor:
    counts = [0] * num_classes
    for i in indices:
        counts[samples[i][1]] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(spec: ModelSpec):
    data_dir = SCRIPT_DIR / spec.data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    base = datasets.ImageFolder(str(data_dir))  # transform=None: just a directory scan
    class_names = base.classes
    if class_names != spec.class_labels:
        raise ValueError(
            f"{spec.test_id}: dataset classes on disk {class_names} do not match "
            f"expected {spec.class_labels}. Check the subfolder names under {data_dir} "
            "-- this would otherwise silently swap what 'risk' and 'normal' mean."
        )

    samples = verify_samples(base.samples)
    if not samples:
        raise ValueError(f"{spec.test_id}: no readable images found under {data_dir}")

    n = len(samples)
    train_idx, val_idx = stratified_split_indices(samples, spec.val_split, SEED)
    train_n, val_n = len(train_idx), len(val_idx)

    def class_counts(indices: list[int]) -> str:
        counts = Counter(samples[i][1] for i in indices)
        return ", ".join(f"{class_names[c]}={counts.get(c, 0)}" for c in sorted(counts))

    LOGGER.info(
        "Dataset %s: %d train, %d val, classes=%s  (train: %s; val: %s)",
        spec.test_id, train_n, val_n, class_names, class_counts(train_idx), class_counts(val_idx),
    )

    train_ds = PallorDataset(samples, train_idx, build_transforms(spec, train=True), spec)
    val_ds = PallorDataset(samples, val_idx, build_transforms(spec, train=False), spec)

    class_weights = None
    if spec.class_weights == "balanced":
        class_weights = compute_class_weights(samples, train_idx, len(class_names))

    pin = torch.cuda.is_available()
    persistent = spec.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=spec.batch_size, shuffle=True,
        num_workers=spec.num_workers, pin_memory=pin, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=spec.batch_size, shuffle=False,
        num_workers=spec.num_workers, pin_memory=pin, persistent_workers=persistent,
    )

    return train_loader, val_loader, class_names, class_weights


# ─── MODEL ─────────────────────────────────────────────────────────────


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif backbone == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ─── TRAINING ──────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        preds = outputs.argmax(1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion[t.long(), p.long()] += 1
    return total_loss / total, correct / total, confusion


def train_model(
    spec: ModelSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    class_weights: torch.Tensor | None,
) -> tuple[Path, float]:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    model = build_model(spec.backbone, len(spec.class_labels), pretrained=spec.pretrained).to(device)
    LOGGER.info("Using device: %s (amp=%s)", device, use_amp)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        LOGGER.info("Class weights: %s", class_weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ckpt_dir = CHECKPOINTS_DIR / spec.test_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    best_val_acc = 0.0
    best_confusion = None

    # Freeze everything except the classifier head for the first
    # spec.freeze_backbone_epochs epochs.
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name or name.startswith("fc")

    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=spec.learning_rate, weight_decay=spec.weight_decay,
    )
    scheduler = None  # only meaningful once the backbone unfreezes (see below)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    LOGGER.info("Training %s (%s, %s, %d epochs)", spec.test_id, spec.backbone, spec.color_space, spec.epochs)

    for epoch in range(1, spec.epochs + 1):
        t0 = time.time()

        if epoch == spec.freeze_backbone_epochs + 1:
            LOGGER.info("  Unfreezing backbone at epoch %d", epoch)
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(
                model.parameters(), lr=spec.learning_rate * spec.fine_tune_lr_factor, weight_decay=spec.weight_decay,
            )
            remaining = max(1, spec.epochs - spec.freeze_backbone_epochs)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc, confusion = evaluate(model, val_loader, criterion, device, len(class_names), use_amp)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        LOGGER.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f  lr=%.1e  %.1fs",
            epoch, spec.epochs, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_confusion = confusion.clone()
            torch.save(model.state_dict(), best_path)
            LOGGER.info("  Saved best model (val_acc=%.3f)", val_acc)

        if scheduler is not None:
            scheduler.step()

    LOGGER.info("Training complete. Best val_acc=%.3f  Saved to %s", best_val_acc, best_path)
    if best_confusion is not None:
        LOGGER.info("  Confusion matrix (rows=true, cols=pred), classes=%s:\n%s", class_names, best_confusion)
    return best_path, best_val_acc


# ─── EXPORT ────────────────────────────────────────────────────────────


def _back_up_existing(output_dir: Path, keep: int = 5) -> None:
    """Timestamped copies of an existing model.pt/model.json before overwrite.

    Keeps the last *keep* generations per extension so a bad run never
    silently destroys the last working production model.
    """
    pt_path, json_path = output_dir / "model.pt", output_dir / "model.json"
    if not pt_path.exists() and not json_path.exists():
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if pt_path.exists():
        shutil.copy2(pt_path, output_dir / f"model.{stamp}.pt")
    if json_path.exists():
        shutil.copy2(json_path, output_dir / f"model.{stamp}.json")
    for extension in ("pt", "json"):
        old = sorted(output_dir.glob(f"model.*.{extension}"))[:-keep]
        for stale in old:
            stale.unlink()
    LOGGER.info("Backed up previous model files in %s (stamp %s, keeping %d)", output_dir, stamp, keep)


def export_model(
    spec: ModelSpec,
    checkpoint: Path,
    val_loader: DataLoader,
    best_val_acc: float,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> None:
    output_dir = output_root / spec.test_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _back_up_existing(output_dir)

    model = build_model(spec.backbone, len(spec.class_labels), pretrained=False)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.randn(1, 3, spec.image_size, spec.image_size)
    traced = torch.jit.trace(model, example_input)

    pt_path = output_dir / "model.pt"
    traced.save(str(pt_path))
    size_mb = pt_path.stat().st_size / 1_048_576
    LOGGER.info("Exported %s -> %s (%.1f MB)", spec.test_id, pt_path, size_mb)

    contract = {
        "contract_version": 1,
        "test_id": spec.test_id,
        "input_shape": [1, 3, spec.image_size, spec.image_size],
        "preprocessing": {
            "color_space": spec.color_space,
            "lab_conversion": (
                "Standard sRGB->linear->XYZ(D65)->CIE Lab, NOT PIL's Image.convert('LAB'). "
                "L in [0,100] -> divide by 100. a,b roughly in [-128,127] -> (val+128)/255, clipped to [0,1]."
            ),
            "resize": spec.resize,
            "center_crop": spec.center_crop,
            "scale": spec.scale,
            "mean": spec.mean,
            "std": spec.std,
            "capture": {
                "center_zoom": spec.capture_zoom,
                "saturation": spec.capture_saturation,
                "contrast": spec.capture_contrast,
            },
        },
        "output_class_labels": spec.class_labels,
        "output_activation": spec.output_activation,
        "primary_score_key": spec.primary_score_key,
        "provenance": {
            "architecture": spec.backbone,
            "training_data": f"Fine-tuned from ImageNet pre-trained {spec.backbone} (CIELAB color space)",
            "checkpoint": str(checkpoint.name),
        },
    }
    json_path = output_dir / "model.json"
    json_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    LOGGER.info("  contract: %s", json_path)

    # Real sanity check: reload the exact file just written and run it
    # against actual validation images (not random noise), so any
    # trace/export drift -- or a still-bad decision boundary -- shows up
    # here instead of only after it's deployed.
    traced_cpu = torch.jit.load(str(pt_path), map_location="cpu")
    traced_cpu.eval()
    correct, total, shown = 0, 0, 0
    confusion = torch.zeros(len(spec.class_labels), len(spec.class_labels), dtype=torch.long)
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = traced_cpu(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1
            if shown < 8:
                take = min(8 - shown, images.size(0))
                for i in range(take):
                    LOGGER.info(
                        "    sample: true=%s pred=%s probs=%s",
                        spec.class_labels[labels[i].item()],
                        spec.class_labels[preds[i].item()],
                        [round(p, 3) for p in probs[i].tolist()],
                    )
                shown += take
    acc = correct / total if total else 0.0
    contract["provenance"]["validation"] = {
        "best_epoch_accuracy": round(best_val_acc, 6),
        "exported_model_accuracy": round(acc, 6),
        "sample_count": total,
        "confusion_matrix": confusion.tolist(),
        "class_order": spec.class_labels,
    }
    # Persist the metrics beside the exact TorchScript file that produced
    # them. This makes a deployed model auditable without hunting logs.
    json_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    LOGGER.info("  Exported-model accuracy on real val data: %.3f (%d samples)", acc, total)
    LOGGER.info("  Exported-model confusion matrix (rows=true, cols=pred): %s\n%s", spec.class_labels, confusion)


# ─── MAIN ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export pallor models")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="2-epoch eye export to training/smoke_exports only")
    mode.add_argument("--only", choices=[spec.test_id for spec in MODELS], help="train and deploy one model only")
    args = parser.parse_args()
    SMOKE_EPOCHS = 2
    smoke_mode = args.smoke
    if smoke_mode:
        LOGGER.info("SMOKE MODE: tiny pass (2 epochs, smallest dataset) + export sanity check.")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input size -> free speedup
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
    LOGGER.info("Device: %s", device)
    LOGGER.info("Color space: Lab (proper sRGB->Lab, see module docstring)")
    LOGGER.info("Loaded %d model configs from %s", len(MODELS), SCRIPT_DIR / "configs")

    specs: list[ModelSpec] = MODELS
    if smoke_mode:
        # Pallor eye is the smallest dataset (710 images): fastest full pipeline check.
        specs = [replace(MODELS[1], epochs=SMOKE_EPOCHS)]
        LOGGER.info("Smoke-testing only: %s (%d epochs)", specs[0].test_id, SMOKE_EPOCHS)
    elif args.only:
        specs = [spec for spec in MODELS if spec.test_id == args.only]
        LOGGER.info("Training only: %s", args.only)

    results: list[tuple[str, float, Path]] = []
    for spec in specs:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("TRAINING: %s", spec.test_id)
        LOGGER.info("=" * 60)
        try:
            train_loader, val_loader, class_names, class_weights = build_dataloaders(spec)
            ckpt, val_acc = train_model(spec, train_loader, val_loader, class_names, class_weights)

            LOGGER.info("Exporting %s...", spec.test_id)
            output_root = SMOKE_OUTPUT_ROOT if smoke_mode else DEFAULT_OUTPUT_ROOT
            export_model(spec, ckpt, val_loader, val_acc, output_root)

            results.append((spec.test_id, val_acc, ckpt))
        except Exception:
            LOGGER.error("Training/export FAILED for %s -- skipping to the next model.", spec.test_id)
            LOGGER.error(traceback.format_exc())
            continue

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("RUN COMPLETE")
    LOGGER.info("=" * 60)
    for test_id, val_acc, ckpt in results:
        LOGGER.info("  %s  val_acc=%.3f  checkpoint=%s", test_id, val_acc, ckpt)
        destination = SMOKE_OUTPUT_ROOT if smoke_mode else DEFAULT_OUTPUT_ROOT
        LOGGER.info("  exported -> %s", destination / test_id)
    failed = {spec.test_id for spec in specs} - {r[0] for r in results}
    if failed:
        LOGGER.warning("  Did NOT complete: %s -- see errors above.", sorted(failed))
    LOGGER.info("\nRestart the backend to pick up the new models.")


if __name__ == "__main__":
    main()
