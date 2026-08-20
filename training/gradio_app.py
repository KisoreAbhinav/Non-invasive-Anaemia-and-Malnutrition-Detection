#!/usr/bin/env python3
"""Gradio UI for the three deployed pallor models (palm, eye, nail).

Runs standalone from the training venv. Loads model.pt + model.json
from backend/models/vision/<test_id>/ once at startup and classifies
one uploaded image per body site on a single page.

Usage:
    python gradio_app.py              # launch UI on http://127.0.0.1:7860
    python gradio_app.py --port 7861  # custom port
    python gradio_app.py --smoke      # headless: one prediction per site
    python gradio_app.py --check-lab  # numeric parity vs train_all_v2 LabTensor
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent / "backend" / "models" / "vision"

MODEL_IDS = ["pallor_palm", "pallor_eye", "pallor_nail"]
SITE_TITLES = {
    "pallor_palm": "Palm palmar pallor",
    "pallor_eye": "Conjunctival (eye) pallor",
    "pallor_nail": "Nail bed pallor",
}
DATA_DIRS = {
    "pallor_palm": "data/pallor_palm",
    "pallor_eye": "data/pallor_eye",
    "pallor_nail": "data/pallor_fingernail",
}

# ─── CIE Lab conversion ────────────────────────────────────────────────
# Identical math to backend/app/flows/vision_inference.py and
# train_all_v2.py::LabTensor. PIL's Image.convert("LAB") is deliberately
# NOT used: it wraps negative a/b values to the top of the byte range,
# corrupting the exact color signal pallor detection needs.

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


def rgb_to_lab_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB image -> (1, 3, H, W) float tensor in [0, 1] (L/100, a,b offset)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb_f = rgb.astype(np.float64) / 255.0
    linear = _srgb_to_linear(rgb_f)
    xyz = linear @ _SRGB_TO_XYZ.T
    fx = _f(xyz[..., 0] / _XN)
    fy = _f(xyz[..., 1] / _YN)
    fz = _f(xyz[..., 2] / _ZN)
    l_ = 116.0 * fy - 16.0
    a_ = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    lab = np.stack([l_, a_, b_], axis=-1)
    lab[..., 0] = lab[..., 0] / 100.0
    lab[..., 1:] = np.clip((lab[..., 1:] + 128.0) / 255.0, 0.0, 1.0)
    return torch.from_numpy(lab.transpose(2, 0, 1)).float().unsqueeze(0)


# ─── Contract-driven preprocessing ─────────────────────────────────────

def preprocess(image: Image.Image, metadata: dict[str, Any]) -> torch.Tensor:
    """Match backend _image_tensor: resize -> center crop -> Lab -> normalize."""
    preprocessing = metadata["preprocessing"]
    capture = preprocessing.get("capture", {})
    zoom = float(capture.get("center_zoom", 1.0))
    if zoom > 1.0:
        width, height = image.size
        side = max(1, int(min(width, height) / zoom))
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    saturation = float(capture.get("saturation", 1.0))
    contrast = float(capture.get("contrast", 1.0))
    if saturation != 1.0:
        image = ImageEnhance.Color(image).enhance(saturation)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    shape = metadata["input_shape"]
    resize = preprocessing.get("resize", [shape[3], shape[2]])
    target_w, target_h = int(resize[0]), int(resize[1])
    center_crop = preprocessing.get("center_crop")
    if center_crop:
        w, h = image.size
        if target_w == target_h:
            scale_factor = target_w / min(w, h)
        else:
            scale_factor = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        cc = int(center_crop)
        left = (new_w - cc) // 2
        top = (new_h - cc) // 2
        image = image.crop((left, top, left + cc, top + cc))
    else:
        image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)

    color_space = preprocessing.get("color_space", "RGB")
    if color_space.upper() == "LAB":
        tensor = rgb_to_lab_tensor(image)  # already in [0, 1]
    else:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
        if preprocessing.get("scale") == [0.0, 1.0] or preprocessing.get("scale") == [0, 1]:
            tensor = tensor / 255.0

    mean, std = preprocessing.get("mean"), preprocessing.get("std")
    if mean is not None and std is not None:
        mt = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
        st = torch.tensor(std, dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mt) / st
    return tensor


# ─── Model loading (once at startup) ───────────────────────────────────

class ModelEntry:
    """One deployed model: metadata contract, TorchScript module, error state."""

    def __init__(self, directory: Path) -> None:
        self.test_id = directory.name
        self.directory = directory
        self.error: str | None = None
        self.metadata: dict[str, Any] | None = None
        self.model: Any = None
        pt = directory / "model.pt"
        json_path = directory / "model.json"
        if not pt.is_file() or not json_path.is_file():
            self.error = "model.pt and model.json must both be present"
            return
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            if metadata.get("test_id") != self.test_id:
                self.error = "model.json test_id must match folder name"
                return
            self.metadata = metadata
            self.model = torch.jit.load(str(pt), map_location="cpu")
            self.model.eval()
        except Exception as exc:  # noqa: BLE001 - malformed contract degrades this card only
            self.error = f"load failed: {exc}"

    @property
    def model_path(self) -> Path:
        return self.directory / "model.pt"


def load_models() -> dict[str, ModelEntry]:
    entries: dict[str, ModelEntry] = {}
    for test_id in MODEL_IDS:
        entry = ModelEntry(MODEL_ROOT / test_id)
        entries[test_id] = entry
        if entry.error:
            LOGGER.warning("Pallor model %s unavailable: %s", test_id, entry.error)
        else:
            LOGGER.info("Loaded %s from %s", test_id, entry.model_path)
    return entries


# ─── Prediction ────────────────────────────────────────────────────────

def predict(entry: ModelEntry, image: Image.Image | str | Path) -> tuple[str, float, str | None]:
    """Return (badge_label, risk_probability, error_message).

    Image may be a PIL image or an on-disk path (gradio type="filepath" or
    smoke-test samples). Badge label is the argmax class ("RISK 88%" /
    "NORMAL 62%"); the number shown is always the primary score key's
    probability (risk).
    """
    if entry.error:
        return "MODEL UNAVAILABLE", 0.0, entry.error
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        tensor = preprocess(image, entry.metadata)
        with torch.inference_mode():
            output = entry.model(tensor)
        values = output.detach().cpu().reshape(-1)
        labels = entry.metadata["output_class_labels"]
        if len(labels) != len(values):
            raise ValueError("model output size does not match output_class_labels")
        activation = entry.metadata.get("output_activation", "softmax")
        if activation == "softmax":
            probs = torch.softmax(values, dim=0)
        elif activation == "sigmoid":
            probs = torch.sigmoid(values)
        elif activation != "none":
            raise ValueError("output_activation must be softmax, sigmoid, or none")
        else:
            probs = values
        scores = {label: float(v) for label, v in zip(labels, probs.tolist())}
        primary_key = entry.metadata.get("primary_score_key", labels[-1])
        risk_prob = scores[primary_key]
        top_label = max(scores, key=scores.get)
        pct = round(risk_prob * 100)
        return f"{top_label.upper()} {pct}%", risk_prob, None
    except Exception as exc:  # noqa: BLE001 - per-card isolation
        return "ERROR", 0.0, str(exc)


# ─── Verification modes ────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _sample_image(test_id: str) -> Image.Image:
    data_dir = SCRIPT_DIR / DATA_DIRS[test_id]
    candidates = sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not candidates:
        raise FileNotFoundError(f"No images found under {data_dir}")
    return Image.open(candidates[0]).convert("RGB")


def run_smoke(entries: dict[str, ModelEntry]) -> None:
    """Headless one-shot: load all models, predict one image per site."""
    failures = 0
    for test_id in MODEL_IDS:
        entry = entries[test_id]
        if entry.error:
            LOGGER.error("SMOKE %s: %s", test_id, entry.error)
            failures += 1
            continue
        try:
            img = _sample_image(test_id)
            label, risk, err = predict(entry, img)
            LOGGER.info("SMOKE %s -> %s (risk=%.3f) [%s]", test_id, label, risk, img.size)
            if err:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("SMOKE %s raised: %s", test_id, exc)
            failures += 1
    if failures:
        LOGGER.error("SMOKE FAILED (%d failure(s))", failures)
        raise SystemExit(1)
    LOGGER.info("SMOKE PASSED")


TEST_METADATA = {
    "input_shape": [1, 3, 224, 224],
    "preprocessing": {
        "color_space": "Lab",
        "resize": [256, 256],
        "center_crop": 224,
        "scale": [0.0, 1.0],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.25, 0.25],
        "capture": {"center_zoom": 2.2, "saturation": 1.12, "contrast": 1.03},
    },
}


def run_check_lab() -> None:
    """Verify this app's preprocessing equals train_all_v2's LabTensor pipeline."""
    spec = importlib.util.spec_from_file_location("train_all_v2", SCRIPT_DIR / "train_all_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    img = _sample_image("pallor_palm")
    pallor_spec = next(item for item in mod.MODELS if item.test_id == "pallor_palm")
    expected = mod.build_transforms(pallor_spec, train=False)(img)  # normalized tensor
    mean = torch.tensor(pallor_spec.mean).view(3, 1, 1)
    std = torch.tensor(pallor_spec.std).view(3, 1, 1)
    expected_un = expected * std + mean  # back to [0, 1] Lab values

    actual = preprocess(img, TEST_METADATA)
    actual_un = actual * std + mean  # back to [0, 1] Lab values
    diff = (actual_un - expected_un).abs().max().item()
    LOGGER.info("Lab parity max abs diff: %.3e (threshold 1e-4)", diff)
    if diff >= 1e-4:
        LOGGER.error("CHECK-LAB FAILED: preprocessing drifted from training")
        raise SystemExit(1)
    LOGGER.info("CHECK-LAB PASSED")


# ─── UI ────────────────────────────────────────────────────────────────

def build_ui(entries: dict[str, ModelEntry], port: int = 7860) -> None:
    import gradio as gr

    def make_handler(test_id: str):
        def handler(img):
            entry = entries[test_id]
            path_md = f"**model:** `{entry.model_path}`" if not entry.error else "**model:** not installed"
            if img is None:
                return "<span style='color:#888'>Awaiting upload</span>", path_md
            label, risk_prob, err = predict(entry, img)
            if err:
                return f"<span style='color:#c0392b'>Error: {html.escape(err)}</span>", path_md
            # Threshold on the primary score key (risk) probability.
            color = "#c0392b" if risk_prob >= 0.5 else "#1a7f37"
            return (
                f"<span style='color:{color};font-weight:bold;font-size:1.5em'>{label}</span>",
                path_md,
            )
        return handler

    with gr.Blocks(title="Pallor Checker") as demo:
        gr.Markdown("# Pallor Checker — Anaemia risk screening")
        with gr.Row():
            for test_id in MODEL_IDS:
                with gr.Column():
                    gr.Markdown(f"## {SITE_TITLES[test_id]}")
                    # type="filepath": corrupt uploads reach the handler as a path and
                    # fail inside predict() at Image.open(), surfacing as the per-card
                    # red error instead of a gradio-side decode failure.
                    img_in = gr.Image(type="filepath", label="Upload image", height=280)
                    out_badge = gr.HTML("<span style='color:#888'>Awaiting upload</span>")
                    out_path = gr.Markdown()
                    img_in.change(make_handler(test_id), img_in, [out_badge, out_path])
    demo.launch(server_name="127.0.0.1", server_port=port)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Pallor Checker gradio UI")
    parser.add_argument("--port", type=int, default=7860, help="UI port (default: 7860)")
    parser.add_argument("--smoke", action="store_true", help="headless: one prediction per site")
    parser.add_argument("--check-lab", action="store_true", help="numeric parity vs train_all_v2 LabTensor")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        run_smoke(load_models())
    elif args.check_lab:
        run_check_lab()
    else:
        build_ui(load_models(), args.port)


if __name__ == "__main__":
    main()
