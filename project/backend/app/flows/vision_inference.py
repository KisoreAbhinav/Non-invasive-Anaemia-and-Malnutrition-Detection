"""CPU preprocessing and inference for registered TorchScript image models."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.flows import model_registry

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_D65_REFERENCE_WHITE = (0.95047, 1.0, 1.08883)
_LAB_DELTA = 6.0 / 29.0


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def _lab_curve(value: np.ndarray) -> np.ndarray:
    return np.where(
        value > _LAB_DELTA**3,
        np.cbrt(value),
        value / (3 * _LAB_DELTA**2) + 4.0 / 29.0,
    )


def _rgb_to_lab_tensor(image: Image.Image) -> Any:
    """Match the trainer's standards-based D65 CIE Lab conversion exactly."""

    import torch

    rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    linear = _srgb_to_linear(rgb.astype(np.float64) / 255.0)
    xyz = linear @ _SRGB_TO_XYZ.T
    xn, yn, zn = _D65_REFERENCE_WHITE
    fx = _lab_curve(xyz[..., 0] / xn)
    fy = _lab_curve(xyz[..., 1] / yn)
    fz = _lab_curve(xyz[..., 2] / zn)

    lightness = (116.0 * fy - 16.0) / 100.0
    a_channel = np.clip((500.0 * (fx - fy) + 128.0) / 255.0, 0.0, 1.0)
    b_channel = np.clip((200.0 * (fy - fz) + 128.0) / 255.0, 0.0, 1.0)
    lab = np.stack([lightness, a_channel, b_channel], axis=-1)
    return torch.from_numpy(lab.transpose(2, 0, 1).copy()).float().unsqueeze(0)


def _resize_and_crop(image: Image.Image, preprocessing: dict[str, Any]) -> Image.Image:
    resize = preprocessing.get("resize")
    if not isinstance(resize, list) or len(resize) != 2:
        raise ValueError("preprocessing.resize must be [width, height]")
    target_width, target_height = (int(resize[0]), int(resize[1]))
    if target_width <= 0 or target_height <= 0:
        raise ValueError("preprocessing.resize dimensions must be positive")

    center_crop = preprocessing.get("center_crop")
    if center_crop is None:
        return image.resize((target_width, target_height), Image.Resampling.BILINEAR)

    crop_size = int(center_crop)
    if crop_size <= 0:
        raise ValueError("preprocessing.center_crop must be positive")
    width, height = image.size
    if target_width == target_height:
        # torchvision.transforms.Resize(integer) makes the shorter side the target.
        scale = target_width / min(width, height)
    else:
        # Rectangular metadata describes a bounding box; cover it before cropping.
        scale = max(target_width / width, target_height / height)
    resized_width = max(crop_size, int(width * scale))
    resized_height = max(crop_size, int(height * scale))
    image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (resized_width - crop_size) // 2
    top = (resized_height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def _image_tensor(image_bytes: bytes, metadata: dict[str, Any]) -> Any:
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded:
            image = uploaded.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("upload must be a readable image") from exc

    shape = metadata.get("input_shape")
    if not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [1, 3]:
        raise ValueError("only [1, 3, height, width] image models are supported")
    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("preprocessing must be an object")
    if "resize" not in preprocessing:
        preprocessing = {
            **preprocessing,
            "resize": [int(shape[3]), int(shape[2])],
        }
    image = _resize_and_crop(image, preprocessing)

    import torch

    if str(preprocessing.get("color_space", "RGB")).upper() == "LAB":
        tensor = _rgb_to_lab_tensor(image)
    else:
        rgb = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        if preprocessing.get("scale") in ([0, 1], [0.0, 1.0]):
            tensor = tensor / 255.0

    mean = preprocessing.get("mean")
    std = preprocessing.get("std")
    if (mean is None) != (std is None):
        raise ValueError("preprocessing.mean and preprocessing.std must be provided together")
    if mean is not None and std is not None:
        if len(mean) != 3 or len(std) != 3 or any(float(value) == 0 for value in std):
            raise ValueError("preprocessing mean/std must contain three values and non-zero std")
        mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor

    expected_shape = tuple(int(dimension) for dimension in shape)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"preprocessing produced {tuple(tensor.shape)}, expected {expected_shape}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("preprocessing produced non-finite input values")
    return tensor


def predict_image(test_id: str, image_bytes: bytes) -> dict[str, Any]:
    metadata = model_registry.get_metadata(test_id)
    model = model_registry.get_model(test_id)
    if metadata is None or model is None:
        raise LookupError("no usable model is installed for this test")

    import torch

    tensor = _image_tensor(image_bytes, metadata)
    with torch.inference_mode():
        output = model(tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError("model output must be a tensor")
    values = output.detach().cpu().reshape(-1)
    labels = metadata["output_class_labels"]
    if len(labels) != len(values):
        raise ValueError("model output size does not match output_class_labels")
    if not torch.isfinite(values).all():
        raise ValueError("model output contains non-finite values")

    activation = metadata.get("output_activation", "softmax")
    if activation == "softmax":
        values = torch.softmax(values, dim=0)
    elif activation == "sigmoid":
        values = torch.sigmoid(values)
    elif activation != "none":
        raise ValueError("output_activation must be softmax, sigmoid, or none")

    scores = {
        label: round(float(value), 6)
        for label, value in zip(labels, values.tolist(), strict=True)
    }
    primary_key = metadata.get("primary_score_key", labels[-1])
    if primary_key not in scores:
        raise ValueError("primary_score_key must name an output label")
    top_label = max(scores, key=scores.get)
    return {
        "test_id": test_id,
        "scores": scores,
        "primary_score": scores[primary_key],
        "primary_score_key": primary_key,
        "classification": top_label,
        "confidence": scores[top_label],
        "provenance": metadata.get("provenance", {}),
    }


def predict_batch(
    sub_captures: list[dict[str, Any]],
    image_bytes_list: list[bytes],
) -> dict[str, Any]:
    """Run each site-specific model and return an unweighted mean result."""

    if not sub_captures:
        raise ValueError("at least one sub-capture is required")
    if len(sub_captures) != len(image_bytes_list):
        raise ValueError(f"Expected {len(sub_captures)} images, got {len(image_bytes_list)}")

    per_image: list[dict[str, Any]] = []
    for capture, image_bytes in zip(sub_captures, image_bytes_list, strict=True):
        result = predict_image(capture["id"], image_bytes)
        result["label"] = capture.get("label", capture["id"])
        per_image.append(result)

    labels = list(per_image[0]["scores"])
    primary_key = per_image[0]["primary_score_key"]
    if any(list(result["scores"]) != labels for result in per_image[1:]):
        raise ValueError("sub-capture models must use the same ordered class labels")
    if any(result["primary_score_key"] != primary_key for result in per_image[1:]):
        raise ValueError("sub-capture models must use the same primary score key")

    combined_scores = {
        label: round(sum(result["scores"][label] for result in per_image) / len(per_image), 6)
        for label in labels
    }
    top_label = max(combined_scores, key=combined_scores.get)
    return {
        "test_id": "pallor",
        "scores": combined_scores,
        "primary_score": combined_scores[primary_key],
        "primary_score_key": primary_key,
        "classification": top_label,
        "confidence": combined_scores[top_label],
        "per_image": per_image,
        "provenance": per_image[0].get("provenance", {}),
    }
