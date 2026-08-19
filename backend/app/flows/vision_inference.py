"""CPU image preprocessing and inference for registered TorchScript models."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.flows import model_registry


def _image_tensor(image_bytes: bytes, metadata: dict[str, Any]) -> Any:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("upload must be a readable image") from exc

    shape = metadata["input_shape"]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ValueError("only [1, 3, height, width] image models are supported")
    preprocessing = metadata["preprocessing"]
    resize = preprocessing.get("resize", [shape[3], shape[2]])
    if not isinstance(resize, list) or len(resize) != 2:
        raise ValueError("preprocessing.resize must be [width, height]")
    target_w, target_h = int(resize[0]), int(resize[1])
    # Resize preserving aspect ratio (shorter side to target, matching torchvision Resize)
    w, h = image.size
    if target_w == target_h:
        scale_factor = target_w / min(w, h)
    else:
        scale_factor = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    image = image.resize((new_w, new_h))

    # Center crop if specified
    center_crop = preprocessing.get("center_crop")
    if center_crop:
        cc = int(center_crop)
        left = (new_w - cc) // 2
        top = (new_h - cc) // 2
        image = image.crop((left, top, left + cc, top + cc))

    if not center_crop and (new_w != target_w or new_h != target_h):
        image = image.resize((target_w, target_h))

    final_w, final_h = image.size
    import torch

    tensor = torch.frombuffer(memoryview(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(final_h, final_w, 3).permute(2, 0, 1).unsqueeze(0).float()
    if preprocessing.get("scale") == [0, 1] or preprocessing.get("scale") == [0.0, 1.0]:
        tensor = tensor / 255.0
    mean, std = preprocessing.get("mean"), preprocessing.get("std")
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor
    return tensor


def predict_batch(sub_captures: list[dict[str, Any]], image_bytes_list: list[bytes]) -> dict[str, Any]:
    """Run inference on multiple images (e.g. pallor eye/nail/tongue) and combine.

    Each entry in *sub_captures* has ``id`` (model dir name), ``label``, etc.
    Returns a single combined result with averaged scores.
    """
    if len(sub_captures) != len(image_bytes_list):
        raise ValueError(f"Expected {len(sub_captures)} images, got {len(image_bytes_list)}")

    per_image: list[dict[str, Any]] = []
    for capture, img_bytes in zip(sub_captures, image_bytes_list):
        result = predict_image(capture["id"], img_bytes)
        result["label"] = capture.get("label", capture["id"])
        per_image.append(result)

    # Average primary scores across all body sites
    primary_key = per_image[0]["primary_score_key"]
    all_scores = [r["primary_score"] for r in per_image]
    avg_primary = sum(all_scores) / len(all_scores)

    # Build combined class scores by averaging each label across images
    combined_scores: dict[str, float] = {}
    for label in per_image[0]["scores"]:
        label_vals = [r["scores"][label] for r in per_image if label in r["scores"]]
        combined_scores[label] = round(sum(label_vals) / len(label_vals), 6)

    top_label = max(combined_scores, key=combined_scores.get)
    return {
        "test_id": "pallor",
        "scores": combined_scores,
        "primary_score": round(avg_primary, 6),
        "primary_score_key": primary_key,
        "classification": top_label,
        "confidence": combined_scores[top_label],
        "per_image": per_image,
        "provenance": per_image[0].get("provenance", {}),
    }


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
    activation = metadata.get("output_activation", "softmax")
    if activation == "softmax":
        values = torch.softmax(values, dim=0)
    elif activation == "sigmoid":
        values = torch.sigmoid(values)
    elif activation != "none":
        raise ValueError("output_activation must be softmax, sigmoid, or none")
    scores = {label: round(float(value), 6) for label, value in zip(labels, values.tolist())}
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
