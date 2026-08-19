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
    width, height = int(resize[0]), int(resize[1])
    image = image.resize((width, height))

    import torch

    tensor = torch.frombuffer(memoryview(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0).float()
    if preprocessing.get("scale") == [0, 1] or preprocessing.get("scale") == [0.0, 1.0]:
        tensor = tensor / 255.0
    mean, std = preprocessing.get("mean"), preprocessing.get("std")
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor
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
