from io import BytesIO

import pytest
import torch
from PIL import Image

from app.flows.vision_inference import _image_tensor, _rgb_to_lab_tensor


PALLOR_METADATA = {
    "input_shape": [1, 3, 224, 224],
    "preprocessing": {
        "color_space": "Lab",
        "resize": [256, 256],
        "center_crop": 224,
        "scale": [0.0, 1.0],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.25, 0.25],
    },
}


def image_bytes(size: tuple[int, int] = (640, 480)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, (180, 130, 120)).save(output, "PNG")
    return output.getvalue()


def test_non_square_image_matches_declared_shape_without_crop_padding() -> None:
    tensor = _image_tensor(image_bytes(), PALLOR_METADATA)

    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all()
    # A uniform source stays spatially uniform. The historical resize bug padded
    # a 640x480 image with black rows before the 224x224 crop.
    assert all(channel.max() - channel.min() < 1e-6 for channel in tensor[0])


def test_lab_conversion_uses_standard_scaled_d65_values() -> None:
    tensor = _rgb_to_lab_tensor(Image.new("RGB", (1, 1), "red"))

    assert tensor[0, 0, 0, 0].item() == pytest.approx(0.5324, abs=1e-4)
    assert tensor[0, 1, 0, 0].item() == pytest.approx(0.8160, abs=1e-4)
    assert tensor[0, 2, 0, 0].item() == pytest.approx(0.7655, abs=1e-4)


def test_invalid_mean_std_contract_is_rejected() -> None:
    metadata = {
        **PALLOR_METADATA,
        "preprocessing": {**PALLOR_METADATA["preprocessing"], "std": [0.5, 0.0, 0.25]},
    }

    with pytest.raises(ValueError, match="non-zero std"):
        _image_tensor(image_bytes((224, 224)), metadata)
