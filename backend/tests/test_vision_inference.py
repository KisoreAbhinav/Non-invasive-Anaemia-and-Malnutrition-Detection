import json
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from app.flows import model_registry
from app.main import app
from tests.asgi_client import ASGITestClient


class FixedLogits(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32)


def camera_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_camera_image_runs_registered_model_and_returns_named_scores(tmp_path: Path) -> None:
    model_dir = tmp_path / "pallor"
    model_dir.mkdir()
    torch.jit.trace(FixedLogits(), torch.zeros(1, 3, 8, 8)).save(model_dir / "model.pt")
    (model_dir / "model.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "test_id": "pallor",
                "input_shape": [1, 3, 8, 8],
                "preprocessing": {"color_space": "RGB", "resize": [8, 8], "scale": [0, 1]},
                "output_class_labels": ["normal", "pallor_signal"],
                "output_activation": "softmax",
                "primary_score_key": "pallor_signal",
                "provenance": {"kind": "test"},
            }
        ),
        encoding="utf-8",
    )
    try:
        model_registry.scan_models(tmp_path)
        response = ASGITestClient(app).post(
            "/api/flows/screening/vision/pallor?population=child_under5",
            content=camera_image(),
            headers={"content-type": "image/jpeg"},
        )
    finally:
        model_registry.scan_models()

    assert response.status_code == 200
    body = response.json()
    assert body["test_id"] == "pallor"
    assert body["scores"]["pallor_signal"] > body["scores"]["normal"]
    assert body["primary_score"] == body["scores"]["pallor_signal"]
    assert body["provenance"] == {"kind": "test"}
