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
    # hair_skin has no sub_captures, so a single raw image body is the correct upload contract
    model_dir = tmp_path / "hair_skin"
    model_dir.mkdir()
    torch.jit.trace(FixedLogits(), torch.zeros(1, 3, 8, 8)).save(model_dir / "model.pt")
    (model_dir / "model.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "test_id": "hair_skin",
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
            "/api/flows/screening/vision/hair_skin?population=child_under5",
            content=camera_image(),
            headers={"content-type": "image/jpeg"},
        )
    finally:
        model_registry.scan_models()

    assert response.status_code == 200
    body = response.json()
    assert body["test_id"] == "hair_skin"
    assert body["scores"]["pallor_signal"] > body["scores"]["normal"]
    assert body["primary_score"] == body["scores"]["pallor_signal"]
    assert body["provenance"] == {"kind": "test"}


def test_multipart_sub_captures_runs_all_models_and_aggregates(tmp_path: Path) -> None:
    for sub_id in ("pallor_eye", "pallor_nail", "pallor_palm"):
        model_dir = tmp_path / sub_id
        model_dir.mkdir()
        torch.jit.trace(FixedLogits(), torch.zeros(1, 3, 8, 8)).save(model_dir / "model.pt")
        (model_dir / "model.json").write_text(
            json.dumps(
                {
                    "contract_version": 1,
                    "test_id": sub_id,
                    "input_shape": [1, 3, 8, 8],
                    "preprocessing": {"color_space": "RGB", "resize": [8, 8], "scale": [0, 1]},
                    "output_class_labels": ["normal", "pallor_signal"],
                    "output_activation": "softmax",
                    "primary_score_key": "pallor_signal",
                    "provenance": {"kind": "test", "sub_id": sub_id},
                }
            ),
            encoding="utf-8",
        )
    try:
        model_registry.scan_models(tmp_path)
        img = camera_image()
        response = ASGITestClient(app).post(
            "/api/flows/screening/vision/pallor?population=child_under5",
            files={
                "pallor_eye": ("eye.jpg", img, "image/jpeg"),
                "pallor_nail": ("nail.jpg", img, "image/jpeg"),
                "pallor_palm": ("palm.jpg", img, "image/jpeg"),
            },
        )
    finally:
        model_registry.scan_models()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    body = response.json()
    assert body["test_id"] == "pallor"
    assert len(body["per_image"]) == 3
    assert [p["label"] for p in body["per_image"]] == ["EYE", "NAILBED", "PALM"]
    assert "scores" in body
    assert body["scores"]["pallor_signal"] > body["scores"]["normal"]
    assert body["primary_score"] == body["scores"]["pallor_signal"]

