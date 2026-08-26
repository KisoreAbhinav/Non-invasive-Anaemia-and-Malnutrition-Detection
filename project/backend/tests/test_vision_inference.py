import json
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from app.flows import model_registry
from app.flows.screening import build_plan
from app.main import app
from tests.asgi_client import ASGITestClient


class FixedLogits(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32)


def camera_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def install_test_model(root: Path, test_id: str) -> None:
    model_dir = root / test_id
    model_dir.mkdir()
    torch.jit.trace(FixedLogits(), torch.zeros(1, 3, 8, 8)).save(model_dir / "model.pt")
    (model_dir / "model.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "test_id": test_id,
                "input_shape": [1, 3, 8, 8],
                "preprocessing": {
                    "color_space": "RGB",
                    "resize": [8, 8],
                    "scale": [0, 1],
                },
                "output_class_labels": ["normal", "risk"],
                "output_activation": "softmax",
                "primary_score_key": "risk",
                "provenance": {"kind": "test", "site": test_id},
            }
        ),
        encoding="utf-8",
    )


def test_pallor_plan_requires_all_three_site_models(tmp_path: Path) -> None:
    try:
        for test_id in ("pallor_eye", "pallor_nail"):
            install_test_model(tmp_path, test_id)
        model_registry.scan_models(tmp_path)
        pallor = next(
            test for test in build_plan("child_under5", {}, {})["visual_cues"]
            if test["id"] == "pallor"
        )
        assert not pallor["availability"]["available"]
        assert "pallor_palm" in pallor["availability"]["message"]

        install_test_model(tmp_path, "pallor_palm")
        model_registry.scan_models(tmp_path)
        pallor = next(
            test for test in build_plan("child_under5", {}, {})["visual_cues"]
            if test["id"] == "pallor"
        )
        assert pallor["availability"]["available"]
    finally:
        model_registry.scan_models()


def test_multipart_sub_captures_runs_all_models_and_aggregates(tmp_path: Path) -> None:
    for test_id in ("pallor_eye", "pallor_nail", "pallor_palm"):
        install_test_model(tmp_path, test_id)
    try:
        model_registry.scan_models(tmp_path)
        image = camera_image()
        response = ASGITestClient(app).post(
            "/api/flows/screening/vision/pallor?population=child_under5",
            files={
                "pallor_eye": ("eye.jpg", image, "image/jpeg"),
                "pallor_nail": ("nail.jpg", image, "image/jpeg"),
                "pallor_palm": ("palm.jpg", image, "image/jpeg"),
            },
        )
    finally:
        model_registry.scan_models()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["classification"] == "risk"
    assert body["primary_score"] == body["scores"]["risk"]
    assert [item["label"] for item in body["per_image"]] == ["EYE", "NAILBED", "PALM"]


def test_multipart_rejects_missing_site_image(tmp_path: Path) -> None:
    for test_id in ("pallor_eye", "pallor_nail", "pallor_palm"):
        install_test_model(tmp_path, test_id)
    try:
        model_registry.scan_models(tmp_path)
        image = camera_image()
        response = ASGITestClient(app).post(
            "/api/flows/screening/vision/pallor?population=child_under5",
            files={
                "pallor_eye": ("eye.jpg", image, "image/jpeg"),
                "pallor_nail": ("nail.jpg", image, "image/jpeg"),
            },
        )
    finally:
        model_registry.scan_models()

    assert response.status_code == 422
    assert response.json()["detail"] == "Missing image for PALM"
