#!/usr/bin/env python3
"""Smoke-test the installed pallor models through the FastAPI upload route.

Pass three real site images for a meaningful operator check, or omit all three
arguments to use generated color swatches for a pipeline-only smoke test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

# Allow `python scripts/verify_vision_inference.py` from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.flows import model_registry
from app.main import app

SITES = (
    ("pallor_eye", "EYE"),
    ("pallor_nail", "NAILBED"),
    ("pallor_palm", "PALM"),
)


def _generated_image(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (640, 480), color).save(output, format="JPEG", quality=95)
    return output.getvalue()


def _load_images(arguments: argparse.Namespace) -> tuple[list[bytes], str]:
    paths = [arguments.eye, arguments.nail, arguments.palm]
    if any(paths) and not all(paths):
        raise ValueError("provide --eye, --nail, and --palm together")
    if all(paths):
        return [path.read_bytes() for path in paths], "operator-supplied images"
    colors = [(224, 174, 166), (206, 151, 146), (190, 130, 120)]
    return [_generated_image(color) for color in colors], "generated color swatches"


def _validate_result(result: dict[str, Any]) -> None:
    if result.get("test_id") != "pallor":
        raise RuntimeError("response did not identify the pallor test")
    if len(result.get("per_image", [])) != len(SITES):
        raise RuntimeError("response did not include all three site results")
    for item in [result, *result["per_image"]]:
        scores = item.get("scores", {})
        if set(scores) != {"normal", "risk"}:
            raise RuntimeError(f"unexpected score labels: {sorted(scores)}")
        values = [float(value) for value in scores.values()]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise RuntimeError("model returned a non-finite or out-of-range probability")
        if not math.isclose(sum(values), 1.0, abs_tol=2e-6):
            raise RuntimeError("softmax probabilities do not sum to one")
    if result["primary_score"] != result["scores"]["risk"]:
        raise RuntimeError("primary score is not the declared risk probability")


async def _run(images: list[bytes], population: str) -> dict[str, Any]:
    model_registry.initialize_registry()
    status = model_registry.status([site_id for site_id, _ in SITES])
    missing = [site_id for site_id, entry in status.items() if not entry["available"]]
    if missing:
        raise RuntimeError(f"missing model artifacts: {', '.join(missing)}")

    files = {
        site_id: (f"{site_id}.jpg", image, "image/jpeg")
        for (site_id, _), image in zip(SITES, images, strict=True)
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://verify.local") as client:
        response = await client.post(
            f"/api/flows/screening/vision/pallor?population={population}",
            files=files,
        )
    if response.status_code != 200:
        raise RuntimeError(f"inference endpoint returned {response.status_code}: {response.text}")
    result = response.json()
    _validate_result(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eye", type=Path, help="conjunctiva image")
    parser.add_argument("--nail", type=Path, help="nail-bed image")
    parser.add_argument("--palm", type=Path, help="palm image")
    parser.add_argument(
        "--population",
        default="child_under5",
        choices=("child_under5", "child_5_12", "pregnant_woman", "adult_nonpregnant"),
    )
    arguments = parser.parse_args()
    try:
        images, source = _load_images(arguments)
        result = asyncio.run(_run(images, arguments.population))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"verification failed: {exc}\n")

    print(f"Verified three-site inference using {source}.")
    print(json.dumps(result, indent=2, sort_keys=True))
    if source == "generated color swatches":
        print("Pipeline smoke test only: use real labeled site images for a clinical-quality check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
