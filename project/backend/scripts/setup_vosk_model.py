#!/usr/bin/env python3
"""Provision the lightweight Indian-English Vosk model for clean deployments."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

LOGGER = logging.getLogger("setup_vosk_model")
MODEL_NAME = "vosk-model-small-en-in-0.4"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"


def _is_model(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_dir() for name in ("am", "conf", "graph"))


def setup_model(model_dir: Path) -> Path:
    for candidate in sorted(model_dir.glob("*/")) if model_dir.is_dir() else []:
        if _is_model(candidate):
            LOGGER.info("Vosk model already present: %s", candidate)
            return candidate

    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / MODEL_NAME
    archive = model_dir / f".{MODEL_NAME}.zip.download"
    extraction_root = model_dir / f".{MODEL_NAME}.extracting"
    LOGGER.info("Downloading lightweight Indian-English Vosk model from %s", MODEL_URL)
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Nourish-Vosk-Setup/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        extraction_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                resolved = (extraction_root / member.filename).resolve()
                if extraction_root.resolve() not in resolved.parents and resolved != extraction_root.resolve():
                    raise ValueError("Vosk archive contains an unsafe path")
            bundle.extractall(extraction_root)
        extracted = extraction_root / MODEL_NAME
        if not _is_model(extracted):
            raise ValueError("downloaded Vosk archive does not contain a valid model")
        extracted.replace(destination)
    finally:
        archive.unlink(missing_ok=True)
        if extraction_root.exists():
            shutil.rmtree(extraction_root)
    LOGGER.info("Installed Vosk model: %s", destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "vosk",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        setup_model(args.model_dir.resolve())
    except Exception as exc:
        LOGGER.error("Vosk model setup failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
