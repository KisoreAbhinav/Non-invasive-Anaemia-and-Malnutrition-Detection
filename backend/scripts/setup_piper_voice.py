#!/usr/bin/env python3
"""Download the default Piper voice and config when neither is present.

Vosk acoustic models are intentionally user-supplied because they are much
larger and deployments may choose different Indian-English model variants.
Piper voices are small, fixed assets with stable official download URLs, so the
TTS voice can be provisioned automatically during a Docker build.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger("setup_piper_voice")

# General-purpose single-speaker US English voice.
VOICE_NAME = "en_US-lessac-medium"
VOICE_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium"
)
FILES = {
    f"{VOICE_NAME}.onnx": f"{VOICE_BASE_URL}/{VOICE_NAME}.onnx?download=true",
    f"{VOICE_NAME}.onnx.json": f"{VOICE_BASE_URL}/{VOICE_NAME}.onnx.json?download=true",
}


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Screening-Piper-Setup/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def setup_voice(model_dir: Path) -> Path:
    target = model_dir / f"{VOICE_NAME}.onnx"
    if target.is_file() and Path(f"{target}.json").is_file():
        LOGGER.info("Piper voice already present: %s", target)
        return target

    model_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Installing the general US-English Piper voice %s", VOICE_NAME)
    temporary_files: list[tuple[Path, Path]] = []
    try:
        for filename, url in FILES.items():
            destination = model_dir / filename
            temporary = model_dir / f".{filename}.download"
            LOGGER.info("Downloading %s", url)
            _download(url, temporary)
            temporary_files.append((temporary, destination))
        for temporary, destination in temporary_files:
            temporary.replace(destination)
    except Exception:
        for temporary, _ in temporary_files:
            temporary.unlink(missing_ok=True)
        raise

    installed = target
    LOGGER.info("Installed Piper voice: %s", installed)
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "piper",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        setup_voice(args.model_dir.resolve())
    except Exception as exc:
        LOGGER.error("Piper voice setup failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
