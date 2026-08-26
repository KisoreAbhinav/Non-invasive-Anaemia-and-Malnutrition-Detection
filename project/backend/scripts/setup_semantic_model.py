#!/usr/bin/env python3
"""Cache the answer matcher's compact sentence-transformer for offline use."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger("setup_semantic_model")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def setup_model(model_dir: Path) -> Path:
    from sentence_transformers import SentenceTransformer

    model_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Caching %s under %s", MODEL_NAME, model_dir)
    try:
        model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(model_dir),
            local_files_only=True,
        )
        LOGGER.info("Using the complete semantic model already in the local cache")
    except Exception:
        LOGGER.info("Local semantic cache is incomplete; downloading the model")
        model = SentenceTransformer(MODEL_NAME, cache_folder=str(model_dir))
    model.encode(["yes", "no"], normalize_embeddings=True)
    LOGGER.info("Semantic answer model is ready for offline inference")
    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "sentence-transformers",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        setup_model(args.model_dir.resolve())
    except Exception as exc:
        LOGGER.error("Semantic model setup failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
