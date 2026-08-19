"""Discovery and lazy loading for drop-in Stage 3+ vision models.

Availability means the fixed TorchScript contract is structurally valid. A model
is loaded on first use and cached. A load failure is retained and makes the test
unavailable until the process is restarted after replacing the model.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.settings import settings

LOGGER = logging.getLogger(__name__)
MODEL_FILENAME = "model.pt"
METADATA_FILENAME = "model.json"
SUPPORTED_METADATA_VERSION = 1


@dataclass
class ModelEntry:
    test_id: str
    directory: Path
    model_path: Path | None = None
    metadata: dict[str, Any] | None = None
    model: Any = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.model_path is not None and self.metadata is not None and self.error is None


_entries: dict[str, ModelEntry] = {}
_registry_lock = threading.RLock()


def _validate_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model.json must contain a JSON object")
    if value.get("contract_version") != SUPPORTED_METADATA_VERSION:
        raise ValueError("contract_version must be 1")
    shape = value.get("input_shape")
    if not isinstance(shape, list) or not shape or any(
        not isinstance(dimension, int) or dimension <= 0 for dimension in shape
    ):
        raise ValueError("input_shape must be a non-empty list of positive integers")
    if not isinstance(value.get("preprocessing"), dict):
        raise ValueError("preprocessing must be an object")
    labels = value.get("output_class_labels")
    if not isinstance(labels, list) or not labels or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError("output_class_labels must be a non-empty list of strings")
    return value


def scan_models(root: Path | None = None) -> dict[str, ModelEntry]:
    """Rescan immediate test folders without importing or loading PyTorch."""

    model_root = root or settings.vision_models_path
    discovered: dict[str, ModelEntry] = {}
    if not model_root.is_dir():
        with _registry_lock:
            _entries.clear()
        return discovered

    for directory in sorted(path for path in model_root.iterdir() if path.is_dir()):
        entry = ModelEntry(test_id=directory.name, directory=directory)
        model_path = directory / MODEL_FILENAME
        metadata_path = directory / METADATA_FILENAME
        if not model_path.is_file() and not metadata_path.is_file():
            entry.error = "no model installed"
        elif not model_path.is_file() or not metadata_path.is_file():
            entry.error = "model.pt and model.json must both be present"
        else:
            try:
                metadata = _validate_metadata(
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                )
                declared_test = metadata.get("test_id")
                if declared_test is not None and declared_test != directory.name:
                    raise ValueError("model.json test_id must match its folder name")
                entry.model_path = model_path
                entry.metadata = metadata
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                entry.error = str(exc)
        discovered[entry.test_id] = entry

    with _registry_lock:
        _entries.clear()
        _entries.update(discovered)
    for entry in discovered.values():
        if entry.error and entry.error != "no model installed":
            LOGGER.warning("Vision model %s unavailable: %s", entry.test_id, entry.error)
    return discovered


def initialize_registry() -> None:
    scan_models()


def _entry(test_id: str) -> ModelEntry | None:
    with _registry_lock:
        return _entries.get(test_id)


def is_available(test_id: str) -> bool:
    entry = _entry(test_id)
    return bool(entry and entry.available)


def get_model(test_id: str) -> Any | None:
    """Return a cached TorchScript module, or None for an unavailable test."""

    entry = _entry(test_id)
    if entry is None or not entry.available:
        return None
    if entry.model is not None:
        return entry.model
    with _registry_lock:
        if entry.model is not None:
            return entry.model
        try:
            import torch

            entry.model = torch.jit.load(str(entry.model_path), map_location="cpu")
            entry.model.eval()
            return entry.model
        except Exception as exc:  # pragma: no cover - needs a real model artifact
            entry.error = f"model load failed: {exc}"
            LOGGER.exception("Unable to load vision model for %s", test_id)
            return None


def status(test_ids: list[str] | None = None) -> dict[str, dict[str, Any]]:
    ids = test_ids if test_ids is not None else sorted(_entries)
    result: dict[str, dict[str, Any]] = {}
    for test_id in ids:
        entry = _entry(test_id)
        available = bool(entry and entry.available)
        result[test_id] = {
            "available": available,
            "status": "will_run" if available else "no_model_installed",
            "message": "will run" if available else "no model installed — test skipped",
            "model": str(entry.model_path) if available and entry else None,
            "metadata": dict(entry.metadata) if available and entry and entry.metadata else None,
            "error": None if available else (entry.error if entry else "test folder not found"),
        }
    return result
