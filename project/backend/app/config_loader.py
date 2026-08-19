import json
from pathlib import Path
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return loaded
