import json
import os

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
