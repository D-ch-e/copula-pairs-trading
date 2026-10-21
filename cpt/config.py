from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str = PROJECT_ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def db_path(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["db_path"]
