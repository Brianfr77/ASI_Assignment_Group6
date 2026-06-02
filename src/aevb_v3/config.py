from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively merged copy of two dictionaries."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def project_path(config: dict[str, Any], key: str, default: str) -> Path:
    """Resolve a path from config relative to the project root."""
    root = Path(config.get("project_root", ".")).expanduser().resolve()
    return (root / config.get(key, default)).resolve()


def map_l2_scale(config: dict[str, Any], n_train: int) -> float:
    """Resolve the explicit MAP L2 coefficient used with per-datapoint losses."""
    raw = config.get("training", {}).get("map_l2_scale", "1_over_n")
    if raw == "1_over_n":
        return 1.0 / float(n_train)
    return float(raw)
