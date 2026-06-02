from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config, project_path
from .utils import read_json, write_json


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if str(config.get("project_root", ".")) == ".":
        config["project_root"] = str(config_path.parents[1])
    return config


def selected_lr_path(config: dict[str, Any]) -> Path:
    return project_path(config, "results_dir", "results") / "manifests" / "selected_lr.json"


def write_selected_lr(config: dict[str, Any], lr: float) -> Path:
    path = selected_lr_path(config)
    write_json(path, {"selected_lr": float(lr)})
    return path


def resolve_lr(config: dict[str, Any], cli_lr: float | None = None) -> float:
    if cli_lr is not None:
        return float(cli_lr)
    raw = config.get("training", {}).get("lr")
    if raw is not None:
        return float(raw)
    path = selected_lr_path(config)
    if path.exists():
        return float(read_json(path)["selected_lr"])
    raise FileNotFoundError(
        f"No learning rate was provided. Run scripts/run_pilot.py first or pass --lr. Missing: {path}"
    )
