from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .utils import safe_torch_load, write_json


TRAINING_COLUMNS = [
    "method",
    "dataset",
    "binarization",
    "z_dim",
    "hidden_dim",
    "seed",
    "epoch",
    "update",
    "samples_seen",
    "split",
    "lower_bound",
    "negative_elbo",
    "bce",
    "kl",
    "lr",
]


def run_name(
    method: str,
    binarization: str,
    z_dim: int,
    hidden_dim: int,
    seed: int,
    lr: float,
    epochs: int,
    extra: str | None = None,
) -> str:
    lr_s = str(lr).replace(".", "p")
    base = f"{method}_{binarization}_z{z_dim}_h{hidden_dim}_seed{seed}_lr{lr_s}_e{epochs}"
    return f"{base}_{extra}" if extra else base


def save_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def append_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    columns = list(rows[0].keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_metrics(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    if optimizers:
        payload["optimizer_states"] = {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    if optimizers and "optimizer_states" in payload:
        for name, optimizer in optimizers.items():
            if name in payload["optimizer_states"]:
                optimizer.load_state_dict(payload["optimizer_states"][name])
    return payload.get("metadata", {})


def latest_intermediate_checkpoint(checkpoint_dir: str | Path, run_stem: str) -> Path | None:
    """Return the latest `{run_stem}_epochN.pt` checkpoint if one exists."""
    checkpoint_dir = Path(checkpoint_dir)
    pattern = re.compile(rf"^{re.escape(run_stem)}_epoch(\d+)\.pt$")
    best: tuple[int, Path] | None = None
    for path in checkpoint_dir.glob(f"{run_stem}_epoch*.pt"):
        match = pattern.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if best is None or epoch > best[0]:
            best = (epoch, path)
    return best[1] if best else None


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
