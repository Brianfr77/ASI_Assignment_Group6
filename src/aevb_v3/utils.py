from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def package_versions() -> dict[str, str]:
    versions = {
        "python": ".".join(map(str, os.sys.version_info[:3])),
        "torch": str(torch.__version__),
        "cuda_available": str(torch.cuda.is_available()),
    }
    try:
        import torchvision

        versions["torchvision"] = str(torchvision.__version__)
    except Exception as exc:  # pragma: no cover - informational only
        versions["torchvision"] = f"unavailable: {exc}"
    return versions


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load trusted local checkpoints across PyTorch 2.6+ weights_only defaults."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # This project only loads checkpoints it has created locally.
        return torch.load(path, map_location=map_location, weights_only=False)


def finite_or_raise(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise FloatingPointError(f"{name} is not finite: {value}")
