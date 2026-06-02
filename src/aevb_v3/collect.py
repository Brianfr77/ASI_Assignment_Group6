from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import package_versions, write_json


def collect_training_metrics(results_dir: str | Path) -> pd.DataFrame:
    metrics_dir = Path(results_dir) / "metrics"
    frames = []
    for path in metrics_dir.glob("*.csv"):
        if path.name.startswith("learning_rate") or path.name.startswith("marginal"):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if {"method", "split", "lower_bound"}.issubset(df.columns):
            df["source_file"] = path.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_final_tables(results_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = collect_training_metrics(results_dir)
    written = []
    if not df.empty:
        final = (
            df.sort_values("epoch")
            .groupby(["method", "z_dim", "hidden_dim", "seed", "split"], as_index=False)
            .tail(1)
        )
        per_seed_rows = []
        for (method, z_dim, hidden_dim, seed), group in final.groupby(["method", "z_dim", "hidden_dim", "seed"]):
            train = group[group["split"] == "train"]
            test = group[group["split"] == "test"]
            if train.empty or test.empty:
                continue
            train = train.iloc[0]
            test = test.iloc[0]
            per_seed_rows.append(
                {
                    "method": method,
                    "z_dim": z_dim,
                    "hidden_dim": hidden_dim,
                    "seed": seed,
                    "final_train_lb": train["lower_bound"],
                    "final_test_lb": test["lower_bound"],
                    "final_test_bce": test["bce"],
                    "final_test_kl": test["kl"],
                }
            )
        per_seed_path = tables_dir / "final_summary_by_seed.csv"
        pd.DataFrame(per_seed_rows).to_csv(per_seed_path, index=False)
        written.append(per_seed_path)
        summary = (
            final[final["split"] == "test"]
            .groupby(["method", "z_dim", "hidden_dim"])
            .agg(
                final_test_lower_bound_mean=("lower_bound", "mean"),
                final_test_lower_bound_std=("lower_bound", "std"),
                final_test_negative_elbo_mean=("negative_elbo", "mean"),
                final_test_negative_elbo_std=("negative_elbo", "std"),
                final_test_bce_mean=("bce", "mean"),
                final_test_kl_mean=("kl", "mean"),
            )
            .reset_index()
        )
        path = tables_dir / "final_test_summary_by_method_z.csv"
        summary.to_csv(path, index=False)
        written.append(path)
    return written


def validate_metrics(results_dir: str | Path) -> dict[str, object]:
    results_dir = Path(results_dir)
    csvs = sorted((results_dir / "metrics").glob("*.csv"))
    issues = []
    for path in csvs:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            issues.append(f"{path.name}: failed to read CSV: {exc}")
            continue
        numeric = df.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            issues.append(f"{path.name}: contains NaN or inf")
    return {"csv_count": len(csvs), "issues": issues}


def write_manifest(results_dir: str | Path, config: dict | None = None) -> Path:
    results_dir = Path(results_dir)
    payload = {
        "config": config or {},
        "package_versions": package_versions(),
        "metrics": sorted(p.name for p in (results_dir / "metrics").glob("*.csv")),
        "figures": sorted(p.name for p in (results_dir / "figures").glob("*.png")),
        "tables": sorted(p.name for p in (results_dir / "tables").glob("*.csv")),
        "checkpoints": sorted(p.name for p in (results_dir / "checkpoints").glob("*.pt")),
        "validation": validate_metrics(results_dir),
    }
    path = results_dir / "manifests" / "experiment_manifest.json"
    write_json(path, payload)
    return path
