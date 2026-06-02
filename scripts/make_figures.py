from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aevb_v3.collect import collect_training_metrics
from aevb_v3.config import project_path
from aevb_v3.io import load_checkpoint
from aevb_v3.models import VAE
from aevb_v3.plots import (
    save_latent_manifold,
    save_lower_bound_comparison,
    save_marginal_likelihood_comparison,
    save_random_samples,
)
from aevb_v3.script_utils import load_experiment_config
from aevb_v3.utils import get_device, safe_torch_load


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report-ready figures from metrics and checkpoints.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    results_dir = project_path(config, "results_dir", "results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = collect_training_metrics(results_dir)
    if not metrics.empty:
        save_lower_bound_comparison(
            metrics,
            figures_dir / "figure2_style_lower_bound_comparison.png",
            "AEVB and Wake-Sleep lower bound on binarized MNIST",
        )
    marginal_path = results_dir / "metrics" / "marginal_likelihood_summary.csv"
    if marginal_path.exists():
        marginal = pd.read_csv(marginal_path)
        save_marginal_likelihood_comparison(
            marginal,
            figures_dir / "figure3_style_marginal_likelihood_comparison.png",
        )

    device = get_device(config.get("device", "auto"))
    ckpt_dir = results_dir / "checkpoints"
    for z_dim in [2, 5, 10, 20, 200]:
        candidates = sorted(ckpt_dir.glob(f"aevb_*_z{z_dim}_h*_seed0_*.pt"))
        if not candidates:
            continue
        ckpt = candidates[-1]
        metadata_probe = safe_torch_load(ckpt, map_location="cpu").get("metadata", {})
        hidden_dim = int(metadata_probe.get("hidden_dim", config["model"].get("hidden_dim", 500)))
        model = VAE(z_dim=z_dim, hidden_dim=hidden_dim, init_std=float(config["model"]["init_std"]))
        load_checkpoint(ckpt, model, device)
        model = model.to(device)
        save_random_samples(model, z_dim, figures_dir / f"aevb_binarized_random_samples_z{z_dim}_seed0.png", device)
        if z_dim == 2:
            save_latent_manifold(model, figures_dir / "aevb_binarized_z2_latent_manifold.png", device, grid=30)
    print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
