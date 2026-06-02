from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .config import project_path
from .data import load_mnist_tensors
from .hmc import bridge_log_marginal_from_posterior_samples, hmc_sample
from .io import save_metrics_csv
from .utils import get_device


def estimate_marginal_likelihood(
    model,
    config: dict[str, Any],
    method: str,
    n_train: int,
    seed: int,
    split: str,
    max_points: int = 1000,
) -> dict[str, float | int | str]:
    """Estimate log p(x) using the paper's low-dimensional HMC-based recipe."""
    device = get_device(config.get("device", "auto"))
    model = model.to(device)
    model.eval()
    data_dir = project_path(config, "data_dir", "data")
    x_all, _ = load_mnist_tensors(
        data_dir,
        split=split,
        binarization=config["data"]["binarization"],
        binarization_seed=int(config["data"]["binarization_seed"]),
    )
    x_all = x_all[:max_points]
    hmc_cfg = config["marginal_likelihood"]
    chunk_size = int(hmc_cfg.get("chunk_size", 100))
    num_samples = int(hmc_cfg.get("posterior_samples", 50))
    burn_in = int(hmc_cfg.get("burn_in", 50))
    step_size = float(hmc_cfg.get("initial_step_size", 0.05))
    leapfrog_steps = int(hmc_cfg.get("leapfrog_steps", 4))
    target_acceptance = float(hmc_cfg.get("target_acceptance", 0.90))

    log_liks = []
    accept_rates = []
    for start in tqdm(range(0, x_all.shape[0], chunk_size), desc=f"ml_{method}_{split}_n{n_train}"):
        x = x_all[start : start + chunk_size].to(device)
        init_z = None
        if hasattr(model, "encode"):
            with torch.no_grad():
                init_z, _ = model.encode(x)
        fit = hmc_sample(
            model,
            x,
            num_samples=num_samples,
            burn_in=burn_in,
            step_size=step_size,
            n_leapfrog=leapfrog_steps,
            target_acceptance=target_acceptance,
            adapt=True,
            init_z=init_z,
        )
        score = hmc_sample(
            model,
            x,
            num_samples=num_samples,
            burn_in=max(5, burn_in // 2),
            step_size=fit.final_step_size,
            n_leapfrog=leapfrog_steps,
            target_acceptance=None,
            adapt=False,
            init_z=fit.samples[-1],
        )
        log_px = bridge_log_marginal_from_posterior_samples(model, x, fit.samples, score.samples)
        log_liks.append(log_px.detach().cpu())
        accept_rates.extend([fit.acceptance_rate, score.acceptance_rate])
    all_log_liks = torch.cat(log_liks)
    return {
        "method": method,
        "n_train": n_train,
        "seed": seed,
        "samples_seen": int(config.get("samples_seen", int(config["training"]["epochs"]) * int(n_train))),
        "split": split,
        "mean_log_likelihood": float(all_log_liks.mean()),
        "std_log_likelihood": float(all_log_liks.std(unbiased=False)),
        "hmc_acceptance": float(sum(accept_rates) / max(len(accept_rates), 1)),
    }


def save_marginal_likelihood_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    save_metrics_csv(path, rows)
