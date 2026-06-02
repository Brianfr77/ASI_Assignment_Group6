from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm


def save_lower_bound_comparison(metrics: pd.DataFrame, path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for (method, z_dim, split), group in metrics.groupby(["method", "z_dim", "split"]):
        if split != "test":
            continue
        summary = group.groupby("samples_seen")["lower_bound"].agg(["mean", "std"]).reset_index()
        label = f"{method}, z={z_dim}"
        plt.plot(summary["samples_seen"], summary["mean"], label=label)
        if "std" in summary:
            std = summary["std"].fillna(0.0)
            plt.fill_between(summary["samples_seen"], summary["mean"] - std, summary["mean"] + std, alpha=0.15)
    plt.xlabel("Training samples evaluated")
    plt.ylabel("Average variational lower bound per datapoint")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_marginal_likelihood_comparison(metrics: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, n_train in zip(axes, sorted(metrics["n_train"].unique())):
        sub = metrics[(metrics["n_train"] == n_train) & (metrics["split"] == "test")]
        for method, group in sub.groupby("method"):
            summary = group.groupby("samples_seen")["mean_log_likelihood"].agg(["mean", "std"]).reset_index()
            ax.plot(summary["samples_seen"], summary["mean"], marker="o", label=method)
            std = summary["std"].fillna(0.0)
            ax.fill_between(summary["samples_seen"], summary["mean"] - std, summary["mean"] + std, alpha=0.15)
        ax.set_title(f"N_train={int(n_train)}")
        ax.set_xlabel("Training samples evaluated")
        ax.set_ylabel("Estimated marginal log-likelihood")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def save_random_samples(model, z_dim: int, path: str | Path, device: torch.device, grid: int = 10) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    z = torch.randn(grid * grid, z_dim, device=device)
    probs = torch.sigmoid(model.decode_logits(z)).view(grid * grid, 28, 28).cpu().numpy()
    fig, axes = plt.subplots(grid, grid, figsize=(grid, grid))
    for ax, image in zip(axes.flat, probs):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout(pad=0.05)
    plt.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def save_latent_manifold(model, path: str | Path, device: torch.device, grid: int = 30) -> None:
    if model.z_dim != 2:
        raise ValueError("Latent manifold visualization requires z_dim=2.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = norm.ppf(np.linspace(0.05, 0.95, grid))
    canvas = np.zeros((28 * grid, 28 * grid))
    model.eval()
    for i, yi in enumerate(coords):
        for j, xi in enumerate(coords):
            z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
            image = torch.sigmoid(model.decode_logits(z)).view(28, 28).cpu().numpy()
            canvas[(grid - 1 - i) * 28 : (grid - i) * 28, j * 28 : (j + 1) * 28] = image
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200)
    plt.close()
