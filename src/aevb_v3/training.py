from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import Adagrad
from tqdm.auto import tqdm
import pandas as pd

from .config import map_l2_scale, project_path
from .data import make_mnist_loaders
from .io import (
    TRAINING_COLUMNS,
    latest_intermediate_checkpoint,
    load_checkpoint,
    run_name,
    save_checkpoint,
    save_metrics_csv,
)
from .losses import elbo_components_from_logits, vae_negative_elbo_loss
from .models import VAE
from .utils import ensure_dirs, get_device, package_versions, set_seed


@torch.no_grad()
def evaluate_elbo(
    model: VAE,
    loader,
    device: torch.device,
    eval_samples: int = 1,
) -> dict[str, float]:
    model.eval()
    total_n = 0
    sums = {"negative_elbo": 0.0, "bce": 0.0, "kl": 0.0}
    for x, _ in loader:
        x = x.to(device)
        batch_n = x.shape[0]
        batch_sums = {"negative_elbo": 0.0, "bce": 0.0, "kl": 0.0}
        for _ in range(eval_samples):
            logits, mu, logvar, _ = model(x)
            nelbo, bce, kl = elbo_components_from_logits(logits, x, mu, logvar)
            batch_sums["negative_elbo"] += float(nelbo.sum().cpu())
            batch_sums["bce"] += float(bce.sum().cpu())
            batch_sums["kl"] += float(kl.sum().cpu())
        for key in sums:
            sums[key] += batch_sums[key] / float(eval_samples)
        total_n += batch_n
    out = {key: value / float(total_n) for key, value in sums.items()}
    out["lower_bound"] = -out["negative_elbo"]
    return out


def train_aevb_one_epoch(
    model: VAE,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    map_scale: float,
) -> dict[str, float]:
    model.train()
    total_n = 0
    sums = {"negative_elbo": 0.0, "bce": 0.0, "kl": 0.0}
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        objective, metrics = vae_negative_elbo_loss(model, x, map_scale)
        objective.backward()
        optimizer.step()
        n = x.shape[0]
        total_n += n
        for key in sums:
            sums[key] += metrics[key] * n
    out = {key: value / float(total_n) for key, value in sums.items()}
    out["lower_bound"] = -out["negative_elbo"]
    return out


def _metric_row(
    method: str,
    config: dict[str, Any],
    z_dim: int,
    hidden_dim: int,
    seed: int,
    epoch: int,
    update: int,
    samples_seen: int,
    split: str,
    metrics: dict[str, float],
    lr: float,
) -> dict[str, Any]:
    row = {
        "method": method,
        "dataset": "mnist",
        "binarization": config["data"]["binarization"],
        "z_dim": z_dim,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "epoch": epoch,
        "update": update,
        "samples_seen": samples_seen,
        "split": split,
        "lower_bound": metrics["lower_bound"],
        "negative_elbo": metrics["negative_elbo"],
        "bce": metrics["bce"],
        "kl": metrics["kl"],
        "lr": lr,
    }
    return {key: row[key] for key in TRAINING_COLUMNS}


def fit_aevb_run(
    config: dict[str, Any],
    z_dim: int,
    seed: int,
    lr: float,
    epochs: int | None = None,
    hidden_dim: int | None = None,
    train_subset_size: int | None = None,
    extra_name: str | None = None,
    force: bool = False,
    resume: bool | None = None,
) -> tuple[Path, Path]:
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    data_dir = project_path(config, "data_dir", "data")
    results_dir = project_path(config, "results_dir", "results")
    metrics_dir = results_dir / "metrics"
    ckpt_dir = results_dir / "checkpoints"
    ensure_dirs(metrics_dir, ckpt_dir)

    train_loader, test_loader, info = make_mnist_loaders(
        data_dir=data_dir,
        batch_size=int(config["training"]["batch_size"]),
        binarization=config["data"]["binarization"],
        binarization_seed=int(config["data"]["binarization_seed"]),
        seed=seed,
        train_subset_size=train_subset_size,
        num_workers=int(config["data"].get("num_workers", 2)),
    )
    epochs = int(epochs or config["training"]["epochs"])
    hidden_dim = int(hidden_dim or config["model"]["hidden_dim"])
    name = run_name("aevb", config["data"]["binarization"], z_dim, hidden_dim, seed, lr, epochs, extra_name)
    metrics_path = metrics_dir / f"{name}.csv"
    ckpt_path = ckpt_dir / f"{name}.pt"
    if metrics_path.exists() and ckpt_path.exists() and not force:
        return metrics_path, ckpt_path

    model = VAE(
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        input_dim=info.input_dim,
        init_std=float(config["model"]["init_std"]),
    ).to(device)
    optimizer = Adagrad(model.parameters(), lr=lr)
    map_scale = map_l2_scale(config, info.n_train)
    eval_samples = int(config["training"].get("eval_samples", 1))
    eval_every = int(config["training"].get("eval_every", 1))
    resume = bool(config["training"].get("resume", True) if resume is None else resume)

    rows: list[dict[str, Any]] = []
    updates = 0
    samples_seen = 0
    start_epoch = 1
    if resume and not force:
        resume_ckpt = latest_intermediate_checkpoint(ckpt_dir, name)
        if resume_ckpt is not None:
            metadata = load_checkpoint(
                resume_ckpt,
                model,
                device,
                optimizers={"optimizer": optimizer},
            )
            start_epoch = int(metadata.get("epoch", 0)) + 1
            updates = int(metadata.get("updates", (start_epoch - 1) * len(train_loader)))
            samples_seen = int(metadata.get("samples_seen", (start_epoch - 1) * info.n_train))
            if metrics_path.exists():
                rows = pd.read_csv(metrics_path).to_dict("records")
            print(f"Resuming AEVB run from {resume_ckpt.name} at epoch {start_epoch}.")

    checkpoint_epochs = {int(e) for e in config["training"].get("checkpoint_epochs", [])}
    checkpoint_every = int(config["training"].get("checkpoint_every", 0) or 0)
    progress = tqdm(range(start_epoch, epochs + 1), desc=name)
    for epoch in progress:
        train_metrics = train_aevb_one_epoch(model, train_loader, optimizer, device, map_scale)
        updates += len(train_loader)
        samples_seen += info.n_train
        if epoch == 1 or epoch == epochs or epoch % eval_every == 0:
            test_metrics = evaluate_elbo(model, test_loader, device, eval_samples)
            rows.append(_metric_row("aevb", config, z_dim, hidden_dim, seed, epoch, updates, samples_seen, "train", train_metrics, lr))
            rows.append(_metric_row("aevb", config, z_dim, hidden_dim, seed, epoch, updates, samples_seen, "test", test_metrics, lr))
            save_metrics_csv(metrics_path, rows)
            progress.set_postfix(test_nelbo=f"{test_metrics['negative_elbo']:.2f}")
        should_save_intermediate = epoch in checkpoint_epochs or (
            checkpoint_every > 0 and epoch % checkpoint_every == 0 and epoch != epochs
        )
        if should_save_intermediate:
            save_checkpoint(
                ckpt_dir / f"{name}_epoch{epoch}.pt",
                model,
                {
                    "method": "aevb",
                    "config": config,
                    "z_dim": z_dim,
                    "hidden_dim": hidden_dim,
                    "seed": seed,
                    "lr": lr,
                    "epoch": epoch,
                    "updates": updates,
                    "samples_seen": samples_seen,
                    "map_l2_scale": map_scale,
                    "package_versions": package_versions(),
                },
                optimizers={"optimizer": optimizer},
            )

    save_metrics_csv(metrics_path, rows)
    save_checkpoint(
        ckpt_path,
        model,
        {
            "method": "aevb",
            "config": config,
            "z_dim": z_dim,
            "hidden_dim": hidden_dim,
            "seed": seed,
            "lr": lr,
            "epochs": epochs,
            "epoch": epochs,
            "updates": updates,
            "samples_seen": samples_seen,
            "map_l2_scale": map_scale,
            "package_versions": package_versions(),
        },
        optimizers={"optimizer": optimizer},
    )
    return metrics_path, ckpt_path


def run_lr_pilot(config: dict[str, Any]) -> float:
    candidates = [float(x) for x in config["training"]["lr_candidates"]]
    z_dim = int(config["pilot"].get("z_dim", 10))
    seed = int(config["pilot"].get("seed", 0))
    epochs = int(config["pilot"].get("epochs", 5))
    rows = []
    for lr in candidates:
        metrics_path, _ = fit_aevb_run(
            config,
            z_dim=z_dim,
            seed=seed,
            lr=lr,
            epochs=epochs,
            extra_name="pilot",
            force=bool(config.get("force", False)),
            resume=False,
        )
        import pandas as pd

        df = pd.read_csv(metrics_path)
        final_train = df[(df["split"] == "train")].sort_values("epoch").iloc[-1]
        rows.append({"lr": lr, "train_negative_elbo": float(final_train["negative_elbo"])})
    selected = min(rows, key=lambda row: row["train_negative_elbo"])["lr"]
    results_dir = project_path(config, "results_dir", "results")
    save_metrics_csv(results_dir / "metrics" / "learning_rate_pilot_summary.csv", rows)
    return float(selected)
