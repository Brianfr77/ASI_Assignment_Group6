from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import Adagrad
from tqdm.auto import tqdm

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
from .losses import encoder_log_prob, map_l2_penalty
from .models import VAE, sample_bernoulli_from_logits
from .training import evaluate_elbo
from .utils import ensure_dirs, get_device, package_versions, set_seed


def train_wake_sleep_one_epoch(
    model: VAE,
    loader,
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    device: torch.device,
    map_scale: float,
) -> dict[str, float]:
    model.train()
    total_n = 0
    wake_loss_sum = 0.0
    sleep_loss_sum = 0.0
    for x, _ in loader:
        x = x.to(device)
        n = x.shape[0]

        # Wake phase: infer z from real x, then update only the generative model.
        with torch.no_grad():
            mu, logvar = model.encode(x)
            z_wake = model.reparameterize(mu, logvar)
        decoder_optimizer.zero_grad(set_to_none=True)
        logits = model.decode_logits(z_wake)
        wake_reconstruction = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, x, reduction="none"
        ).sum(dim=1).mean()
        wake_objective = wake_reconstruction + map_l2_penalty(model.decoder_parameters(), map_scale)
        wake_objective.backward()
        decoder_optimizer.step()

        # Sleep phase: sample from the model, then update only the recognition model.
        with torch.no_grad():
            z_sleep = torch.randn(n, model.z_dim, device=device)
            x_sleep = sample_bernoulli_from_logits(model.decode_logits(z_sleep))
        encoder_optimizer.zero_grad(set_to_none=True)
        sleep_objective = -encoder_log_prob(model, x_sleep, z_sleep).mean()
        sleep_objective = sleep_objective + map_l2_penalty(model.encoder_parameters(), map_scale)
        sleep_objective.backward()
        encoder_optimizer.step()

        total_n += n
        wake_loss_sum += float(wake_reconstruction.detach().cpu()) * n
        sleep_loss_sum += float(sleep_objective.detach().cpu()) * n

    return {
        "wake_reconstruction": wake_loss_sum / float(total_n),
        "sleep_negative_log_q": sleep_loss_sum / float(total_n),
    }


def _metric_row(
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
        "method": "wake_sleep",
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


def fit_wake_sleep_run(
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
    name = run_name("wake_sleep", config["data"]["binarization"], z_dim, hidden_dim, seed, lr, epochs, extra_name)
    metrics_path = metrics_dir / f"{name}.csv"
    ckpt_path = ckpt_dir / f"{name}.pt"
    if metrics_path.exists() and ckpt_path.exists() and not force:
        return metrics_path, ckpt_path

    model = VAE(z_dim, hidden_dim=hidden_dim, input_dim=info.input_dim, init_std=float(config["model"]["init_std"])).to(device)
    encoder_optimizer = Adagrad(model.encoder_parameters(), lr=lr)
    decoder_optimizer = Adagrad(model.decoder_parameters(), lr=lr)
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
                optimizers={
                    "encoder_optimizer": encoder_optimizer,
                    "decoder_optimizer": decoder_optimizer,
                },
            )
            start_epoch = int(metadata.get("epoch", 0)) + 1
            updates = int(metadata.get("updates", (start_epoch - 1) * len(train_loader)))
            samples_seen = int(metadata.get("samples_seen", (start_epoch - 1) * info.n_train))
            if metrics_path.exists():
                rows = pd.read_csv(metrics_path).to_dict("records")
            print(f"Resuming Wake-Sleep run from {resume_ckpt.name} at epoch {start_epoch}.")

    checkpoint_epochs = {int(e) for e in config["training"].get("checkpoint_epochs", [])}
    checkpoint_every = int(config["training"].get("checkpoint_every", 0) or 0)
    progress = tqdm(range(start_epoch, epochs + 1), desc=name)
    for epoch in progress:
        train_wake_sleep_one_epoch(model, train_loader, encoder_optimizer, decoder_optimizer, device, map_scale)
        updates += len(train_loader)
        samples_seen += info.n_train
        if epoch == 1 or epoch == epochs or epoch % eval_every == 0:
            train_metrics = evaluate_elbo(model, train_loader, device, eval_samples)
            test_metrics = evaluate_elbo(model, test_loader, device, eval_samples)
            rows.append(_metric_row(config, z_dim, hidden_dim, seed, epoch, updates, samples_seen, "train", train_metrics, lr))
            rows.append(_metric_row(config, z_dim, hidden_dim, seed, epoch, updates, samples_seen, "test", test_metrics, lr))
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
                    "method": "wake_sleep",
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
                optimizers={
                    "encoder_optimizer": encoder_optimizer,
                    "decoder_optimizer": decoder_optimizer,
                },
            )

    save_metrics_csv(metrics_path, rows)
    save_checkpoint(
        ckpt_path,
        model,
        {
            "method": "wake_sleep",
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
        optimizers={
            "encoder_optimizer": encoder_optimizer,
            "decoder_optimizer": decoder_optimizer,
        },
    )
    return metrics_path, ckpt_path
