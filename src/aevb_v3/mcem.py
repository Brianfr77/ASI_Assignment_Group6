from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import Adagrad
from tqdm.auto import tqdm

from .config import map_l2_scale, project_path
from .data import make_mnist_loaders
from .hmc import hmc_step
from .io import latest_intermediate_checkpoint, load_checkpoint, save_checkpoint, save_metrics_csv
from .losses import map_l2_penalty
from .models import GenerativeModel
from .utils import ensure_dirs, get_device, package_versions, set_seed


@torch.no_grad()
def evaluate_decoder_reconstruction(model: GenerativeModel, loader, device: torch.device) -> dict[str, float]:
    """A reconstruction proxy for MCEM checkpoints; marginal likelihood is evaluated separately."""
    model.eval()
    total_n = 0
    total_bce = 0.0
    for x, _ in loader:
        x = x.to(device)
        z = torch.randn(x.shape[0], model.z_dim, device=device)
        logits = model.decode_logits(z)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)
        total_bce += float(bce.sum().cpu())
        total_n += x.shape[0]
    return {"prior_sample_bce": total_bce / float(total_n)}


def fit_mcem_run(
    config: dict[str, Any],
    n_train: int,
    seed: int,
    lr: float,
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

    hidden_dim = int(config["model"]["hidden_dim"])
    z_dim = int(config["model"]["z_dim"])
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    name = f"mcem_{config['data']['binarization']}_z{z_dim}_h{hidden_dim}_n{n_train}_seed{seed}_lr{str(lr).replace('.', 'p')}_e{epochs}"
    metrics_path = metrics_dir / f"{name}.csv"
    ckpt_path = ckpt_dir / f"{name}.pt"
    if metrics_path.exists() and ckpt_path.exists() and not force:
        return metrics_path, ckpt_path

    train_loader, test_loader, info = make_mnist_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        binarization=config["data"]["binarization"],
        binarization_seed=int(config["data"]["binarization_seed"]),
        seed=seed,
        train_subset_size=n_train,
        num_workers=int(config["data"].get("num_workers", 2)),
    )
    model = GenerativeModel(z_dim=z_dim, hidden_dim=hidden_dim, init_std=float(config["model"]["init_std"])).to(device)
    optimizer = Adagrad(model.parameters(), lr=lr)
    map_scale = map_l2_scale(config, info.n_train)
    hmc_cfg = config["hmc"]
    step_size = float(hmc_cfg.get("initial_step_size", 0.05))
    rows: list[dict[str, Any]] = []
    samples_seen = 0
    updates = 0
    resume = bool(config["training"].get("resume", True) if resume is None else resume)
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
            updates = int(metadata.get("updates", 0))
            samples_seen = int(metadata.get("samples_seen", (start_epoch - 1) * info.n_train))
            step_size = float(metadata.get("hmc_step_size", step_size))
            if metrics_path.exists():
                rows = pd.read_csv(metrics_path).to_dict("records")
            print(f"Resuming MCEM run from {resume_ckpt.name} at epoch {start_epoch}.")

    checkpoint_epochs = {int(e) for e in config["training"].get("checkpoint_epochs", [])}
    checkpoint_every = int(config["training"].get("checkpoint_every", 0) or 0)
    progress = tqdm(range(start_epoch, epochs + 1), desc=name)
    for epoch in progress:
        accept_rates = []
        for x, _ in train_loader:
            x = x.to(device)
            z = torch.randn(x.shape[0], z_dim, device=device)
            for _ in range(int(hmc_cfg.get("posterior_steps", 10))):
                z, accept = hmc_step(model, x, z, step_size, int(hmc_cfg.get("leapfrog_steps", 10)))
                accept_rate = float(accept.float().mean().detach().cpu())
                accept_rates.append(accept_rate)
                if accept_rate > float(hmc_cfg.get("target_acceptance", 0.90)):
                    step_size *= 1.02
                else:
                    step_size *= 0.98
                step_size = min(max(step_size, 1e-4), 1.0)
            for _ in range(int(config["training"].get("weight_updates_per_sample", 5))):
                optimizer.zero_grad(set_to_none=True)
                logits = model.decode_logits(z.detach())
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()
                objective = bce + map_l2_penalty(model.parameters(), map_scale)
                objective.backward()
                optimizer.step()
                updates += 1
            samples_seen += x.shape[0]
        if epoch == 1 or epoch == epochs or epoch % int(config["training"].get("eval_every", 1)) == 0:
            proxy = evaluate_decoder_reconstruction(model, test_loader, device)
            rows.append(
                {
                    "method": "mcem",
                    "dataset": "mnist",
                    "binarization": config["data"]["binarization"],
                    "z_dim": z_dim,
                    "hidden_dim": hidden_dim,
                    "seed": seed,
                    "epoch": epoch,
                    "update": updates,
                    "samples_seen": samples_seen,
                    "n_train": n_train,
                    "prior_sample_bce": proxy["prior_sample_bce"],
                    "hmc_step_size": step_size,
                    "hmc_acceptance": sum(accept_rates) / max(len(accept_rates), 1),
                    "lr": lr,
                }
            )
            save_metrics_csv(metrics_path, rows)
            progress.set_postfix(accept=f"{rows[-1]['hmc_acceptance']:.2f}")
        should_save_intermediate = epoch in checkpoint_epochs or (
            checkpoint_every > 0 and epoch % checkpoint_every == 0 and epoch != epochs
        )
        if should_save_intermediate:
            save_checkpoint(
                ckpt_dir / f"{name}_epoch{epoch}.pt",
                model,
                {
                    "method": "mcem",
                    "config": config,
                    "n_train": n_train,
                    "seed": seed,
                    "lr": lr,
                    "epoch": epoch,
                    "updates": updates,
                    "samples_seen": samples_seen,
                    "hmc_step_size": step_size,
                    "package_versions": package_versions(),
                },
                optimizers={"optimizer": optimizer},
            )
    save_metrics_csv(metrics_path, rows)
    save_checkpoint(
        ckpt_path,
        model,
        {
            "method": "mcem",
            "config": config,
            "n_train": n_train,
            "seed": seed,
            "lr": lr,
            "epochs": epochs,
            "epoch": epochs,
            "updates": updates,
            "samples_seen": samples_seen,
            "hmc_step_size": step_size,
            "package_versions": package_versions(),
        },
        optimizers={"optimizer": optimizer},
    )
    return metrics_path, ckpt_path
