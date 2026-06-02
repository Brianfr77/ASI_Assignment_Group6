from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aevb_v3.config import map_l2_scale, project_path
from aevb_v3.data import load_mnist_tensors, make_mnist_loaders
from aevb_v3.losses import vae_negative_elbo_loss
from aevb_v3.models import VAE
from aevb_v3.script_utils import load_experiment_config
from aevb_v3.training import fit_aevb_run
from aevb_v3.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local smoke tests for the v3 reproduction code.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    set_seed(0)
    device = get_device(config.get("device", "auto"))
    data_dir = project_path(config, "data_dir", "data")
    train_x1, _ = load_mnist_tensors(data_dir, "train", config["data"]["binarization"], int(config["data"]["binarization_seed"]))
    train_x2, _ = load_mnist_tensors(data_dir, "train", config["data"]["binarization"], int(config["data"]["binarization_seed"]))
    assert train_x1.shape == (60000, 784)
    assert torch.equal(train_x1, train_x2), "Static binarization is not deterministic."
    assert set(torch.unique(train_x1).tolist()).issubset({0.0, 1.0})

    loader, _, info = make_mnist_loaders(
        data_dir,
        batch_size=32,
        binarization=config["data"]["binarization"],
        binarization_seed=int(config["data"]["binarization_seed"]),
        seed=0,
        train_subset_size=256,
        test_subset_size=256,
        num_workers=0,
    )
    x, _ = next(iter(loader))
    model = VAE(z_dim=10, hidden_dim=32, init_std=0.01).to(device)
    logits, mu, logvar, _ = model(x.to(device))
    assert logits.shape == (x.shape[0], 784)
    assert mu.shape == (x.shape[0], 10)
    assert logvar.shape == (x.shape[0], 10)
    loss, metrics = vae_negative_elbo_loss(model, x.to(device), map_l2_scale(config, info.n_train))
    assert torch.isfinite(loss)
    assert metrics["negative_elbo"] > 0
    assert metrics["kl"] >= 0

    before = metrics["negative_elbo"]
    fit_aevb_run(
        config,
        z_dim=10,
        seed=0,
        lr=0.01,
        epochs=3,
        hidden_dim=32,
        train_subset_size=256,
        extra_name="smoke",
        force=True,
    )
    print(f"Smoke tests passed. Initial mini-batch NELBO: {before:.2f}")


if __name__ == "__main__":
    main()
