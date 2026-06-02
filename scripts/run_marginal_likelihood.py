from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aevb_v3.config import project_path
from aevb_v3.io import load_checkpoint
from aevb_v3.marginal_likelihood import estimate_marginal_likelihood, save_marginal_likelihood_rows
from aevb_v3.mcem import fit_mcem_run
from aevb_v3.models import GenerativeModel, VAE
from aevb_v3.script_utils import load_experiment_config, resolve_lr
from aevb_v3.training import fit_aevb_run
from aevb_v3.utils import get_device
from aevb_v3.wake_sleep import fit_wake_sleep_run


def _load_model_for_method(method: str, ckpt_path: Path, config: dict, device):
    z_dim = int(config["model"]["z_dim"])
    hidden_dim = int(config["model"]["hidden_dim"])
    if method == "mcem":
        model = GenerativeModel(z_dim=z_dim, hidden_dim=hidden_dim, init_std=float(config["model"]["init_std"]))
    else:
        model = VAE(z_dim=z_dim, hidden_dim=hidden_dim, init_std=float(config["model"]["init_std"]))
    load_checkpoint(ckpt_path, model, device)
    return model.to(device)


def _scheduled_checkpoints(final_ckpt: Path, config: dict) -> list[tuple[int, Path]]:
    checkpoints = []
    for epoch in [int(e) for e in config["training"].get("checkpoint_epochs", [])]:
        candidate = final_ckpt.with_name(f"{final_ckpt.stem}_epoch{epoch}.pt")
        if candidate.exists():
            checkpoints.append((epoch, candidate))
    if not checkpoints:
        checkpoints.append((int(config["training"]["epochs"]), final_ckpt))
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Run z=3/h=100 marginal likelihood experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--lr", type=float, default=None, help="Override Adagrad global learning rate.")
    parser.add_argument("--force", action="store_true", help="Retrain even if metrics/checkpoint exist.")
    parser.add_argument("--resume", action="store_true", help="Resume training methods from intermediate checkpoints.")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    lr = resolve_lr(config, args.lr)
    device = get_device(config.get("device", "auto"))
    results_dir = project_path(config, "results_dir", "results")
    output = results_dir / "metrics" / "marginal_likelihood_summary.csv"
    rows = []
    for n_train in [int(n) for n in config["experiment"]["n_train_values"]]:
        for seed in [int(s) for s in config["experiment"]["seeds"]]:
            method_ckpts: dict[str, Path] = {}
            _, aevb_ckpt = fit_aevb_run(
                config,
                z_dim=int(config["model"]["z_dim"]),
                hidden_dim=int(config["model"]["hidden_dim"]),
                seed=seed,
                lr=lr,
                train_subset_size=n_train,
                extra_name=f"n{n_train}_ml",
                force=args.force,
                resume=args.resume or bool(config["training"].get("resume", True)),
            )
            method_ckpts["aevb"] = aevb_ckpt
            _, ws_ckpt = fit_wake_sleep_run(
                config,
                z_dim=int(config["model"]["z_dim"]),
                hidden_dim=int(config["model"]["hidden_dim"]),
                seed=seed,
                lr=lr,
                train_subset_size=n_train,
                extra_name=f"n{n_train}_ml",
                force=args.force,
                resume=args.resume or bool(config["training"].get("resume", True)),
            )
            method_ckpts["wake_sleep"] = ws_ckpt
            _, mcem_ckpt = fit_mcem_run(
                config,
                n_train=n_train,
                seed=seed,
                lr=lr,
                force=args.force,
                resume=args.resume or bool(config["training"].get("resume", True)),
            )
            method_ckpts["mcem"] = mcem_ckpt

            for method, final_ckpt in method_ckpts.items():
                for checkpoint_epoch, ckpt in _scheduled_checkpoints(final_ckpt, config):
                    model = _load_model_for_method(method, ckpt, config, device)
                    eval_config = dict(config)
                    eval_config["samples_seen"] = int(checkpoint_epoch) * int(n_train)
                    for split in ["train", "test"]:
                        row = estimate_marginal_likelihood(
                            model,
                            eval_config,
                            method=method,
                            n_train=n_train,
                            seed=seed,
                            split=split,
                            max_points=int(config["marginal_likelihood"].get("max_points", 1000)),
                        )
                        row["checkpoint_epoch"] = checkpoint_epoch
                        row["checkpoint_file"] = ckpt.name
                        rows.append(row)
                        save_marginal_likelihood_rows(output, rows)
                        print(row)
    save_marginal_likelihood_rows(output, rows)
    print(f"Wrote marginal likelihood summary: {output}")


if __name__ == "__main__":
    main()
