from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aevb_v3.script_utils import load_experiment_config, resolve_lr
from aevb_v3.training import fit_aevb_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AEVB lower-bound sweep.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--lr", type=float, default=None, help="Override Adagrad global learning rate.")
    parser.add_argument("--force", action="store_true", help="Retrain even if metrics/checkpoint exist.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest intermediate checkpoint.")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    lr = resolve_lr(config, args.lr)
    z_dims = [int(z) for z in config["experiment"]["z_dims"]]
    seeds = [int(s) for s in config["experiment"]["seeds"]]
    run_pairs = [(z_dim, seed) for z_dim in z_dims for seed in seeds]
    qualitative_seeds = [int(s) for s in config["experiment"].get("qualitative_seeds", [0])]
    run_pairs.extend(
        (int(z_dim), seed)
        for z_dim in config["experiment"].get("qualitative_z_dims", [])
        for seed in qualitative_seeds
    )
    seen = set()
    for z_dim, seed in run_pairs:
        key = (z_dim, seed)
        if key in seen:
            continue
        seen.add(key)
        metrics, ckpt = fit_aevb_run(
            config,
            z_dim=z_dim,
            seed=seed,
            lr=lr,
            force=args.force,
            resume=args.resume or bool(config["training"].get("resume", True)),
        )
        print(f"AEVB run complete: metrics={metrics} checkpoint={ckpt}")


if __name__ == "__main__":
    main()
