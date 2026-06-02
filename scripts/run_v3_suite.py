from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def backup_results(backup_dir: Path, label: str) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "scripts/backup_results.py",
            "--results-dir",
            "results",
            "--output-dir",
            str(backup_dir),
            "--name",
            f"AEVB_MNIST_Reproduction_v3_{label}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v3 practical reproduction suite with backups.")
    parser.add_argument("--backup-dir", default="/content/drive/MyDrive/AEVB_MNIST_Reproduction_v3_backups")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke tests.")
    parser.add_argument("--skip-marginal", action="store_true", help="Skip marginal likelihood experiment.")
    args = parser.parse_args()
    backup_dir = Path(args.backup_dir)

    stages = []
    if not args.skip_smoke:
        stages.append(("smoke", [sys.executable, "scripts/run_smoke_tests.py", "--config", "configs/quick_debug.yaml"]))
    stages.extend(
        [
            ("pilot", [sys.executable, "scripts/run_pilot.py", "--config", "configs/aevb_binarized_practical.yaml"]),
            (
                "aevb",
                [
                    sys.executable,
                    "scripts/run_aevb_sweep.py",
                    "--config",
                    "configs/aevb_binarized_practical.yaml",
                    "--resume",
                ],
            ),
            (
                "wake_sleep",
                [
                    sys.executable,
                    "scripts/run_wake_sleep.py",
                    "--config",
                    "configs/wake_sleep_binarized_practical.yaml",
                    "--resume",
                ],
            ),
        ]
    )
    if not args.skip_marginal:
        stages.append(
            (
                "marginal_likelihood",
                [
                    sys.executable,
                    "scripts/run_marginal_likelihood.py",
                    "--config",
                    "configs/marginal_likelihood_z3_h100_practical.yaml",
                    "--resume",
                ],
            )
        )
    stages.extend(
        [
            ("figures", [sys.executable, "scripts/make_figures.py", "--config", "configs/aevb_binarized_practical.yaml"]),
            ("collect", [sys.executable, "scripts/collect_results.py", "--results-dir", "results"]),
        ]
    )

    for label, command in stages:
        try:
            run_command(command)
        finally:
            if (PROJECT_ROOT / "results").exists():
                backup_results(backup_dir, label)


if __name__ == "__main__":
    main()
