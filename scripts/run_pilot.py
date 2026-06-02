from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aevb_v3.script_utils import load_experiment_config, write_selected_lr
from aevb_v3.training import run_lr_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Adagrad learning-rate pilot.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    selected = run_lr_pilot(config)
    path = write_selected_lr(config, selected)
    print(f"Selected Adagrad learning rate: {selected}")
    print(f"Wrote selected lr manifest: {path}")


if __name__ == "__main__":
    main()
