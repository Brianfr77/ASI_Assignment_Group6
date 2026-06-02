from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aevb_v3.collect import validate_metrics, write_final_tables, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect tables, validate metrics, and write a manifest.")
    parser.add_argument("--results-dir", default="results", help="Results directory.")
    args = parser.parse_args()
    written = write_final_tables(args.results_dir)
    validation = validate_metrics(args.results_dir)
    manifest = write_manifest(args.results_dir)
    print(f"Wrote tables: {[str(p) for p in written]}")
    print(f"Validation: {validation}")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
