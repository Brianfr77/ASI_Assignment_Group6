from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a timestamped archive of the current results folder.")
    parser.add_argument("--results-dir", default="results", help="Results directory to archive.")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive", help="Directory where the archive is copied.")
    parser.add_argument("--name", default="AEVB_MNIST_Reproduction_v3_partial", help="Archive filename prefix.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir.resolve()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aevb_v3_backup_") as tmpdir:
        archive_base = Path(tmpdir) / f"{args.name}_{timestamp}"
        archive_path = Path(
            shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=results_dir.parent,
                base_dir=results_dir.name,
            )
        )
        destination = output_dir / archive_path.name
        shutil.copy2(archive_path, destination)
    print(f"Wrote backup archive: {destination}")


if __name__ == "__main__":
    main()
