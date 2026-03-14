"""CLI entry point for the master experiment summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.reporting.experiment_summary import write_experiment_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the master experiment summary."""

    parser = argparse.ArgumentParser(
        description="Build a master table across saved benchmark runs."
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
        help="Root directory containing saved experiment runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments"),
        help="Directory where the summary CSV and JSON inventory will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, json_path = write_experiment_summary(
        experiments_root=args.experiments_root,
        output_dir=args.output_dir,
    )
    print(f"Master experiment matrix saved to: {csv_path}")
    print(f"Master experiment inventory saved to: {json_path}")


if __name__ == "__main__":
    main()
