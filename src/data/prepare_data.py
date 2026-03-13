"""CLI entry point for raw-data validation and cleaning."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.io import ensure_project_directories, load_config
from src.data.prepare import load_and_prepare_raw_dataset
from src.evaluation.metrics import save_json_report


def run_prepare_data(config_path: str | Path) -> None:
    """Read the raw dataset, validate it, and write interim outputs."""

    config = load_config(config_path)
    ensure_project_directories(config)

    prepared = load_and_prepare_raw_dataset(config.data)

    prepared.cleaned_frame.to_parquet(config.data.cleaned_dataset_path, index=False)

    if not prepared.invalid_rows.empty:
        prepared.invalid_rows.to_csv(config.data.invalid_rows_path, index=False)

    save_json_report(prepared.validation_summary, config.data.validation_report_path)

    print(f"Cleaned dataset saved to: {config.data.cleaned_dataset_path}")
    print(f"Validation report saved to: {config.data.validation_report_path}")
    if not prepared.invalid_rows.empty:
        print(f"Invalid rows saved to: {config.data.invalid_rows_path}")
    else:
        print("No invalid rows were detected during preparation.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the preparation step."""

    parser = argparse.ArgumentParser(description="Prepare the raw NISQ fault dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/baseline.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by `python -m src.data.prepare_data`."""

    args = parse_args()
    run_prepare_data(config_path=args.config)


if __name__ == "__main__":
    main()
