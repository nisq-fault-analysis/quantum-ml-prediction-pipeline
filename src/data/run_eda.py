"""Command-line entry point for a first-pass exploratory data analysis run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.io import load_config
from src.data.dataset import prepare_research_table, read_tabular_dataset, validate_required_columns
from src.evaluation.metrics import save_json_report
from src.features.gate_sequence import engineer_gate_sequence_features
from src.visualization.plots import plot_categorical_distribution, plot_numeric_histogram


def build_eda_summary(
    frame: pd.DataFrame, label_column: str, qubit_count_column: str | None = None
) -> dict[str, Any]:
    """Create a compact summary that is easy to inspect and easy to version."""

    summary: dict[str, Any] = {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "columns": frame.columns.tolist(),
        "missing_values_top_20": frame.isna()
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .astype(int)
        .to_dict(),
        "label_distribution_top_20": frame[label_column]
        .astype(str)
        .value_counts()
        .head(20)
        .astype(int)
        .to_dict(),
    }

    if qubit_count_column and qubit_count_column in frame.columns:
        summary["qubit_count_distribution"] = (
            frame[qubit_count_column].astype(str).value_counts().sort_index().astype(int).to_dict()
        )

    return summary


def run_eda(config_path: str | Path) -> None:
    """Run lightweight EDA and save reusable figures."""

    config = load_config(config_path)
    dataset = prepare_research_table(read_tabular_dataset(config.data), config.data)

    required_columns = [config.data.label_column, config.data.gate_sequence_column]
    validate_required_columns(dataset, required_columns)

    eda_output_dir = config.output.output_dir / "eda"
    figure_dir = config.output.figures_dir / "eda"
    eda_output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary = build_eda_summary(
        frame=dataset,
        label_column=config.data.label_column,
        qubit_count_column=config.data.qubit_count_column,
    )
    save_json_report(summary, eda_output_dir / "eda_summary.json")

    plot_categorical_distribution(
        values=dataset[config.data.label_column],
        output_path=figure_dir / "label_distribution.png",
        title="Fault type distribution",
        xlabel=config.data.label_column,
    )

    if config.data.qubit_count_column in dataset.columns:
        plot_categorical_distribution(
            values=dataset[config.data.qubit_count_column],
            output_path=figure_dir / "qubit_count_distribution.png",
            title="Qubit count distribution",
            xlabel=config.data.qubit_count_column,
        )

    gate_features = engineer_gate_sequence_features(
        frame=dataset,
        sequence_column=config.data.gate_sequence_column,
        feature_config=config.features,
    )
    plot_numeric_histogram(
        values=gate_features["gate_token_count"],
        output_path=figure_dir / "gate_token_count_histogram.png",
        title="Gate token count per circuit",
        xlabel="Token count",
    )

    print(f"EDA outputs saved to: {eda_output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the EDA runner."""

    parser = argparse.ArgumentParser(description="Run starter EDA for the NISQ fault dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/baseline.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by `python -m src.data.run_eda`."""

    args = parse_args()
    run_eda(config_path=args.config)


if __name__ == "__main__":
    main()
