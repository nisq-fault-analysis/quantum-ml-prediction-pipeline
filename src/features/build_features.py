"""CLI entry point for building baseline and topology-aware feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.io import ensure_project_directories, load_config
from src.config.schema import ProjectConfig
from src.data.dataset import read_tabular_file, validate_required_columns
from src.evaluation.metrics import save_json_report
from src.features.gate_sequence import (
    engineer_enhanced_classification_features,
    engineer_gate_sequence_features,
)


def build_feature_sets(
    cleaned_frame: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, pd.DataFrame]:
    """Create two saved feature views from the cleaned interim dataset."""

    validate_required_columns(
        cleaned_frame,
        [
            config.data.id_column,
            config.data.label_column,
            *config.features.baseline_numeric_columns,
            *config.features.categorical_feature_columns,
            config.data.gate_sequence_column,
            "bitstring_aligned",
            "ideal_bitstring_aligned",
        ],
    )

    baseline_columns = [
        config.data.id_column,
        config.data.label_column,
        *config.features.baseline_numeric_columns,
        *config.features.categorical_feature_columns,
    ]
    baseline_feature_set = cleaned_frame.loc[:, baseline_columns].copy()

    topology_feature_frame = engineer_gate_sequence_features(
        frame=cleaned_frame,
        sequence_column=config.data.gate_sequence_column,
        feature_config=config.features,
        qubit_count_column=config.data.qubit_count_column,
        bitstring_column="bitstring_aligned",
        ideal_bitstring_column="ideal_bitstring_aligned",
    )
    topology_feature_set = pd.concat(
        [baseline_feature_set, topology_feature_frame],
        axis=1,
    )
    enhanced_feature_frame = engineer_enhanced_classification_features(
        frame=cleaned_frame,
        topology_feature_frame=topology_feature_frame,
        qubit_count_column=config.data.qubit_count_column,
    )
    enhanced_feature_set = pd.concat(
        [topology_feature_set, enhanced_feature_frame],
        axis=1,
    )

    return {
        "baseline_raw": baseline_feature_set,
        "topology_aware": topology_feature_set,
        "enhanced_topology": enhanced_feature_set,
    }


def run_build_features(config_path: str | Path) -> None:
    """Read the cleaned dataset, build feature sets, and save them to disk."""

    config = load_config(config_path)
    ensure_project_directories(config)

    cleaned_frame = read_tabular_file(config.data.cleaned_dataset_path, file_format="auto")
    feature_sets = build_feature_sets(cleaned_frame, config)

    feature_sets["baseline_raw"].to_parquet(config.features.baseline_feature_path, index=False)
    feature_sets["topology_aware"].to_parquet(config.features.topology_feature_path, index=False)
    feature_sets["enhanced_topology"].to_parquet(config.features.enhanced_feature_path, index=False)

    feature_report = {}
    for feature_set_name, feature_frame in feature_sets.items():
        candidate_feature_columns = [
            column
            for column in feature_frame.columns
            if column not in {config.data.id_column, config.data.label_column}
        ]
        zero_variance_columns = [
            column
            for column in candidate_feature_columns
            if feature_frame[column].nunique(dropna=False) <= 1
        ]
        feature_report[feature_set_name] = {
            "row_count": int(len(feature_frame)),
            "feature_column_count": int(len(candidate_feature_columns)),
            "feature_columns": candidate_feature_columns,
            "zero_variance_columns": zero_variance_columns,
        }

    save_json_report(feature_report, config.features.feature_report_path)
    print(f"Baseline feature set saved to: {config.features.baseline_feature_path}")
    print(f"Topology-aware feature set saved to: {config.features.topology_feature_path}")
    print(f"Enhanced topology feature set saved to: {config.features.enhanced_feature_path}")
    print(f"Feature report saved to: {config.features.feature_report_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the feature-building step."""

    parser = argparse.ArgumentParser(description="Build feature tables for the RF baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/baseline.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by `python -m src.features.build_features`."""

    args = parse_args()
    run_build_features(config_path=args.config)


if __name__ == "__main__":
    main()
