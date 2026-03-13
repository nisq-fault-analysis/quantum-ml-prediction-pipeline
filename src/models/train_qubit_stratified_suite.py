"""CLI entry point for qubit-count-stratified classification benchmarks."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config.io import (
    build_run_directory,
    ensure_project_directories,
    load_config,
    save_resolved_config,
)
from src.config.schema import ProjectConfig
from src.data.dataset import read_tabular_file, validate_required_columns
from src.evaluation.metrics import save_json_report
from src.models.classification_features import (
    build_classification_feature_policy,
    build_classification_features,
)
from src.models.model_suite import train_model_suite
from src.models.model_suite_artifacts import save_model_suite_artifacts
from src.models.splitting import build_split_summary


def _select_feature_path(config_path: str | Path) -> tuple[Path, ProjectConfig]:
    config = load_config(config_path)
    feature_paths = {
        "baseline_raw": config.features.baseline_feature_path,
        "topology_aware": config.features.topology_feature_path,
        "enhanced_topology": config.features.enhanced_feature_path,
    }
    return feature_paths[config.training.feature_set_name], config


def run_qubit_stratified_suite(config_path: str | Path) -> None:
    """Train the classification suite separately for each qubit-count subgroup."""

    feature_path, config = _select_feature_path(config_path)
    ensure_project_directories(config)
    run_directory = build_run_directory(config)
    config.output.run_name = run_directory.name
    save_resolved_config(config, run_directory / "run_config.yaml")

    feature_frame = read_tabular_file(feature_path, file_format="auto")
    validate_required_columns(
        feature_frame,
        [config.data.id_column, config.data.label_column, config.data.qubit_count_column],
    )
    save_json_report(
        {
            "prediction_context": config.training.prediction_context,
            "excluded_feature_columns": list(config.training.excluded_feature_columns),
        },
        run_directory / "feature_policy.json",
    )

    comparison_frames: list[pd.DataFrame] = []
    split_summaries: dict[str, dict[str, object]] = {}
    feature_policy_by_qubit: dict[str, dict[str, object]] = {}
    subset_metadata_by_qubit: dict[str, dict[str, object]] = {}
    qubit_counts = sorted(feature_frame[config.data.qubit_count_column].dropna().unique().tolist())
    for qubit_count in qubit_counts:
        subset = feature_frame.loc[
            feature_frame[config.data.qubit_count_column] == qubit_count
        ].copy()
        qubit_key = str(int(qubit_count))
        qubit_directory = run_directory / f"qubit_count_{qubit_key}"
        save_resolved_config(config, qubit_directory / "run_config.yaml")

        X, labels = build_classification_features(subset, config)
        feature_policy = build_classification_feature_policy(subset, config, X)

        split, results = train_model_suite(X, labels, config)
        comparison_frame = save_model_suite_artifacts(
            qubit_directory,
            results,
            build_split_summary(split),
            feature_policy,
        )
        subset_metadata = {
            "subset_type": "qubit_count",
            "filter_column": config.data.qubit_count_column,
            "filter_value": int(qubit_count),
            "row_count": int(len(subset)),
            "label_distribution": {
                label: int(count)
                for label, count in Counter(
                    subset[config.data.label_column].astype(str).tolist()
                ).items()
            },
        }
        save_json_report(subset_metadata, qubit_directory / "subset_metadata.json")
        comparison_frame.insert(0, "artifact_subdirectory", qubit_directory.name)
        comparison_frame.insert(0, "qubit_count", int(qubit_count))
        comparison_frames.append(comparison_frame)
        split_summaries[qubit_key] = build_split_summary(split)
        feature_policy_by_qubit[qubit_key] = feature_policy
        subset_metadata_by_qubit[qubit_key] = subset_metadata

    aggregated_frame = pd.concat(comparison_frames, ignore_index=True)
    aggregated_frame.to_csv(run_directory / "qubit_model_comparison.csv", index=False)

    best_rows = (
        aggregated_frame.sort_values(
            by=["qubit_count", "validation_macro_f1", "validation_accuracy"],
            ascending=[True, False, False],
        )
        .groupby("qubit_count", as_index=False)
        .first()
    )
    best_rows.to_csv(run_directory / "best_model_by_qubit_count.csv", index=False)
    save_json_report(split_summaries, run_directory / "split_summaries.json")
    save_json_report(feature_policy_by_qubit, run_directory / "feature_policy_by_qubit.json")
    save_json_report(subset_metadata_by_qubit, run_directory / "subset_metadata_by_qubit.json")
    print(f"Qubit-stratified benchmark artifacts saved to: {run_directory}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the qubit-stratified benchmark."""

    parser = argparse.ArgumentParser(description="Train classification baselines per qubit count.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/qubit_stratified.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_qubit_stratified_suite(config_path=args.config)


if __name__ == "__main__":
    main()
