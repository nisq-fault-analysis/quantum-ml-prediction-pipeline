"""CLI entry point for training and comparing the baseline model suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.io import (
    build_run_directory,
    ensure_project_directories,
    load_config,
    save_resolved_config,
)
from src.config.schema import ProjectConfig
from src.data.dataset import read_tabular_file, validate_required_columns
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
    feature_path = feature_paths[config.training.feature_set_name]
    return feature_path, config


def run_model_suite(config_path: str | Path) -> None:
    """Train the configured model suite and save comparable experiment artifacts."""

    feature_path, config = _select_feature_path(config_path)
    ensure_project_directories(config)
    run_directory = build_run_directory(config)
    config.output.run_name = run_directory.name
    save_resolved_config(config, run_directory / "run_config.yaml")

    feature_frame = read_tabular_file(feature_path, file_format="auto")
    validate_required_columns(feature_frame, [config.data.id_column, config.data.label_column])

    X, labels = build_classification_features(feature_frame, config)
    feature_policy = build_classification_feature_policy(feature_frame, config, X)

    split, results = train_model_suite(X, labels, config)
    save_model_suite_artifacts(
        run_directory,
        results,
        build_split_summary(split),
        feature_policy,
    )

    print(f"Baseline model suite artifacts saved to: {run_directory}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the model suite script."""

    parser = argparse.ArgumentParser(description="Train and compare the thesis baseline models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/model_suite.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by `python -m src.models.train_model_suite`."""

    args = parse_args()
    run_model_suite(config_path=args.config)


if __name__ == "__main__":
    main()
