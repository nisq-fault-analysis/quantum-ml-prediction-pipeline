"""CLI entry point for the first Random Forest baseline experiment."""

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
from src.evaluation.metrics import save_json_report
from src.models.random_forest import save_model_artifact, train_random_forest
from src.models.splitting import build_split_summary, split_dataset
from src.visualization.plots import plot_confusion_matrix, plot_feature_importance


def _select_feature_path(config_path: str | Path) -> tuple[Path, ProjectConfig]:
    config = load_config(config_path)
    feature_path = (
        config.features.baseline_feature_path
        if config.training.feature_set_name == "baseline_raw"
        else config.features.topology_feature_path
    )
    return feature_path, config


def run_rf_baseline(config_path: str | Path) -> None:
    """Train, evaluate, and save the Random Forest baseline artifacts."""

    feature_path, config = _select_feature_path(config_path)
    ensure_project_directories(config)
    run_directory = build_run_directory(config)
    config.output.run_name = run_directory.name
    save_resolved_config(config, run_directory / "run_config.yaml")

    feature_frame = read_tabular_file(feature_path, file_format="auto")
    validate_required_columns(feature_frame, [config.data.id_column, config.data.label_column])

    labels = feature_frame[config.data.label_column].astype(str)
    candidate_drop_columns = {
        config.data.id_column,
        config.data.label_column,
    }
    X = feature_frame.drop(columns=list(candidate_drop_columns), errors="ignore")
    X = X.loc[:, X.nunique(dropna=False) > 1].copy()

    split = split_dataset(X, labels, config)
    save_json_report(build_split_summary(split), run_directory / "split_summary.json")
    result = train_random_forest(X, labels, config)

    save_model_artifact(result.pipeline, run_directory / "model.joblib")
    save_json_report(
        {
            "model_name": "random_forest",
            "model_display_name": "Random Forest",
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
        },
        run_directory / "metrics.json",
    )
    (run_directory / "validation_classification_report.txt").write_text(
        result.validation_report_text,
        encoding="utf-8",
    )
    (run_directory / "test_classification_report.txt").write_text(
        result.test_report_text,
        encoding="utf-8",
    )

    plot_confusion_matrix(
        y_true=result.y_validation.tolist(),
        y_pred=result.y_pred_validation.tolist(),
        labels=result.labels,
        output_path=run_directory / "validation_confusion_matrix.png",
        title="Random Forest validation confusion matrix",
    )
    plot_confusion_matrix(
        y_true=result.y_test.tolist(),
        y_pred=result.y_pred_test.tolist(),
        labels=result.labels,
        output_path=run_directory / "test_confusion_matrix.png",
        title="Random Forest test confusion matrix",
    )
    plot_feature_importance(
        importance_frame=result.feature_importance_frame,
        output_path=run_directory / "feature_importance.png",
    )

    print(f"Random Forest baseline artifacts saved to: {run_directory}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline training script."""

    parser = argparse.ArgumentParser(description="Train the Random Forest baseline model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/baseline.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by `python -m src.models.train_rf_baseline`."""

    args = parse_args()
    run_rf_baseline(config_path=args.config)


if __name__ == "__main__":
    main()
