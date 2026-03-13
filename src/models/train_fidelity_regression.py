"""CLI entry point for fidelity regression benchmarks."""

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
from src.models.random_forest import save_model_artifact
from src.models.regression_suite import (
    build_regression_comparison_frame,
    build_regression_features,
    build_regression_split_summary,
    get_regression_display_name,
    train_regression_suite,
)
from src.visualization.plots import plot_actual_vs_predicted, plot_feature_importance


def _select_feature_path(config_path: str | Path) -> tuple[Path, ProjectConfig]:
    config = load_config(config_path)
    feature_paths = {
        "baseline_raw": config.features.baseline_feature_path,
        "topology_aware": config.features.topology_feature_path,
        "enhanced_topology": config.features.enhanced_feature_path,
    }
    return feature_paths[config.training.feature_set_name], config


def run_fidelity_regression(config_path: str | Path) -> None:
    """Train the fidelity regression suite and save comparable artifacts."""

    feature_path, config = _select_feature_path(config_path)
    ensure_project_directories(config)
    run_directory = build_run_directory(config)
    config.output.run_name = run_directory.name
    save_resolved_config(config, run_directory / "run_config.yaml")

    feature_frame = read_tabular_file(feature_path, file_format="auto")
    validate_required_columns(
        feature_frame, [config.data.id_column, config.data.label_column, "fidelity"]
    )

    X, target = build_regression_features(feature_frame, config)
    split, results = train_regression_suite(X, target, config)
    save_json_report(build_regression_split_summary(split), run_directory / "split_summary.json")

    for result in results:
        model_directory = run_directory / result.model_name
        model_directory.mkdir(parents=True, exist_ok=True)

        save_model_artifact(result.pipeline, model_directory / "model.joblib")
        save_json_report(
            {
                "model_name": result.model_name,
                "model_display_name": get_regression_display_name(result.model_name),
                "validation_metrics": result.validation_metrics,
                "test_metrics": result.test_metrics,
            },
            model_directory / "metrics.json",
        )
        plot_actual_vs_predicted(
            y_true=result.y_validation.tolist(),
            y_pred=result.y_pred_validation.tolist(),
            output_path=model_directory / "validation_actual_vs_predicted.png",
            title=f"{get_regression_display_name(result.model_name)} validation predictions",
        )
        plot_actual_vs_predicted(
            y_true=result.y_test.tolist(),
            y_pred=result.y_pred_test.tolist(),
            output_path=model_directory / "test_actual_vs_predicted.png",
            title=f"{get_regression_display_name(result.model_name)} test predictions",
        )

        if result.feature_importance_frame is not None:
            result.feature_importance_frame.to_csv(
                model_directory / "feature_importance.csv",
                index=False,
            )
            plot_feature_importance(
                importance_frame=result.feature_importance_frame,
                output_path=model_directory / "feature_importance.png",
                title=f"{get_regression_display_name(result.model_name)} feature importance",
            )

    comparison_frame = build_regression_comparison_frame(results)
    comparison_frame.to_csv(run_directory / "model_comparison.csv", index=False)
    print(f"Fidelity regression artifacts saved to: {run_directory}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the regression suite."""

    parser = argparse.ArgumentParser(description="Train fidelity regression baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/fidelity_regression.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fidelity_regression(config_path=args.config)


if __name__ == "__main__":
    main()
