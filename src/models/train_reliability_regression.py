"""CLI entry point for leakage-free pre-run reliability regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.io import (
    build_run_directory,
    ensure_project_directories,
    load_config,
    save_resolved_config,
)
from src.config.schema import ProjectConfig
from src.data.dataset import read_tabular_file
from src.evaluation.metrics import save_json_report
from src.models.random_forest import save_model_artifact
from src.models.regression_suite import (
    RegressionResult,
    build_regression_comparison_frame,
    build_regression_split_summary,
    get_regression_display_name,
    train_regression_suite,
)
from src.models.reliability_regression import (
    RELIABILITY_ALLOWED_ENGINEERED_FEATURE_COLUMNS,
    RELIABILITY_ALLOWED_RAW_FEATURE_COLUMNS,
    build_reliability_feature_policy,
    build_reliability_features,
    build_reliability_target_frame,
    build_reliability_target_summary,
    resolve_reliability_target_source_columns,
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


def _format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def _build_results_table(comparison_frame: pd.DataFrame) -> str:
    header = "| Model | Validation R2 | Test R2 | Validation MAE | Test MAE |"
    separator = "| --- | ---: | ---: | ---: | ---: |"
    rows = [header, separator]

    for _, row in comparison_frame.iterrows():
        rows.append(
            "| "
            f"{row['model_display_name']} | "
            f"{_format_metric(row['validation_r2'])} | "
            f"{_format_metric(row['test_r2'])} | "
            f"{_format_metric(row['validation_mae'])} | "
            f"{_format_metric(row['test_mae'])} |"
        )

    return "\n".join(rows)


def _build_summary_markdown(
    *,
    comparison_frame: pd.DataFrame,
    feature_policy: dict[str, object],
    target_summary: dict[str, object],
    best_result: RegressionResult,
) -> str:
    forbidden_columns_present = ", ".join(feature_policy["forbidden_column_reasons"]) or "none"
    other_non_allowed_columns = (
        ", ".join(feature_policy["excluded_non_allowed_columns_present_in_feature_table"]) or "none"
    )

    return "\n".join(
        [
            "# Leakage-Free Reliability Regression",
            "",
            "## Target",
            "- Target name: reliability",
            "- Definition: `reliability = 1 - (bit_errors / qubit_count)`",
            "- Target bounds: `[0, 1]`",
            "- Observed bitstring column used only for target construction: "
            f"`{target_summary['bit_error_source_columns']['observed']}`",
            "- Ideal bitstring column used only for target construction: "
            f"`{target_summary['bit_error_source_columns']['ideal']}`",
            "- Row count: " f"{target_summary['row_count']}",
            "- Reliability mean: " f"{_format_metric(target_summary['reliability_mean'])}",
            "- Reliability median: " f"{_format_metric(target_summary['reliability_median'])}",
            "- Reliability min/max: "
            f"{_format_metric(target_summary['reliability_min'])} / "
            f"{_format_metric(target_summary['reliability_max'])}",
            "",
            "## Leakage-Free Feature Policy",
            "- Prediction context: pre-execution",
            "- Allowed raw features: " f"{', '.join(RELIABILITY_ALLOWED_RAW_FEATURE_COLUMNS)}",
            "- Allowed engineered pre-run features: "
            f"{', '.join(RELIABILITY_ALLOWED_ENGINEERED_FEATURE_COLUMNS)}",
            "- Used feature columns: " f"{', '.join(feature_policy['used_feature_columns'])}",
            "- Dropped zero-variance columns: "
            f"{', '.join(feature_policy['dropped_zero_variance_columns']) or 'none'}",
            "- Forbidden columns present in source feature table: " f"{forbidden_columns_present}",
            "- Other non-allowed columns present in source feature table: "
            f"{other_non_allowed_columns}",
            "",
            "## Results",
            "- Best validation model: " f"{get_regression_display_name(best_result.model_name)}",
            "- Best validation R2: " f"{_format_metric(best_result.validation_metrics['r2'])}",
            "- Best held-out test R2: " f"{_format_metric(best_result.test_metrics['r2'])}",
            "- Best held-out test MAE: " f"{_format_metric(best_result.test_metrics['mae'])}",
            "",
            _build_results_table(comparison_frame),
            "",
            "## Notes",
            "- Post-run columns such as `bitstring`, `ideal_bitstring`, `fidelity`, `bit_errors`, "
            "`observed_error_rate`, and target-derived fields are excluded from model inputs.",
            "- Root-level `metrics.json`, `model.joblib`, and `prediction_scatter.png` correspond "
            "to the best validation model for quick thesis reuse.",
            "- Root-level `feature_importance.png` comes from the best available tree model when "
            "the selected best model does not expose importances.",
            "- Per-model subdirectories preserve the full comparison suite.",
        ]
    )


def _save_best_model_artifacts(
    *,
    run_directory: Path,
    best_result: RegressionResult,
    best_tree_result: RegressionResult | None,
) -> None:
    save_model_artifact(best_result.pipeline, run_directory / "model.joblib")
    save_json_report(
        {
            "selected_model_name": best_result.model_name,
            "selected_model_display_name": get_regression_display_name(best_result.model_name),
            "selection_rule": "highest validation_r2, then lowest validation_mae",
            "feature_importance_source_model": (
                get_regression_display_name(best_tree_result.model_name)
                if best_tree_result is not None
                else None
            ),
            "validation_metrics": best_result.validation_metrics,
            "test_metrics": best_result.test_metrics,
        },
        run_directory / "metrics.json",
    )
    plot_actual_vs_predicted(
        y_true=best_result.y_test.tolist(),
        y_pred=best_result.y_pred_test.tolist(),
        output_path=run_directory / "prediction_scatter.png",
        title=f"{get_regression_display_name(best_result.model_name)} test predictions",
    )

    importance_result = (
        best_result if best_result.feature_importance_frame is not None else best_tree_result
    )
    if importance_result is not None and importance_result.feature_importance_frame is not None:
        importance_result.feature_importance_frame.to_csv(
            run_directory / "feature_importance.csv",
            index=False,
        )
        plot_feature_importance(
            importance_frame=importance_result.feature_importance_frame,
            output_path=run_directory / "feature_importance.png",
            title=f"{get_regression_display_name(importance_result.model_name)} feature importance",
        )


def run_reliability_regression(config_path: str | Path) -> None:
    """Train the leakage-free reliability regression suite and save artifacts."""

    feature_path, config = _select_feature_path(config_path)
    ensure_project_directories(config)
    run_directory = build_run_directory(config)
    config.output.run_name = run_directory.name
    save_resolved_config(config, run_directory / "run_config.yaml")

    cleaned_frame = read_tabular_file(config.data.cleaned_dataset_path, file_format="auto")
    feature_frame = read_tabular_file(feature_path, file_format="auto")

    observed_column, ideal_column = resolve_reliability_target_source_columns(cleaned_frame, config)
    target_frame = build_reliability_target_frame(cleaned_frame, config)
    target_summary = build_reliability_target_summary(
        target_frame,
        observed_column=observed_column,
        ideal_column=ideal_column,
    )
    X, target = build_reliability_features(feature_frame, target_frame, config)
    feature_policy = build_reliability_feature_policy(feature_frame, X, config)

    split, results = train_regression_suite(X, target, config)
    comparison_frame = build_regression_comparison_frame(results)
    comparison_frame.to_csv(run_directory / "model_comparison.csv", index=False)
    save_json_report(build_regression_split_summary(split), run_directory / "split_summary.json")
    save_json_report(feature_policy, run_directory / "feature_policy.json")
    save_json_report(target_summary, run_directory / "target_summary.json")

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

    best_model_name = str(comparison_frame.iloc[0]["model_name"])
    best_result = next(result for result in results if result.model_name == best_model_name)
    best_tree_result = next(
        (
            result
            for model_name in comparison_frame["model_name"].tolist()
            for result in results
            if result.model_name == model_name and result.feature_importance_frame is not None
        ),
        None,
    )
    _save_best_model_artifacts(
        run_directory=run_directory,
        best_result=best_result,
        best_tree_result=best_tree_result,
    )

    summary_markdown = _build_summary_markdown(
        comparison_frame=comparison_frame,
        feature_policy=feature_policy,
        target_summary=target_summary,
        best_result=best_result,
    )
    (run_directory / "summary.md").write_text(summary_markdown, encoding="utf-8")
    print(f"Reliability regression artifacts saved to: {run_directory}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the reliability regression suite."""

    parser = argparse.ArgumentParser(
        description="Train leakage-free pre-run reliability regression baselines."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/reliability_regression.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_reliability_regression(config_path=args.config)


if __name__ == "__main__":
    main()
