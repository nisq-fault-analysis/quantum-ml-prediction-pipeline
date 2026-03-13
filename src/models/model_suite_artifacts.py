"""Shared artifact persistence for classification benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import save_json_report
from src.models.model_suite import ModelSuiteResult, build_model_comparison_frame, get_display_name
from src.models.random_forest import save_model_artifact
from src.visualization.plots import plot_confusion_matrix, plot_feature_importance


def save_model_suite_artifacts(
    run_directory: str | Path,
    results: list[ModelSuiteResult],
    split_summary: dict[str, Any],
    feature_policy: dict[str, Any],
    *,
    comparison_filename: str = "model_comparison.csv",
) -> pd.DataFrame:
    """Persist a full classification benchmark run in the standard artifact shape."""

    output_directory = Path(run_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    save_json_report(split_summary, output_directory / "split_summary.json")
    save_json_report(feature_policy, output_directory / "feature_policy.json")

    for result in results:
        model_directory = output_directory / result.model_name
        model_directory.mkdir(parents=True, exist_ok=True)

        metrics_payload = {
            "model_name": result.model_name,
            "model_display_name": get_display_name(result.model_name),
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
        }
        save_model_artifact(result.pipeline, model_directory / "model.joblib")
        save_json_report(metrics_payload, model_directory / "metrics.json")
        (model_directory / "validation_classification_report.txt").write_text(
            result.validation_report_text,
            encoding="utf-8",
        )
        (model_directory / "test_classification_report.txt").write_text(
            result.test_report_text,
            encoding="utf-8",
        )

        plot_confusion_matrix(
            y_true=result.y_validation.tolist(),
            y_pred=result.y_pred_validation.tolist(),
            labels=result.labels,
            output_path=model_directory / "validation_confusion_matrix.png",
            title=f"{get_display_name(result.model_name)} validation confusion matrix",
        )
        plot_confusion_matrix(
            y_true=result.y_test.tolist(),
            y_pred=result.y_pred_test.tolist(),
            labels=result.labels,
            output_path=model_directory / "test_confusion_matrix.png",
            title=f"{get_display_name(result.model_name)} test confusion matrix",
        )

        if result.feature_importance_frame is not None:
            result.feature_importance_frame.to_csv(
                model_directory / "feature_importance.csv",
                index=False,
            )
            plot_title = (
                f"{get_display_name(result.model_name)} coefficient magnitude"
                if result.model_name == "logistic_regression"
                else f"{get_display_name(result.model_name)} feature importance"
            )
            plot_feature_importance(
                importance_frame=result.feature_importance_frame,
                output_path=model_directory / "feature_importance.png",
                title=plot_title,
            )

    comparison_frame = build_model_comparison_frame(results)
    comparison_frame.to_csv(output_directory / comparison_filename, index=False)
    return comparison_frame
