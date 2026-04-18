"""CLI entry point for packaged release-dataset regression benchmarks."""

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
from src.data.release_package import ReleaseSplitBundle, load_release_split_bundle
from src.evaluation.metrics import save_json_report
from src.models.random_forest import save_model_artifact
from src.models.regression_suite import (
    RegressionResult,
    build_regression_comparison_frame,
    build_regression_split_from_precomputed_frames,
    build_regression_split_summary,
    get_regression_display_name,
    train_regression_suite_on_split,
)
from src.visualization.plots import plot_actual_vs_predicted, plot_feature_importance


def _resolve_target_column(bundle: ReleaseSplitBundle, configured_target_column: str | None) -> str:
    target_column = configured_target_column or "reliability"
    manifest_targets = (
        bundle.feature_manifest.get("target_columns", []) if bundle.feature_manifest else []
    )
    if manifest_targets and target_column not in manifest_targets:
        manifest_target_text = ", ".join(manifest_targets)
        raise ValueError(
            f"Configured target column {target_column!r} is not listed in the feature manifest. "
            f"Available manifest targets: {manifest_target_text}"
        )
    return target_column


def _resolve_candidate_feature_columns(
    bundle: ReleaseSplitBundle,
    *,
    target_column: str,
    excluded_feature_columns: list[str],
) -> tuple[list[str], str]:
    if bundle.feature_manifest and bundle.feature_manifest.get("input_feature_columns"):
        manifest_columns = list(bundle.feature_manifest["input_feature_columns"])
        candidate_columns = [
            column
            for column in manifest_columns
            if column != target_column and column not in excluded_feature_columns
        ]
        return candidate_columns, "feature_manifest"

    shared_columns = [
        column
        for column in bundle.train_frame.columns
        if column in bundle.validation_frame.columns and column in bundle.test_frame.columns
    ]
    candidate_columns = [
        column
        for column in shared_columns
        if column != target_column and column not in excluded_feature_columns
    ]
    return candidate_columns, "shared_split_columns"


def _build_feature_policy(
    *,
    bundle: ReleaseSplitBundle,
    target_column: str,
    feature_source: str,
    candidate_feature_columns: list[str],
    used_feature_columns: list[str],
    excluded_feature_columns: list[str],
) -> dict[str, object]:
    return {
        "target_column": target_column,
        "feature_source": feature_source,
        "excluded_feature_columns": excluded_feature_columns,
        "candidate_feature_columns": candidate_feature_columns,
        "candidate_feature_column_count": int(len(candidate_feature_columns)),
        "used_feature_columns": used_feature_columns,
        "used_feature_column_count": int(len(used_feature_columns)),
        "dropped_zero_variance_columns": [
            column for column in candidate_feature_columns if column not in used_feature_columns
        ],
        "manifest_declared_target_columns": (
            list(bundle.feature_manifest.get("target_columns", []))
            if bundle.feature_manifest
            else []
        ),
        "manifest_recommended_group_columns": (
            list(bundle.feature_manifest.get("recommended_group_columns", []))
            if bundle.feature_manifest
            else []
        ),
    }


def _build_target_summary(bundle: ReleaseSplitBundle, target_column: str) -> dict[str, object]:
    combined_target = pd.concat(
        [
            pd.to_numeric(bundle.train_frame[target_column], errors="coerce"),
            pd.to_numeric(bundle.validation_frame[target_column], errors="coerce"),
            pd.to_numeric(bundle.test_frame[target_column], errors="coerce"),
        ],
        ignore_index=True,
    )
    if combined_target.isna().any():
        missing_target_count = int(combined_target.isna().sum())
        raise ValueError(
            f"Target column {target_column!r} contains {missing_target_count} missing or "
            "non-numeric values across the packaged splits."
        )

    return {
        "target_name": target_column,
        "target_source": "packaged_release_split",
        "row_count": int(len(combined_target)),
        "train_rows": int(len(bundle.train_frame)),
        "validation_rows": int(len(bundle.validation_frame)),
        "test_rows": int(len(bundle.test_frame)),
        "target_min": float(combined_target.min()),
        "target_max": float(combined_target.max()),
        "target_mean": float(combined_target.mean()),
        "target_median": float(combined_target.median()),
        "dataset_id": (
            bundle.split_manifest.get("dataset_id") if bundle.split_manifest is not None else None
        ),
        "profile_name": (
            bundle.split_manifest.get("profile_name") if bundle.split_manifest is not None else None
        ),
    }


def _build_split_summary(bundle: ReleaseSplitBundle, split) -> dict[str, object]:
    split_summary = build_regression_split_summary(split)
    split_summary.update(
        {
            "split_source": "packaged_release_split",
            "split_files": {name: str(path) for name, path in bundle.split_paths.items()},
            "group_columns": (
                list(bundle.split_manifest.get("group_columns", []))
                if bundle.split_manifest is not None
                else []
            ),
            "manifest_row_counts": (
                dict(bundle.split_manifest.get("row_counts", {}))
                if bundle.split_manifest is not None
                else {}
            ),
        }
    )
    return split_summary


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
    target_summary: dict[str, object],
    split_summary: dict[str, object],
    feature_policy: dict[str, object],
    comparison_frame: pd.DataFrame,
    best_result: RegressionResult,
) -> str:
    return "\n".join(
        [
            "# Packaged Release Regression",
            "",
            "## Dataset",
            f"- Dataset id: {target_summary['dataset_id'] or 'unknown'}",
            f"- Profile name: {target_summary['profile_name'] or 'unknown'}",
            f"- Split source: {split_summary['split_source']}",
            "- Group columns: " f"{', '.join(split_summary['group_columns']) or 'none'}",
            "",
            "## Target",
            f"- Target column: `{target_summary['target_name']}`",
            f"- Row count: {target_summary['row_count']}",
            f"- Target mean: {_format_metric(target_summary['target_mean'])}",
            f"- Target median: {_format_metric(target_summary['target_median'])}",
            "- Target min/max: "
            f"{_format_metric(target_summary['target_min'])} / "
            f"{_format_metric(target_summary['target_max'])}",
            "",
            "## Features",
            f"- Feature source: {feature_policy['feature_source']}",
            f"- Candidate feature count: {feature_policy['candidate_feature_column_count']}",
            f"- Used feature count: {feature_policy['used_feature_column_count']}",
            "- Dropped zero-variance columns: "
            f"{', '.join(feature_policy['dropped_zero_variance_columns']) or 'none'}",
            "",
            "## Results",
            "- Best validation model: " f"{get_regression_display_name(best_result.model_name)}",
            "- Best validation R2: " f"{_format_metric(best_result.validation_metrics['r2'])}",
            "- Best held-out test R2: " f"{_format_metric(best_result.test_metrics['r2'])}",
            "- Best held-out test MAE: " f"{_format_metric(best_result.test_metrics['mae'])}",
            "",
            _build_results_table(comparison_frame),
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


def run_release_regression(config_path: str | Path) -> None:
    """Train regression baselines on a packaged release dataset with fixed splits."""

    config = load_config(config_path)
    ensure_project_directories(config)
    run_directory = build_run_directory(config)
    config.output.run_name = run_directory.name
    save_resolved_config(config, run_directory / "run_config.yaml")

    bundle = load_release_split_bundle(config.data)
    target_column = _resolve_target_column(bundle, config.training.target_column)
    candidate_feature_columns, feature_source = _resolve_candidate_feature_columns(
        bundle,
        target_column=target_column,
        excluded_feature_columns=list(config.training.excluded_feature_columns),
    )
    split = build_regression_split_from_precomputed_frames(
        train_frame=bundle.train_frame,
        validation_frame=bundle.validation_frame,
        test_frame=bundle.test_frame,
        target_column=target_column,
        feature_columns=candidate_feature_columns,
    )
    feature_policy = _build_feature_policy(
        bundle=bundle,
        target_column=target_column,
        feature_source=feature_source,
        candidate_feature_columns=candidate_feature_columns,
        used_feature_columns=split.X_train.columns.tolist(),
        excluded_feature_columns=list(config.training.excluded_feature_columns),
    )
    target_summary = _build_target_summary(bundle, target_column)
    split_summary = _build_split_summary(bundle, split)

    split, results = train_regression_suite_on_split(split, config)
    comparison_frame = build_regression_comparison_frame(results)
    comparison_frame.to_csv(run_directory / "model_comparison.csv", index=False)
    save_json_report(split_summary, run_directory / "split_summary.json")
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
        target_summary=target_summary,
        split_summary=split_summary,
        feature_policy=feature_policy,
        comparison_frame=comparison_frame,
        best_result=best_result,
    )
    (run_directory / "summary.md").write_text(summary_markdown, encoding="utf-8")
    print(f"Release regression artifacts saved to: {run_directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train regression baselines on a packaged release dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/release_reliability_125k.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_release_regression(config_path=args.config)


if __name__ == "__main__":
    main()
