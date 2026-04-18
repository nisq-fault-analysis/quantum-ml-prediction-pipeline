"""CLI entry point for grouped ablation studies on packaged thesis releases."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.config.io import (
    build_run_directory,
    ensure_project_directories,
    load_config,
    save_resolved_config,
)
from src.data.release_package import load_release_split_bundle
from src.evaluation.metrics import save_json_report
from src.models.grouped_split_validation import (
    assert_grouped_split_integrity,
    build_split_membership_frame,
)
from src.models.mlflow_tracking import (
    log_artifact_path,
    log_metrics,
    log_params,
    resolve_git_metadata,
    start_mlflow_run,
)
from src.models.random_forest import save_model_artifact
from src.models.release_evaluation import (
    apply_difficulty_bucketer,
    build_local_feature_gain_diagnostic,
    build_prediction_frame,
    compute_family_dominance_frame,
    compute_permutation_importance_frame,
    compute_slice_metrics,
    compute_variant_gap_diagnostic,
    compute_worst_slice_frame,
    fit_difficulty_bucketer,
    save_release_shap_artifacts,
)
from src.models.release_thesis_pipeline import (
    build_fixed_release_split,
    build_model_importance_frame,
    build_regression_model_specs,
    build_release_ablation_specs,
    build_release_feature_policy,
    filter_release_split_frames,
    fit_release_regression_model,
    resolve_release_feature_context,
    resolve_release_target_column,
    resolve_repo_root,
    select_ablation_feature_columns,
    select_best_model_rows,
)
from src.visualization.plots import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
    plot_slice_metric_bars,
)


def _configure_training_logger(run_directory: Path) -> logging.Logger:
    """Create a timestamped logger that writes to console and the run folder."""

    logger = logging.getLogger(f"qfc.release_ablation.{run_directory.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_directory / "training.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _close_training_logger(logger: logging.Logger) -> None:
    """Flush and close logger handlers so log files are released promptly."""

    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _log_split_overview(
    logger: logging.Logger,
    *,
    split_frames: dict[str, pd.DataFrame],
    group_column: str,
    prefix: str,
) -> None:
    """Log row, group, and compiler-variant coverage for a split mapping."""

    for split_name, frame in split_frames.items():
        variant_counts = (
            frame["compiler_variant"].value_counts().sort_index().to_dict()
            if "compiler_variant" in frame.columns
            else {}
        )
        logger.info(
            "%s %s split: %s rows, %s unique %s values, variants=%s",
            prefix,
            split_name,
            len(frame),
            frame[group_column].nunique(dropna=False),
            group_column,
            variant_counts or "n/a",
        )


def _build_suite_summary_markdown(
    *,
    audit_report: dict[str, object],
    best_model_by_ablation: pd.DataFrame,
    local_feature_gain: dict[str, object],
    worst_slices: pd.DataFrame,
) -> str:
    rows = [
        "# Grouped Release Ablation Summary",
        "",
        "## Leakage Audit",
        f"- Grouped split passed: {audit_report['passed']}",
        f"- Group column: `{audit_report['group_column']}`",
        "- Overlapping groups found: "
        f"{sum(len(groups) for groups in audit_report['overlapping_groups'].values())}",
        "",
        "## Best Model By Ablation",
    ]
    for row in best_model_by_ablation.itertuples():
        rows.append(
            "- "
            f"{row.ablation_name}: {row.model_display_name} "
            f"(validation R2={row.validation_r2:.4f}, test R2={row.test_r2:.4f}, "
            f"test MAE={row.test_mae:.4f})"
        )

    rows.extend(
        [
            "",
            "## Diagnostics",
            "- Local-feature gain diagnostic available: "
            f"{local_feature_gain.get('available', False)}",
        ]
    )
    if local_feature_gain.get("available"):
        rows.append(
            "- Routing-sensitive test delta with local features: "
            f"delta R2={float(local_feature_gain['delta_r2_with_local_minus_without_local']):.4f}, "
            f"delta MAE={float(local_feature_gain['delta_mae_with_local_minus_without_local']):.4f}"
        )

    rows.extend(["", "## Worst Slices"])
    if worst_slices.empty:
        rows.append("- No slice diagnostics were available.")
    else:
        for row in worst_slices.itertuples():
            r2_value = "n/a" if pd.isna(row.r2) else f"{float(row.r2):.4f}"
            rows.append(
                "- "
                f"{row.slice_name}={row.slice_value} (n={int(row.sample_count)}): "
                f"MAE={float(row.mae):.4f}, R2={r2_value}"
            )

    return "\n".join(rows)


def _build_audit_summary() -> dict[str, object]:
    """Return the static leakage audit narrative for the upgraded pipeline."""

    return {
        "leakage_risks_found": [
            "Legacy release training regenerated splits internally instead of relying on grouped "
            "`base_circuit_id` boundaries.",
            "Legacy hyperparameter tuning in the broader repository used standard splits rather "
            "than grouped folds.",
            "Variant-specific local hardware features were null on raw rows without an explicit "
            "ablation protocol to test whether they genuinely helped.",
        ],
        "fixes_applied": [
            "Packaged release experiments now validate disjoint `base_circuit_id` groups across "
            "train, validation, and test.",
            "All imputers, encoders, scalers, and estimators live inside one sklearn Pipeline "
            "fit only on training data or grouped CV training folds.",
            "Grouped CV is now used for hyperparameter tuning on the training split.",
            "Ablation modes now separate raw-only, transpiled-only, mixed without variant info, "
            "mixed with compiler variant, and mixed with compiler variant plus local features.",
            "Evaluation now reports global and sliced metrics with sample counts and stores exact "
            "split membership as an artifact.",
        ],
        "remaining_methodological_caveats": [
            "The dataset remains simulator-generated rather than collected from live hardware.",
            "Outer train/validation/test partitions come from the supplied manifest rather than "
            "being regenerated here, so robustness to alternative grouped outer splits still "
            "requires additional reruns.",
            "Permutation importance is computed on a bounded evaluation sample for practicality, "
            "so the ranking is approximate rather than exhaustive.",
        ],
    }


def run_release_ablation(config_path: str | Path) -> None:
    """Run grouped ablations for the packaged thesis dataset."""

    run_started_at = time.perf_counter()
    config = load_config(config_path)
    ensure_project_directories(config)
    run_directory = build_run_directory(config)
    config.output.run_name = run_directory.name
    save_resolved_config(config, run_directory / "run_config.yaml")
    logger = _configure_training_logger(run_directory)
    logger.info("Starting grouped release ablation run in %s", run_directory)
    logger.info("Resolved config saved to %s", run_directory / "run_config.yaml")
    logger.info("Source config: %s", Path(config_path).resolve())

    bundle = load_release_split_bundle(config.data)
    target_column = resolve_release_target_column(bundle, config)
    context = resolve_release_feature_context(bundle, config, target_column=target_column)
    repo_root = resolve_repo_root()
    git_metadata = resolve_git_metadata(repo_root)
    logger.info(
        "Loaded release dataset_id=%s, profile=%s, target=%s, feature source=%s",
        bundle.split_manifest.get("dataset_id") if bundle.split_manifest else "unknown",
        bundle.split_manifest.get("profile_name") if bundle.split_manifest else "unknown",
        target_column,
        context.candidate_feature_source,
    )

    split_frames = {
        "train": bundle.train_frame,
        "validation": bundle.validation_frame,
        "test": bundle.test_frame,
    }
    _log_split_overview(
        logger,
        split_frames=split_frames,
        group_column=context.group_column,
        prefix="Loaded",
    )
    audit = assert_grouped_split_integrity(
        split_frames,
        group_column=context.group_column,
        row_id_columns=context.row_id_columns,
    )
    audit_report = audit.to_dict()
    if bundle.split_manifest and "twin_leakage_check" in bundle.split_manifest:
        audit_report["manifest_twin_leakage_check"] = bundle.split_manifest["twin_leakage_check"]
    save_json_report(audit_report, run_directory / "grouped_split_audit.json")
    logger.info(
        "Grouped split audit passed for %s with %s total overlapping groups.",
        context.group_column,
        sum(len(groups) for groups in audit_report["overlapping_groups"].values()),
    )

    split_membership = build_split_membership_frame(
        split_frames,
        group_column=context.group_column,
        row_id_columns=context.row_id_columns,
    )
    split_membership.to_parquet(run_directory / "split_membership.parquet", index=False)
    if bundle.split_manifest is not None:
        save_json_report(bundle.split_manifest, run_directory / "split_manifest_used.json")
    if bundle.feature_manifest is not None:
        save_json_report(bundle.feature_manifest, run_directory / "feature_manifest_used.json")

    difficulty_bucketer = fit_difficulty_bucketer(
        bundle.train_frame,
        reference_column=config.training.difficulty_reference_column,
        bucket_count=config.training.difficulty_bucket_count,
    )
    save_json_report(difficulty_bucketer.to_dict(), run_directory / "difficulty_bucket_config.json")
    logger.info(
        "Fitted %s difficulty buckets from training column %s.",
        config.training.difficulty_bucket_count,
        config.training.difficulty_reference_column,
    )

    ablation_specs = build_release_ablation_specs()
    model_specs = build_regression_model_specs(config)
    logger.info(
        "Planned run: %s ablation(s), %s model(s), GridSearchCV verbose=%s.",
        len(config.training.ablation_modes),
        len(model_specs),
        config.training.grid_search_verbose,
    )
    suite_summary_rows: list[dict[str, object]] = []
    suite_slice_metrics_frames: list[pd.DataFrame] = []
    suite_variant_diagnostics: list[dict[str, object]] = []
    audit_summary = _build_audit_summary()
    save_json_report(audit_summary, run_directory / "audit_summary.json")

    for ablation_name in config.training.ablation_modes:
        ablation_started_at = time.perf_counter()
        ablation_spec = ablation_specs[ablation_name]
        ablation_directory = run_directory / ablation_name
        ablation_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Starting ablation %s: %s", ablation_name, ablation_spec.description)

        filtered_frames = filter_release_split_frames(bundle, ablation_spec)
        ablation_audit = assert_grouped_split_integrity(
            filtered_frames,
            group_column=context.group_column,
            row_id_columns=context.row_id_columns,
        )
        save_json_report(
            ablation_audit.to_dict(),
            ablation_directory / "grouped_split_audit.json",
        )
        _log_split_overview(
            logger,
            split_frames=filtered_frames,
            group_column=context.group_column,
            prefix=f"Ablation {ablation_name}",
        )

        feature_columns = select_ablation_feature_columns(context, ablation_spec)
        split_bundle = build_fixed_release_split(
            filtered_frames,
            feature_columns=feature_columns,
            target_column=target_column,
            group_column=context.group_column,
        )
        feature_policy = build_release_feature_policy(
            context=context,
            ablation_spec=ablation_spec,
            feature_columns=split_bundle["feature_columns"],
        )
        save_json_report(feature_policy, ablation_directory / "feature_policy.json")
        logger.info(
            "Ablation %s uses %s feature columns before encoding (%s local feature columns).",
            ablation_name,
            len(split_bundle["feature_columns"]),
            len(feature_policy["used_local_feature_columns"]),
        )

        evaluation_frames = {
            split_name: apply_difficulty_bucketer(frame, difficulty_bucketer)
            for split_name, frame in filtered_frames.items()
        }
        ablation_summary_rows: list[dict[str, object]] = []
        ablation_slice_frames: list[pd.DataFrame] = []

        for model_spec in model_specs:
            model_started_at = time.perf_counter()
            model_directory = ablation_directory / model_spec.name
            model_directory.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Training %s within ablation %s. Artifacts: %s",
                model_spec.display_name,
                ablation_name,
                model_directory,
            )

            result = fit_release_regression_model(
                split_bundle=split_bundle,
                model_spec=model_spec,
                config=config,
                logger=logger,
            )
            save_model_artifact(result.pipeline, model_directory / "model.joblib")
            result.cv_results_frame.to_csv(model_directory / "grouped_cv_results.csv", index=False)

            validation_prediction_frame = build_prediction_frame(
                evaluation_frames["validation"],
                y_true=split_bundle["y_validation"],
                y_pred=result.validation_predictions,
                split_name="validation",
                ablation_name=ablation_name,
                model_name=result.model_name,
                target_column=target_column,
            )
            test_prediction_frame = build_prediction_frame(
                evaluation_frames["test"],
                y_true=split_bundle["y_test"],
                y_pred=result.test_predictions,
                split_name="test",
                ablation_name=ablation_name,
                model_name=result.model_name,
                target_column=target_column,
            )
            validation_prediction_frame.to_parquet(
                model_directory / "validation_predictions.parquet",
                index=False,
            )
            test_prediction_frame.to_parquet(
                model_directory / "test_predictions.parquet",
                index=False,
            )

            validation_slice_metrics = compute_slice_metrics(validation_prediction_frame)
            test_slice_metrics = compute_slice_metrics(test_prediction_frame)
            slice_metrics_frame = pd.concat(
                [validation_slice_metrics, test_slice_metrics],
                ignore_index=True,
            )
            slice_metrics_frame.to_csv(model_directory / "slice_metrics.csv", index=False)
            ablation_slice_frames.append(slice_metrics_frame)
            suite_slice_metrics_frames.append(slice_metrics_frame)

            family_dominance_frame = compute_family_dominance_frame(test_prediction_frame)
            family_dominance_frame.to_csv(model_directory / "family_dominance.csv", index=False)
            worst_slice_frame = compute_worst_slice_frame(
                test_slice_metrics,
                slice_names=["family", "qubit_count"],
            )
            worst_slice_frame.to_csv(model_directory / "worst_slices.csv", index=False)

            variant_gap_diagnostic = compute_variant_gap_diagnostic(test_prediction_frame)
            variant_gap_diagnostic.update(
                {
                    "ablation_name": ablation_name,
                    "model_name": result.model_name,
                }
            )
            suite_variant_diagnostics.append(variant_gap_diagnostic)
            save_json_report(
                variant_gap_diagnostic,
                model_directory / "variant_gap_diagnostic.json",
            )

            metrics_payload = {
                "ablation_name": ablation_name,
                "model_name": result.model_name,
                "model_display_name": result.model_display_name,
                "best_params": result.best_params,
                "validation_metrics": result.validation_metrics,
                "test_metrics": result.test_metrics,
            }
            save_json_report(metrics_payload, model_directory / "metrics.json")

            importance_frame = build_model_importance_frame(result.pipeline, result.model_name)
            if importance_frame is not None:
                importance_frame.to_csv(model_directory / "feature_importance.csv", index=False)
                plot_feature_importance(
                    importance_frame=importance_frame.rename(columns={"importance": "importance"}),
                    output_path=model_directory / "feature_importance.png",
                    title=f"{result.model_display_name} feature importance",
                )

            try:
                permutation_importance_frame = compute_permutation_importance_frame(
                    result.pipeline,
                    split_bundle["X_validation"],
                    split_bundle["y_validation"],
                    max_rows=config.training.permutation_importance_max_rows,
                    n_repeats=config.training.permutation_importance_repeats,
                    random_state=config.training.random_state,
                )
            except Exception as exc:
                logger.exception(
                    "Permutation importance failed for %s in ablation %s. "
                    "Continuing without permutation diagnostics.",
                    result.model_display_name,
                    ablation_name,
                )
                save_json_report(
                    {
                        "available": False,
                        "error": str(exc),
                        "ablation_name": ablation_name,
                        "model_name": result.model_name,
                    },
                    model_directory / "permutation_importance_error.json",
                )
            else:
                permutation_importance_frame.to_csv(
                    model_directory / "permutation_importance.csv",
                    index=False,
                )
                plot_feature_importance(
                    importance_frame=permutation_importance_frame.rename(
                        columns={"importance_mean": "importance"}
                    ),
                    output_path=model_directory / "permutation_importance.png",
                    title=f"{result.model_display_name} permutation importance",
                )

            if config.training.enable_shap:
                explained_split_name = config.training.shap_explained_split
                explained_features = (
                    split_bundle["X_validation"]
                    if explained_split_name == "validation"
                    else split_bundle["X_test"]
                )
                try:
                    shap_directory = save_release_shap_artifacts(
                        result.pipeline,
                        model_name=result.model_name,
                        model_display_name=result.model_display_name,
                        X_background=split_bundle["X_train"],
                        X_explained=explained_features,
                        explained_split=explained_split_name,
                        output_directory=model_directory,
                        max_rows=config.training.shap_max_rows,
                        background_max_rows=config.training.shap_background_max_rows,
                        random_state=config.training.random_state,
                    )
                except Exception as exc:
                    logger.exception(
                        "SHAP analysis failed for %s in ablation %s. Continuing without SHAP "
                        "artifacts.",
                        result.model_display_name,
                        ablation_name,
                    )
                    save_json_report(
                        {
                            "available": False,
                            "error": str(exc),
                            "explained_split": config.training.shap_explained_split,
                            "ablation_name": ablation_name,
                            "model_name": result.model_name,
                        },
                        model_directory / "shap_analysis_error.json",
                    )
                else:
                    logger.info(
                        "Saved SHAP artifacts for %s in ablation %s to %s",
                        result.model_display_name,
                        ablation_name,
                        shap_directory,
                    )

            plot_actual_vs_predicted(
                y_true=split_bundle["y_validation"].tolist(),
                y_pred=result.validation_predictions.tolist(),
                output_path=model_directory / "validation_actual_vs_predicted.png",
                title=f"{result.model_display_name} validation predictions",
            )
            plot_actual_vs_predicted(
                y_true=split_bundle["y_test"].tolist(),
                y_pred=result.test_predictions.tolist(),
                output_path=model_directory / "test_actual_vs_predicted.png",
                title=f"{result.model_display_name} test predictions",
            )
            plot_residuals(
                y_true=split_bundle["y_test"].tolist(),
                y_pred=result.test_predictions.tolist(),
                output_path=model_directory / "test_residuals.png",
                title=f"{result.model_display_name} test residuals",
            )
            plot_slice_metric_bars(
                test_slice_metrics.loc[test_slice_metrics["slice_name"] == "family"],
                output_path=model_directory / "family_mae_slices.png",
                metric_column="mae",
                title=f"{result.model_display_name} family MAE slices",
            )
            plot_slice_metric_bars(
                test_slice_metrics.loc[test_slice_metrics["slice_name"] == "qubit_count"],
                output_path=model_directory / "qubit_mae_slices.png",
                metric_column="mae",
                title=f"{result.model_display_name} qubit-count MAE slices",
            )

            run_name_prefix = config.training.mlflow_run_name_prefix or run_directory.name
            with start_mlflow_run(
                config,
                run_name=f"{run_name_prefix}-{ablation_name}-{result.model_name}",
                tags={
                    "dataset_id": (
                        bundle.split_manifest.get("dataset_id") if bundle.split_manifest else None
                    ),
                    "profile_name": (
                        bundle.split_manifest.get("profile_name") if bundle.split_manifest else None
                    ),
                    "profile_hash_sha256": (
                        bundle.split_manifest.get("profile_hash_sha256")
                        if bundle.split_manifest
                        else None
                    ),
                    "target_name": target_column,
                    "model_type": result.model_name,
                    "ablation_name": ablation_name,
                    "split_strategy": "precomputed_grouped_release_manifest",
                    "group_key": context.group_column,
                    **git_metadata,
                },
            ) as mlflow:
                log_params(
                    mlflow,
                    {
                        "dataset_id": (
                            bundle.split_manifest.get("dataset_id")
                            if bundle.split_manifest
                            else None
                        ),
                        "manifest_profile_hash": (
                            bundle.split_manifest.get("profile_hash_sha256")
                            if bundle.split_manifest
                            else None
                        ),
                        "random_seed": config.training.random_state,
                        "split_strategy": "precomputed_grouped_release_manifest",
                        "group_key": context.group_column,
                        "feature_set_name": ablation_name,
                        "target_name": target_column,
                        "model_type": result.model_name,
                        "grouped_cv_splits": config.training.grouped_cv_splits,
                        "enable_local_features": ablation_spec.include_local_features,
                        "include_compiler_variant": ablation_spec.include_compiler_variant,
                        **result.best_params,
                    },
                )
                log_metrics(mlflow, result.validation_metrics, prefix="validation")
                log_metrics(mlflow, result.test_metrics, prefix="test")
                log_artifact_path(mlflow, model_directory, artifact_path="artifacts")
                log_artifact_path(
                    mlflow,
                    run_directory / "split_membership.parquet",
                    artifact_path="artifacts",
                )
            logger.info(
                "Finished %s in ablation %s in %.1f seconds. Validation R2=%.4f, Test R2=%.4f",
                result.model_display_name,
                ablation_name,
                time.perf_counter() - model_started_at,
                result.validation_metrics["r2"],
                result.test_metrics["r2"],
            )

            ablation_summary_rows.append(
                {
                    "ablation_name": ablation_name,
                    "ablation_description": ablation_spec.description,
                    "model_name": result.model_name,
                    "model_display_name": result.model_display_name,
                    "target_name": target_column,
                    "feature_column_count": int(len(split_bundle["feature_columns"])),
                    "used_local_feature_count": int(
                        len(
                            [
                                column
                                for column in split_bundle["feature_columns"]
                                if column in context.local_feature_columns
                            ]
                        )
                    ),
                    "validation_mae": result.validation_metrics["mae"],
                    "validation_rmse": result.validation_metrics["rmse"],
                    "validation_r2": result.validation_metrics["r2"],
                    "test_mae": result.test_metrics["mae"],
                    "test_rmse": result.test_metrics["rmse"],
                    "test_r2": result.test_metrics["r2"],
                    "cv_best_r2": result.validation_metrics["cv_best_r2"],
                    "best_params": str(result.best_params),
                }
            )

        ablation_summary_frame = pd.DataFrame(ablation_summary_rows).sort_values(
            by=["validation_r2", "validation_mae"],
            ascending=[False, True],
        )
        ablation_summary_frame.to_csv(ablation_directory / "model_comparison.csv", index=False)
        suite_summary_rows.extend(ablation_summary_rows)

        if ablation_slice_frames:
            pd.concat(ablation_slice_frames, ignore_index=True).to_csv(
                ablation_directory / "slice_metrics.csv",
                index=False,
            )
        if not ablation_summary_frame.empty:
            best_ablation_row = ablation_summary_frame.iloc[0]
            logger.info(
                "Completed ablation %s in %.1f seconds. Best validation model: %s (R2=%.4f).",
                ablation_name,
                time.perf_counter() - ablation_started_at,
                best_ablation_row["model_display_name"],
                float(best_ablation_row["validation_r2"]),
            )

    suite_summary_frame = pd.DataFrame(suite_summary_rows).sort_values(
        by=["ablation_name", "validation_r2", "validation_mae"],
        ascending=[True, False, True],
    )
    suite_summary_frame.to_csv(run_directory / "ablation_model_comparison.csv", index=False)
    best_model_by_ablation = select_best_model_rows(suite_summary_frame)
    best_model_by_ablation.to_csv(run_directory / "best_model_by_ablation.csv", index=False)

    suite_slice_metrics = pd.concat(suite_slice_metrics_frames, ignore_index=True)
    suite_slice_metrics.to_csv(run_directory / "suite_slice_metrics.csv", index=False)
    best_model_keys = set(
        zip(
            best_model_by_ablation["ablation_name"],
            best_model_by_ablation["model_name"],
            strict=True,
        )
    )
    best_slice_metrics = suite_slice_metrics.loc[
        suite_slice_metrics.apply(
            lambda row: (row["ablation_name"], row["model_name"]) in best_model_keys,
            axis=1,
        )
    ].copy()
    best_slice_metrics.to_csv(run_directory / "best_model_slice_metrics.csv", index=False)

    variant_gap_frame = pd.DataFrame(suite_variant_diagnostics)
    variant_gap_frame.to_csv(run_directory / "variant_gap_diagnostics.csv", index=False)
    local_feature_gain = build_local_feature_gain_diagnostic(best_slice_metrics)
    save_json_report(local_feature_gain, run_directory / "local_feature_gain_diagnostic.json")

    best_overall_row = (
        suite_summary_frame.sort_values(
            by=["validation_r2", "validation_mae"],
            ascending=[False, True],
        )
        .iloc[0]
        .to_dict()
    )
    best_overall_slice_metrics = best_slice_metrics.loc[
        (best_slice_metrics["ablation_name"] == best_overall_row["ablation_name"])
        & (best_slice_metrics["model_name"] == best_overall_row["model_name"])
        & (best_slice_metrics["split_name"] == "test")
    ].copy()
    worst_slices = compute_worst_slice_frame(
        best_overall_slice_metrics,
        slice_names=["family", "qubit_count"],
    )
    worst_slices.to_csv(run_directory / "best_overall_worst_slices.csv", index=False)

    summary_markdown = _build_suite_summary_markdown(
        audit_report=audit_report,
        best_model_by_ablation=best_model_by_ablation,
        local_feature_gain=local_feature_gain,
        worst_slices=worst_slices,
    )
    (run_directory / "summary.md").write_text(summary_markdown, encoding="utf-8")

    logger.info(
        "Release ablation finished in %.1f seconds. Summary: %s",
        time.perf_counter() - run_started_at,
        run_directory / "summary.md",
    )
    logger.info("Training log written to %s", run_directory / "training.log")
    logger.info("Leakage risks found:")
    for item in audit_summary["leakage_risks_found"]:
        logger.info("- %s", item)
    logger.info("Fixes applied:")
    for item in audit_summary["fixes_applied"]:
        logger.info("- %s", item)
    logger.info("Remaining methodological caveats:")
    for item in audit_summary["remaining_methodological_caveats"]:
        logger.info("- %s", item)
    _close_training_logger(logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped thesis ablations on a packaged release dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/release_reliability_125k_ablation.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_release_ablation(config_path=args.config)


if __name__ == "__main__":
    main()
