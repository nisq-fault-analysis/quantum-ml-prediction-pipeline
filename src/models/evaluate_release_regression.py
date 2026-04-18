"""Recompute slice metrics and diagnostics for a saved release-ablation run."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config.io import load_config
from src.data.release_package import load_release_split_bundle
from src.evaluation.metrics import save_json_report
from src.models.release_evaluation import (
    build_local_feature_gain_diagnostic,
    compute_family_dominance_frame,
    compute_slice_metrics,
    compute_variant_gap_diagnostic,
    compute_worst_slice_frame,
    save_release_shap_artifacts,
)
from src.models.release_thesis_pipeline import (
    build_fixed_release_split,
    build_release_ablation_specs,
    filter_release_split_frames,
    resolve_release_feature_context,
    resolve_release_target_column,
    select_ablation_feature_columns,
    select_best_model_rows,
)


def evaluate_release_run(run_directory: str | Path, *, include_shap: bool = False) -> None:
    """Rebuild evaluation artifacts from saved validation/test prediction tables."""

    root = Path(run_directory)
    comparison_path = root / "ablation_model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Expected ablation comparison file not found: {comparison_path}")

    comparison_frame = pd.read_csv(comparison_path)
    prediction_paths = sorted(root.glob("*/*/validation_predictions.parquet"))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No saved validation prediction artifacts were found under {root}."
        )

    config = load_config(root / "run_config.yaml") if include_shap else None
    bundle = load_release_split_bundle(config.data) if config is not None else None
    target_column = (
        resolve_release_target_column(bundle, config)
        if bundle is not None and config is not None
        else None
    )
    context = (
        resolve_release_feature_context(bundle, config, target_column=target_column)
        if bundle is not None and config is not None and target_column is not None
        else None
    )
    ablation_specs = build_release_ablation_specs() if include_shap else {}
    split_bundles_by_ablation: dict[str, dict[str, object]] = {}

    suite_slice_frames: list[pd.DataFrame] = []
    suite_variant_diagnostics: list[dict[str, object]] = []
    for validation_prediction_path in prediction_paths:
        model_directory = validation_prediction_path.parent
        ablation_name = model_directory.parent.name
        model_name = model_directory.name
        test_prediction_path = model_directory / "test_predictions.parquet"
        validation_prediction_frame = pd.read_parquet(validation_prediction_path)
        test_prediction_frame = pd.read_parquet(test_prediction_path)

        validation_slice_metrics = compute_slice_metrics(validation_prediction_frame)
        test_slice_metrics = compute_slice_metrics(test_prediction_frame)
        combined_slice_metrics = pd.concat(
            [validation_slice_metrics, test_slice_metrics],
            ignore_index=True,
        )
        combined_slice_metrics.to_csv(model_directory / "slice_metrics.csv", index=False)
        compute_family_dominance_frame(test_prediction_frame).to_csv(
            model_directory / "family_dominance.csv",
            index=False,
        )
        compute_worst_slice_frame(
            test_slice_metrics,
            slice_names=["family", "qubit_count"],
        ).to_csv(model_directory / "worst_slices.csv", index=False)

        variant_gap = compute_variant_gap_diagnostic(test_prediction_frame)
        save_json_report(variant_gap, model_directory / "variant_gap_diagnostic.json")

        suite_slice_frames.append(combined_slice_metrics)
        variant_gap.update(
            {
                "ablation_name": str(test_prediction_frame["ablation_name"].iloc[0]),
                "model_name": str(test_prediction_frame["model_name"].iloc[0]),
            }
        )
        suite_variant_diagnostics.append(variant_gap)

        if include_shap and config is not None and bundle is not None and context is not None:
            split_bundle = split_bundles_by_ablation.get(ablation_name)
            if split_bundle is None:
                ablation_spec = ablation_specs[ablation_name]
                filtered_frames = filter_release_split_frames(bundle, ablation_spec)
                requested_feature_columns = select_ablation_feature_columns(context, ablation_spec)
                split_bundle = build_fixed_release_split(
                    filtered_frames,
                    feature_columns=requested_feature_columns,
                    target_column=target_column,
                    group_column=context.group_column,
                )
                split_bundles_by_ablation[ablation_name] = split_bundle

            comparison_row = comparison_frame.loc[
                (comparison_frame["ablation_name"] == ablation_name)
                & (comparison_frame["model_name"] == model_name)
            ]
            model_display_name = (
                str(comparison_row.iloc[0]["model_display_name"])
                if not comparison_row.empty
                else model_name
            )
            explained_features = (
                split_bundle["X_validation"]
                if config.training.shap_explained_split == "validation"
                else split_bundle["X_test"]
            )
            pipeline = joblib.load(model_directory / "model.joblib")
            try:
                save_release_shap_artifacts(
                    pipeline,
                    model_name=model_name,
                    model_display_name=model_display_name,
                    X_background=split_bundle["X_train"],
                    X_explained=explained_features,
                    explained_split=config.training.shap_explained_split,
                    output_directory=model_directory,
                    max_rows=config.training.shap_max_rows,
                    background_max_rows=config.training.shap_background_max_rows,
                    random_state=config.training.random_state,
                )
            except Exception as exc:
                save_json_report(
                    {
                        "available": False,
                        "error": str(exc),
                        "explained_split": config.training.shap_explained_split,
                        "ablation_name": ablation_name,
                        "model_name": model_name,
                    },
                    model_directory / "shap_analysis_error.json",
                )

    suite_slice_metrics = pd.concat(suite_slice_frames, ignore_index=True)
    suite_slice_metrics.to_csv(root / "suite_slice_metrics.csv", index=False)
    best_model_by_ablation = select_best_model_rows(comparison_frame)
    best_model_by_ablation.to_csv(root / "best_model_by_ablation.csv", index=False)
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
    best_slice_metrics.to_csv(root / "best_model_slice_metrics.csv", index=False)

    pd.DataFrame(suite_variant_diagnostics).to_csv(
        root / "variant_gap_diagnostics.csv",
        index=False,
    )
    save_json_report(
        build_local_feature_gain_diagnostic(best_slice_metrics),
        root / "local_feature_gain_diagnostic.json",
    )
    if include_shap:
        print(f"Release evaluation and SHAP artifacts refreshed under: {root}")
    else:
        print(f"Release evaluation artifacts refreshed under: {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute slice metrics for a saved release-ablation run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a saved experiments/release_ablation/<run_name> directory.",
    )
    parser.add_argument(
        "--include-shap",
        action="store_true",
        help="Also backfill SHAP artifacts for each saved model without retraining.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_release_run(args.run_dir, include_shap=args.include_shap)


if __name__ == "__main__":
    main()
