from __future__ import annotations

import logging

import pandas as pd

from src.config.schema import ProjectConfig, TrainingConfig
from src.data.release_package import ReleaseSplitBundle
from src.models.release_evaluation import (
    compute_permutation_importance_frame,
    compute_release_shap_artifacts,
)
from src.models.release_thesis_pipeline import (
    RegressionModelSpec,
    ReleaseFeatureContext,
    build_fixed_release_split,
    build_release_ablation_specs,
    fit_release_regression_model,
    resolve_release_feature_context,
    select_ablation_feature_columns,
)


def test_select_ablation_feature_columns_separates_variant_and_local_features() -> None:
    context = ReleaseFeatureContext(
        target_column="reliability",
        group_column="base_circuit_id",
        row_id_columns=["circuit_id", "compiler_variant"],
        leakage_excluded_columns=["reliability"],
        shared_feature_columns=["qubit_count", "t1_mean"],
        local_feature_columns=["local_t1_mean", "coupling_path_length"],
        variant_feature_columns=["compiler_variant"],
        candidate_feature_source="feature_manifest",
    )
    ablation_specs = build_release_ablation_specs()

    both_columns = select_ablation_feature_columns(context, ablation_specs["both"])
    with_local_columns = select_ablation_feature_columns(
        context,
        ablation_specs["both_with_local_features"],
    )

    assert both_columns == ["qubit_count", "t1_mean"]
    assert with_local_columns == [
        "qubit_count",
        "t1_mean",
        "compiler_variant",
        "local_t1_mean",
        "coupling_path_length",
    ]


def test_fit_release_regression_model_uses_training_only_preprocessing_statistics() -> None:
    split_frames = {
        "train": pd.DataFrame(
            {
                "base_circuit_id": ["g1", "g2", "g3", "g4"],
                "compiler_variant": ["raw", "transpiled", "raw", "transpiled"],
                "qubit_count": [2, 3, 4, 5],
                "local_t1_mean": [1.0, None, 3.0, 5.0],
                "reliability": [0.9, 0.8, 0.7, 0.6],
            }
        ),
        "validation": pd.DataFrame(
            {
                "base_circuit_id": ["g5"],
                "compiler_variant": ["unseen_variant"],
                "qubit_count": [6],
                "local_t1_mean": [100.0],
                "reliability": [0.5],
            }
        ),
        "test": pd.DataFrame(
            {
                "base_circuit_id": ["g6"],
                "compiler_variant": ["raw"],
                "qubit_count": [7],
                "local_t1_mean": [200.0],
                "reliability": [0.4],
            }
        ),
    }
    split_bundle = build_fixed_release_split(
        split_frames,
        feature_columns=["qubit_count", "compiler_variant", "local_t1_mean"],
        target_column="reliability",
        group_column="base_circuit_id",
    )
    config = ProjectConfig(
        training=TrainingConfig(
            regression_model_names=["ridge_regression"],
            grouped_cv_splits=2,
            tune_hyperparameters=False,
            ridge_alpha=1.0,
            grid_search_verbose=1,
        )
    )
    result = fit_release_regression_model(
        split_bundle=split_bundle,
        model_spec=RegressionModelSpec(
            name="ridge_regression",
            display_name="Ridge Regression",
            param_grid=[{"model__alpha": [1.0]}],
        ),
        config=config,
        logger=logging.getLogger("tests.release_thesis_pipeline"),
    )

    preprocessor = result.pipeline.named_steps["preprocessor"]
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    categorical_encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]

    assert numeric_imputer.statistics_.tolist() == [3.5, 3.0]
    assert categorical_encoder.categories_[0].tolist() == ["raw", "transpiled"]


def test_resolve_release_feature_context_excludes_forbidden_pre_run_columns() -> None:
    frame = pd.DataFrame(
        {
            "base_circuit_id": ["g1"],
            "circuit_id": ["g1-raw"],
            "compiler_variant": ["raw"],
            "qubit_count": [2],
            "noise_regime": ["high"],
            "noise_dominant_channel": ["readout"],
            "exact_match_probability": [0.9],
            "local_t1_mean": [12.0],
            "reliability": [0.8],
        }
    )
    bundle = ReleaseSplitBundle(
        train_frame=frame,
        validation_frame=frame,
        test_frame=frame,
        split_paths={},
        split_manifest={"group_columns": ["base_circuit_id"]},
        feature_manifest={
            "input_feature_columns": [
                "qubit_count",
                "compiler_variant",
                "noise_regime",
                "noise_dominant_channel",
                "exact_match_probability",
                "local_t1_mean",
            ],
            "target_columns": ["reliability"],
            "recommended_group_columns": ["base_circuit_id"],
        },
    )

    context = resolve_release_feature_context(
        bundle,
        ProjectConfig(training=TrainingConfig(target_column="reliability")),
        target_column="reliability",
    )

    assert "noise_regime" not in context.shared_feature_columns
    assert "noise_dominant_channel" not in context.shared_feature_columns
    assert "exact_match_probability" not in context.shared_feature_columns
    assert "local_t1_mean" in context.local_feature_columns


def test_compute_permutation_importance_frame_uses_input_feature_names() -> None:
    split_frames = {
        "train": pd.DataFrame(
            {
                "base_circuit_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
                "compiler_variant": [
                    "raw",
                    "transpiled",
                    "raw",
                    "transpiled",
                    "raw",
                    "transpiled",
                ],
                "qubit_count": [2, 3, 4, 5, 6, 7],
                "local_t1_mean": [1.0, None, 3.0, 5.0, 7.0, 9.0],
                "reliability": [0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
            }
        ),
        "validation": pd.DataFrame(
            {
                "base_circuit_id": ["g7", "g8"],
                "compiler_variant": ["raw", "transpiled"],
                "qubit_count": [8, 9],
                "local_t1_mean": [11.0, 13.0],
                "reliability": [0.45, 0.35],
            }
        ),
        "test": pd.DataFrame(
            {
                "base_circuit_id": ["g9", "g10"],
                "compiler_variant": ["raw", "transpiled"],
                "qubit_count": [10, 11],
                "local_t1_mean": [15.0, 17.0],
                "reliability": [0.25, 0.15],
            }
        ),
    }
    split_bundle = build_fixed_release_split(
        split_frames,
        feature_columns=["qubit_count", "compiler_variant", "local_t1_mean"],
        target_column="reliability",
        group_column="base_circuit_id",
    )
    config = ProjectConfig(
        training=TrainingConfig(
            regression_model_names=["ridge_regression"],
            grouped_cv_splits=3,
            tune_hyperparameters=False,
            ridge_alpha=1.0,
            grid_search_verbose=0,
            permutation_importance_repeats=2,
            permutation_importance_max_rows=100,
        )
    )
    result = fit_release_regression_model(
        split_bundle=split_bundle,
        model_spec=RegressionModelSpec(
            name="ridge_regression",
            display_name="Ridge Regression",
            param_grid=[{"model__alpha": [1.0]}],
        ),
        config=config,
        logger=logging.getLogger("tests.release_thesis_pipeline"),
    )

    importance_frame = compute_permutation_importance_frame(
        result.pipeline,
        split_bundle["X_validation"],
        split_bundle["y_validation"],
        max_rows=100,
        n_repeats=2,
        random_state=42,
    )

    assert set(importance_frame["feature"].tolist()) == {
        "qubit_count",
        "compiler_variant",
        "local_t1_mean",
    }
    assert len(importance_frame) == 3


def test_compute_release_shap_artifacts_aggregates_transformed_features() -> None:
    split_frames = {
        "train": pd.DataFrame(
            {
                "base_circuit_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
                "compiler_variant": [
                    "raw",
                    "transpiled",
                    "raw",
                    "transpiled",
                    "raw",
                    "transpiled",
                ],
                "qubit_count": [2, 3, 4, 5, 6, 7],
                "local_t1_mean": [1.0, 2.0, 3.0, 5.0, 7.0, 9.0],
                "reliability": [0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
            }
        ),
        "validation": pd.DataFrame(
            {
                "base_circuit_id": ["g7", "g8"],
                "compiler_variant": ["raw", "transpiled"],
                "qubit_count": [8, 9],
                "local_t1_mean": [11.0, 13.0],
                "reliability": [0.45, 0.35],
            }
        ),
        "test": pd.DataFrame(
            {
                "base_circuit_id": ["g9", "g10"],
                "compiler_variant": ["raw", "transpiled"],
                "qubit_count": [10, 11],
                "local_t1_mean": [15.0, 17.0],
                "reliability": [0.25, 0.15],
            }
        ),
    }
    split_bundle = build_fixed_release_split(
        split_frames,
        feature_columns=["qubit_count", "compiler_variant", "local_t1_mean"],
        target_column="reliability",
        group_column="base_circuit_id",
    )
    config = ProjectConfig(
        training=TrainingConfig(
            regression_model_names=["ridge_regression"],
            grouped_cv_splits=3,
            tune_hyperparameters=False,
            ridge_alpha=1.0,
            grid_search_verbose=0,
        )
    )
    result = fit_release_regression_model(
        split_bundle=split_bundle,
        model_spec=RegressionModelSpec(
            name="ridge_regression",
            display_name="Ridge Regression",
            param_grid=[{"model__alpha": [1.0]}],
        ),
        config=config,
        logger=logging.getLogger("tests.release_thesis_pipeline"),
    )

    shap_artifacts = compute_release_shap_artifacts(
        result.pipeline,
        model_name=result.model_name,
        X_background=split_bundle["X_train"],
        X_explained=split_bundle["X_validation"],
        explained_split="validation",
        max_rows=100,
        background_max_rows=100,
        random_state=42,
    )

    assert set(shap_artifacts.source_importance_frame["feature"].tolist()) == {
        "qubit_count",
        "compiler_variant",
        "local_t1_mean",
    }
    assert any(
        feature_name.startswith("compiler_variant_")
        for feature_name in shap_artifacts.transformed_importance_frame["feature"].tolist()
    )
    assert shap_artifacts.metadata["explained_split"] == "validation"
