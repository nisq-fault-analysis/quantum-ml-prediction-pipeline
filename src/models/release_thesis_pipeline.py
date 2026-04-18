"""Core training helpers for grouped thesis regression on packaged releases."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.schema import ProjectConfig, ReleaseAblationMode
from src.data.dataset import validate_required_columns
from src.data.release_package import ReleaseSplitBundle
from src.evaluation.metrics import compute_regression_metrics
from src.models.grouped_split_validation import build_grouped_cv

LOCAL_FEATURE_COLUMNS = [
    "local_readout_error_mean",
    "local_readout_error_max",
    "local_t1_mean",
    "local_t2_mean",
    "local_two_qubit_error_mean",
    "local_two_qubit_error_max",
    "coupling_path_length",
]

VARIANT_FEATURE_COLUMNS = ["compiler_variant"]
DEFAULT_ROW_ID_COLUMNS = ["circuit_id", "compiler_variant"]
FORBIDDEN_PRE_RUN_COLUMNS = [
    "exact_match_probability",
    "mitigated_reliability",
    "mitigation_gain",
    "noise_regime",
    "noise_dominant_channel",
]


@dataclass(frozen=True, slots=True)
class ReleaseAblationSpec:
    """One thesis ablation regime for the release dataset."""

    name: ReleaseAblationMode
    description: str
    allowed_variants: tuple[str, ...]
    include_compiler_variant: bool
    include_local_features: bool


@dataclass(frozen=True, slots=True)
class ReleaseFeatureContext:
    """Resolved feature groups for the packaged release dataset."""

    target_column: str
    group_column: str
    row_id_columns: list[str]
    leakage_excluded_columns: list[str]
    shared_feature_columns: list[str]
    local_feature_columns: list[str]
    variant_feature_columns: list[str]
    candidate_feature_source: str


@dataclass(frozen=True, slots=True)
class RegressionModelSpec:
    """Model metadata used by the grouped thesis regression runner."""

    name: str
    display_name: str
    param_grid: list[dict[str, Any]]


@dataclass(slots=True)
class FittedReleaseModelResult:
    """Artifacts from one fitted model under one ablation."""

    model_name: str
    model_display_name: str
    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_results_frame: pd.DataFrame
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_predictions: pd.Series
    test_predictions: pd.Series


class _LoggerWriter:
    """File-like adapter that forwards line-buffered text into a logger."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        return len(message)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.rstrip())
        self._buffer = ""


def build_release_ablation_specs() -> dict[ReleaseAblationMode, ReleaseAblationSpec]:
    """Return the supported ablation modes for the thesis dataset."""

    return {
        "raw_only": ReleaseAblationSpec(
            name="raw_only",
            description="Raw rows only, shared pre-run features, no transpilation-local features.",
            allowed_variants=("raw",),
            include_compiler_variant=False,
            include_local_features=False,
        ),
        "transpiled_only": ReleaseAblationSpec(
            name="transpiled_only",
            description=(
                "Transpiled rows only, shared pre-run features plus local mapping-aware features."
            ),
            allowed_variants=("transpiled",),
            include_compiler_variant=False,
            include_local_features=True,
        ),
        "both": ReleaseAblationSpec(
            name="both",
            description=(
                "Raw and transpiled rows together, but only shared variant-agnostic pre-run "
                "features."
            ),
            allowed_variants=("raw", "transpiled"),
            include_compiler_variant=False,
            include_local_features=False,
        ),
        "both_without_local_features": ReleaseAblationSpec(
            name="both_without_local_features",
            description=(
                "Raw and transpiled rows together, shared pre-run features plus explicit "
                "`compiler_variant`, without local mapping-aware features."
            ),
            allowed_variants=("raw", "transpiled"),
            include_compiler_variant=True,
            include_local_features=False,
        ),
        "both_with_local_features": ReleaseAblationSpec(
            name="both_with_local_features",
            description=(
                "Raw and transpiled rows together, explicit `compiler_variant`, and local "
                "mapping-aware features with train-only imputation."
            ),
            allowed_variants=("raw", "transpiled"),
            include_compiler_variant=True,
            include_local_features=True,
        ),
    }


def resolve_release_target_column(
    bundle: ReleaseSplitBundle,
    config: ProjectConfig,
) -> str:
    """Resolve the thesis target column and validate it against the manifest."""

    target_column = config.training.target_column or "reliability"
    manifest_targets = (
        list(bundle.feature_manifest.get("target_columns", [])) if bundle.feature_manifest else []
    )
    if manifest_targets and target_column not in manifest_targets:
        raise ValueError(
            f"Configured target column {target_column!r} is not listed in the release feature "
            f"manifest. Available targets: {manifest_targets}"
        )
    return target_column


def resolve_release_feature_context(
    bundle: ReleaseSplitBundle,
    config: ProjectConfig,
    *,
    target_column: str,
) -> ReleaseFeatureContext:
    """Resolve the safe feature groups for the packaged release dataset."""

    if bundle.feature_manifest and bundle.feature_manifest.get("input_feature_columns"):
        source_columns = list(bundle.feature_manifest["input_feature_columns"])
        candidate_source = "feature_manifest"
    else:
        source_columns = [
            column
            for column in bundle.train_frame.columns
            if column in bundle.validation_frame.columns and column in bundle.test_frame.columns
        ]
        candidate_source = "shared_split_columns"

    leakage_excluded_columns = sorted(
        {
            target_column,
            config.data.id_column,
            config.data.label_column,
            *FORBIDDEN_PRE_RUN_COLUMNS,
            *config.training.excluded_feature_columns,
        }
    )
    safe_columns = [column for column in source_columns if column not in leakage_excluded_columns]
    local_feature_columns = [column for column in LOCAL_FEATURE_COLUMNS if column in safe_columns]
    shared_feature_columns = [
        column for column in safe_columns if column not in local_feature_columns
    ]
    variant_feature_columns = [
        column for column in VARIANT_FEATURE_COLUMNS if column in bundle.train_frame.columns
    ]
    if not shared_feature_columns:
        raise ValueError("No safe shared feature columns remain after leakage exclusions.")

    group_column = (
        config.training.group_column
        or (
            bundle.feature_manifest.get("recommended_group_columns", [None])[0]
            if bundle.feature_manifest
            else None
        )
        or "base_circuit_id"
    )
    row_id_columns = [
        column
        for column in DEFAULT_ROW_ID_COLUMNS
        if column in bundle.train_frame.columns and column != group_column
    ]
    if not row_id_columns:
        row_id_columns = [config.data.id_column]

    return ReleaseFeatureContext(
        target_column=target_column,
        group_column=group_column,
        row_id_columns=row_id_columns,
        leakage_excluded_columns=leakage_excluded_columns,
        shared_feature_columns=shared_feature_columns,
        local_feature_columns=local_feature_columns,
        variant_feature_columns=variant_feature_columns,
        candidate_feature_source=candidate_source,
    )


def filter_release_split_frames(
    bundle: ReleaseSplitBundle,
    ablation_spec: ReleaseAblationSpec,
) -> dict[str, pd.DataFrame]:
    """Filter train/validation/test frames to the requested compiler variants."""

    allowed_variants = set(ablation_spec.allowed_variants)
    split_frames = {
        "train": bundle.train_frame,
        "validation": bundle.validation_frame,
        "test": bundle.test_frame,
    }
    filtered: dict[str, pd.DataFrame] = {}
    for split_name, frame in split_frames.items():
        validate_required_columns(frame, ["compiler_variant"])
        filtered_frame = frame.loc[frame["compiler_variant"].isin(allowed_variants)].copy()
        if filtered_frame.empty:
            raise ValueError(
                f"Ablation {ablation_spec.name!r} produced an empty {split_name} split."
            )
        filtered[split_name] = filtered_frame
    return filtered


def select_ablation_feature_columns(
    context: ReleaseFeatureContext,
    ablation_spec: ReleaseAblationSpec,
) -> list[str]:
    """Select the feature columns used by one ablation mode."""

    feature_columns = list(context.shared_feature_columns)
    if ablation_spec.include_compiler_variant:
        feature_columns.extend(context.variant_feature_columns)
    if ablation_spec.include_local_features:
        feature_columns.extend(context.local_feature_columns)
    deduplicated_columns = list(dict.fromkeys(feature_columns))
    if not deduplicated_columns:
        raise ValueError(f"Ablation {ablation_spec.name!r} has no feature columns.")
    return deduplicated_columns


def build_release_feature_policy(
    *,
    context: ReleaseFeatureContext,
    ablation_spec: ReleaseAblationSpec,
    feature_columns: list[str],
) -> dict[str, object]:
    """Describe how one ablation constructs its design matrix."""

    used_local_features = [
        column for column in context.local_feature_columns if column in feature_columns
    ]
    return {
        "ablation_name": ablation_spec.name,
        "ablation_description": ablation_spec.description,
        "allowed_variants": list(ablation_spec.allowed_variants),
        "candidate_feature_source": context.candidate_feature_source,
        "target_column": context.target_column,
        "group_column": context.group_column,
        "leakage_excluded_columns": list(context.leakage_excluded_columns),
        "shared_feature_columns": list(context.shared_feature_columns),
        "variant_feature_columns": list(context.variant_feature_columns),
        "local_feature_columns": list(context.local_feature_columns),
        "used_feature_columns": feature_columns,
        "used_feature_column_count": int(len(feature_columns)),
        "used_local_feature_columns": used_local_features,
        "used_local_feature_column_count": int(len(used_local_features)),
        "missing_value_strategy": {
            "numeric": "median_imputation_on_training_data_only",
            "categorical": "most_frequent_imputation_on_training_data_only",
            "missing_indicators_added": False,
            "local_feature_columns_with_expected_raw_nulls": list(context.local_feature_columns),
        },
    }


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = X.select_dtypes(include=["object", "string", "category", "bool"]).columns
    categorical_columns = categorical_columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_regression_model_specs(config: ProjectConfig) -> list[RegressionModelSpec]:
    """Return the grouped thesis regression baselines."""

    requested_model_names = set(config.training.regression_model_names)
    specs = [
        RegressionModelSpec(
            name="dummy_mean",
            display_name="Dummy Mean",
            param_grid=[{}],
        ),
        RegressionModelSpec(
            name="ridge_regression",
            display_name="Ridge Regression",
            param_grid=(
                [{"model__alpha": [0.1, 1.0, 10.0]}]
                if config.training.tune_hyperparameters
                else [{"model__alpha": [config.training.ridge_alpha]}]
            ),
        ),
        RegressionModelSpec(
            name="random_forest_regressor",
            display_name="Random Forest Regressor",
            param_grid=(
                [
                    {
                        "model__max_depth": [8, 12, None],
                        "model__min_samples_leaf": [1, 2],
                    }
                ]
                if config.training.tune_hyperparameters
                else [{"model__max_depth": [config.training.max_depth]}]
            ),
        ),
        RegressionModelSpec(
            name="xgboost_regressor",
            display_name="XGBoost Regressor",
            param_grid=(
                [
                    {
                        "model__max_depth": [4, 6],
                        "model__learning_rate": [0.05, 0.1],
                    }
                ]
                if config.training.tune_hyperparameters
                else [
                    {
                        "model__max_depth": [config.training.xgboost_max_depth],
                        "model__learning_rate": [config.training.xgboost_learning_rate],
                    }
                ]
            ),
        ),
    ]
    return [spec for spec in specs if spec.name in requested_model_names]


def _build_estimator(model_name: str, config: ProjectConfig):
    if model_name == "dummy_mean":
        return DummyRegressor(strategy="mean")

    if model_name == "ridge_regression":
        return Ridge(alpha=config.training.ridge_alpha)

    if model_name == "random_forest_regressor":
        return RandomForestRegressor(
            n_estimators=config.training.n_estimators,
            max_depth=config.training.max_depth,
            min_samples_split=config.training.min_samples_split,
            min_samples_leaf=config.training.min_samples_leaf,
            random_state=config.training.random_state,
            n_jobs=-1,
        )

    if model_name == "xgboost_regressor":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=config.training.xgboost_n_estimators,
            max_depth=config.training.xgboost_max_depth,
            learning_rate=config.training.xgboost_learning_rate,
            subsample=config.training.xgboost_subsample,
            colsample_bytree=config.training.xgboost_colsample_bytree,
            reg_lambda=config.training.xgboost_reg_lambda,
            random_state=config.training.random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
            verbosity=0,
        )

    raise ValueError(f"Unsupported release regression model: {model_name}")


def build_fixed_release_split(
    split_frames: dict[str, pd.DataFrame],
    *,
    feature_columns: list[str],
    target_column: str,
    group_column: str,
) -> dict[str, object]:
    """Build one fixed grouped split from already separated release frames."""

    required_columns = [target_column, group_column, *feature_columns]
    for frame in split_frames.values():
        validate_required_columns(frame, required_columns)

    X_train = split_frames["train"].loc[:, feature_columns].copy()
    selected_feature_columns = [
        column for column in feature_columns if X_train[column].nunique(dropna=False) > 1
    ]
    if not selected_feature_columns:
        raise ValueError("All selected feature columns are zero-variance on the training split.")

    return {
        "X_train": split_frames["train"].loc[:, selected_feature_columns].copy(),
        "y_train": pd.to_numeric(
            split_frames["train"][target_column],
            errors="raise",
        ).astype(float),
        "groups_train": split_frames["train"][group_column].astype(str).copy(),
        "X_validation": split_frames["validation"].loc[:, selected_feature_columns].copy(),
        "y_validation": pd.to_numeric(
            split_frames["validation"][target_column], errors="raise"
        ).astype(float),
        "groups_validation": split_frames["validation"][group_column].astype(str).copy(),
        "X_test": split_frames["test"].loc[:, selected_feature_columns].copy(),
        "y_test": pd.to_numeric(split_frames["test"][target_column], errors="raise").astype(float),
        "groups_test": split_frames["test"][group_column].astype(str).copy(),
        "feature_columns": selected_feature_columns,
    }


def fit_release_regression_model(
    *,
    split_bundle: dict[str, object],
    model_spec: RegressionModelSpec,
    config: ProjectConfig,
    logger: logging.Logger | None = None,
) -> FittedReleaseModelResult:
    """Tune and fit one grouped thesis regression model."""

    logger = logger or logging.getLogger(__name__)
    X_train = split_bundle["X_train"]
    y_train = split_bundle["y_train"]
    groups_train = split_bundle["groups_train"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(X_train)),
            ("model", _build_estimator(model_spec.name, config)),
        ]
    )
    unique_groups = int(pd.Series(groups_train).nunique())
    if unique_groups < 2:
        raise ValueError("Grouped CV requires at least two unique groups in the training split.")
    cv_splits = min(config.training.grouped_cv_splits, unique_groups)
    grouped_cv = build_grouped_cv(task_type="regression", n_splits=cv_splits)
    candidate_count = len(ParameterGrid(model_spec.param_grid))
    total_fit_count = candidate_count * cv_splits
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_spec.param_grid,
        scoring={
            "r2": "r2",
            "neg_mae": "neg_mean_absolute_error",
            "neg_rmse": "neg_root_mean_squared_error",
        },
        refit="r2",
        cv=grouped_cv,
        n_jobs=1,
        return_train_score=False,
        verbose=config.training.grid_search_verbose,
    )
    logger.info(
        "Starting grouped CV for %s: %s candidate setting(s), %s fold(s), %s total fit(s), "
        "%s training rows, %s unique groups.",
        model_spec.display_name,
        candidate_count,
        cv_splits,
        total_fit_count,
        len(X_train),
        unique_groups,
    )
    fit_started_at = time.perf_counter()
    if config.training.grid_search_verbose > 0:
        progress_stream = _LoggerWriter(logger)
        with (
            contextlib.redirect_stdout(progress_stream),
            contextlib.redirect_stderr(progress_stream),
        ):
            search.fit(X_train, y_train, groups=groups_train)
        progress_stream.flush()
    else:
        search.fit(X_train, y_train, groups=groups_train)
    fit_elapsed_seconds = time.perf_counter() - fit_started_at

    best_pipeline: Pipeline = search.best_estimator_
    validation_predictions = pd.Series(
        best_pipeline.predict(split_bundle["X_validation"]),
        index=split_bundle["X_validation"].index,
        name="predicted_value",
    ).astype(float)
    test_predictions = pd.Series(
        best_pipeline.predict(split_bundle["X_test"]),
        index=split_bundle["X_test"].index,
        name="predicted_value",
    ).astype(float)

    validation_metrics = _compute_safe_global_metrics(
        split_bundle["y_validation"],
        validation_predictions,
    )
    validation_metrics.update(
        {
            "train_rows": int(len(split_bundle["X_train"])),
            "validation_rows": int(len(split_bundle["X_validation"])),
            "test_rows": int(len(split_bundle["X_test"])),
            "feature_columns_before_encoding": int(len(split_bundle["feature_columns"])),
            "cv_best_r2": float(search.best_score_),
        }
    )
    test_metrics = _compute_safe_global_metrics(split_bundle["y_test"], test_predictions)
    test_metrics.update(
        {
            "train_rows": int(len(split_bundle["X_train"])),
            "validation_rows": int(len(split_bundle["X_validation"])),
            "test_rows": int(len(split_bundle["X_test"])),
            "feature_columns_before_encoding": int(len(split_bundle["feature_columns"])),
            "cv_best_r2": float(search.best_score_),
        }
    )

    cv_results_frame = pd.DataFrame(search.cv_results_).sort_values(
        by=["rank_test_r2", "mean_test_neg_mae"],
        ascending=[True, False],
    )
    best_params = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
    logger.info(
        "Finished grouped CV for %s in %.1f seconds. Best CV R2=%.4f. Best params=%s",
        model_spec.display_name,
        fit_elapsed_seconds,
        float(search.best_score_),
        best_params or {},
    )
    logger.info(
        "Scored %s: validation R2=%.4f, validation MAE=%.4f, test R2=%.4f, test MAE=%.4f",
        model_spec.display_name,
        validation_metrics["r2"],
        validation_metrics["mae"],
        test_metrics["r2"],
        test_metrics["mae"],
    )
    return FittedReleaseModelResult(
        model_name=model_spec.name,
        model_display_name=model_spec.display_name,
        pipeline=best_pipeline,
        best_params=best_params,
        cv_results_frame=cv_results_frame,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
    )


def select_best_model_rows(summary_frame: pd.DataFrame) -> pd.DataFrame:
    """Choose the best validation model per ablation."""

    return (
        summary_frame.sort_values(
            by=["ablation_name", "validation_r2", "validation_mae"],
            ascending=[True, False, True],
        )
        .groupby("ablation_name", as_index=False)
        .first()
    )


def resolve_repo_root() -> Path:
    """Return the repository root inferred from this module location."""

    return Path(__file__).resolve().parents[2]


def build_model_importance_frame(pipeline: Pipeline, model_name: str) -> pd.DataFrame | None:
    """Extract a readable feature-importance frame when the model exposes one."""

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name in {"random_forest_regressor", "xgboost_regressor"} and hasattr(
        model, "feature_importances_"
    ):
        importance_values = np.asarray(model.feature_importances_, dtype=float)
    elif model_name == "ridge_regression" and hasattr(model, "coef_"):
        importance_values = np.abs(np.asarray(model.coef_, dtype=float))
    else:
        return None

    frame = pd.DataFrame({"feature": feature_names, "importance": importance_values})
    return frame.sort_values(by="importance", ascending=False).reset_index(drop=True)


def _compute_safe_global_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics while avoiding undefined small-sample R2 warnings."""

    if len(y_true) < 2:
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
            "r2": 0.0,
        }

    return compute_regression_metrics(y_true, y_pred)
