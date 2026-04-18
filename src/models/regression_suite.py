"""Regression helpers for predicting fidelity from circuit and noise features."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config.schema import ProjectConfig
from src.data.dataset import validate_required_columns
from src.evaluation.metrics import compute_regression_metrics

REGRESSION_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "dummy_mean": "Dummy Mean",
    "random_forest_regressor": "Random Forest Regressor",
    "xgboost_regressor": "XGBoost Regressor",
}


OUTCOME_DERIVED_COLUMNS = {
    "fidelity_loss",
    "bit_errors",
    "observed_error_rate",
    "bit_error_density",
    "exact_match_probability",
    "mitigated_reliability",
    "mitigation_gain",
}


@dataclass(slots=True)
class RegressionSplit:
    """Store one reproducible train/validation/test partition for regression."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(slots=True)
class RegressionResult:
    """Artifacts produced by one fitted regression model."""

    model_name: str
    pipeline: Pipeline
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importance_frame: pd.DataFrame | None
    y_validation: pd.Series
    y_pred_validation: pd.Series
    y_test: pd.Series
    y_pred_test: pd.Series


def get_regression_display_name(model_name: str) -> str:
    """Return a readable label for plots and reports."""

    return REGRESSION_MODEL_DISPLAY_NAMES[model_name]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
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
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_regression_features(
    feature_frame: pd.DataFrame,
    config: ProjectConfig,
    *,
    target_column: str | None = None,
    candidate_feature_columns: list[str] | None = None,
    excluded_feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare an input matrix for fidelity regression without target leakage."""

    target_name = target_column or config.training.target_column or "fidelity"
    excluded_columns = {
        config.data.id_column,
        config.data.label_column,
        target_name,
        *OUTCOME_DERIVED_COLUMNS,
        *config.training.excluded_feature_columns,
        *(excluded_feature_columns or []),
    }
    validate_required_columns(feature_frame, [target_name])
    target = pd.to_numeric(feature_frame[target_name], errors="coerce")
    if target.isna().any():
        missing_target_count = int(target.isna().sum())
        raise ValueError(
            f"Regression target column {target_name!r} contains {missing_target_count} missing "
            "or non-numeric values."
        )

    if candidate_feature_columns is not None:
        validate_required_columns(feature_frame, candidate_feature_columns)
        X = feature_frame.loc[:, candidate_feature_columns].copy()
        X = X.drop(columns=list(excluded_columns), errors="ignore")
    else:
        X = feature_frame.drop(columns=list(excluded_columns), errors="ignore")

    candidate_drop_columns = {
        column for column in X.columns if X[column].nunique(dropna=False) <= 1
    }
    X = X.drop(columns=list(candidate_drop_columns), errors="ignore")
    return X, target.astype(float)


def build_regression_split_from_precomputed_frames(
    *,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> RegressionSplit:
    """Build a fixed regression split from packaged train/validation/test frames."""

    validate_required_columns(train_frame, [target_column, *feature_columns])
    validate_required_columns(validation_frame, [target_column, *feature_columns])
    validate_required_columns(test_frame, [target_column, *feature_columns])

    X_train = train_frame.loc[:, feature_columns].copy()
    X_validation = validation_frame.loc[:, feature_columns].copy()
    X_test = test_frame.loc[:, feature_columns].copy()

    selected_columns = [
        column for column in feature_columns if X_train[column].nunique(dropna=False) > 1
    ]
    if not selected_columns:
        raise ValueError("No usable feature columns remain after dropping zero-variance columns.")

    y_train = pd.to_numeric(train_frame[target_column], errors="coerce")
    y_validation = pd.to_numeric(validation_frame[target_column], errors="coerce")
    y_test = pd.to_numeric(test_frame[target_column], errors="coerce")
    for split_name, target in [
        ("train", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ]:
        if target.isna().any():
            missing_target_count = int(target.isna().sum())
            raise ValueError(
                f"Regression target column {target_column!r} contains {missing_target_count} "
                f"missing or non-numeric values in the {split_name} split."
            )

    return RegressionSplit(
        X_train=X_train.loc[:, selected_columns].copy(),
        X_validation=X_validation.loc[:, selected_columns].copy(),
        X_test=X_test.loc[:, selected_columns].copy(),
        y_train=y_train.astype(float),
        y_validation=y_validation.astype(float),
        y_test=y_test.astype(float),
    )


def split_regression_dataset(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    config: ProjectConfig,
) -> RegressionSplit:
    """Create a simple train/validation/test split for regression."""

    X_temp, X_test, y_temp, y_test = train_test_split(
        feature_frame,
        target,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
    )
    validation_fraction_within_temp = config.training.validation_size / (
        1.0 - config.training.test_size
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_fraction_within_temp,
        random_state=config.training.random_state,
    )

    return RegressionSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train.astype(float),
        y_validation=y_validation.astype(float),
        y_test=y_test.astype(float),
    )


def build_regression_split_summary(split: RegressionSplit) -> dict[str, float | int]:
    """Return a small summary of split sizes for regression runs."""

    total_rows = len(split.y_train) + len(split.y_validation) + len(split.y_test)
    return {
        "total_rows": int(total_rows),
        "train_rows": int(len(split.y_train)),
        "validation_rows": int(len(split.y_validation)),
        "test_rows": int(len(split.y_test)),
        "train_fraction": round(len(split.y_train) / total_rows, 6),
        "validation_fraction": round(len(split.y_validation) / total_rows, 6),
        "test_fraction": round(len(split.y_test) / total_rows, 6),
    }


def _build_estimator(model_name: str, config: ProjectConfig):
    if model_name == "dummy_mean":
        return DummyRegressor(strategy="mean")

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
            verbosity=0,
        )

    raise ValueError(f"Unsupported regression model: {model_name}")


def _build_feature_importance_frame(pipeline: Pipeline, model_name: str) -> pd.DataFrame | None:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name in {"random_forest_regressor", "xgboost_regressor"} and hasattr(
        model, "feature_importances_"
    ):
        importance_frame = pd.DataFrame(
            {"feature": feature_names, "importance": np.asarray(model.feature_importances_)}
        )
        return importance_frame.sort_values(by="importance", ascending=False).reset_index(drop=True)

    return None


def train_regression_suite(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    config: ProjectConfig,
) -> tuple[RegressionSplit, list[RegressionResult]]:
    """Train the fidelity regression suite and collect validation/test metrics."""

    split = split_regression_dataset(feature_frame, target, config)
    return train_regression_suite_on_split(split, config)


def train_regression_suite_on_split(
    split: RegressionSplit,
    config: ProjectConfig,
) -> tuple[RegressionSplit, list[RegressionResult]]:
    """Train the regression suite on a caller-supplied split."""

    results: list[RegressionResult] = []
    feature_column_count = int(split.X_train.shape[1])

    for model_name in ["dummy_mean", "random_forest_regressor", "xgboost_regressor"]:
        started_at = perf_counter()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(split.X_train)),
                ("model", _build_estimator(model_name, config)),
            ]
        )
        pipeline.fit(split.X_train, split.y_train)
        training_seconds = round(perf_counter() - started_at, 4)

        y_pred_validation = pd.Series(
            pipeline.predict(split.X_validation),
            index=split.X_validation.index,
            name="predicted_value",
        ).astype(float)
        validation_metrics = compute_regression_metrics(split.y_validation, y_pred_validation)
        validation_metrics.update(
            {
                "train_rows": int(len(split.X_train)),
                "validation_rows": int(len(split.X_validation)),
                "test_rows": int(len(split.X_test)),
                "feature_columns_before_encoding": feature_column_count,
                "training_seconds": training_seconds,
            }
        )

        y_pred_test = pd.Series(
            pipeline.predict(split.X_test),
            index=split.X_test.index,
            name="predicted_value",
        ).astype(float)
        test_metrics = compute_regression_metrics(split.y_test, y_pred_test)
        test_metrics.update(
            {
                "train_rows": int(len(split.X_train)),
                "validation_rows": int(len(split.X_validation)),
                "test_rows": int(len(split.X_test)),
                "feature_columns_before_encoding": feature_column_count,
                "training_seconds": training_seconds,
            }
        )

        results.append(
            RegressionResult(
                model_name=model_name,
                pipeline=pipeline,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                feature_importance_frame=_build_feature_importance_frame(pipeline, model_name),
                y_validation=split.y_validation,
                y_pred_validation=y_pred_validation,
                y_test=split.y_test,
                y_pred_test=y_pred_test,
            )
        )

    return split, results


def build_regression_comparison_frame(results: list[RegressionResult]) -> pd.DataFrame:
    """Create a compact comparison table for fidelity regression."""

    rows = []
    for result in results:
        rows.append(
            {
                "model_name": result.model_name,
                "model_display_name": get_regression_display_name(result.model_name),
                **{f"validation_{key}": value for key, value in result.validation_metrics.items()},
                **{f"test_{key}": value for key, value in result.test_metrics.items()},
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by=["validation_r2", "validation_mae"], ascending=[False, True])
        .reset_index(drop=True)
    )
