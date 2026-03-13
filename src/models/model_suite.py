"""Shared helpers for comparing several baseline classifiers on one split.

The main teaching idea in this module is fairness:
- every model sees the same engineered features
- every model uses the same train/test split
- every run saves artifacts in the same shape

That makes the comparison suitable for a thesis chapter instead of a
"many scripts, many slightly different results" workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.config.schema import ModelName, ProjectConfig
from src.evaluation.metrics import build_classification_report_text, compute_classification_metrics
from src.models.splitting import TrainValidationTestSplit, split_dataset

MODEL_DISPLAY_NAMES: dict[ModelName, str] = {
    "dummy_most_frequent": "Dummy Most Frequent",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


@dataclass(slots=True)
class ModelSuiteResult:
    """In-memory artifacts produced by one fitted model."""

    model_name: ModelName
    pipeline: Pipeline
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_report_text: str
    test_report_text: str
    feature_importance_frame: pd.DataFrame | None
    y_validation: pd.Series
    y_pred_validation: pd.Series
    y_test: pd.Series
    y_pred_test: pd.Series
    labels: list[str]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Encode categorical columns only; keep numeric columns unchanged."""

    categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _class_weight_or_none(class_weight: str) -> str | None:
    return None if class_weight == "none" else class_weight


def get_display_name(model_name: ModelName) -> str:
    """Map the stable internal key to a readable title for plots and docs."""

    return MODEL_DISPLAY_NAMES[model_name]


def _build_estimator(model_name: ModelName, config: ProjectConfig, class_count: int):
    """Construct the requested estimator using lightweight, reproducible defaults."""

    if model_name == "dummy_most_frequent":
        return DummyClassifier(
            strategy=config.training.dummy_strategy,
            random_state=config.training.random_state,
        )

    if model_name == "logistic_regression":
        return LogisticRegression(
            C=config.training.logistic_c,
            max_iter=config.training.logistic_max_iter,
            class_weight=_class_weight_or_none(config.training.logistic_class_weight),
            random_state=config.training.random_state,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.training.n_estimators,
            max_depth=config.training.max_depth,
            min_samples_split=config.training.min_samples_split,
            min_samples_leaf=config.training.min_samples_leaf,
            class_weight=_class_weight_or_none(config.training.class_weight),
            random_state=config.training.random_state,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        objective = "binary:logistic" if class_count == 2 else "multi:softprob"
        extra_args = {} if class_count == 2 else {"num_class": class_count}
        return XGBClassifier(
            objective=objective,
            n_estimators=config.training.xgboost_n_estimators,
            max_depth=config.training.xgboost_max_depth,
            learning_rate=config.training.xgboost_learning_rate,
            subsample=config.training.xgboost_subsample,
            colsample_bytree=config.training.xgboost_colsample_bytree,
            reg_lambda=config.training.xgboost_reg_lambda,
            random_state=config.training.random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
            **extra_args,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def _fit_pipeline_for_model(
    model_name: ModelName,
    split: TrainValidationTestSplit,
    config: ProjectConfig,
) -> tuple[
    Pipeline,
    pd.Series,
    list[list[float]] | None,
    pd.Series,
    list[list[float]] | None,
]:
    """Fit one model and return predictions in the original label space."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(split.X_train)),
            ("model", _build_estimator(model_name, config, len(split.labels))),
        ]
    )

    if model_name == "xgboost":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(split.y_train)
        pipeline.fit(split.X_train, y_train_encoded)

        raw_validation_predictions = np.asarray(pipeline.predict(split.X_validation)).astype(int)
        decoded_validation_predictions = label_encoder.inverse_transform(raw_validation_predictions)
        y_pred_validation = pd.Series(
            decoded_validation_predictions,
            index=split.X_validation.index,
            name="predicted_label",
        )
        y_score_validation = (
            pipeline.predict_proba(split.X_validation).tolist()
            if hasattr(pipeline, "predict_proba")
            else None
        )
        raw_test_predictions = np.asarray(pipeline.predict(split.X_test)).astype(int)
        decoded_test_predictions = label_encoder.inverse_transform(raw_test_predictions)
        y_pred_test = pd.Series(
            decoded_test_predictions,
            index=split.X_test.index,
            name="predicted_label",
        )
        y_score_test = (
            pipeline.predict_proba(split.X_test).tolist()
            if hasattr(pipeline, "predict_proba")
            else None
        )
        return (
            pipeline,
            y_pred_validation.astype(str),
            y_score_validation,
            y_pred_test.astype(str),
            y_score_test,
        )

    pipeline.fit(split.X_train, split.y_train)
    y_pred_validation = pd.Series(
        pipeline.predict(split.X_validation),
        index=split.X_validation.index,
        name="predicted_label",
    ).astype(str)
    y_score_validation = (
        pipeline.predict_proba(split.X_validation).tolist()
        if hasattr(pipeline, "predict_proba")
        else None
    )
    y_pred_test = pd.Series(
        pipeline.predict(split.X_test),
        index=split.X_test.index,
        name="predicted_label",
    ).astype(str)
    y_score_test = (
        pipeline.predict_proba(split.X_test).tolist()
        if hasattr(pipeline, "predict_proba")
        else None
    )
    return pipeline, y_pred_validation, y_score_validation, y_pred_test, y_score_test


def _build_feature_importance_frame(
    pipeline: Pipeline,
    model_name: ModelName,
) -> pd.DataFrame | None:
    """Extract a simple importance view when the estimator supports one.

    For logistic regression we use absolute coefficient magnitude. The values
    are not the same thing as tree importances, but they still help answer
    which inputs the linear model relied on most strongly.
    """

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name in {"random_forest", "xgboost"} and hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif model_name == "logistic_regression" and hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coefficients).mean(axis=0)
    else:
        return None

    importance_frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    return importance_frame.sort_values(by="importance", ascending=False).reset_index(drop=True)


def train_model_suite(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    config: ProjectConfig,
) -> tuple[TrainValidationTestSplit, list[ModelSuiteResult]]:
    """Train the requested baselines on one shared split and collect artifacts."""

    split = split_dataset(feature_frame, labels, config)
    results: list[ModelSuiteResult] = []

    for model_name in config.training.model_names:
        started_at = perf_counter()
        (
            pipeline,
            y_pred_validation,
            y_score_validation,
            y_pred_test,
            y_score_test,
        ) = _fit_pipeline_for_model(model_name, split, config)
        training_seconds = perf_counter() - started_at

        validation_metrics = compute_classification_metrics(
            y_true=split.y_validation.astype(str),
            y_pred=y_pred_validation.astype(str),
            y_score=y_score_validation,
            labels=split.labels,
            compute_roc_auc=config.training.compute_roc_auc,
        )
        validation_metrics.update(
            {
                "train_rows": int(len(split.X_train)),
                "validation_rows": int(len(split.X_validation)),
                "test_rows": int(len(split.X_test)),
                "feature_columns_before_encoding": int(feature_frame.shape[1]),
                "training_seconds": round(training_seconds, 4),
            }
        )
        test_metrics = compute_classification_metrics(
            y_true=split.y_test.astype(str),
            y_pred=y_pred_test.astype(str),
            y_score=y_score_test,
            labels=split.labels,
            compute_roc_auc=config.training.compute_roc_auc,
        )
        test_metrics.update(
            {
                "train_rows": int(len(split.X_train)),
                "validation_rows": int(len(split.X_validation)),
                "test_rows": int(len(split.X_test)),
                "feature_columns_before_encoding": int(feature_frame.shape[1]),
                "training_seconds": round(training_seconds, 4),
            }
        )

        validation_report_text = build_classification_report_text(
            y_true=split.y_validation.astype(str),
            y_pred=y_pred_validation.astype(str),
            labels=split.labels,
        )
        test_report_text = build_classification_report_text(
            y_true=split.y_test.astype(str),
            y_pred=y_pred_test.astype(str),
            labels=split.labels,
        )

        results.append(
            ModelSuiteResult(
                model_name=model_name,
                pipeline=pipeline,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                validation_report_text=validation_report_text,
                test_report_text=test_report_text,
                feature_importance_frame=_build_feature_importance_frame(pipeline, model_name),
                y_validation=split.y_validation.astype(str),
                y_pred_validation=y_pred_validation.astype(str),
                y_test=split.y_test.astype(str),
                y_pred_test=y_pred_test.astype(str),
                labels=split.labels,
            )
        )

    return split, results


def build_model_comparison_frame(results: list[ModelSuiteResult]) -> pd.DataFrame:
    """Create a compact table that is easy to paste into thesis notes."""

    rows = []
    for result in results:
        rows.append(
            {
                "model_name": result.model_name,
                "model_display_name": get_display_name(result.model_name),
                **{f"validation_{key}": value for key, value in result.validation_metrics.items()},
                **{f"test_{key}": value for key, value in result.test_metrics.items()},
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["validation_macro_f1", "validation_accuracy"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
