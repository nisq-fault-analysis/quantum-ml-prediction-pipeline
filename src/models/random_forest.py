"""Training helpers for the first Random Forest baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config.schema import ProjectConfig
from src.evaluation.metrics import build_classification_report_text, compute_classification_metrics
from src.models.splitting import split_dataset


@dataclass(slots=True)
class RandomForestResult:
    """Artifacts kept in memory after training one Random Forest experiment."""

    pipeline: Pipeline
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_report_text: str
    test_report_text: str
    feature_importance_frame: pd.DataFrame
    y_validation: pd.Series
    y_pred_validation: pd.Series
    y_test: pd.Series
    y_pred_test: pd.Series
    labels: list[str]


def _class_weight_or_none(class_weight: str) -> str | None:
    return None if class_weight == "none" else class_weight


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Encode only the categorical columns and pass numeric columns through."""

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


def _build_feature_importance_frame(pipeline: Pipeline) -> pd.DataFrame:
    """Map tree importances back to readable feature names."""

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model: RandomForestClassifier = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return importance_frame.sort_values(by="importance", ascending=False).reset_index(drop=True)


def train_random_forest(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    config: ProjectConfig,
) -> RandomForestResult:
    """Train the baseline Random Forest model and return reusable artifacts."""

    split = split_dataset(feature_frame, labels, config)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(split.X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config.training.n_estimators,
                    max_depth=config.training.max_depth,
                    min_samples_split=config.training.min_samples_split,
                    min_samples_leaf=config.training.min_samples_leaf,
                    class_weight=_class_weight_or_none(config.training.class_weight),
                    random_state=config.training.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(split.X_train, split.y_train)

    y_pred_validation = pd.Series(
        pipeline.predict(split.X_validation),
        index=split.X_validation.index,
        name="predicted_label",
    ).astype(str)
    y_score_validation = (
        pipeline.predict_proba(split.X_validation) if hasattr(pipeline, "predict_proba") else None
    )
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
        }
    )

    y_pred_test = pd.Series(
        pipeline.predict(split.X_test),
        index=split.X_test.index,
        name="predicted_label",
    ).astype(str)
    y_score_test = (
        pipeline.predict_proba(split.X_test) if hasattr(pipeline, "predict_proba") else None
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

    return RandomForestResult(
        pipeline=pipeline,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_report_text=validation_report_text,
        test_report_text=test_report_text,
        feature_importance_frame=_build_feature_importance_frame(pipeline),
        y_validation=split.y_validation.astype(str),
        y_pred_validation=y_pred_validation.astype(str),
        y_test=split.y_test.astype(str),
        y_pred_test=y_pred_test.astype(str),
        labels=split.labels,
    )


def save_model_artifact(pipeline: Pipeline, destination: str | Path) -> None:
    """Persist the fitted model with joblib."""

    dump(pipeline, destination, compress=3)
