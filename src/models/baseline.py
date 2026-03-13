"""Baseline classical ML models for NISQ fault classification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config.schema import ProjectConfig, TrainingConfig
from src.evaluation.metrics import build_classification_report_frame, compute_classification_metrics


@dataclass(slots=True)
class ModelRun:
    """Container for everything we usually want to save from a single model run."""

    model_name: str
    estimator: Any
    metrics: dict[str, float]
    report_frame: pd.DataFrame
    predictions: pd.DataFrame
    X_test: pd.DataFrame
    label_names: list[str]
    y_true_labels: pd.Series
    y_pred_labels: pd.Series


def _choose_stratify_target(labels: pd.Series, test_size: float) -> pd.Series | None:
    """Use label stratification only when the split is statistically feasible."""

    label_counts = labels.value_counts()
    if label_counts.empty or label_counts.min() < 2:
        return None

    minimum_test_rows = math.ceil(len(labels) * test_size)
    if minimum_test_rows < label_counts.size:
        return None

    return labels


def _class_weight_or_none(class_weight: str) -> str | None:
    return "balanced" if class_weight == "balanced" else None


def build_candidate_models(training_config: TrainingConfig, num_classes: int) -> dict[str, Any]:
    """Create the baseline model family used for the first comparative experiments."""

    models: dict[str, Any] = {}
    class_weight = _class_weight_or_none(training_config.class_weight)

    if "logreg" in training_config.enable_models:
        models["logreg"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight=class_weight,
                        multi_class="auto",
                    ),
                ),
            ]
        )

    if "random_forest" in training_config.enable_models:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=training_config.n_estimators,
            max_depth=training_config.max_depth,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1,
        )

    if "xgboost" in training_config.enable_models:
        xgboost_kwargs: dict[str, Any] = {
            "n_estimators": training_config.n_estimators,
            "max_depth": training_config.max_depth,
            "learning_rate": training_config.learning_rate,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        }

        if num_classes > 2:
            xgboost_kwargs["objective"] = "multi:softprob"
            xgboost_kwargs["eval_metric"] = "mlogloss"
            xgboost_kwargs["num_class"] = num_classes
        else:
            xgboost_kwargs["objective"] = "binary:logistic"
            xgboost_kwargs["eval_metric"] = "logloss"

        models["xgboost"] = XGBClassifier(**xgboost_kwargs)

    return models


def train_and_evaluate_models(
    feature_table: pd.DataFrame, labels: pd.Series, config: ProjectConfig
) -> list[ModelRun]:
    """Train the configured baseline models on one dataset slice.

    A single slice can be the global dataset or one qubit-count stratum.
    """

    label_encoder = LabelEncoder()
    encoded_labels = pd.Series(
        label_encoder.fit_transform(labels.astype(str)),
        index=labels.index,
        name="encoded_label",
    )
    stratify_target = (
        _choose_stratify_target(encoded_labels, config.data.test_size)
        if config.data.stratify_by_label
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        feature_table,
        encoded_labels,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=stratify_target,
    )

    candidate_models = build_candidate_models(
        config.training,
        num_classes=len(label_encoder.classes_),
    )

    model_runs: list[ModelRun] = []
    label_names = label_encoder.classes_.tolist()

    for model_name, estimator in candidate_models.items():
        estimator.fit(X_train, y_train)
        raw_predictions = estimator.predict(X_test)
        y_pred = np.asarray(raw_predictions).astype(int)

        metrics = compute_classification_metrics(y_test, y_pred)
        report_frame = build_classification_report_frame(y_test, y_pred, target_names=label_names)

        y_true_labels = pd.Series(
            label_encoder.inverse_transform(y_test.to_numpy()),
            index=X_test.index,
            name="true_label",
        )
        y_pred_labels = pd.Series(
            label_encoder.inverse_transform(y_pred),
            index=X_test.index,
            name="predicted_label",
        )

        predictions = pd.DataFrame(
            {
                "row_index": X_test.index,
                "true_label": y_true_labels,
                "predicted_label": y_pred_labels,
            }
        )

        if hasattr(estimator, "predict_proba"):
            prediction_probabilities = estimator.predict_proba(X_test)
            predictions["prediction_confidence"] = prediction_probabilities.max(axis=1)

        model_runs.append(
            ModelRun(
                model_name=model_name,
                estimator=estimator,
                metrics=metrics,
                report_frame=report_frame,
                predictions=predictions,
                X_test=X_test,
                label_names=label_names,
                y_true_labels=y_true_labels,
                y_pred_labels=y_pred_labels,
            )
        )

    return model_runs
