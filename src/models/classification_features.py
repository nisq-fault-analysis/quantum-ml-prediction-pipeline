"""Helpers for leakage-free classification feature selection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.schema import ProjectConfig

CLASSIFICATION_EXCLUSION_REASONS = {
    "fidelity": (
        "post-observation outcome quality and therefore unavailable for "
        "pre-execution prediction"
    ),
    "fidelity_loss": "deterministic transform of observed fidelity and therefore leakage",
    "bit_errors": "computed from observed versus ideal outputs and therefore leakage",
    "observed_error_rate": "derived from measured bit errors and therefore leakage",
    "bit_error_density": "derived from measured bit errors and therefore leakage",
    "timestamp": (
        "collection-order metadata rather than a scientific pre-execution "
        "circuit property"
    ),
}


def build_classification_features(
    feature_frame: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the configured classification inputs and labels.

    The default thesis setting is pre-execution classification, so the function
    removes outcome-derived columns before fitting any classifier.
    """

    labels = feature_frame[config.data.label_column].astype(str)
    candidate_drop_columns = {
        config.data.id_column,
        config.data.label_column,
        *config.training.excluded_feature_columns,
    }
    X = feature_frame.drop(columns=list(candidate_drop_columns), errors="ignore")
    X = X.loc[:, X.nunique(dropna=False) > 1].copy()
    return X, labels


def build_classification_feature_policy(
    feature_frame: pd.DataFrame,
    config: ProjectConfig,
    X: pd.DataFrame,
) -> dict[str, Any]:
    """Describe the active feature policy for reproducible run artifacts."""

    excluded_columns_present = [
        column
        for column in config.training.excluded_feature_columns
        if column in feature_frame.columns
    ]
    exclusion_reasons = {
        column: CLASSIFICATION_EXCLUSION_REASONS.get(column, "user-configured exclusion")
        for column in excluded_columns_present
    }
    return {
        "prediction_context": config.training.prediction_context,
        "excluded_feature_columns": list(config.training.excluded_feature_columns),
        "excluded_columns_present_in_feature_table": excluded_columns_present,
        "exclusion_reasons": exclusion_reasons,
        "used_feature_columns": X.columns.tolist(),
        "used_feature_column_count": int(X.shape[1]),
    }
