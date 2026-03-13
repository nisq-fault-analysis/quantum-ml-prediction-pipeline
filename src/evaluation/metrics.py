"""Evaluation helpers used by the Random Forest baseline experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    y_score: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str] | None = None,
    compute_roc_auc: bool = False,
) -> dict[str, float]:
    """Compute the main metrics used in the thesis baseline tables."""

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if compute_roc_auc and y_score is not None:
        score_frame = pd.DataFrame(y_score)
        distinct_labels = list(labels) if labels is not None else sorted({*y_true, *y_pred})

        if len(distinct_labels) == 2 and score_frame.shape[1] >= 2:
            positive_label = distinct_labels[-1]
            y_true_binary = pd.Series(y_true).eq(positive_label).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(y_true_binary, score_frame.iloc[:, 1]))
        elif len(distinct_labels) > 2 and score_frame.shape[1] == len(distinct_labels):
            label_binarizer = LabelBinarizer()
            y_true_binarized = label_binarizer.fit_transform(y_true)
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true_binarized,
                    score_frame.to_numpy(),
                    multi_class="ovr",
                    average="macro",
                )
            )

    return metrics


def build_classification_report_text(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> str:
    """Return the text report that is easy to inspect in experiment folders."""

    return classification_report(
        y_true,
        y_pred,
        labels=list(labels) if labels is not None else None,
        zero_division=0,
    )


def compute_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Compute the main regression metrics for fidelity prediction."""

    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json_report(payload: Mapping[str, Any], destination: str | Path) -> None:
    """Write experiment metadata or metrics to JSON."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
