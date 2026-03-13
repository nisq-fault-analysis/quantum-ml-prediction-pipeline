"""Evaluation helpers used by the baseline experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, float]:
    """Compute a compact set of metrics suitable for experiment comparison tables."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def build_classification_report_frame(
    y_true: Sequence[int], y_pred: Sequence[int], target_names: list[str]
) -> pd.DataFrame:
    """Return the per-class report as a DataFrame so it can be saved to CSV."""

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    frame = pd.DataFrame(report).transpose().reset_index()
    frame = frame.rename(columns={"index": "label"})
    return frame


def save_json_report(payload: Mapping[str, Any], destination: str | Path) -> None:
    """Write experiment metadata or metrics to JSON."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
