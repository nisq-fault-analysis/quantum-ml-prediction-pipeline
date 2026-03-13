"""Reusable plotting utilities for baseline experiments and EDA."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def _save_figure(figure: plt.Figure, output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_categorical_distribution(
    values: pd.Series, output_path: str | Path, title: str, xlabel: str
) -> None:
    """Plot a simple bar chart for class or qubit-count distributions."""

    counts = values.astype(str).value_counts().sort_values(ascending=False)

    figure, axis = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=axis, color="#1f77b4")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()

    _save_figure(figure, output_path)


def plot_numeric_histogram(
    values: pd.Series, output_path: str | Path, title: str, xlabel: str
) -> None:
    """Plot a histogram for a numeric exploratory feature."""

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(values.dropna(), bins=30, color="#4c956c", edgecolor="black")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Frequency")
    figure.tight_layout()

    _save_figure(figure, output_path)


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot a confusion matrix that can be cited in the thesis."""

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()

    _save_figure(figure, output_path)


def plot_feature_importance(
    importance_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Random Forest feature importance",
    top_n: int = 15,
) -> None:
    """Plot the top feature importances from the fitted Random Forest model."""

    top_features = importance_frame.head(top_n).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(top_features["feature"], top_features["importance"], color="#dd8452")
    axis.set_title(title)
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()

    _save_figure(figure, output_path)


def plot_actual_vs_predicted(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot regression predictions against the ground truth."""

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(y_true, y_pred, alpha=0.35, color="#2a6f97", edgecolors="none")
    lower_bound = min(min(y_true), min(y_pred))
    upper_bound = max(max(y_true), max(y_pred))
    axis.plot([lower_bound, upper_bound], [lower_bound, upper_bound], color="#c1121f")
    axis.set_title(title)
    axis.set_xlabel("Actual value")
    axis.set_ylabel("Predicted value")
    figure.tight_layout()

    _save_figure(figure, output_path)
