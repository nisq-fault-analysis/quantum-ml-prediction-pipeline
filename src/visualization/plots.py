"""Reusable plotting utilities for reports and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix


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

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


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

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot a confusion matrix that can be dropped into a thesis chapter."""

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

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def plot_shap_summary(model: object, features: pd.DataFrame, output_path: str | Path) -> None:
    """Generate a SHAP summary plot for a fitted tree-based model."""

    sample = features if len(features) <= 500 else features.sample(500, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    current_figure = plt.gcf()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    current_figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close("all")
