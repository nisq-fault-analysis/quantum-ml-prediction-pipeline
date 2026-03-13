"""Dataset splitting helpers shared by baseline training scripts.

The main idea is to keep splitting logic in one place so every model family
uses the same methodology:
- reserve the test set first
- split the remaining data into train and validation
- stratify by label when the class counts make that feasible
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.schema import ProjectConfig


@dataclass(slots=True)
class TrainValidationTestSplit:
    """Store one reproducible train/validation/test partition."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    labels: list[str]


def choose_stratify_target(labels: pd.Series, split_size: float) -> pd.Series | None:
    """Use label stratification only when each class can appear in both sides."""

    label_counts = labels.value_counts()
    if label_counts.empty or label_counts.min() < 2:
        return None

    minimum_split_rows = math.ceil(len(labels) * split_size)
    if minimum_split_rows < label_counts.size:
        return None

    return labels


def split_dataset(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    config: ProjectConfig,
) -> TrainValidationTestSplit:
    """Create an explicit train/validation/test split for thesis experiments."""

    test_stratify_target = (
        choose_stratify_target(labels, config.training.test_size)
        if config.training.stratify_by_label
        else None
    )
    X_temp, X_test, y_temp, y_test = train_test_split(
        feature_frame,
        labels,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=test_stratify_target,
    )

    validation_fraction_within_temp = config.training.validation_size / (
        1.0 - config.training.test_size
    )
    validation_stratify_target = (
        choose_stratify_target(y_temp, validation_fraction_within_temp)
        if config.training.stratify_by_label
        else None
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_fraction_within_temp,
        random_state=config.training.random_state,
        stratify=validation_stratify_target,
    )

    labels_sorted = sorted(labels.astype(str).unique().tolist())
    return TrainValidationTestSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train.astype(str),
        y_validation=y_validation.astype(str),
        y_test=y_test.astype(str),
        labels=labels_sorted,
    )


def build_split_summary(split: TrainValidationTestSplit) -> dict[str, object]:
    """Return a compact JSON-friendly summary of split sizes and class balance."""

    def label_counts(series: pd.Series) -> dict[str, int]:
        return {
            str(label): int(count) for label, count in series.value_counts().sort_index().items()
        }

    total_rows = len(split.y_train) + len(split.y_validation) + len(split.y_test)
    return {
        "total_rows": int(total_rows),
        "train_rows": int(len(split.y_train)),
        "validation_rows": int(len(split.y_validation)),
        "test_rows": int(len(split.y_test)),
        "train_fraction": round(len(split.y_train) / total_rows, 6),
        "validation_fraction": round(len(split.y_validation) / total_rows, 6),
        "test_fraction": round(len(split.y_test) / total_rows, 6),
        "train_label_counts": label_counts(split.y_train),
        "validation_label_counts": label_counts(split.y_validation),
        "test_label_counts": label_counts(split.y_test),
    }
