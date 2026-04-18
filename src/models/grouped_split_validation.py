"""Validation helpers for grouped train/validation/test split integrity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold

from src.data.dataset import validate_required_columns

TaskType = Literal["classification", "regression"]


@dataclass(slots=True)
class GroupedSplitAudit:
    """Structured audit of grouped split integrity."""

    group_column: str
    row_id_columns: list[str]
    split_row_counts: dict[str, int]
    split_group_counts: dict[str, int]
    overlapping_groups: dict[str, list[str]]
    duplicate_row_ids_within_split: dict[str, int]

    @property
    def passed(self) -> bool:
        return not any(self.overlapping_groups.values()) and not any(
            self.duplicate_row_ids_within_split.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "group_column": self.group_column,
            "row_id_columns": self.row_id_columns,
            "split_row_counts": self.split_row_counts,
            "split_group_counts": self.split_group_counts,
            "overlapping_groups": self.overlapping_groups,
            "duplicate_row_ids_within_split": self.duplicate_row_ids_within_split,
        }


def build_grouped_cv(
    *,
    task_type: TaskType,
    n_splits: int,
) -> GroupKFold | StratifiedGroupKFold:
    """Return a grouped cross-validator suitable for the requested task."""

    if task_type == "classification":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    return GroupKFold(n_splits=n_splits)


def grouped_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    group_column: str,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Create grouped train/validation/test frames from one dataset."""

    validate_required_columns(frame, [group_column])
    groups = frame[group_column].astype(str)

    first_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_validation_indices, test_indices = next(first_splitter.split(frame, groups=groups))
    train_validation_frame = frame.iloc[train_validation_indices].copy()
    test_frame = frame.iloc[test_indices].copy()

    validation_fraction_within_train_validation = validation_size / (1.0 - test_size)
    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction_within_train_validation,
        random_state=random_state,
    )
    train_indices, validation_indices = next(
        second_splitter.split(
            train_validation_frame,
            groups=train_validation_frame[group_column].astype(str),
        )
    )
    train_frame = train_validation_frame.iloc[train_indices].copy()
    validation_frame = train_validation_frame.iloc[validation_indices].copy()

    return {
        "train": train_frame,
        "validation": validation_frame,
        "test": test_frame,
    }


def build_split_membership_frame(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    group_column: str,
    row_id_columns: list[str],
) -> pd.DataFrame:
    """Flatten split membership into one artifact-friendly table."""

    membership_frames: list[pd.DataFrame] = []
    required_columns = [group_column, *row_id_columns]
    for split_name, frame in split_frames.items():
        validate_required_columns(frame, required_columns)
        membership = frame.loc[:, required_columns].copy()
        membership.insert(0, "split_name", split_name)
        membership_frames.append(membership)

    return pd.concat(membership_frames, ignore_index=True)


def audit_grouped_split_frames(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    group_column: str,
    row_id_columns: list[str],
) -> GroupedSplitAudit:
    """Validate that grouped splits are disjoint and row identifiers are unique per split."""

    split_row_counts: dict[str, int] = {}
    split_group_counts: dict[str, int] = {}
    groups_by_split: dict[str, set[str]] = {}
    duplicate_row_ids_within_split: dict[str, int] = {}

    required_columns = [group_column, *row_id_columns]
    for split_name, frame in split_frames.items():
        validate_required_columns(frame, required_columns)
        split_row_counts[split_name] = int(len(frame))
        groups = frame[group_column].astype(str)
        groups_by_split[split_name] = set(groups.tolist())
        split_group_counts[split_name] = int(groups.nunique())

        duplicate_mask = frame.loc[:, row_id_columns].duplicated()
        duplicate_row_ids_within_split[split_name] = int(duplicate_mask.sum())

    overlapping_groups: dict[str, list[str]] = {}
    split_names = list(split_frames)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = sorted(groups_by_split[left_name] & groups_by_split[right_name])
            overlapping_groups[f"{left_name}__{right_name}"] = overlap

    return GroupedSplitAudit(
        group_column=group_column,
        row_id_columns=row_id_columns,
        split_row_counts=split_row_counts,
        split_group_counts=split_group_counts,
        overlapping_groups=overlapping_groups,
        duplicate_row_ids_within_split=duplicate_row_ids_within_split,
    )


def assert_grouped_split_integrity(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    group_column: str,
    row_id_columns: list[str],
) -> GroupedSplitAudit:
    """Raise when grouped split integrity checks fail."""

    audit = audit_grouped_split_frames(
        split_frames,
        group_column=group_column,
        row_id_columns=row_id_columns,
    )
    if not audit.passed:
        raise ValueError(
            "Grouped split integrity validation failed. "
            f"Overlapping groups: {audit.overlapping_groups}. "
            f"Duplicate row ids: {audit.duplicate_row_ids_within_split}."
        )
    return audit
