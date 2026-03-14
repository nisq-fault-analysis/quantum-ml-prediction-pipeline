"""Helpers for reproducible filtered experiment subsets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype


def _coerce_subset_value(series: pd.Series, raw_value: str) -> Any:
    """Convert a CLI-style string value into the column's natural scalar type."""

    if is_bool_dtype(series):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"Could not interpret boolean subset value: {raw_value!r}")

    if is_integer_dtype(series):
        return int(raw_value)

    if is_float_dtype(series):
        return float(raw_value)

    return raw_value


def filter_frame_by_subset(
    frame: pd.DataFrame,
    *,
    subset_column: str | None,
    subset_value: str | None,
    label_column: str | None = None,
    subset_type: str = "user_filter",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Optionally filter a frame to a single subgroup and describe the result."""

    if subset_column is None and subset_value is None:
        return frame, None
    if subset_column is None or subset_value is None:
        raise ValueError("Both subset_column and subset_value must be provided together.")
    if subset_column not in frame.columns:
        raise ValueError(f"Subset column {subset_column!r} is not present in the feature table.")

    typed_subset_value = _coerce_subset_value(frame[subset_column], subset_value)
    filtered_frame = frame.loc[frame[subset_column] == typed_subset_value].copy()
    if filtered_frame.empty:
        raise ValueError(
            f"Subset filter {subset_column} == {typed_subset_value!r} produced no rows."
        )

    subset_metadata: dict[str, Any] = {
        "subset_type": subset_type,
        "filter_column": subset_column,
        "filter_value": typed_subset_value,
        "row_count": int(len(filtered_frame)),
    }
    if label_column and label_column in filtered_frame.columns:
        subset_metadata["label_distribution"] = {
            label: int(count)
            for label, count in Counter(filtered_frame[label_column].astype(str).tolist()).items()
        }
    return filtered_frame, subset_metadata


def filter_frame_from_saved_metadata(
    frame: pd.DataFrame, run_path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply a previously saved subset filter from a run directory."""

    subset_metadata_path = Path(run_path) / "subset_metadata.json"
    if not subset_metadata_path.exists():
        return frame, None

    subset_metadata = json.loads(subset_metadata_path.read_text(encoding="utf-8"))
    filter_column = str(subset_metadata["filter_column"])
    filter_value = subset_metadata["filter_value"]
    filtered_frame = frame.loc[frame[filter_column] == filter_value].copy()
    if filtered_frame.empty:
        raise ValueError(
            f"Subset filter {filter_column} == {filter_value!r} produced no rows for {run_path}"
        )
    return filtered_frame, subset_metadata
