"""Dataset loading and schema validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.config.schema import DataConfig


def infer_file_format(dataset_path: Path, configured_format: str) -> str:
    """Infer the tabular file format unless the config already fixes it explicitly."""

    if configured_format != "auto":
        return configured_format

    if dataset_path.suffix.lower() == ".parquet":
        return "parquet"

    return "csv"


def read_tabular_file(
    dataset_path: str | Path,
    file_format: str = "auto",
    *,
    string_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV or Parquet file while preserving selected string columns."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    resolved_format = infer_file_format(path, configured_format=file_format)
    if resolved_format == "csv":
        dtype_mapping = {column: "string" for column in (string_columns or [])}
        return pd.read_csv(path, dtype=dtype_mapping or None)

    if resolved_format == "parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {resolved_format}")


def read_tabular_dataset(config: DataConfig) -> pd.DataFrame:
    """Read the raw Kaggle dataset from disk."""

    string_columns = [
        config.id_column,
        config.gate_sequence_column,
        config.bitstring_column,
        config.ideal_bitstring_column,
        config.timestamp_column,
        config.label_column,
        *config.categorical_columns,
    ]
    return read_tabular_file(
        dataset_path=config.dataset_path,
        file_format=config.file_format,
        string_columns=string_columns,
    )


def validate_required_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Fail early when the configured schema does not match the actual dataset."""

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns. "
            f"Expected columns not found: {missing_columns}."
        )


def prepare_research_table(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Return a copy of the frame with configured columns removed."""

    table = frame.copy()
    if config.drop_columns:
        table = table.drop(columns=config.drop_columns, errors="ignore")

    return table
