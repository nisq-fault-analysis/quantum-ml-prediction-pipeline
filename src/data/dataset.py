"""Dataset loading helpers for the NISQ fault classification project."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config.schema import DataConfig


def infer_file_format(dataset_path: Path, configured_format: str) -> str:
    """Infer the tabular file format unless the config already fixes it explicitly."""

    if configured_format != "auto":
        return configured_format

    if dataset_path.suffix.lower() == ".parquet":
        return "parquet"

    return "csv"


def read_tabular_dataset(config: DataConfig) -> pd.DataFrame:
    """Read the raw dataset from disk.

    The function intentionally supports only CSV and Parquet to keep the baseline
    workflow simple and well-documented.
    """

    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset file not found. Update 'data.dataset_path' in the experiment config "
            f"to point at your Kaggle export. Missing path: {dataset_path}"
        )

    file_format = infer_file_format(dataset_path=dataset_path, configured_format=config.file_format)

    if file_format == "csv":
        return pd.read_csv(dataset_path)

    if file_format == "parquet":
        return pd.read_parquet(dataset_path)

    raise ValueError(f"Unsupported file format: {file_format}")


def validate_required_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Fail early when the config does not match the actual dataset schema."""

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns. "
            f"Expected columns not found: {missing_columns}. "
            "Update the config once you know the Kaggle schema."
        )


def prepare_research_table(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Return a modelling table with configured columns removed.

    This is a light-weight preprocessing step. It is intentionally conservative:
    the raw file stays untouched, while scripts operate on a clean in-memory copy.
    """

    table = frame.copy()

    if config.drop_columns:
        table = table.drop(columns=config.drop_columns, errors="ignore")

    return table
