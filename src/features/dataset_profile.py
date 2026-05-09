"""Dataset-profile helpers for future feature-table and experiment reruns."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from src.config.schema import ProjectConfig
from src.features.gate_sequence import parse_gate_types


def _series_distribution(values: pd.Series) -> dict[str, int]:
    counts = Counter(values.astype(str).tolist())
    return {
        key: int(value)
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }


def _top_sequences(normalized_sequences: pd.Series, limit: int = 10) -> list[dict[str, Any]]:
    counts = normalized_sequences.value_counts(dropna=False).head(limit)
    return [{"sequence": str(sequence), "count": int(count)} for sequence, count in counts.items()]


def _gate_count_summary(token_lists: pd.Series) -> dict[str, float | int]:
    gate_counts = token_lists.apply(len)
    if gate_counts.empty:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(gate_counts.min()),
        "max": int(gate_counts.max()),
        "mean": round(float(gate_counts.mean()), 4),
    }


def build_dataset_profile(
    cleaned_frame: pd.DataFrame,
    config: ProjectConfig,
    feature_report: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Describe dataset diversity and zero-variance warnings for future reruns."""

    sequence_column = config.data.gate_sequence_column
    token_lists = cleaned_frame[sequence_column].apply(
        lambda value: parse_gate_types(value, config.features.gate_delimiters)
    )
    normalized_sequences = token_lists.apply(lambda tokens: ",".join(tokens))
    gate_type_counts = Counter(
        token for tokens in token_lists.tolist() for token in tokens if token
    )
    qubit_counts = cleaned_frame[config.data.qubit_count_column].dropna()

    qubit_profile: dict[str, Any] = {
        "distribution": _series_distribution(cleaned_frame[config.data.qubit_count_column]),
    }
    if not qubit_counts.empty:
        qubit_profile.update(
            {
                "min": int(qubit_counts.min()),
                "max": int(qubit_counts.max()),
                "unique_values": sorted(int(value) for value in qubit_counts.unique().tolist()),
            }
        )

    return {
        "dataset_path": config.data.dataset_path.as_posix(),
        "cleaned_dataset_path": config.data.cleaned_dataset_path.as_posix(),
        "row_count": int(len(cleaned_frame)),
        "label_distribution": _series_distribution(cleaned_frame[config.data.label_column]),
        "qubit_count_profile": qubit_profile,
        "gate_sequence_profile": {
            "sequence_column": sequence_column,
            "non_empty_sequence_rows": int(token_lists.apply(bool).sum()),
            "unique_normalized_sequence_count": int(normalized_sequences.nunique(dropna=False)),
            "constant_normalized_sequence": bool(normalized_sequences.nunique(dropna=False) <= 1),
            "top_normalized_sequences": _top_sequences(normalized_sequences),
            "gate_count_per_circuit": _gate_count_summary(token_lists),
            "unique_gate_type_count": int(len(gate_type_counts)),
            "gate_type_frequencies": {
                key: int(value)
                for key, value in sorted(
                    gate_type_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            },
        },
        "feature_set_warnings": {
            feature_set_name: {
                "feature_column_count": int(report["feature_column_count"]),
                "zero_variance_columns": list(report["zero_variance_columns"]),
            }
            for feature_set_name, report in feature_report.items()
        },
    }
