"""Feature engineering from gate-sequence strings.

The baseline features are intentionally transparent: counts, diversity, and gate-frequency
signals are easy to inspect, easy to explain in a thesis, and useful as a starting point
before moving to richer sequence models.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Sequence

import pandas as pd

from src.config.schema import FeatureConfig


def _delimiter_pattern(delimiters: Sequence[str]) -> re.Pattern[str]:
    escaped_delimiters = [re.escape(delimiter) for delimiter in delimiters]
    return re.compile("|".join(escaped_delimiters))


def _sanitize_feature_name(token: str) -> str:
    return re.sub(r"\W+", "_", token).strip("_") or "unknown_gate"


def normalize_gate_token(token: str, lowercase: bool = True) -> str:
    """Normalize tokens so the same gate is counted consistently."""

    cleaned = token.strip()
    if lowercase:
        cleaned = cleaned.lower()

    return cleaned


def tokenize_gate_sequence(
    sequence: object, delimiters: Sequence[str], lowercase: bool = True
) -> list[str]:
    """Split a gate-sequence field into normalized tokens."""

    if pd.isna(sequence):
        return []

    text = str(sequence).strip()
    if not text:
        return []

    normalized_text = _delimiter_pattern(delimiters).sub(" ", text)

    tokens: list[str] = []
    for token in normalized_text.split():
        cleaned = normalize_gate_token(token, lowercase=lowercase)
        if cleaned:
            tokens.append(cleaned)

    return tokens


def engineer_gate_sequence_features(
    frame: pd.DataFrame,
    sequence_column: str,
    feature_config: FeatureConfig,
    qubit_count_column: str | None = None,
) -> pd.DataFrame:
    """Convert gate-sequence strings into a numeric feature table.

    TODO: After inspecting the real dataset, extend this module with domain-informed
    features such as depth, entangling-gate ratios, or temporal fault-position features.
    """

    normalized_top_gates = list(
        dict.fromkeys(
            normalize_gate_token(gate, lowercase=feature_config.lowercase_tokens)
            for gate in feature_config.top_gates
        )
    )

    rows: list[dict[str, float]] = []
    for sequence in frame[sequence_column]:
        tokens = tokenize_gate_sequence(
            sequence=sequence,
            delimiters=feature_config.gate_delimiters,
            lowercase=feature_config.lowercase_tokens,
        )
        token_counts = Counter(tokens)

        row: dict[str, float] = {
            "gate_token_count": float(len(tokens)),
            "gate_unique_count": float(len(token_counts)),
            "gate_diversity_ratio": float(len(token_counts) / len(tokens)) if tokens else 0.0,
            "contains_measure": float("measure" in token_counts),
        }

        for gate in normalized_top_gates:
            feature_name = f"gate_count__{_sanitize_feature_name(gate)}"
            row[feature_name] = float(token_counts.get(gate, 0))

        rows.append(row)

    features = pd.DataFrame.from_records(rows, index=frame.index)

    if qubit_count_column and qubit_count_column in frame.columns:
        # Keeping qubit count as a feature helps the global baseline, while stratified
        # runs let us study whether models behave differently at each circuit size.
        features["qubit_count"] = (
            pd.to_numeric(frame[qubit_count_column], errors="coerce").fillna(-1)
        )

    return features
