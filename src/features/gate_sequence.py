"""Feature engineering from NISQ gate-type strings and measurement outcomes."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.config.schema import FeatureConfig

GATE_HEAD_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a safe ratio and avoid noisy division-by-zero failures."""

    if denominator == 0 or math.isnan(denominator):
        return 0.0
    return float(numerator / denominator)


def normalize_gate_token(token: str) -> str:
    """Normalize gate names to a consistent lowercase representation."""

    cleaned = token.strip().lower().strip("\"'`")
    if not cleaned:
        return ""

    if ":" in cleaned:
        colon_parts = [part.strip() for part in cleaned.split(":") if part.strip()]
        if colon_parts:
            cleaned = colon_parts[-1]

    head = re.split(r"[\s(\[{<]", cleaned, maxsplit=1)[0].strip(")]}>,")
    if head:
        match = GATE_HEAD_PATTERN.match(head)
        if match is not None:
            return match.group(0)

    match = GATE_HEAD_PATTERN.search(cleaned)
    return match.group(0) if match is not None else cleaned


def split_gate_sequence(gate_types: str, delimiters: Sequence[str]) -> list[str]:
    """Split a gate string while keeping delimiters inside brackets untouched."""

    ordered_delimiters = sorted(
        {delimiter for delimiter in delimiters if delimiter},
        key=len,
        reverse=True,
    )
    if not ordered_delimiters:
        return [gate_types]

    segments: list[str] = []
    current_segment: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    index = 0

    while index < len(gate_types):
        if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            matched_delimiter = next(
                (
                    delimiter
                    for delimiter in ordered_delimiters
                    if gate_types.startswith(delimiter, index)
                ),
                None,
            )
            if matched_delimiter is not None:
                segment = "".join(current_segment).strip()
                if segment:
                    segments.append(segment)
                current_segment = []
                index += len(matched_delimiter)
                continue

        character = gate_types[index]
        if character == "(":
            paren_depth += 1
        elif character == ")" and paren_depth > 0:
            paren_depth -= 1
        elif character == "[":
            bracket_depth += 1
        elif character == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif character == "{":
            brace_depth += 1
        elif character == "}" and brace_depth > 0:
            brace_depth -= 1

        current_segment.append(character)
        index += 1

    final_segment = "".join(current_segment).strip()
    if final_segment:
        segments.append(final_segment)

    return segments


def parse_gate_types(gate_types: object, delimiters: Sequence[str]) -> list[str]:
    """Split a gate string such as `H,CX,RX,CX` into normalized tokens."""

    if gate_types is None or pd.isna(gate_types):
        return []

    text = str(gate_types).strip()
    if not text:
        return []

    tokens = [normalize_gate_token(token) for token in split_gate_sequence(text, delimiters)]
    return [token for token in tokens if token]


def count_specific_gate(tokens: Sequence[str], gate_name: str) -> int:
    """Count how often a specific gate appears in the token sequence."""

    normalized_gate_name = normalize_gate_token(gate_name)
    return sum(1 for token in tokens if token == normalized_gate_name)


def count_unique_gates(tokens: Sequence[str]) -> int:
    """Count the number of unique gate types used in the circuit."""

    return len(set(tokens))


def compute_two_qubit_ratio(tokens: Sequence[str], two_qubit_gates: Sequence[str]) -> float:
    """Estimate the fraction of gates that are two-qubit operations."""

    normalized_two_qubit_gates = {normalize_gate_token(gate) for gate in two_qubit_gates}
    two_qubit_count = sum(1 for token in tokens if token in normalized_two_qubit_gates)
    return safe_divide(two_qubit_count, len(tokens))


def compute_bit_errors(observed_bitstring: object, ideal_bitstring: object) -> int | None:
    """Compute the Hamming distance between aligned observed and ideal bitstrings."""

    if observed_bitstring is None or ideal_bitstring is None:
        return None
    if pd.isna(observed_bitstring) or pd.isna(ideal_bitstring):
        return None

    observed_text = str(observed_bitstring)
    ideal_text = str(ideal_bitstring)
    if len(observed_text) != len(ideal_text):
        return None

    return sum(
        left_bit != right_bit
        for left_bit, right_bit in zip(observed_text, ideal_text, strict=False)
    )


def engineer_gate_sequence_features(
    frame: pd.DataFrame,
    sequence_column: str,
    feature_config: FeatureConfig,
    qubit_count_column: str | None = None,
    bitstring_column: str = "bitstring_aligned",
    ideal_bitstring_column: str = "ideal_bitstring_aligned",
) -> pd.DataFrame:
    """Build the thesis-aligned engineered features used by the RF baseline.

    Some engineered gate features may be constant for the current Kaggle file because
    `gate_types` appears fixed in the public sample. We still compute them because:
    1. the feature logic belongs in the pipeline
    2. richer or future datasets may vary
    3. documenting zero-variance features is itself useful thesis evidence
    """

    observed_column = (
        bitstring_column
        if bitstring_column in frame.columns
        else "bitstring" if "bitstring" in frame.columns else bitstring_column
    )
    ideal_column = (
        ideal_bitstring_column
        if ideal_bitstring_column in frame.columns
        else "ideal_bitstring" if "ideal_bitstring" in frame.columns else ideal_bitstring_column
    )

    feature_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        tokens = parse_gate_types(row.get(sequence_column), feature_config.gate_delimiters)
        num_cx = count_specific_gate(tokens, "cx")
        unique_gates = count_unique_gates(tokens)
        qubit_count_value = row.get(qubit_count_column) if qubit_count_column else None
        qubit_count = (
            float(qubit_count_value)
            if qubit_count_value is not None and not pd.isna(qubit_count_value)
            else 0.0
        )
        gate_depth_value = row.get("gate_depth")
        gate_depth = float(gate_depth_value) if pd.notna(gate_depth_value) else 0.0
        t1_time_value = row.get("t1_time")
        t1_time = float(t1_time_value) if pd.notna(t1_time_value) else 0.0
        t2_time_value = row.get("t2_time")
        t2_time = float(t2_time_value) if pd.notna(t2_time_value) else 0.0

        bit_errors = compute_bit_errors(
            row.get(observed_column),
            row.get(ideal_column),
        )
        observed_error_rate = (
            safe_divide(float(bit_errors), qubit_count) if bit_errors is not None else 0.0
        )

        feature_rows.append(
            {
                "num_cx": float(num_cx),
                "two_qubit_ratio": compute_two_qubit_ratio(tokens, feature_config.two_qubit_gates),
                "unique_gates": float(unique_gates),
                "cx_density": safe_divide(float(num_cx), gate_depth),
                "t2_t1_ratio": safe_divide(t2_time, t1_time),
                "bit_errors": float(bit_errors) if bit_errors is not None else 0.0,
                "observed_error_rate": observed_error_rate,
            }
        )

    return pd.DataFrame.from_records(feature_rows, index=frame.index)


def engineer_enhanced_classification_features(
    frame: pd.DataFrame,
    topology_feature_frame: pd.DataFrame,
    qubit_count_column: str = "qubit_count",
) -> pd.DataFrame:
    """Build a small set of normalized features for harder classification runs.

    These features are intentionally simple. They try to express relationships
    that may matter more than raw magnitudes alone:
    - circuit depth relative to qubit count
    - interaction between gate depth and gate error rate
    - relative scale of readout noise vs gate noise
    - coherence margins from T1/T2 values
    - observed-output mismatch normalized by depth
    """

    enhanced_feature_rows: list[dict[str, float]] = []
    for row_index, row in frame.iterrows():
        qubit_count_value = row.get(qubit_count_column)
        qubit_count = (
            float(qubit_count_value)
            if qubit_count_value is not None and not pd.isna(qubit_count_value)
            else 0.0
        )
        gate_depth_value = row.get("gate_depth")
        gate_depth = float(gate_depth_value) if pd.notna(gate_depth_value) else 0.0
        error_rate_gate_value = row.get("error_rate_gate")
        error_rate_gate = float(error_rate_gate_value) if pd.notna(error_rate_gate_value) else 0.0
        readout_error_value = row.get("readout_error")
        readout_error = float(readout_error_value) if pd.notna(readout_error_value) else 0.0
        t1_time_value = row.get("t1_time")
        t1_time = float(t1_time_value) if pd.notna(t1_time_value) else 0.0
        t2_time_value = row.get("t2_time")
        t2_time = float(t2_time_value) if pd.notna(t2_time_value) else 0.0
        fidelity_value = row.get("fidelity")
        fidelity = float(fidelity_value) if pd.notna(fidelity_value) else 0.0
        bit_errors = float(topology_feature_frame.loc[row_index, "bit_errors"])

        enhanced_feature_rows.append(
            {
                "depth_per_qubit": safe_divide(gate_depth, qubit_count),
                "error_depth_load": float(error_rate_gate * gate_depth),
                "readout_to_gate_error_ratio": safe_divide(readout_error, error_rate_gate),
                "coherence_gap": float(t1_time - t2_time),
                "coherence_min": float(min(t1_time, t2_time)),
                "fidelity_loss": float(max(0.0, 1.0 - fidelity)),
                "bit_error_density": safe_divide(bit_errors, gate_depth),
            }
        )

    return pd.DataFrame.from_records(enhanced_feature_rows, index=frame.index)
