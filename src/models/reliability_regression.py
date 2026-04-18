"""Leakage-free helpers for pre-run reliability regression."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.schema import ProjectConfig
from src.data.dataset import validate_required_columns
from src.features.gate_sequence import compute_bit_errors

RELIABILITY_TARGET_COLUMN = "reliability"

RELIABILITY_ALLOWED_RAW_FEATURE_COLUMNS = [
    "qubit_count",
    "gate_depth",
    "error_rate_gate",
    "t1_time",
    "t2_time",
    "readout_error",
    "device_type",
]

RELIABILITY_ALLOWED_ENGINEERED_FEATURE_COLUMNS = [
    "num_cx",
    "two_qubit_ratio",
    "unique_gates",
    "cx_density",
    "t2_t1_ratio",
]

RELIABILITY_ALLOWED_FEATURE_COLUMNS = [
    *RELIABILITY_ALLOWED_RAW_FEATURE_COLUMNS,
    *RELIABILITY_ALLOWED_ENGINEERED_FEATURE_COLUMNS,
]

RELIABILITY_FORBIDDEN_FEATURE_REASONS = {
    "bitstring": "observed circuit output and therefore unavailable before execution",
    "ideal_bitstring": "ideal output string used only to define the target",
    "bitstring_aligned": "aligned observed output used only to define the target",
    "ideal_bitstring_aligned": "aligned ideal output used only to define the target",
    "fidelity": "post-observation quality metric and therefore leakage",
    "bit_errors": "target-construction intermediate derived from observed outputs",
    "observed_error_rate": "derived from observed outputs and therefore leakage",
    "fidelity_loss": "derived from observed fidelity and therefore leakage",
    "bit_error_density": "derived from observed bit errors and therefore leakage",
    RELIABILITY_TARGET_COLUMN: "regression target and therefore unavailable as input",
}


def resolve_reliability_target_source_columns(
    cleaned_frame: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[str, str]:
    """Return the observed and ideal bitstring columns used to build the target."""

    observed_column = (
        "bitstring_aligned"
        if "bitstring_aligned" in cleaned_frame.columns
        else config.data.bitstring_column
    )
    ideal_column = (
        "ideal_bitstring_aligned"
        if "ideal_bitstring_aligned" in cleaned_frame.columns
        else config.data.ideal_bitstring_column
    )
    validate_required_columns(
        cleaned_frame,
        [config.data.id_column, config.data.qubit_count_column, observed_column, ideal_column],
    )
    return observed_column, ideal_column


def build_reliability_target_frame(
    cleaned_frame: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Build a regression target from post-run outcomes without exposing them as inputs."""

    observed_column, ideal_column = resolve_reliability_target_source_columns(cleaned_frame, config)
    target_rows: list[dict[str, float | int | str]] = []

    for _, row in cleaned_frame.iterrows():
        circuit_id = row[config.data.id_column]
        qubit_count = int(row[config.data.qubit_count_column])
        if qubit_count <= 0:
            raise ValueError(
                f"Cannot build reliability target with non-positive qubit_count for {circuit_id}."
            )

        bit_errors = compute_bit_errors(row[observed_column], row[ideal_column])
        if bit_errors is None:
            raise ValueError(
                "Cannot build reliability target because bit error computation failed for "
                f"{circuit_id}."
            )

        reliability = 1.0 - (float(bit_errors) / float(qubit_count))
        if reliability < 0.0 or reliability > 1.0:
            raise ValueError(
                "Reliability target fell outside [0, 1] for "
                f"{circuit_id}: reliability={reliability}."
            )

        target_rows.append(
            {
                config.data.id_column: str(circuit_id),
                "bit_errors": int(bit_errors),
                RELIABILITY_TARGET_COLUMN: float(reliability),
            }
        )

    return pd.DataFrame.from_records(target_rows)


def build_reliability_target_summary(
    target_frame: pd.DataFrame,
    *,
    observed_column: str,
    ideal_column: str,
) -> dict[str, Any]:
    """Summarize the constructed reliability target for reproducible experiment artifacts."""

    reliability = target_frame[RELIABILITY_TARGET_COLUMN].astype(float)
    bit_errors = target_frame["bit_errors"].astype(int)
    return {
        "target_name": RELIABILITY_TARGET_COLUMN,
        "target_definition": "1 - (bit_errors / qubit_count)",
        "target_bounds": [0.0, 1.0],
        "bit_error_source_columns": {
            "observed": observed_column,
            "ideal": ideal_column,
        },
        "row_count": int(len(target_frame)),
        "bit_errors_min": int(bit_errors.min()),
        "bit_errors_max": int(bit_errors.max()),
        "reliability_min": float(reliability.min()),
        "reliability_max": float(reliability.max()),
        "reliability_mean": float(reliability.mean()),
        "reliability_median": float(reliability.median()),
    }


def build_reliability_features(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a leakage-free design matrix and reliability target series."""

    validate_required_columns(feature_frame, [config.data.id_column])
    validate_required_columns(target_frame, [config.data.id_column, RELIABILITY_TARGET_COLUMN])

    if target_frame[config.data.id_column].duplicated().any():
        raise ValueError("Reliability target frame contains duplicate circuit identifiers.")

    target_lookup = target_frame.set_index(config.data.id_column)[RELIABILITY_TARGET_COLUMN]
    target = feature_frame[config.data.id_column].astype(str).map(target_lookup)
    if target.isna().any():
        missing_count = int(target.isna().sum())
        raise ValueError(
            "Reliability target alignment failed because some feature rows could not be matched "
            f"to the computed target. Missing matches: {missing_count}."
        )

    allowed_columns_present = [
        column for column in RELIABILITY_ALLOWED_FEATURE_COLUMNS if column in feature_frame.columns
    ]
    if not allowed_columns_present:
        raise ValueError(
            "The selected feature table does not contain any allowed leakage-free reliability "
            "features."
        )

    X = feature_frame.loc[:, allowed_columns_present].copy()
    X = X.loc[:, X.nunique(dropna=False) > 1].copy()
    if X.empty:
        raise ValueError("All allowed reliability features were removed as zero-variance columns.")

    return X, target.astype(float)


def build_reliability_feature_policy(
    feature_frame: pd.DataFrame,
    X: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Describe the leakage-free feature policy for reliability regression runs."""

    allowed_columns_present = [
        column for column in RELIABILITY_ALLOWED_FEATURE_COLUMNS if column in feature_frame.columns
    ]
    dropped_zero_variance_columns = [
        column for column in allowed_columns_present if column not in X.columns
    ]
    missing_allowed_columns = [
        column
        for column in RELIABILITY_ALLOWED_FEATURE_COLUMNS
        if column not in feature_frame.columns
    ]
    forbidden_columns_present = [
        column
        for column in feature_frame.columns
        if column in RELIABILITY_FORBIDDEN_FEATURE_REASONS
    ]
    excluded_non_allowed_columns = [
        column
        for column in feature_frame.columns
        if column not in RELIABILITY_ALLOWED_FEATURE_COLUMNS
        and column not in RELIABILITY_FORBIDDEN_FEATURE_REASONS
        and column != config.data.id_column
        and column != config.data.label_column
    ]

    return {
        "prediction_context": "pre_execution",
        "target_name": RELIABILITY_TARGET_COLUMN,
        "target_definition": "1 - (bit_errors / qubit_count)",
        "selected_feature_set": config.training.feature_set_name,
        "allowed_raw_feature_columns": list(RELIABILITY_ALLOWED_RAW_FEATURE_COLUMNS),
        "allowed_engineered_feature_columns": list(RELIABILITY_ALLOWED_ENGINEERED_FEATURE_COLUMNS),
        "forbidden_feature_columns": list(RELIABILITY_FORBIDDEN_FEATURE_REASONS),
        "forbidden_column_reasons": {
            column: RELIABILITY_FORBIDDEN_FEATURE_REASONS[column]
            for column in forbidden_columns_present
        },
        "allowed_columns_present_in_feature_table": allowed_columns_present,
        "missing_allowed_columns": missing_allowed_columns,
        "dropped_zero_variance_columns": dropped_zero_variance_columns,
        "excluded_non_allowed_columns_present_in_feature_table": excluded_non_allowed_columns,
        "used_feature_columns": X.columns.tolist(),
        "used_feature_column_count": int(X.shape[1]),
    }
