from __future__ import annotations

import pandas as pd
import pytest

from src.config.schema import ProjectConfig
from src.features.gate_sequence import compute_bit_errors
from src.models.reliability_regression import (
    RELIABILITY_TARGET_COLUMN,
    build_reliability_feature_policy,
    build_reliability_features,
    build_reliability_target_frame,
)


def test_compute_bit_errors_supports_reliability_target_definition() -> None:
    assert compute_bit_errors("10110", "10011") == 2


def test_build_reliability_target_frame_computes_bounded_target() -> None:
    cleaned_frame = pd.DataFrame(
        {
            "circuit_id": ["circ_1", "circ_2"],
            "qubit_count": [4, 5],
            "bitstring_aligned": ["1010", "11100"],
            "ideal_bitstring_aligned": ["1000", "11000"],
        }
    )

    target_frame = build_reliability_target_frame(cleaned_frame, ProjectConfig())

    assert target_frame["bit_errors"].tolist() == [1, 1]
    assert target_frame[RELIABILITY_TARGET_COLUMN].tolist() == [0.75, 0.8]
    assert target_frame[RELIABILITY_TARGET_COLUMN].between(0.0, 1.0).all()


def test_build_reliability_target_frame_rejects_out_of_bounds_target() -> None:
    cleaned_frame = pd.DataFrame(
        {
            "circuit_id": ["circ_1"],
            "qubit_count": [2],
            "bitstring_aligned": ["111"],
            "ideal_bitstring_aligned": ["000"],
        }
    )

    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        build_reliability_target_frame(cleaned_frame, ProjectConfig())


def test_build_reliability_features_excludes_leaky_columns() -> None:
    feature_frame = pd.DataFrame(
        {
            "circuit_id": ["circ_1", "circ_2"],
            "error_type": ["readout", "depolarizing"],
            "qubit_count": [4, 5],
            "gate_depth": [12, 16],
            "error_rate_gate": [0.01, 0.04],
            "t1_time": [80.0, 70.0],
            "t2_time": [60.0, 50.0],
            "readout_error": [0.02, 0.03],
            "device_type": ["superconducting", "trapped_ion"],
            "num_cx": [3.0, 5.0],
            "two_qubit_ratio": [0.25, 0.31],
            "unique_gates": [4.0, 5.0],
            "cx_density": [0.25, 0.3125],
            "t2_t1_ratio": [0.75, 0.714285714],
            "shots": [1024, 1024],
            "fidelity": [0.95, 0.84],
            "bit_errors": [1.0, 2.0],
            "observed_error_rate": [0.25, 0.4],
            "fidelity_loss": [0.05, 0.16],
            "bit_error_density": [0.08, 0.125],
            "depth_per_qubit": [3.0, 3.2],
        }
    )
    target_frame = pd.DataFrame(
        {
            "circuit_id": ["circ_1", "circ_2"],
            RELIABILITY_TARGET_COLUMN: [0.75, 0.6],
        }
    )

    X, y = build_reliability_features(feature_frame, target_frame, ProjectConfig())
    feature_policy = build_reliability_feature_policy(feature_frame, X, ProjectConfig())

    assert y.tolist() == [0.75, 0.6]
    assert set(X.columns) == {
        "qubit_count",
        "gate_depth",
        "error_rate_gate",
        "t1_time",
        "t2_time",
        "readout_error",
        "device_type",
        "num_cx",
        "two_qubit_ratio",
        "unique_gates",
        "cx_density",
        "t2_t1_ratio",
    }
    assert "fidelity" not in X.columns
    assert "bit_errors" not in X.columns
    assert "observed_error_rate" not in X.columns
    assert "shots" in feature_policy["excluded_non_allowed_columns_present_in_feature_table"]
    assert "fidelity" in feature_policy["forbidden_column_reasons"]
