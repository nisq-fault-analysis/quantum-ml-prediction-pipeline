from __future__ import annotations

import pandas as pd

from src.config.schema import FeatureConfig
from src.features.gate_sequence import (
    compute_bit_errors,
    compute_two_qubit_ratio,
    engineer_enhanced_classification_features,
    engineer_gate_sequence_features,
    parse_gate_types,
)


def test_parse_gate_types_handles_multiple_delimiters() -> None:
    tokens = parse_gate_types("H->CX;RX", delimiters=[",", ";", "->"])

    assert tokens == ["h", "cx", "rx"]


def test_compute_bit_errors_returns_hamming_distance_for_equal_length_strings() -> None:
    bit_errors = compute_bit_errors("0011", "0110")

    assert bit_errors == 2


def test_compute_bit_errors_returns_none_for_unequal_length_strings() -> None:
    bit_errors = compute_bit_errors("001", "0011")

    assert bit_errors is None


def test_compute_two_qubit_ratio_avoids_division_by_zero() -> None:
    ratio = compute_two_qubit_ratio([], two_qubit_gates=["cx", "cz"])

    assert ratio == 0.0


def test_engineer_gate_sequence_features_builds_required_columns() -> None:
    frame = pd.DataFrame(
        {
            "gate_types": ["H,CX,RX,CX"],
            "qubit_count": [4],
            "gate_depth": [8],
            "t1_time": [50.0],
            "t2_time": [25.0],
            "bitstring_aligned": ["0101"],
            "ideal_bitstring_aligned": ["0001"],
        }
    )
    feature_config = FeatureConfig(two_qubit_gates=["cx", "cz"])

    feature_table = engineer_gate_sequence_features(
        frame=frame,
        sequence_column="gate_types",
        feature_config=feature_config,
        qubit_count_column="qubit_count",
    )

    assert feature_table.loc[0, "num_cx"] == 2.0
    assert feature_table.loc[0, "two_qubit_ratio"] == 0.5
    assert feature_table.loc[0, "unique_gates"] == 3.0
    assert feature_table.loc[0, "cx_density"] == 0.25
    assert feature_table.loc[0, "t2_t1_ratio"] == 0.5
    assert feature_table.loc[0, "bit_errors"] == 1.0
    assert feature_table.loc[0, "observed_error_rate"] == 0.25


def test_engineer_enhanced_classification_features_builds_normalized_columns() -> None:
    frame = pd.DataFrame(
        {
            "qubit_count": [4],
            "gate_depth": [8],
            "error_rate_gate": [0.02],
            "readout_error": [0.01],
            "t1_time": [50.0],
            "t2_time": [25.0],
            "fidelity": [0.6],
        }
    )
    topology_frame = pd.DataFrame({"bit_errors": [2.0]})

    feature_table = engineer_enhanced_classification_features(
        frame=frame,
        topology_feature_frame=topology_frame,
        qubit_count_column="qubit_count",
    )

    assert feature_table.loc[0, "depth_per_qubit"] == 2.0
    assert feature_table.loc[0, "error_depth_load"] == 0.16
    assert feature_table.loc[0, "readout_to_gate_error_ratio"] == 0.5
    assert feature_table.loc[0, "coherence_gap"] == 25.0
    assert feature_table.loc[0, "coherence_min"] == 25.0
    assert feature_table.loc[0, "fidelity_loss"] == 0.4
    assert feature_table.loc[0, "bit_error_density"] == 0.25
