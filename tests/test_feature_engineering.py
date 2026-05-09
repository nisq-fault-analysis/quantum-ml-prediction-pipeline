from __future__ import annotations

import pandas as pd

from src.config.schema import FeatureConfig
from src.features.build_features import build_feature_report
from src.features.dataset_profile import build_dataset_profile
from src.features.gate_sequence import (
    compute_bit_errors,
    compute_two_qubit_ratio,
    engineer_enhanced_classification_features,
    engineer_gate_sequence_features,
    parse_gate_types,
    split_gate_sequence,
)


def test_parse_gate_types_handles_multiple_delimiters() -> None:
    tokens = parse_gate_types("H->CX;RX", delimiters=[",", ";", "->"])

    assert tokens == ["h", "cx", "rx"]


def test_split_gate_sequence_ignores_delimiters_inside_parameters() -> None:
    segments = split_gate_sequence(
        "U3(theta,phi,lambda), CX[q0,q1]; RZZ(pi/2)",
        delimiters=[",", ";"],
    )

    assert segments == ["U3(theta,phi,lambda)", "CX[q0,q1]", "RZZ(pi/2)"]


def test_parse_gate_types_normalizes_parameterized_and_targeted_gates() -> None:
    tokens = parse_gate_types(
        "U3(theta,phi,lambda), CX[q0,q1]; RZZ(pi/2) -> measure",
        delimiters=[",", ";", "->"],
    )

    assert tokens == ["u3", "cx", "rzz", "measure"]


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


def test_build_dataset_profile_reports_gate_diversity_and_zero_variance_warnings() -> None:
    cleaned_frame = pd.DataFrame(
        {
            "circuit_id": ["circ_1", "circ_2", "circ_3"],
            "error_type": ["readout", "depolarizing", "readout"],
            "qubit_count": [5, 6, 6],
            "gate_depth": [8, 10, 12],
            "error_rate_gate": [0.01, 0.02, 0.03],
            "t1_time": [50.0, 45.0, 40.0],
            "t2_time": [25.0, 20.0, 18.0],
            "readout_error": [0.01, 0.02, 0.02],
            "shots": [100, 100, 100],
            "fidelity": [0.95, 0.82, 0.8],
            "device_type": ["superconducting", "trapped_ion", "trapped_ion"],
            "gate_types": [
                "H,CX,RX",
                "U3(theta,phi,lambda),CX[q0,q1]",
                "RZZ(pi/2);SX",
            ],
            "bitstring_aligned": ["01010", "111000", "000111"],
            "ideal_bitstring_aligned": ["00010", "110000", "000101"],
        }
    )
    feature_config = FeatureConfig(two_qubit_gates=["cx", "rzz"])
    topology_feature_frame = engineer_gate_sequence_features(
        frame=cleaned_frame,
        sequence_column="gate_types",
        feature_config=feature_config,
        qubit_count_column="qubit_count",
    )
    feature_sets = {
        "baseline_raw": cleaned_frame.loc[
            :,
            [
                "circuit_id",
                "error_type",
                "qubit_count",
                "gate_depth",
                "error_rate_gate",
                "t1_time",
                "t2_time",
                "readout_error",
                "shots",
                "fidelity",
                "device_type",
            ],
        ].copy(),
        "topology_aware": pd.concat(
            [
                cleaned_frame.loc[
                    :,
                    [
                        "circuit_id",
                        "error_type",
                        "qubit_count",
                        "gate_depth",
                        "error_rate_gate",
                        "t1_time",
                        "t2_time",
                        "readout_error",
                        "shots",
                        "fidelity",
                        "device_type",
                    ],
                ],
                topology_feature_frame,
            ],
            axis=1,
        ),
    }

    from src.config.schema import ProjectConfig

    config = ProjectConfig(features=feature_config)
    feature_report = build_feature_report(feature_sets, config)
    dataset_profile = build_dataset_profile(cleaned_frame, config, feature_report)

    assert dataset_profile["row_count"] == 3
    assert dataset_profile["qubit_count_profile"]["min"] == 5
    assert dataset_profile["qubit_count_profile"]["max"] == 6
    assert dataset_profile["gate_sequence_profile"]["unique_normalized_sequence_count"] == 3
    assert dataset_profile["gate_sequence_profile"]["constant_normalized_sequence"] is False
    assert dataset_profile["gate_sequence_profile"]["unique_gate_type_count"] == 6
    assert dataset_profile["gate_sequence_profile"]["gate_type_frequencies"]["cx"] == 2
    assert (
        "shots" in dataset_profile["feature_set_warnings"]["baseline_raw"]["zero_variance_columns"]
    )
