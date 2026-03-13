from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from src.config.io import load_config
from src.config.schema import DataConfig
from src.data.dataset import validate_required_columns
from src.data.prepare import normalize_bitstring, prepare_raw_dataset


def test_load_config_reads_rf_baseline_yaml_into_typed_model(tmp_path: Path) -> None:
    config_path = tmp_path / "example.yaml"
    config_path.write_text(
        dedent("""
            data:
              dataset_path: data/raw/NISQ-FaultLogs-100K.csv
              cleaned_dataset_path: data/interim/cleaned.parquet
              invalid_rows_path: data/interim/invalid.csv
              validation_report_path: data/interim/report.json
              file_format: csv
              id_column: circuit_id
              label_column: error_type
              gate_sequence_column: gate_types
              qubit_count_column: qubit_count
              bitstring_column: bitstring
              ideal_bitstring_column: ideal_bitstring
              timestamp_column: timestamp
              categorical_columns: ["device_type"]
              numeric_columns: ["qubit_count", "gate_depth"]
              drop_columns: []
              drop_invalid_rows: true
              align_short_bitstrings_to_qubit_count: true
            features:
              baseline_feature_path: data/processed/raw.parquet
              topology_feature_path: data/processed/topology.parquet
              feature_report_path: data/processed/report.json
              gate_delimiters: [","]
              two_qubit_gates: ["cx"]
              baseline_numeric_columns: ["qubit_count", "gate_depth"]
              categorical_feature_columns: ["device_type"]
            training:
              feature_set_name: topology_aware
              validation_size: 0.15
              test_size: 0.05
              random_state: 42
              stratify_by_label: true
              n_estimators: 300
              max_depth:
              min_samples_split: 2
              min_samples_leaf: 1
              class_weight: none
              compute_roc_auc: false
            output:
              experiment_root: experiments/rf_baseline
              run_name: demo_run
              figures_dir: reports/figures/rf_baseline
            """),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.dataset_path == Path("data/raw/NISQ-FaultLogs-100K.csv")
    assert config.training.feature_set_name == "topology_aware"
    assert config.training.validation_size == 0.15
    assert config.output.experiment_root == Path("experiments/rf_baseline")


def test_validate_required_columns_raises_for_missing_columns() -> None:
    frame = pd.DataFrame({"present": [1]})

    with pytest.raises(ValueError):
        validate_required_columns(frame, ["present", "missing"])


def test_normalize_bitstring_left_pads_short_strings() -> None:
    normalized, was_padded, issue = normalize_bitstring(
        "101",
        expected_length=5,
        align_short_strings=True,
    )

    assert normalized == "00101"
    assert was_padded is True
    assert issue is None


def test_prepare_raw_dataset_reports_expected_ranges_for_invalid_numeric_rows() -> None:
    frame = pd.DataFrame(
        {
            "circuit_id": ["circ_bad"],
            "qubit_count": [4],
            "gate_depth": [8],
            "gate_types": ["H,CX,RX"],
            "error_rate_gate": [0.02],
            "t1_time": [-1.5],
            "t2_time": [-2.5],
            "readout_error": [0.01],
            "shots": [100],
            "bitstring": ["101"],
            "ideal_bitstring": ["0011"],
            "fidelity": [0.5],
            "timestamp": ["2026-01-01T00:00:00"],
            "device_type": ["superconducting"],
            "error_type": ["readout"],
        }
    )

    prepared = prepare_raw_dataset(frame, DataConfig())

    invalid_details = prepared.validation_summary["invalid_issue_details"]
    assert invalid_details["out_of_range_t1_time"]["expected_condition"] == "> 0.0"
    assert invalid_details["out_of_range_t1_time"]["observed_min"] == -1.5
    assert invalid_details["out_of_range_t1_time"]["observed_max"] == -1.5
    assert "validation_issue_details" in prepared.invalid_rows.columns
