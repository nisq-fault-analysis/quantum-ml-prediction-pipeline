from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pandas as pd

from src.models.train_qubit_stratified_suite import run_qubit_stratified_suite


def test_qubit_stratified_suite_saves_per_qubit_artifacts(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for qubit_count in [5, 6]:
        for row_index in range(40):
            is_readout = row_index % 2 == 0
            rows.append(
                {
                    "circuit_id": f"circ_{qubit_count}_{row_index}",
                    "error_type": "readout" if is_readout else "depolarizing",
                    "qubit_count": qubit_count,
                    "gate_depth": 10 + row_index,
                    "error_rate_gate": 0.01 if is_readout else 0.2,
                    "t1_time": 100.0 + row_index,
                    "t2_time": 80.0 + row_index,
                    "readout_error": 0.01 if is_readout else 0.15,
                    "device_type": "superconducting" if is_readout else "trapped_ion",
                }
            )

    feature_path = tmp_path / "baseline_raw.parquet"
    pd.DataFrame(rows).to_parquet(feature_path, index=False)

    config_path = tmp_path / "qubit_stratified.yaml"
    config_path.write_text(
        dedent(f"""
            data:
              id_column: circuit_id
              label_column: error_type
              qubit_count_column: qubit_count
            features:
              baseline_feature_path: {feature_path.as_posix()}
              topology_feature_path: {feature_path.as_posix()}
              enhanced_feature_path: {feature_path.as_posix()}
              feature_report_path: {(tmp_path / "feature_report.json").as_posix()}
            training:
              model_names: ["dummy_most_frequent", "logistic_regression"]
              feature_set_name: baseline_raw
              prediction_context: pre_execution
              excluded_feature_columns: ["timestamp", "fidelity"]
              validation_size: 0.15
              test_size: 0.05
              random_state: 7
              stratify_by_label: true
              logistic_max_iter: 200
            output:
              experiment_root: {(tmp_path / "experiments").as_posix()}
              run_name: demo_stratified
              figures_dir: {(tmp_path / "figures").as_posix()}
            """),
        encoding="utf-8",
    )

    run_qubit_stratified_suite(config_path)

    run_directory = tmp_path / "experiments" / "demo_stratified"
    assert (run_directory / "best_model_by_qubit_count.csv").exists()
    assert (run_directory / "subset_metadata_by_qubit.json").exists()

    qubit_directory = run_directory / "qubit_count_5"
    assert (qubit_directory / "model_comparison.csv").exists()
    assert (qubit_directory / "split_summary.json").exists()
    assert (qubit_directory / "feature_policy.json").exists()
    assert (qubit_directory / "subset_metadata.json").exists()
    assert (qubit_directory / "dummy_most_frequent" / "metrics.json").exists()
    assert (qubit_directory / "logistic_regression" / "metrics.json").exists()

    subset_metadata = json.loads((qubit_directory / "subset_metadata.json").read_text("utf-8"))
    assert subset_metadata["filter_column"] == "qubit_count"
    assert subset_metadata["filter_value"] == 5
