from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.reporting.experiment_summary import write_experiment_summary


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_write_experiment_summary_collects_standard_run_families(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    output_dir = tmp_path / "reports" / "experiments"

    model_benchmark_run = experiments_root / "model_benchmark" / "run_a"
    _write_yaml(
        model_benchmark_run / "run_config.yaml",
        {
            "data": {"dataset_path": "data/raw/dataset_a.csv"},
            "training": {
                "model_names": ["dummy_most_frequent", "xgboost"],
                "feature_set_name": "baseline_raw",
                "prediction_context": "pre_execution",
            },
        },
    )
    pd.DataFrame(
        [
            {
                "model_name": "dummy_most_frequent",
                "model_display_name": "Dummy Most Frequent",
                "validation_macro_f1": 0.33,
                "validation_accuracy": 0.5,
                "test_macro_f1": 0.33,
                "test_accuracy": 0.5,
                "validation_feature_columns_before_encoding": 7,
                "test_feature_columns_before_encoding": 7,
            },
            {
                "model_name": "xgboost",
                "model_display_name": "XGBoost",
                "validation_macro_f1": 0.51,
                "validation_accuracy": 0.51,
                "test_macro_f1": 0.52,
                "test_accuracy": 0.52,
                "validation_feature_columns_before_encoding": 7,
                "test_feature_columns_before_encoding": 7,
            },
        ]
    ).to_csv(model_benchmark_run / "model_comparison.csv", index=False)

    stratified_run = experiments_root / "qubit_stratified" / "run_b"
    _write_yaml(
        stratified_run / "run_config.yaml",
        {
            "data": {"dataset_path": "data/raw/dataset_a.csv"},
            "training": {
                "model_names": ["random_forest", "xgboost"],
                "feature_set_name": "baseline_raw",
                "prediction_context": "pre_execution",
            },
        },
    )
    pd.DataFrame(
        [
            {
                "qubit_count": 6,
                "artifact_subdirectory": "qubit_count_6",
                "model_name": "random_forest",
                "model_display_name": "Random Forest",
                "validation_macro_f1": 0.50,
                "validation_accuracy": 0.50,
                "test_macro_f1": 0.49,
                "test_accuracy": 0.49,
                "validation_feature_columns_before_encoding": 6,
                "test_feature_columns_before_encoding": 6,
            },
            {
                "qubit_count": 6,
                "artifact_subdirectory": "qubit_count_6",
                "model_name": "xgboost",
                "model_display_name": "XGBoost",
                "validation_macro_f1": 0.53,
                "validation_accuracy": 0.53,
                "test_macro_f1": 0.54,
                "test_accuracy": 0.54,
                "validation_feature_columns_before_encoding": 6,
                "test_feature_columns_before_encoding": 6,
            },
        ]
    ).to_csv(stratified_run / "qubit_model_comparison.csv", index=False)

    regression_run = experiments_root / "fidelity_regression" / "run_c"
    _write_yaml(
        regression_run / "run_config.yaml",
        {
            "data": {"dataset_path": "data/raw/dataset_a.csv"},
            "training": {
                "feature_set_name": "enhanced_topology",
                "model_names": ["random_forest"],
            },
        },
    )
    pd.DataFrame(
        [
            {
                "model_name": "random_forest_regressor",
                "model_display_name": "Random Forest Regressor",
                "validation_r2": 0.94,
                "validation_mae": 0.04,
                "test_r2": 0.93,
                "test_mae": 0.05,
                "validation_feature_columns_before_encoding": 14,
                "test_feature_columns_before_encoding": 14,
            }
        ]
    ).to_csv(regression_run / "model_comparison.csv", index=False)

    reliability_run = experiments_root / "reliability_baseline" / "run_f"
    _write_yaml(
        reliability_run / "run_config.yaml",
        {
            "data": {"dataset_path": "data/raw/dataset_a.csv"},
            "training": {
                "feature_set_name": "topology_aware",
                "prediction_context": "pre_execution",
            },
        },
    )
    pd.DataFrame(
        [
            {
                "model_name": "random_forest_regressor",
                "model_display_name": "Random Forest Regressor",
                "validation_r2": 0.88,
                "validation_mae": 0.07,
                "test_r2": 0.87,
                "test_mae": 0.08,
                "validation_feature_columns_before_encoding": 9,
                "test_feature_columns_before_encoding": 9,
            }
        ]
    ).to_csv(reliability_run / "model_comparison.csv", index=False)

    tuned_run = experiments_root / "tuned_classification" / "run_d"
    _write_yaml(
        tuned_run / "run_config.yaml",
        {
            "data": {"dataset_path": "data/raw/dataset_a.csv"},
            "training": {
                "model_names": ["random_forest", "xgboost"],
                "feature_set_name": "baseline_raw",
                "prediction_context": "pre_execution",
            },
        },
    )
    pd.DataFrame(
        [
            {
                "model_name": "xgboost",
                "model_display_name": "XGBoost",
                "best_validation_macro_f1": 0.55,
                "test_accuracy": 0.52,
                "test_macro_f1": 0.51,
                "best_params": "{'max_depth': 6}",
            }
        ]
    ).to_csv(tuned_run / "tuned_model_comparison.csv", index=False)
    _write_json(
        tuned_run / "subset_metadata.json",
        {"filter_column": "qubit_count", "filter_value": 8},
    )

    rf_baseline_run = experiments_root / "rf_baseline" / "run_e"
    _write_yaml(
        rf_baseline_run / "run_config.yaml",
        {
            "data": {"dataset_path": "data/raw/dataset_a.csv"},
            "training": {
                "model_names": ["random_forest"],
                "feature_set_name": "topology_aware",
                "prediction_context": "pre_execution",
            },
        },
    )
    _write_json(
        rf_baseline_run / "metrics.json",
        {
            "accuracy": 0.49,
            "macro_f1": 0.48,
            "feature_columns_before_encoding": 16,
        },
    )

    csv_path, json_path = write_experiment_summary(
        experiments_root=experiments_root,
        output_dir=output_dir,
    )

    matrix = pd.read_csv(csv_path)
    inventory = json.loads(json_path.read_text(encoding="utf-8"))

    assert csv_path.exists()
    assert json_path.exists()
    assert set(matrix["experiment_family"]) == {
        "classification_global",
        "classification_stratified",
        "classification_tuned",
        "fidelity_regression",
        "reliability_regression",
        "rf_baseline_single_model",
    }
    assert len(matrix) == 8
    assert inventory["total_rows"] == 8
    assert inventory["total_runs"] == 6
    assert inventory["datasets"] == ["dataset_a.csv"]
    assert (
        matrix.loc[
            matrix["experiment_family"] == "classification_global", "is_best_test_in_scope"
        ].sum()
        == 1
    )
    assert (
        matrix.loc[
            matrix["experiment_family"] == "classification_stratified",
            "is_best_test_in_scope",
        ].sum()
        == 1
    )
    tuned_row = matrix.loc[matrix["experiment_family"] == "classification_tuned"].iloc[0]
    assert tuned_row["subset_column"] == "qubit_count"
    assert tuned_row["subset_value"] == 8
    assert tuned_row["best_params"] == "{'max_depth': 6}"
