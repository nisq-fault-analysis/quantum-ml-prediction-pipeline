from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.reporting.milestone_reports import generate_milestone_report


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_generate_milestone_report_creates_markdown_json_and_schema(tmp_path: Path) -> None:
    global_run = tmp_path / "global_run"
    stratified_run = tmp_path / "stratified_run"
    regression_run = tmp_path / "regression_run"
    tuned_run = tmp_path / "tuned_q8"
    shap_run = stratified_run / "qubit_count_8"
    output_dir = tmp_path / "reports"

    _write_yaml(
        global_run / "run_config.yaml",
        {
            "training": {
                "model_names": ["dummy_most_frequent", "xgboost"],
                "feature_set_name": "baseline_raw",
                "prediction_context": "pre_execution",
            }
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
            },
            {
                "model_name": "xgboost",
                "model_display_name": "XGBoost",
                "validation_macro_f1": 0.49,
                "validation_accuracy": 0.5,
                "test_macro_f1": 0.51,
                "test_accuracy": 0.51,
                "validation_feature_columns_before_encoding": 7,
            },
        ]
    ).to_csv(global_run / "model_comparison.csv", index=False)
    _write_json(
        global_run / "feature_policy.json",
        {
            "excluded_feature_columns": ["fidelity", "timestamp"],
            "prediction_context": "pre_execution",
        },
    )
    _write_json(global_run / "split_summary.json", {"train_rows": 10})

    _write_yaml(
        stratified_run / "run_config.yaml",
        {
            "training": {
                "model_names": ["random_forest", "xgboost"],
                "feature_set_name": "baseline_raw",
                "prediction_context": "pre_execution",
            }
        },
    )
    pd.DataFrame(
        [
            {
                "qubit_count": 6,
                "artifact_subdirectory": "qubit_count_6",
                "model_name": "xgboost",
                "model_display_name": "XGBoost",
                "validation_macro_f1": 0.52,
                "validation_accuracy": 0.52,
                "test_macro_f1": 0.5493,
                "test_accuracy": 0.5494,
                "validation_feature_columns_before_encoding": 6,
            },
            {
                "qubit_count": 8,
                "artifact_subdirectory": "qubit_count_8",
                "model_name": "random_forest",
                "model_display_name": "Random Forest",
                "validation_macro_f1": 0.5261,
                "validation_accuracy": 0.5261,
                "test_macro_f1": 0.5435,
                "test_accuracy": 0.5435,
                "validation_feature_columns_before_encoding": 6,
            },
        ]
    ).to_csv(stratified_run / "best_model_by_qubit_count.csv", index=False)
    pd.DataFrame(
        [
            {
                "qubit_count": 6,
                "artifact_subdirectory": "qubit_count_6",
                "model_name": "xgboost",
                "model_display_name": "XGBoost",
                "validation_macro_f1": 0.52,
                "validation_accuracy": 0.52,
                "test_macro_f1": 0.5493,
                "test_accuracy": 0.5494,
                "validation_feature_columns_before_encoding": 6,
            }
        ]
    ).to_csv(stratified_run / "qubit_model_comparison.csv", index=False)
    _write_json(
        stratified_run / "feature_policy_by_qubit.json",
        {"8": {"excluded_feature_columns": ["fidelity", "timestamp"]}},
    )
    _write_json(stratified_run / "subset_metadata_by_qubit.json", {"8": {"row_count": 100}})

    _write_yaml(
        regression_run / "run_config.yaml",
        {
            "training": {
                "model_names": ["random_forest_regressor", "xgboost_regressor"],
                "feature_set_name": "enhanced_topology",
            }
        },
    )
    pd.DataFrame(
        [
            {
                "model_name": "random_forest_regressor",
                "model_display_name": "Random Forest Regressor",
                "validation_r2": 0.94,
                "validation_mae": 0.043,
                "test_r2": 0.941,
                "test_mae": 0.043,
                "validation_feature_columns_before_encoding": 14,
            }
        ]
    ).to_csv(regression_run / "model_comparison.csv", index=False)

    _write_yaml(
        tuned_run / "run_config.yaml",
        {
            "training": {
                "model_names": ["random_forest", "xgboost"],
                "feature_set_name": "baseline_raw",
                "prediction_context": "pre_execution",
            }
        },
    )
    pd.DataFrame(
        [
            {
                "model_name": "xgboost",
                "model_display_name": "XGBoost",
                "best_validation_macro_f1": 0.5413,
                "test_accuracy": 0.5152,
                "test_macro_f1": 0.5152,
            }
        ]
    ).to_csv(tuned_run / "tuned_model_comparison.csv", index=False)
    _write_json(
        tuned_run / "subset_metadata.json",
        {"filter_column": "qubit_count", "filter_value": 8},
    )

    _write_json(
        shap_run / "shap_analysis" / "shap_metadata.json",
        {
            "selected_model": {"model_display_name": "Random Forest"},
            "explained_split": "test",
            "subset_metadata": {"filter_column": "qubit_count", "filter_value": 8},
        },
    )
    pd.DataFrame(
        [
            {"feature": "t1_time", "mean_abs_shap": 0.13},
            {"feature": "gate_depth", "mean_abs_shap": 0.12},
            {"feature": "readout_error", "mean_abs_shap": 0.11},
        ]
    ).to_csv(shap_run / "shap_analysis" / "shap_feature_importance.csv", index=False)

    config_path = tmp_path / "milestone.yaml"
    _write_yaml(
        config_path,
        {
            "title": "Test Milestone",
            "report_slug": "test_milestone",
            "output_dir": str(output_dir),
            "dataset_used": "data/raw/NISQ-FaultLogs-100K.csv",
            "split_strategy": "Deterministic 80/15/5 split.",
            "experiment_scope": "Synthetic test of the milestone reporter.",
            "subgroup": "qubit_count",
            "artifacts": {
                "global_classification_run": str(global_run),
                "stratified_classification_run": str(stratified_run),
                "regression_run": str(regression_run),
                "tuned_runs": [str(tuned_run)],
                "shap_runs": [str(shap_run)],
            },
            "manual_interpretation": {
                "plain_language_conclusion": "Stratified models beat the global baseline.",
                "scientific_meaning": "Regime-specific structure matters.",
                "negative_results_to_preserve": ["Global classification stayed weak."],
                "caveats": ["Do not overclaim."],
                "methodological_warnings": ["Validation and test winners differ."],
                "thesis_framing": {
                    "headline_result": "Use qubit_count = 8 as the headline subgroup.",
                    "more_trustworthy_result": "Use the untuned q=8 result.",
                    "held_out_test_comparator": "Use qubit_count = 6 as the held-out comparator.",
                },
                "recommended_next_steps": ["Collect richer topology variation."],
                "thesis_reuse_sentences": ["Leakage-free global classification remained weak."],
            },
        },
    )

    outputs = generate_milestone_report(config_path=config_path, repo_root=tmp_path)

    report_json = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    report_markdown = outputs.markdown_path.read_text(encoding="utf-8")
    report_schema = json.loads(outputs.schema_path.read_text(encoding="utf-8"))

    assert outputs.markdown_path.exists()
    assert outputs.json_path.exists()
    assert outputs.schema_path.exists()
    assert report_json["best_raw_results"]["global_test_winner"]["model_display_name"] == "XGBoost"
    assert (
        report_json["best_raw_results"]["stratified_validation_winner"]["subgroup"]
        == "qubit_count = 8"
    )
    assert report_json["best_raw_results"]["tuning_comparisons"][0]["test_delta"] < 0
    assert (
        report_json["best_raw_results"]["shap_highlights"][0]["top_features"][0]["feature"]
        == "t1_time"
    )
    assert "## Main Scientific Takeaway" in report_markdown
    assert "Validation improved but held-out test performance worsened" in report_markdown
    assert report_schema["title"] == "MilestoneReport"
