# Experiments

This repository currently stores experiment runs under:

- `experiments/rf_baseline/`
- `experiments/model_benchmark/`
- `experiments/fidelity_regression/`
- `experiments/reliability_baseline/`
- `experiments/release_ablation/`
- `experiments/qubit_stratified/`
- `experiments/tuned_classification/`

Single-model Random Forest runs follow this structure:

```text
experiments/rf_baseline/<timestamp_or_run_name>/
|-- metrics.json
|-- split_summary.json
|-- validation_classification_report.txt
|-- test_classification_report.txt
|-- validation_confusion_matrix.png
|-- test_confusion_matrix.png
|-- feature_importance.png
|-- model.joblib
`-- run_config.yaml
```

Multi-model benchmark runs follow this structure:

```text
experiments/model_benchmark/<timestamp_or_run_name>/
|-- model_comparison.csv
|-- feature_policy.json
|-- split_summary.json
|-- run_config.yaml
|-- shap_analysis/
|   |-- shap_feature_importance.csv
|   |-- shap_feature_importance.png
|   `-- shap_metadata.json
|-- dummy_most_frequent/
|   |-- metrics.json
|   |-- validation_classification_report.txt
|   |-- test_classification_report.txt
|   |-- validation_confusion_matrix.png
|   |-- test_confusion_matrix.png
|   `-- model.joblib
|-- logistic_regression/
|   |-- metrics.json
|   |-- validation_classification_report.txt
|   |-- test_classification_report.txt
|   |-- validation_confusion_matrix.png
|   |-- test_confusion_matrix.png
|   |-- feature_importance.csv
|   |-- feature_importance.png
|   `-- model.joblib
|-- random_forest/
|   |-- metrics.json
|   |-- validation_classification_report.txt
|   |-- test_classification_report.txt
|   |-- validation_confusion_matrix.png
|   |-- test_confusion_matrix.png
|   |-- feature_importance.csv
|   |-- feature_importance.png
|   `-- model.joblib
`-- xgboost/
    |-- metrics.json
    |-- validation_classification_report.txt
    |-- test_classification_report.txt
    |-- validation_confusion_matrix.png
    |-- test_confusion_matrix.png
    |-- feature_importance.csv
    |-- feature_importance.png
    `-- model.joblib
```

Why this layout matters:

- it keeps model outputs grouped by run
- it preserves the exact config used for each result
- it makes thesis tables and figures traceable

`experiments/configs/` stores planned experiments.
`experiments/rf_baseline/` stores the original single-model baseline runs.
`experiments/model_benchmark/` stores fair, same-split model comparisons.
`experiments/fidelity_regression/` stores regression runs that predict fidelity.
`experiments/reliability_baseline/` stores leakage-free pre-run reliability regression runs.
`experiments/release_ablation/` stores grouped raw/transpiled thesis ablation runs for packaged releases.
`experiments/qubit_stratified/` stores per-qubit-count classification summaries.
`experiments/tuned_classification/` stores validation-driven tuning results.

Stratified classification runs now also create one self-contained benchmark folder
per qubit count, for example:

```text
experiments/qubit_stratified/<timestamp_or_run_name>/
|-- best_model_by_qubit_count.csv
|-- qubit_model_comparison.csv
|-- qubit_count_5/
|   |-- model_comparison.csv
|   |-- split_summary.json
|   |-- feature_policy.json
|   |-- subset_metadata.json
|   `-- <per-model artifacts>
`-- qubit_count_6/
    `-- ...
```

Grouped release ablation runs follow this structure:

```text
experiments/release_ablation/<timestamp_or_run_name>/
|-- ablation_model_comparison.csv
|-- best_model_by_ablation.csv
|-- suite_slice_metrics.csv
|-- split_membership.parquet
|-- grouped_split_audit.json
|-- summary.md
|-- raw_only/
|   |-- feature_policy.json
|   |-- grouped_split_audit.json
|   `-- <per-model artifacts>
`-- both_with_local_features/
    `-- <per-model artifacts>
```
