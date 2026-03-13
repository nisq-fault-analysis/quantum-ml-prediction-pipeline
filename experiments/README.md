# Experiments

This repository currently stores experiment runs under:

- `experiments/rf_baseline/`
- `experiments/model_benchmark/`

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
|-- split_summary.json
|-- run_config.yaml
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
