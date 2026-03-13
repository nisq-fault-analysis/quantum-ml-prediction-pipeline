# Experiment Outputs

This folder keeps the project reproducible by separating two things:

- `experiments/configs/`: the declared plan for an experiment
- `experiments/runs/`: the saved results of actually executing that plan

## Recommended Output Layout

Each run should create a dedicated folder under `experiments/runs/`:

```text
experiments/runs/<experiment_name>/
|-- resolved_config.yaml
|-- run_metadata.json
|-- model_comparison.csv
|-- global/
|   |-- logreg/
|   |   |-- metrics.json
|   |   |-- classification_report.csv
|   |   |-- predictions.csv
|   |   `-- confusion_matrix.png
|   |-- random_forest/
|   `-- xgboost/
`-- qubit_<n>/
    `-- ...
```

## Why This Matters

- `configs/` captures the intent of the experiment
- `runs/` captures the evidence produced by the experiment
- storing both makes the thesis methodology auditable and repeatable

## Naming Convention

Use experiment names that encode the research question, for example:

- `baseline_nisq_fault_classification`
- `baseline_qubit_stratified`
- `xgb_feature_ablation_v1`

## What To Save

Save these by default:

- resolved config file
- metrics summary
- per-class report
- predictions
- confusion matrix
- SHAP summary for the chosen tree model

TODO: If you later add cross-validation or hyperparameter sweeps, create a separate subfolder convention for sweep summaries.
