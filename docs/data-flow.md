# Data Flow

## Big Picture

The first thesis baseline follows this data flow:

`raw CSV -> cleaned interim dataset -> processed feature tables -> Random Forest run artifacts`

Each step writes files to disk on purpose. That makes the pipeline reproducible and lets you inspect intermediate outputs instead of treating the whole workflow as a black box.

## Stage 1: Raw Data

Input:

- `data/raw/NISQ-FaultLogs-100K.csv`

Properties:

- original Kaggle export
- never modified in place
- source of truth for the baseline

## Stage 2: Interim Data

Created by:

- `python -m src.data.prepare_data --config experiments/configs/baseline.yaml`

Outputs:

- cleaned dataset in `data/interim/`
- validation report JSON
- invalid-row CSV when problems are found

What happens here:

- required columns are checked
- numeric columns are coerced explicitly
- timestamps are parsed
- bitstrings are validated and aligned to `qubit_count`
- invalid rows are flagged and optionally dropped

Important note:

- shorter bitstrings are currently left-padded with zeros to `qubit_count`

This rule belongs to the historical Kaggle fault-log baseline. The current HW2 reliability
benchmark uses the generated `thesis_production_125k_v1` split in `data/hw2/` and does
not use bitstrings as model inputs.

## Stage 3: Processed Feature Tables

Created by:

- `python -m src.features.build_features --config experiments/configs/baseline.yaml`

Outputs:

- `data/processed/rf_baseline_raw_features.parquet`
- `data/processed/rf_baseline_topology_aware_features.parquet`
- feature report JSON

Why two feature sets exist:

- `baseline_raw`: numeric raw columns plus categorical device information
- `topology_aware`: baseline features plus engineered ratios and bitstring-derived features

This makes ablation-style comparisons easier later.

## Stage 4: Experiment Artifacts

Created by:

- `python -m src.models.train_rf_baseline --config experiments/configs/baseline.yaml`

Outputs:

- `experiments/rf_baseline/<run_name>/metrics.json`
- `experiments/rf_baseline/<run_name>/classification_report.txt`
- `experiments/rf_baseline/<run_name>/confusion_matrix.png`
- `experiments/rf_baseline/<run_name>/feature_importance.png`
- `experiments/rf_baseline/<run_name>/model.joblib`
- `experiments/rf_baseline/<run_name>/run_config.yaml`

Why this matters for the thesis:

- every reported result can be tied back to a saved config
- every figure can be traced to a specific run
- the methodology chapter can describe the pipeline step by step
