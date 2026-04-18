# quantum-fault-classifier

Research repository for a diploma thesis on machine-learning-based classification of NISQ circuit fault types.

## Project Purpose

This repository turns a raw Kaggle dataset of simulated NISQ circuit fault logs into:

- cleaned interim datasets
- reproducible feature tables
- a comparable suite of baseline classifiers
- fidelity regression baselines
- leakage-free reliability regression baselines
- qubit-count-stratified benchmark tables
- evaluation artifacts for thesis chapters, papers, and figures

The current focus is not model complexity. It is building a clean, auditable baseline that you can explain confidently in a thesis.

## Thesis Context

The thesis studies whether classical machine-learning models can classify fault types in noisy intermediate-scale quantum circuits from simulation-derived fault logs.

This repository currently implements the first full baseline workflow:

`raw Kaggle CSV -> validation/cleaning -> feature engineering -> baseline models -> metrics and plots`

That baseline is intentionally simple and transparent so it can support:

- Chapter 2 background/problem framing
- Chapter 3 dataset and preprocessing methodology
- Chapter 4 feature engineering and first modelling decisions

## Repository Structure

```text
quantum-fault-classifier/
|-- data/
|   |-- raw/                  # Original Kaggle CSV, never edited in place
|   |-- interim/              # Cleaned dataset plus validation logs
|   `-- processed/            # Saved feature tables ready for modelling
|-- notebooks/               # Exploration and quick inspection
|-- src/
|   |-- config/              # Typed YAML config models
|   |-- data/                # Reading, validation, and cleaning
|   |-- features/            # Feature engineering from gate strings and bitstrings
|   |-- models/              # Baseline model training and comparison pipeline
|   |-- evaluation/          # Metrics/report helpers
|   `-- visualization/       # Confusion matrix and feature-importance plots
|-- experiments/
|   |-- configs/             # Experiment YAML files
|   |-- rf_baseline/         # Timestamped Random Forest run artifacts
|   `-- model_benchmark/     # Timestamped multi-model comparison runs
|-- reports/
|   `-- figures/             # Figures selected for thesis chapters
|-- docs/                    # Architecture, data flow, and thesis mapping
`-- tests/                   # Focused tests for validation and features
```

## Raw Data, Interim Data, Processed Data

This repository uses three data stages on purpose:

- `data/raw/`: untouched source data from Kaggle
- `data/interim/`: cleaned and validated rows, plus logs about invalid rows
- `data/processed/`: feature tables used directly for modelling

This separation matters for thesis work because it keeps preprocessing decisions explicit and reproducible.

## Current Baseline Models

The repository now supports a small benchmark suite trained to predict:

- target: `error_type`

The training pipeline:

1. loads one of the saved feature tables
2. performs one reproducible `train/validation/test` split shared by all compared models
3. uses the current thesis-friendly convention of `80/15/5`
4. stratifies by target class when statistically feasible
5. one-hot encodes only categorical inputs that actually need encoding
6. drops leakage-prone columns for the default `pre_execution` classification setting
7. trains `DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`, and `XGBoost`
8. saves validation and test metrics separately for each model
9. saves a `model_comparison.csv` summary and a `split_summary.json` at the run root

The repository also supports:

- fidelity regression with `DummyRegressor`, `RandomForestRegressor`, and `XGBoostRegressor`
- leakage-free reliability regression with `DummyRegressor`, `RandomForestRegressor`, and `XGBoostRegressor`
- qubit-count-stratified classification comparisons
- lightweight validation-driven tuning for Random Forest and XGBoost
- durable milestone reports that preserve raw results, interpretation, caveats, and thesis framing

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the project

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Optional notebook support:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[notebook]"
```

### 3. Confirm the raw dataset exists

Expected file:

- [data/raw/NISQ-FaultLogs-100K.csv](C:/Users/coufa/Documents/GitHub/quantum-fault-classifier/data/raw/NISQ-FaultLogs-100K.csv)

## Typical Workflow

### Step 1: Prepare raw data

```powershell
.venv\Scripts\python.exe -m src.data.prepare_data --config experiments/configs/baseline.yaml
```

Outputs:

- cleaned interim dataset
- validation report JSON
- invalid-row CSV if any rows fail checks

### Step 2: Build feature tables

```powershell
.venv\Scripts\python.exe -m src.features.build_features --config experiments/configs/baseline.yaml
```

Outputs:

- baseline/raw feature set
- topology-aware feature set
- dataset profile for future snapshots
- feature report showing zero-variance columns

### Step 3: Train the multi-model benchmark suite

```powershell
.venv\Scripts\python.exe -m src.models.train_model_suite --config experiments/configs/model_suite.yaml
```

Outputs:

- one subfolder per model
- `model_comparison.csv`
- `split_summary.json`
- `feature_policy.json`
- per-model `metrics.json`
- per-model validation/test classification reports
- per-model validation/test confusion matrices
- per-model `model.joblib`
- per-model importance artifacts when supported
- `run_config.yaml`

The default classification policy is now explicitly leakage-free for pre-execution prediction.
It excludes `fidelity`, `fidelity_loss`, `bit_errors`, `observed_error_rate`,
`bit_error_density`, and `timestamp` from classifier inputs even if those columns are
present in saved feature tables for regression or later post-observation analysis.

The feature-building step now also saves a dataset profile JSON that records:

- gate-sequence diversity
- gate-type frequencies
- qubit-count coverage
- zero-variance warnings by feature set

That profile is meant to answer, at a glance, whether a newly uploaded dataset snapshot
actually contains richer topology signal than the original public Kaggle file.

The original single-model Random Forest command still exists when you want to rerun only that reference model:

```powershell
.venv\Scripts\python.exe -m src.models.train_rf_baseline --config experiments/configs/baseline.yaml
```

## Makefile Shortcuts

If you have `make` available:

```bash
make setup
make prepare-data
make build-features
make train-model-suite
make train-rf-baseline
make train-fidelity-regression
make train-reliability-regression
make train-release-ablation
make train-qubit-stratified
make tune-classifiers
make build-experiment-summary
```

## Master Experiment Summary

To compare saved runs across models, feature sets, and future dataset snapshots, build the
master experiment matrix:

```powershell
.venv\Scripts\python.exe -m src.reporting.build_experiment_summary --experiments-root experiments --output-dir reports\experiments
```

This writes:

- `reports/experiments/master_experiment_matrix.csv`
- `reports/experiments/master_experiment_inventory.json`

The matrix is a row-per-model summary across the standard experiment roots, including
global classification, qubit-stratified classification, tuned subgroup reruns, fidelity
regression, leakage-free reliability regression, and the legacy single-model RF baseline.

## Leakage-Free Reliability Regression

The repository also includes a separate regression task that predicts a bounded
pre-run reliability score:

`reliability = 1 - (bit_errors / qubit_count)`

This target is built from observed versus ideal bitstrings, but those outcome columns
are used only to define the target. They are not allowed into the model inputs.

The allowed inputs are the thesis-friendly pre-execution circuit and device features:

- `qubit_count`
- `gate_depth`
- `error_rate_gate`
- `t1_time`
- `t2_time`
- `readout_error`
- `device_type`
- `num_cx`
- `two_qubit_ratio`
- `unique_gates`
- `cx_density`
- `t2_t1_ratio`

Run it with:

```powershell
.venv\Scripts\python.exe -m src.models.train_reliability_regression --config experiments/configs/reliability_regression.yaml
```

## Packaged Release Pipeline

The packaged thesis dataset `thesis_production_125k_v1` uses a stricter pipeline than the
legacy Kaggle baseline.

What is different:

- train, validation, and test come from the packaged `split_manifest.json`
- split integrity is revalidated locally so `base_circuit_id` never crosses splits
- hyperparameter tuning uses grouped CV on the training split only
- imputers, encoders, scalers, and models all live inside one sklearn `Pipeline`
- regression evaluation is reported globally and sliced by `family`, `qubit_count`,
  `compiler_variant`, and a train-derived `difficulty_bucket`
- exact split membership is saved as an artifact for reproducibility

How leakage is prevented:

- the release runner reads the shipped grouped split instead of resplitting rows naively
- target columns and other outcome-derived columns are excluded explicitly before fitting
- preprocessing is fit only on training rows, or on grouped CV training folds during tuning
- `base_circuit_id` overlap across splits raises an error before training begins

How raw and transpiled null local features are handled:

- `compiler_variant` is treated as an explicit optional feature in mixed-data ablations
- local mapping-aware features are imputed inside the training pipeline with train-only medians
- no missing-indicator features are added, so null patterns are not turned into hidden flags
- the ablation suite compares mixed-data runs with and without local features to show whether
  they help beyond simple variant identity

Run the grouped release ablation study with:

```powershell
.venv\Scripts\python.exe -m src.models.train_release_ablation --config experiments/configs/release_reliability_125k_ablation.yaml
```

or:

```bash
make train-release-ablation
```

Live progress is written to the console and to `experiments/release_ablation/<run_name>/training.log`.
Set `training.grid_search_verbose` in the config if you want quieter or noisier grouped-CV updates.
SHAP artifacts are also written per model under `.../<ablation>/<model>/shap_analysis/`.

Refresh sliced metrics and diagnostics from saved prediction artifacts without retraining:

```powershell
.venv\Scripts\python.exe -m src.models.evaluate_release_regression --run-dir experiments\release_ablation\<run_name>
```

Backfill SHAP artifacts for an already-finished release run without retraining:

```powershell
.venv\Scripts\python.exe -m src.models.evaluate_release_regression --run-dir experiments\release_ablation\<run_name> --include-shap
```

To switch from reliability to fidelity, keep the same config shape and change only:

- `training.target_column: fidelity`

The grouped release runner writes:

- `ablation_model_comparison.csv`
- `best_model_by_ablation.csv`
- `suite_slice_metrics.csv`
- `training.log`
- `split_membership.parquet`
- `grouped_split_audit.json`
- per-model `shap_analysis/shap_feature_importance.csv`
- per-model `shap_analysis/shap_summary.png`
- per-model prediction tables, residual plots, importance artifacts, and grouped CV results

## Durable Milestone Reports

When an experiment becomes important enough to cite later in the thesis, generate a durable milestone report instead of relying on memory or terminal output alone:

```powershell
.venv\Scripts\python.exe -m src.reporting.generate_milestone_report --config reports\milestone_configs\20260313_leakage_free_classification.yaml
```

This writes:

- a Markdown summary for humans
- a JSON summary for reuse
- a JSON schema for the report structure

The output lives under `reports/milestones/`. The config file stores manual interpretation separately from artifact paths so negative results, caveats, and thesis framing survive alongside the metrics.

## How This Repo Connects To Other Repos In The Org

This repository is the analysis and modelling layer in a broader research stack.

- Upstream repos: circuit generation, simulation, or dataset collection
- This repo: cleaning, feature engineering, baseline ML, evaluation, and figure generation
- Downstream repos: thesis writing, paper drafting, presentations

TODO: Replace the generic repo-role descriptions above with the actual repository names used in your organization.

## Beginner Notes

- `circuit_id` is kept for traceability, not used as a model feature.
- `timestamp` is preserved in the cleaned data but excluded from the leakage-free classifier because it may encode simulation order rather than a scientifically meaningful signal.
- The public Kaggle file appears to use the same `gate_types` string in every row, so some engineered gate features may have zero variance. That is okay for the baseline; the feature report makes that visible instead of hiding it.

## Key Docs

- [docs/architecture.md](C:/Users/coufa/Documents/GitHub/quantum-fault-classifier/docs/architecture.md)
- [docs/data-flow.md](C:/Users/coufa/Documents/GitHub/quantum-fault-classifier/docs/data-flow.md)
- [docs/baseline-model.md](C:/Users/coufa/Documents/GitHub/quantum-fault-classifier/docs/baseline-model.md)
- [docs/reliability-regression.md](C:/Users/coufa/Documents/GitHub/quantum-fault-classifier/docs/reliability-regression.md)
- [docs/milestone-reports.md](C:/Users/coufa/Documents/GitHub/quantum-fault-classifier/docs/milestone-reports.md)
- [docs/thesis-chapter-mapping.md](C:/Users/coufa/Documents/GitHub/quantum-fault-classifier/docs/thesis-chapter-mapping.md)
