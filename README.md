# NISQ Reliability Prediction

This repository contains the Homework 2 submission for an MML1 project on predicting
the reliability of noisy quantum circuit executions from pre-run circuit and hardware
features.

## Project Task

The model predicts pre-run circuit reliability:

- **Input**: circuit structure, compiler metadata, and hardware-calibration/noise features
- **Target**: `reliability`
- **Task type**: supervised regression
- **Main HW2 metric**: validation MAE
- **Supporting metrics**: validation RMSE and validation R2

The test set is created and saved for later final evaluation, but it is not used for HW2
model comparison or model selection.

## Datasets

This repository contains two dataset lines. They have different roles.

### 1. Kaggle NISQ Fault Logs 100K

Path:

```text
data/raw/NISQ-FaultLogs-100K.csv
```

This is the earlier public dataset used for initial exploration, cleaning, feature
engineering, and historical fault-type classification baselines with target `error_type`.
It is useful background work, but it is not the official HW2 benchmark dataset.

### 2. Generated Thesis Dataset

Dataset id:

```text
thesis_production_125k_v1
```

Local release paths:

```text
data/raw/thesis_production_125k/thesis_production_125k.parquet
data/raw/thesis_production_125k/release/splits/train.parquet
data/raw/thesis_production_125k/release/splits/validation.parquet
data/raw/thesis_production_125k/release/splits/test.parquet
```

This dataset was produced by the separate dataset-generator workflow. It contains
125,000 base circuits and 250,000 execution-variant rows because each base circuit has
raw and transpiled variants.

This generated dataset is the current thesis dataset and the dataset used for HW2.

## Why The Committed HW2 Data Are A Sample

The full generated release split is large:

- full train: 200,000 rows
- full validation: 25,000 rows
- full test: 25,000 rows

The full data are kept locally under `data/raw/`, which is ignored by Git. For HW2, the
repository commits a smaller grouped sample under `data/hw2/`:

- HW2 train: 20,000 rows
- HW2 validation: 2,500 rows
- HW2 test: 2,500 rows

The sample preserves the official split logic. It is sampled by `base_circuit_id`, so the
same base circuit never appears in more than one split.

## Split Logic

The generated dataset has paired rows for the same base circuit, usually raw and
transpiled variants. A normal random row split would be too optimistic because one variant
of a circuit could appear in train and another variant in validation or test.

Therefore the split is grouped by:

```text
base_circuit_id
```

The saved split summary reports zero overlap:

- train/validation overlap: 0 groups
- train/test overlap: 0 groups
- validation/test overlap: 0 groups

This matches the intended prediction scenario: evaluate on circuits whose base
construction was not seen during fitting.

## Leakage Policy

The prediction context is pre-run reliability estimation. Features must be available before
observing execution outputs.

Allowed inputs include circuit, compiler, and calibration-like hardware features such as:

- `qubit_count`
- circuit depth and gate-count features
- interaction-graph features
- `t1_mean`, `t2_mean`
- `readout_error`
- `single_qubit_gate_error`
- `two_qubit_gate_error`
- local mapped-qubit calibration features when available
- `compiler_variant`

Excluded from model inputs:

- `reliability` itself
- `fidelity`
- `error_rate`
- `algorithmic_success_probability`
- `exact_output_success_rate`
- measured output count payloads
- mitigation-derived outcome fields

All learned preprocessing in `benchmark.ipynb` is inside an sklearn pipeline and is fitted
on the training split only.

## Benchmark

`benchmark.ipynb` compares:

- `DummyRegressor(strategy="mean")`
- `Ridge(alpha=1.0)`

The comparison is validation-only. Current executed notebook results:

```text
ridge_regression validation MAE: 0.072702
dummy_mean       validation MAE: 0.145326
```

These are intentionally simple HW2 baselines, not final thesis models.

## How To Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Optional notebook support:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[notebook]"
```

Open the notebooks:

```powershell
jupyter notebook dataprocessing.ipynb
jupyter notebook benchmark.ipynb
```

The notebooks also contain a small dependency check cell. If the active notebook kernel is
missing `pandas`, `pyarrow`, or `scikit-learn`, it installs the missing package into that
kernel.

## Repository Structure

```text
.
|-- dataprocessing.ipynb
|-- dataprocessing.html
|-- benchmark.ipynb
|-- benchmark.html
|-- data/
|   |-- hw2/                 # Committed HW2 train/validation/test sample
|   |-- raw/                 # Local source datasets, ignored except .gitkeep
|   |-- interim/             # Local intermediate data, ignored except .gitkeep
|   `-- processed/           # Local processed data, ignored except .gitkeep
|-- docs/                    # Supporting methodology notes
|-- experiments/             # Experiment configs and local outputs
|-- src/                     # Project code
`-- tests/                   # Automated tests
```

## Notes
- The Kaggle dataset and generated thesis dataset are both described, but HW2 uses the
  generated `thesis_production_125k_v1` reliability dataset.
