# Reliability Regression

## Why This Task Exists

Fault-type classification and reliability prediction answer different thesis questions.

- classification asks which fault label best describes a circuit outcome
- reliability regression asks how trustworthy the circuit execution is likely to be on a bounded `0..1` scale

This repository defines a leakage-free reliability benchmark so the thesis can discuss pre-run reliability estimation without quietly using observed outputs as model inputs.

## Dataset Roles

The repository contains two dataset lines:

- **Kaggle NISQ Fault Logs 100K**: the earlier public dataset used for initial cleaning,
  feature engineering, and historical fault-type classification experiments.
- **Generated thesis dataset `thesis_production_125k_v1`**: the current thesis dataset
  produced by the separate dataset-generator workflow. It contains 125,000 base circuits
  and 250,000 execution-variant rows because each base circuit has raw and transpiled
  variants.

The current reliability-regression direction uses `thesis_production_125k_v1`. The Kaggle
dataset remains background and historical baseline material, but it is not the official
current reliability benchmark dataset.

## Reliability Definition

For the earlier Kaggle-derived baseline, reliability was constructed as:

`reliability = 1 - (bit_errors / qubit_count)`

where:

- `bit_errors` is the Hamming distance between `bitstring` and `ideal_bitstring`
- `qubit_count` is the circuit width

Interpretation:

- `1.0` means the observed and ideal bitstrings match exactly
- `0.0` means every qubit position disagrees

The pipeline validates that the constructed target remains inside `[0, 1]`.

For the generated `thesis_production_125k_v1` release, `reliability` is shipped directly as
one of the target columns in the packaged dataset. The same modeling rule applies: target
and post-run outcome columns are not used as input features.

## Leakage Policy

This experiment is intentionally leakage-free for pre-run prediction.

Allowed raw inputs:

- `qubit_count`
- `gate_depth`
- `error_rate_gate`
- `t1_time`
- `t2_time`
- `readout_error`
- `device_type`

Allowed engineered pre-run inputs:

- `num_cx`
- `two_qubit_ratio`
- `unique_gates`
- `cx_density`
- `t2_t1_ratio`

Forbidden inputs:

- `bitstring`
- `ideal_bitstring`
- `bitstring_aligned`
- `ideal_bitstring_aligned`
- `fidelity`
- `bit_errors`
- `observed_error_rate`
- `fidelity_loss`
- `bit_error_density`
- `reliability`

Important distinction:

- post-run columns may be used to construct the target
- they must not be passed to the regressor as input features

## Models

The initial reliability benchmark compares:

- `DummyRegressor`
- `RandomForestRegressor`
- `XGBoostRegressor`

Metrics:

- `MAE`
- `RMSE`
- `R2`

Artifacts:

- per-model `metrics.json`
- per-model `model.joblib`
- validation/test actual-vs-predicted plots
- feature importance plots for tree models
- root-level comparison table, feature policy, target summary, and Markdown summary

## How To Run

1. Prepare the cleaned dataset.
2. Build the saved feature tables.
3. Run the reliability regression benchmark.

```powershell
.venv\Scripts\python.exe -m src.data.prepare_data --config experiments/configs/baseline.yaml
.venv\Scripts\python.exe -m src.features.build_features --config experiments/configs/baseline.yaml
.venv\Scripts\python.exe -m src.models.train_reliability_regression --config experiments/configs/reliability_regression.yaml
```

Or with `make`:

```bash
make prepare-data
make build-features
make train-reliability-regression
```

## How This Differs From Fault Classification

- reliability regression predicts a continuous value, not a class label
- its target is constructed from post-run outcomes, but those outcome columns are excluded from the inputs
- it evaluates with regression metrics rather than macro-F1 or accuracy
- it is a separate baseline workflow and should not be conflated with the leakage-free fault-classification headline
