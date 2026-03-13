# Baseline Model Suite

## Why This Suite Exists

The thesis should not jump straight from one model to conclusions. This suite
creates a layered comparison:

- `DummyClassifier` checks the absolute minimum level of performance
- `LogisticRegression` checks whether a simple linear decision boundary is enough
- `RandomForestClassifier` checks whether non-linear tabular learning helps
- `XGBoost` gives a stronger boosted-tree comparison without changing the data pipeline

Random Forest is still a strong first thesis baseline because it is:

- simple to explain
- robust on mixed tabular features
- relatively insensitive to feature scaling
- able to capture non-linear interactions
- widely accepted as a sensible classical benchmark

It is not the only model anymore. It is the first trustworthy reference point
inside a fair comparison suite.

## What The Model Predicts

Target:

- `error_type`

Current public Kaggle data appears to be binary:

- `depolarizing`
- `readout`

## Input Feature Sets

### Baseline Raw

The saved baseline table stays close to the original tabular measurements:

- qubit count
- gate depth
- gate error rate
- T1 / T2 values
- readout error
- shots
- fidelity
- device type

For the default leakage-free classification setting, the training code then
removes `fidelity` before fitting because it is not available for true
pre-execution prediction.

### Topology Aware

This extends the saved raw table with engineered features:

- `num_cx`
- `two_qubit_ratio`
- `unique_gates`
- `cx_density`
- `t2_t1_ratio`
- `bit_errors`
- `observed_error_rate`

For leakage-free classification, the pipeline keeps the structural gate and
coherence features but excludes measurement-derived features such as
`bit_errors`, `observed_error_rate`, `fidelity_loss`, and `bit_error_density`.

## Preprocessing Choices

- one shared `train/validation/test` split uses a fixed `random_state`
- the current repository default is `80/15/5`
- target stratification is applied when feasible
- only categorical columns are encoded
- numeric columns are passed through unchanged

That design is intentionally lightweight and thesis-friendly.

## Saved Artifacts

Each multi-model run saves:

- one folder per model
- split summary JSON
- metrics JSON
- validation and test classification reports
- validation and test confusion matrices
- model artifact
- feature-importance or coefficient-magnitude artifacts when supported
- a root-level comparison table
- the resolved run config

## Important Limitation

The current public CSV appears to contain the same `gate_types` string for every row. That means some gate-derived features may end up constant. This is not a bug in the code; it is a property of the current dataset snapshot.

In thesis terms, this is worth documenting because it affects how much signal topology-aware features can contribute.
