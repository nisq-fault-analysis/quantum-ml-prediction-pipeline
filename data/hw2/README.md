# HW2 Saved Datasets

This folder contains the train, validation, and test datasets used by the root-level
`dataprocessing.ipynb` and `benchmark.ipynb` notebooks.

- `train.parquet`: training split used to fit preprocessing and benchmark models
- `validation.parquet`: validation split used for HW2 benchmark comparison
- `test.parquet`: held-out split saved for later final evaluation, not used for HW2 comparison
- `split_summary.json`: split sizes, grouped sampling details, target summaries, and split policy
- `feature_policy.json`: target, allowed pre-execution features, and excluded leakage columns

The task is leakage-free reliability regression from pre-run circuit, compiler, and
hardware-calibration features. The target is `reliability`, a continuous value in `[0, 1]`.
Outcome-derived columns such as output counts, fidelity, error-rate targets, and mitigation
outputs are excluded from model inputs.

The repository contains two datasets:

- Kaggle NISQ Fault Logs 100K: the earlier public dataset used for historical fault-type
  classification and first baseline experiments.
- `thesis_production_125k_v1`: the generated thesis dataset from the dataset-generator
  workflow. This is the current HW2 dataset and the main reliability-regression dataset.

The full generated release has 125,000 base circuits and 250,000 execution-variant rows
because each base circuit has raw and transpiled variants. The committed HW2 files are a
grouped sample from the official release splits so the repository remains practical to
submit on GitHub.
