# Thesis Chapter Mapping

This document maps repository components to likely thesis chapters so the codebase and written thesis reinforce each other.

## Chapter 2: Background And Problem Framing

Relevant repository parts:

- `README.md`
- `docs/architecture.md`
- `notebooks/01_eda_starter.ipynb`

How it helps:

- frames the NISQ fault classification problem
- explains why classical ML is a useful baseline
- provides early dataset inspection material for motivating the task

## Chapter 3: Dataset And Research Setup

Relevant repository parts:

- `data/raw/`
- `src/data/dataset.py`
- `experiments/configs/baseline.yaml`

How it helps:

- documents where the dataset comes from
- captures the schema assumptions
- provides reproducible loading and validation logic

## Chapter 4: Feature Engineering

Relevant repository parts:

- `src/features/gate_sequence.py`
- `tests/test_feature_engineering.py`

How it helps:

- formalizes how gate sequences become numeric features
- supports discussion of simple interpretable representations before advanced models

## Chapter 5: Modelling Methodology

Relevant repository parts:

- `src/models/baseline.py`
- `src/models/train_baseline.py`
- `experiments/configs/`

How it helps:

- defines the train/test split strategy
- shows which baseline models are compared
- records reproducible modelling choices in config files

## Chapter 6: Evaluation And Interpretability

Relevant repository parts:

- `src/evaluation/metrics.py`
- `src/visualization/plots.py`
- SHAP outputs under `experiments/runs/`

How it helps:

- provides consistent metrics
- creates confusion matrices and SHAP figures
- supports analysis of both performance and model behaviour

## Chapter 7: Results

Relevant repository parts:

- `experiments/runs/`
- `reports/figures/`

How it helps:

- stores model comparison tables
- stores per-group results for qubit-count strata
- generates thesis-ready figures and evidence

## Chapter 8: Discussion, Limitations, And Future Work

Relevant repository parts:

- `docs/architecture.md`
- experiment configs and results history
- TODO markers across the codebase

How it helps:

- highlights where assumptions are still manual
- shows what the baseline does and does not capture
- points naturally toward richer feature sets and stronger models
