# quantum-fault-classifier

Research repository for a diploma thesis on machine-learning-based classification of NISQ circuit fault types.

## Purpose

This repository is the analysis and modelling home for the project. It is designed to support:

- exploratory data analysis on the Kaggle NISQ fault dataset
- feature engineering from circuit gate sequences
- training and comparing classical ML baselines
- SHAP-based interpretability
- stratified modelling by qubit count
- generating reproducible figures and experiment outputs for thesis chapters and papers

The code is deliberately structured as a research workflow rather than a product application. That means reproducibility, clarity, and clean experiment logging matter more than building a large framework.

## Thesis Context

The thesis problem is to classify fault types in noisy intermediate-scale quantum (NISQ) circuits using classical machine learning. In practice, that means we want to:

1. understand the dataset and its biases
2. convert circuit/gate-sequence information into usable tabular features
3. compare multiple baseline models under the same split strategy
4. study how performance changes across qubit-count strata
5. interpret model behaviour with SHAP and convert results into figures/tables for writing

This repository is intended to be the reproducible bridge between raw data and thesis-ready evidence.

## Repository Structure

```text
quantum-fault-classifier/
|-- data/
|   |-- raw/              # Original dataset files, never edited in place
|   |-- interim/          # Intermediate artifacts such as engineered feature tables
|   `-- processed/        # Cleaned datasets ready for stable downstream use
|-- notebooks/           # Exploratory notebooks and quick visual inspection
|-- src/
|   |-- config/          # Typed experiment configuration models and YAML loading
|   |-- data/            # Dataset loading and EDA entry points
|   |-- features/        # Feature engineering from gate sequences
|   |-- models/          # Baseline training pipeline
|   |-- evaluation/      # Metrics and report helpers
|   `-- visualization/   # Plotting utilities for EDA and experiments
|-- experiments/
|   |-- configs/         # YAML experiment definitions
|   `-- runs/            # Saved outputs from executed experiments
|-- reports/
|   `-- figures/         # Figures reused in thesis chapters or papers
|-- tests/               # Small, fast tests for core logic
`-- docs/                # Architecture notes and thesis mapping
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

On PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. Add the dataset

Download the Kaggle dataset manually and place it under `data/raw/`.

Then update `experiments/configs/baseline.yaml`:

- `data.dataset_path`
- `data.label_column`
- `data.gate_sequence_column`
- `data.qubit_count_column`

These names are left as placeholders on purpose because we should not guess the real schema.

## Typical Workflow

### Option A: via `make`

```bash
make setup
make run-eda
make train-baseline
```

### Option B: direct Python commands

```bash
python -m src.data.run_eda --config experiments/configs/baseline.yaml
python -m src.models.train_baseline --config experiments/configs/baseline.yaml
```

Recommended working rhythm:

1. edit the experiment YAML
2. run EDA to inspect class balance, qubit distribution, and sequence lengths
3. run the baseline training pipeline
4. inspect `experiments/runs/` for metrics, reports, predictions, and SHAP outputs
5. move the final figures you want to cite into `reports/figures/`
6. document observations in thesis notes as you go

## How This Repo Connects To Other Repos In The Org

This repository is meant to sit in the middle of a larger research workflow.

- Upstream repos: circuit generation, simulation, hardware execution, or dataset collection
- This repo: feature engineering, classical ML experiments, interpretability, and figure generation
- Downstream repos: thesis/manuscript writing, presentation material, or publication packaging

TODO: Replace the generic descriptions above with the actual repository names used in your organization.

A useful mental model is:

`raw quantum experiment data -> this repository -> thesis figures/tables/manuscript`

## First Files To Read

- `experiments/configs/baseline.yaml`
- `src/data/dataset.py`
- `src/features/gate_sequence.py`
- `src/models/train_baseline.py`
- `docs/architecture.md`
- `docs/thesis-chapter-mapping.md`

## Notes

- No real data is included in the repository scaffold.
- Paths are placeholders until you insert the Kaggle dataset.
- The starter baseline is intentionally simple and transparent, so you can explain it easily in a thesis before adding more advanced models.
