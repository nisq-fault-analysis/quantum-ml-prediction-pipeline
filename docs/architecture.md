# Repository Architecture

## Goal

This repository is built as a research workflow, not a production application.

That means the design priorities are:

1. reproducibility
2. readability
3. easy iteration on experiments
4. clear mapping from code to thesis chapters

## Core Pipeline

The current baseline architecture is:

`raw Kaggle CSV -> cleaned interim dataset -> processed feature tables -> Random Forest experiment artifacts`

The code is split by responsibility so each step can be inspected and explained independently.

## Why The Repository Is Split This Way

### `src/config`

This package holds typed YAML config models.

Why it matters:

- experiments are reproducible
- paths and parameters live in one place
- the methodology chapter can refer to a concrete configuration file

### `src/data`

This package owns:

- loading the raw CSV
- validating required columns
- coercing types
- aligning bitstrings
- writing cleaned interim data

Why it matters:

- raw data handling becomes explicit instead of notebook-only
- invalid rows are logged rather than disappearing silently

### `src/features`

This package builds saved feature tables.

Why it matters:

- feature logic is reusable
- you can compare raw vs engineered feature sets cleanly
- the feature-engineering chapter has a concrete implementation to reference

### `src/models`

This package trains the first Random Forest baseline and saves artifacts.

Why it matters:

- the first benchmark is reproducible
- the run folder contains both metrics and the exact config used

### `src/evaluation`

This package computes metrics and report text.

Why it matters:

- all experiments use a consistent evaluation procedure
- thesis tables are easier to keep comparable

### `src/visualization`

This package saves confusion matrices and feature-importance plots.

Why it matters:

- figures become scriptable rather than manual notebook screenshots

## Data Stages

The repository intentionally separates data into three stages:

- `data/raw`: source data only
- `data/interim`: cleaned, validated, but still close to the source
- `data/processed`: model-ready feature tables

This separation makes it easier to answer a common thesis question:

“Which preprocessing decisions happened before modelling?”

## Experiment Outputs

The current baseline writes to:

- `experiments/rf_baseline/<run_name>/`

Each run folder stores:

- metrics
- classification report
- confusion matrix
- feature importance plot
- serialized model
- resolved config

That is the minimal structure needed for a research-grade baseline.

## Beginner Mental Model

You can think of the repository as three layers:

1. inputs
2. transformations
3. evidence

Inputs:

- raw CSV
- experiment config

Transformations:

- cleaning
- feature building
- model training

Evidence:

- metrics
- plots
- saved model
- run config
