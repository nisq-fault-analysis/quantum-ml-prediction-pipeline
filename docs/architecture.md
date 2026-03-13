# Repository Architecture

## Goal

This repository is built to answer a research question, not to ship a user-facing application. That changes the design priorities.

The priorities here are:

1. reproducibility
2. readability
3. easy experiment iteration
4. clear mapping from code to thesis methodology

## The Core Idea

The repository follows a simple pipeline:

`raw dataset -> typed config -> feature engineering -> model training -> evaluation -> figures/results`

Each step has a dedicated place in the folder structure so that you can explain the workflow without mixing concerns.

## Why The Code Is Split By Responsibility

### `src/config`

This is where experiment choices live. Instead of hardcoding paths and column names in many scripts, we keep them in YAML and validate them with Pydantic. That makes experiments easier to rerun and easier to describe in the methodology chapter.

### `src/data`

This layer reads the dataset and performs light checks. Its job is to answer:

- where is the dataset?
- what file format is it?
- do the expected columns actually exist?

This keeps I/O logic separate from model logic.

### `src/features`

This layer turns raw circuit/gate-sequence strings into numeric features. The starter version uses transparent count-based features because they are easy to validate and easy to explain in a thesis. Later you can extend it with domain-specific features such as depth or entanglement structure.

### `src/models`

This layer owns train/test splitting, model fitting, and experiment output saving. It is kept separate so you can swap model families without rewriting data loading or plotting code.

### `src/evaluation`

This layer converts predictions into metrics and table-friendly reports. In a thesis project, this matters because you will likely reuse the same evaluation tables across many experiment variants.

### `src/visualization`

This layer produces reusable figures. Keeping plotting code in one place avoids notebook-only logic that becomes hard to reproduce later.

## Why There Is Both `notebooks/` And `src/`

This is a common beginner question.

- `notebooks/` is for exploration, quick hypotheses, and visual inspection
- `src/` is for reusable logic that should survive beyond one notebook session

A healthy pattern is:

1. explore an idea in a notebook
2. once the idea stabilizes, move the logic into `src/`
3. keep the notebook as a thin consumer of reusable code

## Why We Save Outputs Under `experiments/runs/`

Research projects produce many artifacts:

- metrics
- confusion matrices
- predictions
- SHAP plots
- feature tables

If those files are not organized, it becomes very hard to answer simple questions like:

- Which config produced this figure?
- Which model won on 8-qubit circuits?
- Can I reproduce the table in Chapter 6?

That is why each experiment gets a named output folder with a resolved config file beside it.

## Beginner Mental Model

You can think of the repository as three layers:

1. inputs
2. transformations
3. evidence

Inputs:

- raw data
- config files

Transformations:

- loading
- feature engineering
- model training
- evaluation

Evidence:

- metrics
- reports
- figures
- saved predictions

## What To Extend Later

Once the baseline is stable, good next extensions are:

- cross-validation
- hyperparameter search
- richer sequence features
- ablation studies
- calibration analysis
- error analysis notebooks
- figure scripts dedicated to final thesis visuals
