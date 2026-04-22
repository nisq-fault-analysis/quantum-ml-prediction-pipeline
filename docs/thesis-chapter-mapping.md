# Thesis Chapter Mapping

This document connects the codebase to the written thesis so implementation and writing stay aligned.

## Chapter 2: Background And Problem Definition

Relevant files:

- [README.md](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/README.md)
- [docs/architecture.md](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/docs/architecture.md)
- [docs/baseline-model.md](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/docs/baseline-model.md)

How they help:

- frame the research problem
- justify why a classical baseline is useful
- explain why Random Forest is a reasonable first benchmark
- explain why multiple baselines are needed for a fair comparison

## Chapter 3: Dataset And Preprocessing

Relevant files:

- [experiments/configs/baseline.yaml](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/experiments/configs/baseline.yaml)
- [experiments/configs/model_suite.yaml](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/experiments/configs/model_suite.yaml)
- [src/data/dataset.py](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/src/data/dataset.py)
- [src/data/prepare.py](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/src/data/prepare.py)
- [docs/data-flow.md](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/docs/data-flow.md)

How they help:

- define the raw schema
- document cleaning assumptions
- log invalid rows and missing values
- separate raw, interim, and processed data

## Chapter 4: Feature Engineering And Baseline Modelling

Relevant files:

- [src/features/gate_sequence.py](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/src/features/gate_sequence.py)
- [src/features/build_features.py](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/src/features/build_features.py)
- [src/models/random_forest.py](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/src/models/random_forest.py)
- [src/models/model_suite.py](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/src/models/model_suite.py)
- [src/models/train_model_suite.py](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/src/models/train_model_suite.py)
- [src/models/train_rf_baseline.py](C:/Users/coufa/Documents/GitHub/nisq-ml-predictor/src/models/train_rf_baseline.py)

How they help:

- implement the first feature sets
- build the Random Forest baseline
- compare Dummy, Logistic Regression, Random Forest, and XGBoost fairly
- define the reproducible train/test workflow

## Later Chapters

This repository also prepares material for later chapters:

- evaluation outputs and figures support results chapters
- run configs and assumptions support discussion and limitations
- saved feature tables support future model-comparison chapters
