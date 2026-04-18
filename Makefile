PYTHON ?= .venv\Scripts\python.exe
CONFIG ?= experiments/configs/baseline.yaml
SUITE_CONFIG ?= experiments/configs/model_suite.yaml
REGRESSION_CONFIG ?= experiments/configs/fidelity_regression.yaml
RELIABILITY_CONFIG ?= experiments/configs/reliability_regression.yaml
RELEASE_REGRESSION_CONFIG ?= experiments/configs/release_reliability_125k.yaml
RELEASE_ABLATION_CONFIG ?= experiments/configs/release_reliability_125k_ablation.yaml
STRATIFIED_CONFIG ?= experiments/configs/qubit_stratified.yaml
TUNING_CONFIG ?= experiments/configs/tuned_classification.yaml
EXPERIMENTS_ROOT ?= experiments
EXPERIMENT_SUMMARY_DIR ?= reports/experiments

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .
	black --check .
	mypy src

format:
	black .
	ruff check . --fix

test:
	pytest

prepare-data:
	$(PYTHON) -m src.data.prepare_data --config $(CONFIG)

build-features:
	$(PYTHON) -m src.features.build_features --config $(CONFIG)

run-eda:
	$(PYTHON) -m src.data.run_eda --config $(CONFIG)

train-rf-baseline:
	$(PYTHON) -m src.models.train_rf_baseline --config $(CONFIG)

train-model-suite:
	$(PYTHON) -m src.models.train_model_suite --config $(SUITE_CONFIG)

train-fidelity-regression:
	$(PYTHON) -m src.models.train_fidelity_regression --config $(REGRESSION_CONFIG)

train-reliability-regression:
	$(PYTHON) -m src.models.train_reliability_regression --config $(RELIABILITY_CONFIG)

train-release-regression:
	$(PYTHON) -m src.models.train_release_regression --config $(RELEASE_REGRESSION_CONFIG)

train-release-ablation:
	$(PYTHON) -m src.models.train_release_ablation --config $(RELEASE_ABLATION_CONFIG)

evaluate-release-regression:
	$(PYTHON) -m src.models.evaluate_release_regression --run-dir $(RUN_DIR)

backfill-release-shap:
	$(PYTHON) -m src.models.evaluate_release_regression --run-dir $(RUN_DIR) --include-shap

train-qubit-stratified:
	$(PYTHON) -m src.models.train_qubit_stratified_suite --config $(STRATIFIED_CONFIG)

tune-classifiers:
	$(PYTHON) -m src.models.tune_classifiers --config $(TUNING_CONFIG)

build-experiment-summary:
	$(PYTHON) -m src.reporting.build_experiment_summary --experiments-root $(EXPERIMENTS_ROOT) --output-dir $(EXPERIMENT_SUMMARY_DIR)
