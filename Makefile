PYTHON ?= .venv\Scripts\python.exe
CONFIG ?= experiments/configs/baseline.yaml
SUITE_CONFIG ?= experiments/configs/model_suite.yaml

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
