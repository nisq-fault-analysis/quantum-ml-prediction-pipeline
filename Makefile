PYTHON ?= python
CONFIG ?= experiments/configs/baseline.yaml

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

run-eda:
	$(PYTHON) -m src.data.run_eda --config $(CONFIG)

train-baseline:
	$(PYTHON) -m src.models.train_baseline --config $(CONFIG)
