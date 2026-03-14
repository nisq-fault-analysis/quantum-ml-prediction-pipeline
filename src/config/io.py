"""YAML configuration and path helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from src.config.schema import ProjectConfig


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load an experiment YAML file into a typed Pydantic model."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    return ProjectConfig.model_validate(raw_config)


def ensure_project_directories(config: ProjectConfig) -> None:
    """Create the core directories used by the research workflow."""

    required_directories = [
        Path("data/raw"),
        Path("data/interim"),
        Path("data/processed"),
        config.output.experiment_root,
        config.output.figures_dir,
        config.data.cleaned_dataset_path.parent,
        config.data.invalid_rows_path.parent,
        config.data.validation_report_path.parent,
        config.features.baseline_feature_path.parent,
        config.features.topology_feature_path.parent,
        config.features.enhanced_feature_path.parent,
        config.features.feature_report_path.parent,
        config.features.dataset_profile_path.parent,
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_resolved_config(config: ProjectConfig, destination: str | Path) -> None:
    """Persist the effective config next to experiment results for reproducibility."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)


def build_run_directory(config: ProjectConfig) -> Path:
    """Build a timestamped experiment folder unless the config pins a run name."""

    run_name = config.output.run_name
    if run_name:
        run_directory = config.output.experiment_root / run_name
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory

    base_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = config.output.experiment_root / base_run_name
    if not run_directory.exists():
        run_directory.mkdir(parents=True, exist_ok=False)
        return run_directory

    suffix = 1
    while True:
        candidate_directory = config.output.experiment_root / f"{base_run_name}_{suffix:02d}"
        if not candidate_directory.exists():
            candidate_directory.mkdir(parents=True, exist_ok=False)
            return candidate_directory
        suffix += 1
