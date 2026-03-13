"""YAML configuration loading utilities."""

from __future__ import annotations

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
    """Create the core output directories used by the research workflow."""

    required_directories = [
        Path("data/interim"),
        Path("data/processed"),
        config.output.output_dir,
        config.output.figures_dir,
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_resolved_config(config: ProjectConfig, destination: str | Path) -> None:
    """Persist the effective config next to experiment results for reproducibility."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
