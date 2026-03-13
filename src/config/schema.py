"""Typed configuration models for the research pipeline.

Keeping experiment choices in a schema has two benefits:
1. scripts become easier to rerun and compare
2. the methodology is easier to describe in the thesis because assumptions live in one place
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Describe where the dataset lives and how the scripts should interpret it."""

    dataset_path: Path
    file_format: Literal["auto", "csv", "parquet"] = "auto"
    label_column: str = "fault_type"
    gate_sequence_column: str = "gate_sequence"
    qubit_count_column: str = "qubit_count"
    drop_columns: list[str] = Field(default_factory=list)
    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    random_state: int = 42
    stratify_by_label: bool = True


class FeatureConfig(BaseModel):
    """Control the simple, interpretable baseline features built from gate sequences."""

    gate_delimiters: list[str] = Field(default_factory=lambda: [" ", ",", ";", "|", "->"])
    top_gates: list[str] = Field(
        default_factory=lambda: ["x", "h", "cx", "cz", "rz", "ry", "measure"]
    )
    lowercase_tokens: bool = True


class TrainingConfig(BaseModel):
    """Training options for the baseline modelling scripts."""

    enable_models: list[Literal["logreg", "random_forest", "xgboost"]] = Field(
        default_factory=lambda: ["logreg", "random_forest", "xgboost"]
    )
    stratify_by_qubit_count: bool = True
    minimum_samples_per_qubit_group: int = Field(default=25, ge=5)
    generate_shap_for: Literal["logreg", "random_forest", "xgboost"] | None = "xgboost"
    class_weight: Literal["balanced", "none"] = "balanced"
    n_estimators: int = Field(default=300, ge=50)
    max_depth: int = Field(default=6, ge=2)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)


class OutputConfig(BaseModel):
    """Choose where experiment artifacts and figures are written."""

    experiment_name: str = "baseline_nisq_fault_classification"
    output_dir: Path = Path("experiments/runs/baseline_nisq_fault_classification")
    figures_dir: Path = Path("reports/figures")
    save_predictions: bool = True
    save_shap: bool = True


class ProjectConfig(BaseModel):
    """Full project configuration used by the CLI entry points."""

    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
