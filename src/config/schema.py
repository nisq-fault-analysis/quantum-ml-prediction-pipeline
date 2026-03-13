"""Typed configuration models for the thesis baseline pipeline.

The configuration is intentionally explicit so that each experiment run answers:
- which raw file was used
- how the raw file was cleaned
- which feature set was built
- which model settings produced the reported metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ModelName = Literal[
    "dummy_most_frequent",
    "logistic_regression",
    "random_forest",
    "xgboost",
]


def default_model_names() -> list[ModelName]:
    """Return the baseline models we want to compare by default."""

    return [
        "dummy_most_frequent",
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]


class DataConfig(BaseModel):
    """Describe the raw dataset and the cleaned interim outputs."""

    dataset_path: Path = Path("data/raw/NISQ-FaultLogs-100K.csv")
    cleaned_dataset_path: Path = Path("data/interim/nisq_fault_logs_cleaned.parquet")
    invalid_rows_path: Path = Path("data/interim/nisq_fault_logs_invalid_rows.csv")
    validation_report_path: Path = Path("data/interim/nisq_fault_logs_validation_report.json")
    file_format: Literal["auto", "csv", "parquet"] = "auto"
    required_columns: list[str] = Field(
        default_factory=lambda: [
            "circuit_id",
            "qubit_count",
            "gate_depth",
            "gate_types",
            "error_rate_gate",
            "t1_time",
            "t2_time",
            "readout_error",
            "shots",
            "bitstring",
            "ideal_bitstring",
            "fidelity",
            "timestamp",
            "device_type",
            "error_type",
        ]
    )
    id_column: str = "circuit_id"
    label_column: str = "error_type"
    gate_sequence_column: str = "gate_types"
    qubit_count_column: str = "qubit_count"
    bitstring_column: str = "bitstring"
    ideal_bitstring_column: str = "ideal_bitstring"
    timestamp_column: str = "timestamp"
    categorical_columns: list[str] = Field(default_factory=lambda: ["device_type"])
    numeric_columns: list[str] = Field(
        default_factory=lambda: [
            "qubit_count",
            "gate_depth",
            "error_rate_gate",
            "t1_time",
            "t2_time",
            "readout_error",
            "shots",
            "fidelity",
        ]
    )
    drop_columns: list[str] = Field(default_factory=list)
    drop_invalid_rows: bool = True
    align_short_bitstrings_to_qubit_count: bool = True


class FeatureConfig(BaseModel):
    """Control how baseline and topology-aware feature sets are derived."""

    baseline_feature_path: Path = Path("data/processed/rf_baseline_raw_features.parquet")
    topology_feature_path: Path = Path("data/processed/rf_baseline_topology_aware_features.parquet")
    feature_report_path: Path = Path("data/processed/rf_feature_report.json")
    gate_delimiters: list[str] = Field(default_factory=lambda: [",", ";", "|", "->"])
    two_qubit_gates: list[str] = Field(default_factory=lambda: ["cx", "cz", "swap"])
    baseline_numeric_columns: list[str] = Field(
        default_factory=lambda: [
            "qubit_count",
            "gate_depth",
            "error_rate_gate",
            "t1_time",
            "t2_time",
            "readout_error",
            "shots",
            "fidelity",
        ]
    )
    categorical_feature_columns: list[str] = Field(default_factory=lambda: ["device_type"])


class TrainingConfig(BaseModel):
    """Training options for the thesis baseline model suite."""

    model_names: list[ModelName] = Field(default_factory=default_model_names)
    feature_set_name: Literal["baseline_raw", "topology_aware"] = "topology_aware"
    validation_size: float = Field(default=0.15, gt=0.0, lt=0.5)
    test_size: float = Field(default=0.05, gt=0.0, lt=0.5)
    random_state: int = 42
    stratify_by_label: bool = True
    dummy_strategy: Literal["most_frequent", "prior", "stratified", "uniform"] = "most_frequent"
    logistic_c: float = Field(default=1.0, gt=0.0)
    logistic_max_iter: int = Field(default=1000, ge=100)
    logistic_class_weight: Literal["none", "balanced"] = "none"
    n_estimators: int = Field(default=200, ge=100)
    max_depth: int | None = 12
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=2, ge=1)
    class_weight: Literal["none", "balanced", "balanced_subsample"] = "none"
    xgboost_n_estimators: int = Field(default=300, ge=50)
    xgboost_max_depth: int = Field(default=6, ge=2)
    xgboost_learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    xgboost_subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    xgboost_colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    xgboost_reg_lambda: float = Field(default=1.0, ge=0.0)
    compute_roc_auc: bool = False

    @model_validator(mode="after")
    def validate_split_sizes(self) -> TrainingConfig:
        """Ensure the configured split leaves enough data for training."""

        if self.validation_size + self.test_size >= 1.0:
            raise ValueError("validation_size + test_size must be smaller than 1.0")

        return self


class OutputConfig(BaseModel):
    """Choose where experiment artifacts and thesis figures are written."""

    experiment_root: Path = Path("experiments/rf_baseline")
    run_name: str | None = None
    figures_dir: Path = Path("reports/figures/rf_baseline")


class ProjectConfig(BaseModel):
    """Top-level configuration shared by all CLI entry points."""

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
