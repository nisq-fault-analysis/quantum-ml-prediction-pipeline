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

from pydantic import BaseModel, Field, field_validator, model_validator

ReleaseAblationMode = Literal[
    "raw_only",
    "transpiled_only",
    "both",
    "both_without_local_features",
    "both_with_local_features",
]

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


def default_release_regression_models() -> list[str]:
    """Return the regression baselines used for packaged thesis datasets."""

    return [
        "dummy_mean",
        "ridge_regression",
        "random_forest_regressor",
        "xgboost_regressor",
    ]


def default_release_ablation_modes() -> list[ReleaseAblationMode]:
    """Return the ablation modes used to audit raw/transpiled effects."""

    return [
        "raw_only",
        "transpiled_only",
        "both",
        "both_without_local_features",
        "both_with_local_features",
    ]


class DataConfig(BaseModel):
    """Describe the raw dataset and the cleaned interim outputs."""

    dataset_path: Path = Path("data/raw/NISQ-FaultLogs-100K.csv")
    cleaned_dataset_path: Path = Path("data/interim/nisq_fault_logs_cleaned.parquet")
    invalid_rows_path: Path = Path("data/interim/nisq_fault_logs_invalid_rows.csv")
    validation_report_path: Path = Path("data/interim/nisq_fault_logs_validation_report.json")
    split_manifest_path: Path | None = None
    feature_manifest_path: Path | None = None
    train_split_path: Path | None = None
    validation_split_path: Path | None = None
    test_split_path: Path | None = None
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

    @model_validator(mode="after")
    def validate_packaged_split_configuration(self) -> DataConfig:
        """Ensure packaged split configuration is either complete or omitted."""

        split_paths = [
            self.train_split_path,
            self.validation_split_path,
            self.test_split_path,
        ]
        has_any_direct_split_path = any(path is not None for path in split_paths)
        has_all_direct_split_paths = all(path is not None for path in split_paths)

        if has_any_direct_split_path and not has_all_direct_split_paths:
            raise ValueError(
                "train_split_path, validation_split_path, and test_split_path must all be set "
                "when configuring direct packaged split files."
            )

        if self.feature_manifest_path is not None and not (
            self.split_manifest_path or has_all_direct_split_paths
        ):
            raise ValueError(
                "feature_manifest_path requires either split_manifest_path or all direct split "
                "paths to be configured."
            )

        return self


class FeatureConfig(BaseModel):
    """Control how baseline and topology-aware feature sets are derived."""

    baseline_feature_path: Path = Path("data/processed/rf_baseline_raw_features.parquet")
    topology_feature_path: Path = Path("data/processed/rf_baseline_topology_aware_features.parquet")
    enhanced_feature_path: Path = Path("data/processed/rf_enhanced_topology_features.parquet")
    feature_report_path: Path = Path("data/processed/rf_feature_report.json")
    dataset_profile_path: Path = Path("data/processed/rf_dataset_profile.json")
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
    regression_model_names: list[str] = Field(default_factory=default_release_regression_models)
    ablation_modes: list[ReleaseAblationMode] = Field(
        default_factory=default_release_ablation_modes
    )
    target_column: str | None = None
    group_column: str | None = None
    feature_set_name: Literal["baseline_raw", "topology_aware", "enhanced_topology"] = (
        "topology_aware"
    )
    prediction_context: Literal["pre_execution", "post_observation"] = "pre_execution"
    excluded_feature_columns: list[str] = Field(
        default_factory=lambda: [
            "fidelity",
            "fidelity_loss",
            "bit_errors",
            "observed_error_rate",
            "bit_error_density",
            "timestamp",
        ]
    )
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
    ridge_alpha: float = Field(default=1.0, gt=0.0)
    grouped_cv_splits: int = Field(default=3, ge=2)
    tune_hyperparameters: bool = True
    difficulty_reference_column: str = "original_circuit_depth"
    difficulty_bucket_count: int = Field(default=4, ge=2, le=10)
    permutation_importance_repeats: int = Field(default=8, ge=1, le=100)
    permutation_importance_max_rows: int = Field(default=5000, ge=100, le=50000)
    enable_shap: bool = True
    shap_explained_split: Literal["validation", "test"] = "test"
    shap_max_rows: int = Field(default=1000, ge=50, le=10000)
    shap_background_max_rows: int = Field(default=500, ge=50, le=5000)
    enable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "nisq-ml-predictor"
    mlflow_run_name_prefix: str | None = None
    grid_search_verbose: int = Field(default=2, ge=0, le=10)
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


_NISQ_TARGET_COLUMNS = (
    "reliability",
    "fidelity",
    "error_rate",
    "algorithmic_success_probability",
    "exact_output_success_rate",
)


class NISQReliabilityConfig(BaseModel):
    """Configuration for the NISQ reliability dataset loader.

    The dataset is delivered as three pre-split parquet files plus two JSON
    sidecar files (split_manifest.json, feature_manifest.json).  Column lists
    are read from feature_manifest.json at load time; nothing is hardcoded.

    Parameters
    ----------
    dataset_dir:
        Directory that contains train.parquet, validation.parquet,
        test.parquet, split_manifest.json, and feature_manifest.json.
    target_column:
        Regression target exposed as ``y``.  Defaults to ``"reliability"``.
        Must be one of the five recognised target columns.
    group_column:
        Column used to verify that no group leaks across splits.  The value
        from split_manifest.json takes precedence when present.
    drop_payload_columns:
        Drop raw JSON payload columns (counts_json, ideal_distribution_json,
        compiler_metadata_json) before returning data.
    """

    dataset_dir: Path = Path("data/nisq_reliability")
    target_column: str = "reliability"
    group_column: str = "base_circuit_id"
    drop_payload_columns: bool = True

    @field_validator("target_column")
    @classmethod
    def _validate_target_column(cls, value: str) -> str:
        if value not in _NISQ_TARGET_COLUMNS:
            raise ValueError(
                f"target_column must be one of {list(_NISQ_TARGET_COLUMNS)}, "
                f"got {value!r}"
            )
        return value


class ProjectConfig(BaseModel):
    """Top-level configuration shared by all CLI entry points."""

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    nisq_reliability: NISQReliabilityConfig | None = None
