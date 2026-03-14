"""Structured models for durable milestone experiment reports."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ComparisonInputsConfig(BaseModel):
    """Optional manual hints for the "what was compared" section."""

    models_compared: list[str] = Field(default_factory=list)
    feature_sets_compared: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReportArtifactsConfig(BaseModel):
    """Artifact paths used to assemble the milestone report."""

    global_classification_run: Path | None = None
    stratified_classification_run: Path | None = None
    regression_run: Path | None = None
    tuned_runs: list[Path] = Field(default_factory=list)
    shap_runs: list[Path] = Field(default_factory=list)
    extra_artifact_paths: list[Path] = Field(default_factory=list)


class ThesisFramingConfig(BaseModel):
    """Manual thesis framing guidance that should survive over time."""

    headline_result: str
    more_trustworthy_result: str
    held_out_test_comparator: str | None = None
    presentation_notes: list[str] = Field(default_factory=list)


class ManualInterpretationConfig(BaseModel):
    """Human interpretation that should remain distinct from raw metrics."""

    plain_language_conclusion: str
    scientific_meaning: str
    negative_results_to_preserve: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    methodological_warnings: list[str] = Field(default_factory=list)
    thesis_framing: ThesisFramingConfig
    recommended_next_steps: list[str] = Field(default_factory=list)
    thesis_reuse_sentences: list[str] = Field(default_factory=list)


class MilestoneReportConfig(BaseModel):
    """Config file used to generate one durable milestone report."""

    title: str
    report_slug: str
    output_dir: Path = Path("reports/milestones")
    run_group_name: str | None = None
    dataset_used: str | None = None
    split_strategy: str
    experiment_scope: str
    subgroup: str | None = None
    what_was_compared: ComparisonInputsConfig = Field(default_factory=ComparisonInputsConfig)
    artifacts: ReportArtifactsConfig
    manual_interpretation: ManualInterpretationConfig


class ReportMetadata(BaseModel):
    """Top-level metadata shown at the start of the report."""

    report_title: str
    report_slug: str
    report_timestamp: str
    run_group_name: str | None = None
    dataset_used: str | None = None
    split_strategy: str
    experiment_scope: str
    subgroup: str | None = None
    report_config_path: str


class ComparisonOverview(BaseModel):
    """What the milestone compared and under which feature policy."""

    models_compared: list[str] = Field(default_factory=list)
    feature_sets_compared: list[str] = Field(default_factory=list)
    tuning_was_run: bool = False
    shap_was_run: bool = False
    prediction_contexts: list[str] = Field(default_factory=list)
    excluded_feature_columns: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ClassificationResultSummary(BaseModel):
    """Compact summary of one classification winner."""

    scope: str
    subgroup: str | None = None
    artifact_subdirectory: str | None = None
    model_name: str
    model_display_name: str
    validation_macro_f1: float | None = None
    validation_accuracy: float | None = None
    test_macro_f1: float | None = None
    test_accuracy: float | None = None
    feature_columns_before_encoding: int | None = None
    run_directory: str
    source_table: str


class RegressionResultSummary(BaseModel):
    """Compact summary of one regression winner."""

    scope: str
    model_name: str
    model_display_name: str
    validation_r2: float | None = None
    validation_mae: float | None = None
    test_r2: float | None = None
    test_mae: float | None = None
    feature_columns_before_encoding: int | None = None
    run_directory: str
    source_table: str


class TuningComparison(BaseModel):
    """Comparison between untuned subgroup winner and tuned rerun."""

    subgroup: str
    untuned_model_display_name: str | None = None
    tuned_model_display_name: str
    untuned_validation_macro_f1: float | None = None
    tuned_validation_macro_f1: float | None = None
    untuned_test_macro_f1: float | None = None
    tuned_test_macro_f1: float | None = None
    validation_delta: float | None = None
    test_delta: float | None = None
    interpretation: str
    untuned_run_directory: str | None = None
    tuned_run_directory: str
    source_table: str


class ShapFeatureImportance(BaseModel):
    """One SHAP feature row kept in the durable summary."""

    feature: str
    mean_abs_shap: float


class ShapRunSummary(BaseModel):
    """Compact summary of a SHAP explanation run."""

    scope: str
    subgroup: str | None = None
    selected_model_display_name: str
    explained_split: str
    top_features: list[ShapFeatureImportance] = Field(default_factory=list)
    source_directory: str
    source_csv: str
    source_metadata: str


class BestRawResults(BaseModel):
    """Raw metric winners that should remain separable from interpretation."""

    global_validation_winner: ClassificationResultSummary | None = None
    global_test_winner: ClassificationResultSummary | None = None
    stratified_validation_winner: ClassificationResultSummary | None = None
    stratified_test_winner: ClassificationResultSummary | None = None
    subgroup_winners: list[ClassificationResultSummary] = Field(default_factory=list)
    regression_reference: RegressionResultSummary | None = None
    tuning_comparisons: list[TuningComparison] = Field(default_factory=list)
    shap_highlights: list[ShapRunSummary] = Field(default_factory=list)


class MainScientificTakeaway(BaseModel):
    """Interpretation section that explains what the results mean."""

    plain_language_conclusion: str
    scientific_meaning: str
    negative_results_to_preserve: list[str] = Field(default_factory=list)


class ThesisFramingRecommendation(BaseModel):
    """How to use the milestone later in the thesis."""

    headline_result: str
    more_trustworthy_result: str
    held_out_test_comparator: str | None = None
    presentation_notes: list[str] = Field(default_factory=list)


class ArtifactReference(BaseModel):
    """Pointer to a source artifact that supports the milestone report."""

    label: str
    path: str


class MilestoneReport(BaseModel):
    """Complete structured milestone report."""

    metadata: ReportMetadata
    what_was_compared: ComparisonOverview
    best_raw_results: BestRawResults
    main_scientific_takeaway: MainScientificTakeaway
    important_caveats: list[str] = Field(default_factory=list)
    methodological_warnings: list[str] = Field(default_factory=list)
    thesis_framing_recommendation: ThesisFramingRecommendation
    recommended_next_steps: list[str] = Field(default_factory=list)
    thesis_reuse_sentences: list[str] = Field(default_factory=list)
    artifact_references: list[ArtifactReference] = Field(default_factory=list)


class GeneratedReportArtifacts(BaseModel):
    """Paths written by the report generator."""

    markdown_path: Path
    json_path: Path
    schema_path: Path
