"""Generate durable thesis milestone reports from saved experiment artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
import yaml

from src.reporting.models import (
    ArtifactReference,
    BestRawResults,
    ClassificationResultSummary,
    ComparisonOverview,
    GeneratedReportArtifacts,
    MainScientificTakeaway,
    MilestoneReport,
    MilestoneReportConfig,
    RegressionResultSummary,
    ReportMetadata,
    ShapFeatureImportance,
    ShapRunSummary,
    ThesisFramingRecommendation,
    TuningComparison,
)

REPORT_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "milestone_report.md.tpl"
MODEL_DISPLAY_NAMES = {
    "dummy_most_frequent": "Dummy Most Frequent",
    "dummy_mean": "Dummy Mean",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "random_forest_regressor": "Random Forest Regressor",
    "xgboost_regressor": "XGBoost Regressor",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def load_milestone_report_config(config_path: str | Path) -> MilestoneReportConfig:
    """Load a report config from YAML."""

    path = Path(config_path)
    return MilestoneReportConfig.model_validate(_load_yaml(path))


def _resolve_config_path(path_value: Path | None, repo_root: Path) -> Path | None:
    if path_value is None:
        return None
    return path_value if path_value.is_absolute() else (repo_root / path_value).resolve()


def _repo_relative(path_value: Path, repo_root: Path) -> str:
    try:
        return str(path_value.resolve().relative_to(repo_root))
    except ValueError:
        return str(path_value.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected artifact at {path}")
    return path


def _ensure_directory(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected run directory at {path}")
    if path.is_file():
        return path.parent
    return path


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Artifact table is empty: {path}")
    return frame


def _read_run_config(run_dir: Path) -> dict[str, Any]:
    run_config_path = _ensure_file(run_dir / "run_config.yaml")
    return _load_yaml(run_config_path)


def _read_feature_policy(run_dir: Path, filename: str) -> dict[str, Any] | None:
    path = run_dir / filename
    if not path.exists():
        return None
    return _load_json(path)


def _format_subset_label(subset_metadata: dict[str, Any] | None) -> str | None:
    if not subset_metadata:
        return None
    return f"{subset_metadata['filter_column']} = {subset_metadata['filter_value']}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _pretty_model_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _maybe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _classification_summary_from_row(
    row: pd.Series,
    *,
    scope: str,
    run_dir: Path,
    source_table: Path,
    repo_root: Path,
    subgroup: str | None = None,
) -> ClassificationResultSummary:
    return ClassificationResultSummary(
        scope=scope,
        subgroup=subgroup,
        artifact_subdirectory=(
            str(row["artifact_subdirectory"]) if "artifact_subdirectory" in row.index else None
        ),
        model_name=str(row["model_name"]),
        model_display_name=str(row["model_display_name"]),
        validation_macro_f1=_maybe_float(row.get("validation_macro_f1")),
        validation_accuracy=_maybe_float(row.get("validation_accuracy")),
        test_macro_f1=_maybe_float(row.get("test_macro_f1")),
        test_accuracy=_maybe_float(row.get("test_accuracy")),
        feature_columns_before_encoding=_maybe_int(
            row.get("validation_feature_columns_before_encoding")
            or row.get("test_feature_columns_before_encoding")
        ),
        run_directory=_repo_relative(run_dir, repo_root),
        source_table=_repo_relative(source_table, repo_root),
    )


def _regression_summary_from_row(
    row: pd.Series,
    *,
    scope: str,
    run_dir: Path,
    source_table: Path,
    repo_root: Path,
) -> RegressionResultSummary:
    return RegressionResultSummary(
        scope=scope,
        model_name=str(row["model_name"]),
        model_display_name=str(row["model_display_name"]),
        validation_r2=_maybe_float(row.get("validation_r2")),
        validation_mae=_maybe_float(row.get("validation_mae")),
        test_r2=_maybe_float(row.get("test_r2")),
        test_mae=_maybe_float(row.get("test_mae")),
        feature_columns_before_encoding=_maybe_int(
            row.get("validation_feature_columns_before_encoding")
            or row.get("test_feature_columns_before_encoding")
        ),
        run_directory=_repo_relative(run_dir, repo_root),
        source_table=_repo_relative(source_table, repo_root),
    )


def _load_classification_run(run_path: Path, repo_root: Path) -> dict[str, Any]:
    run_dir = _ensure_directory(run_path)
    comparison_path = _ensure_file(run_dir / "model_comparison.csv")
    comparison_frame = _load_frame(comparison_path)
    run_config = _read_run_config(run_dir)
    feature_policy = _read_feature_policy(run_dir, "feature_policy.json") or {}

    validation_winner = comparison_frame.sort_values(
        by=["validation_macro_f1", "validation_accuracy"],
        ascending=[False, False],
    ).iloc[0]
    test_winner = comparison_frame.sort_values(
        by=["test_macro_f1", "test_accuracy"],
        ascending=[False, False],
    ).iloc[0]

    return {
        "run_dir": run_dir,
        "comparison_path": comparison_path,
        "comparison_frame": comparison_frame,
        "run_config": run_config,
        "feature_policy": feature_policy,
        "validation_winner": _classification_summary_from_row(
            validation_winner,
            scope="global classification validation winner",
            run_dir=run_dir,
            source_table=comparison_path,
            repo_root=repo_root,
        ),
        "test_winner": _classification_summary_from_row(
            test_winner,
            scope="global classification held-out test winner",
            run_dir=run_dir,
            source_table=comparison_path,
            repo_root=repo_root,
        ),
        "models_compared": comparison_frame["model_display_name"].astype(str).tolist(),
    }


def _load_stratified_run(run_path: Path, repo_root: Path) -> dict[str, Any]:
    run_dir = _ensure_directory(run_path)
    best_by_qubit_path = _ensure_file(run_dir / "best_model_by_qubit_count.csv")
    best_by_qubit = _load_frame(best_by_qubit_path)
    comparison_path = _ensure_file(run_dir / "qubit_model_comparison.csv")
    run_config = _read_run_config(run_dir)
    feature_policy_by_qubit = _read_feature_policy(run_dir, "feature_policy_by_qubit.json") or {}

    validation_winner = best_by_qubit.sort_values(
        by=["validation_macro_f1", "validation_accuracy"],
        ascending=[False, False],
    ).iloc[0]
    test_winner = best_by_qubit.sort_values(
        by=["test_macro_f1", "test_accuracy"],
        ascending=[False, False],
    ).iloc[0]

    subgroup_winners = [
        _classification_summary_from_row(
            row,
            scope="stratified subgroup winner",
            run_dir=run_dir,
            source_table=best_by_qubit_path,
            repo_root=repo_root,
            subgroup=f"qubit_count = {int(row['qubit_count'])}",
        )
        for _, row in best_by_qubit.sort_values(by="qubit_count").iterrows()
    ]

    subgroup_index = {
        str(int(row["qubit_count"])): row
        for _, row in best_by_qubit.sort_values(by="qubit_count").iterrows()
    }

    return {
        "run_dir": run_dir,
        "best_by_qubit_path": best_by_qubit_path,
        "comparison_path": comparison_path,
        "best_by_qubit": best_by_qubit,
        "run_config": run_config,
        "feature_policy_by_qubit": feature_policy_by_qubit,
        "validation_winner": _classification_summary_from_row(
            validation_winner,
            scope="stratified validation winner",
            run_dir=run_dir,
            source_table=best_by_qubit_path,
            repo_root=repo_root,
            subgroup=f"qubit_count = {int(validation_winner['qubit_count'])}",
        ),
        "test_winner": _classification_summary_from_row(
            test_winner,
            scope="stratified held-out test winner",
            run_dir=run_dir,
            source_table=best_by_qubit_path,
            repo_root=repo_root,
            subgroup=f"qubit_count = {int(test_winner['qubit_count'])}",
        ),
        "subgroup_winners": subgroup_winners,
        "subgroup_index": subgroup_index,
    }


def _load_regression_run(run_path: Path, repo_root: Path) -> dict[str, Any]:
    run_dir = _ensure_directory(run_path)
    comparison_path = _ensure_file(run_dir / "model_comparison.csv")
    comparison_frame = _load_frame(comparison_path)
    run_config = _read_run_config(run_dir)

    winner = comparison_frame.sort_values(
        by=["test_r2", "validation_r2", "test_mae"],
        ascending=[False, False, True],
    ).iloc[0]

    return {
        "run_dir": run_dir,
        "comparison_path": comparison_path,
        "comparison_frame": comparison_frame,
        "run_config": run_config,
        "winner": _regression_summary_from_row(
            winner,
            scope="fidelity regression reference",
            run_dir=run_dir,
            source_table=comparison_path,
            repo_root=repo_root,
        ),
        "models_compared": comparison_frame["model_display_name"].astype(str).tolist(),
    }


def _load_tuned_run(run_path: Path, repo_root: Path) -> dict[str, Any]:
    run_dir = _ensure_directory(run_path)
    comparison_path = _ensure_file(run_dir / "tuned_model_comparison.csv")
    comparison_frame = _load_frame(comparison_path)
    run_config = _read_run_config(run_dir)
    subset_metadata = _load_json(_ensure_file(run_dir / "subset_metadata.json"))
    best_row = comparison_frame.sort_values(
        by=["best_validation_macro_f1", "test_accuracy"],
        ascending=[False, False],
    ).iloc[0]

    return {
        "run_dir": run_dir,
        "comparison_path": comparison_path,
        "comparison_frame": comparison_frame,
        "run_config": run_config,
        "subset_metadata": subset_metadata,
        "best_row": best_row,
        "subgroup": _format_subset_label(subset_metadata),
    }


def _resolve_shap_directory(path_value: Path) -> Path:
    if path_value.is_file():
        if path_value.name == "shap_feature_importance.csv":
            return path_value.parent
        return path_value.parent
    if (path_value / "shap_analysis").exists():
        return path_value / "shap_analysis"
    return path_value


def _load_shap_run(path_value: Path, repo_root: Path) -> ShapRunSummary:
    shap_dir = _resolve_shap_directory(path_value)
    importance_path = _ensure_file(shap_dir / "shap_feature_importance.csv")
    metadata_path = _ensure_file(shap_dir / "shap_metadata.json")
    importance_frame = _load_frame(importance_path)
    metadata = _load_json(metadata_path)
    selected_model = metadata.get("selected_model", {})
    subset_metadata = metadata.get("subset_metadata")
    subgroup = _format_subset_label(subset_metadata if isinstance(subset_metadata, dict) else None)
    top_features = [
        ShapFeatureImportance(
            feature=str(row["feature"]),
            mean_abs_shap=float(row["mean_abs_shap"]),
        )
        for _, row in importance_frame.head(5).iterrows()
    ]
    scope = "SHAP explanation"
    if subgroup:
        scope = f"SHAP explanation for {subgroup}"

    return ShapRunSummary(
        scope=scope,
        subgroup=subgroup,
        selected_model_display_name=str(
            selected_model.get("model_display_name", selected_model.get("model_name", "unknown"))
        ),
        explained_split=str(metadata.get("explained_split", "unknown")),
        top_features=top_features,
        source_directory=_repo_relative(shap_dir, repo_root),
        source_csv=_repo_relative(importance_path, repo_root),
        source_metadata=_repo_relative(metadata_path, repo_root),
    )


def _build_tuning_comparison(
    tuned_run: dict[str, Any],
    stratified_run: dict[str, Any] | None,
    repo_root: Path,
) -> TuningComparison:
    best_row = tuned_run["best_row"]
    subgroup = tuned_run["subgroup"] or "filtered subgroup"
    untuned_row: pd.Series | None = None

    if stratified_run is not None:
        subset_metadata = tuned_run["subset_metadata"]
        subgroup_value = str(int(subset_metadata["filter_value"]))
        untuned_row = stratified_run["subgroup_index"].get(subgroup_value)

    tuned_validation = _maybe_float(best_row.get("best_validation_macro_f1"))
    tuned_test = _maybe_float(best_row.get("test_macro_f1"))
    untuned_validation = (
        _maybe_float(untuned_row.get("validation_macro_f1"))
        if untuned_row is not None
        else None
    )
    untuned_test = (
        _maybe_float(untuned_row.get("test_macro_f1"))
        if untuned_row is not None
        else None
    )
    validation_delta = (
        None
        if tuned_validation is None or untuned_validation is None
        else tuned_validation - untuned_validation
    )
    test_delta = None if tuned_test is None or untuned_test is None else tuned_test - untuned_test

    if validation_delta is not None and test_delta is not None:
        if validation_delta > 0 and test_delta < 0:
            interpretation = (
                "Validation improved but held-out test performance worsened, which is consistent "
                "with overfitting during tuning."
            )
        elif validation_delta <= 0 and test_delta < 0:
            interpretation = (
                "Tuning did not improve validation or held-out test performance for this subgroup."
            )
        elif validation_delta > 0 and test_delta >= 0:
            interpretation = (
                "Tuning improved validation and did not harm held-out test "
                "performance for this subgroup."
            )
        else:
            interpretation = "Tuning results were mixed and should be interpreted cautiously."
    else:
        interpretation = (
            "Untuned subgroup metrics were not available for direct comparison, "
            "so the tuning result should be interpreted in isolation."
        )

    return TuningComparison(
        subgroup=subgroup,
        untuned_model_display_name=(
            str(untuned_row["model_display_name"]) if untuned_row is not None else None
        ),
        tuned_model_display_name=str(best_row["model_display_name"]),
        untuned_validation_macro_f1=untuned_validation,
        tuned_validation_macro_f1=tuned_validation,
        untuned_test_macro_f1=untuned_test,
        tuned_test_macro_f1=tuned_test,
        validation_delta=validation_delta,
        test_delta=test_delta,
        interpretation=interpretation,
        untuned_run_directory=(
            _repo_relative(stratified_run["run_dir"], repo_root)
            if stratified_run is not None
            else None
        ),
        tuned_run_directory=_repo_relative(tuned_run["run_dir"], repo_root),
        source_table=_repo_relative(tuned_run["comparison_path"], repo_root),
    )


def _collect_artifact_references(
    *,
    global_run: dict[str, Any] | None,
    stratified_run: dict[str, Any] | None,
    regression_run: dict[str, Any] | None,
    tuned_runs: list[dict[str, Any]],
    shap_runs: list[ShapRunSummary],
    extra_artifact_paths: list[Path],
    repo_root: Path,
) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []

    if global_run is not None:
        references.extend(
            [
                ArtifactReference(
                    label="Global classification comparison table",
                    path=_repo_relative(global_run["comparison_path"], repo_root),
                ),
                ArtifactReference(
                    label="Global classification feature policy",
                    path=_repo_relative(global_run["run_dir"] / "feature_policy.json", repo_root),
                ),
                ArtifactReference(
                    label="Global classification split summary",
                    path=_repo_relative(global_run["run_dir"] / "split_summary.json", repo_root),
                ),
            ]
        )

    if stratified_run is not None:
        references.extend(
            [
                ArtifactReference(
                    label="Stratified best model by qubit count",
                    path=_repo_relative(stratified_run["best_by_qubit_path"], repo_root),
                ),
                ArtifactReference(
                    label="Stratified full comparison table",
                    path=_repo_relative(stratified_run["comparison_path"], repo_root),
                ),
                ArtifactReference(
                    label="Stratified feature policy by qubit count",
                    path=_repo_relative(
                        stratified_run["run_dir"] / "feature_policy_by_qubit.json",
                        repo_root,
                    ),
                ),
                ArtifactReference(
                    label="Stratified subset metadata by qubit count",
                    path=_repo_relative(
                        stratified_run["run_dir"] / "subset_metadata_by_qubit.json",
                        repo_root,
                    ),
                ),
            ]
        )

    if regression_run is not None:
        references.append(
            ArtifactReference(
                label="Fidelity regression comparison table",
                path=_repo_relative(regression_run["comparison_path"], repo_root),
            )
        )

    for tuned_run in tuned_runs:
        subset_label = tuned_run["subgroup"] or "filtered subgroup"
        references.extend(
            [
                ArtifactReference(
                    label=f"Tuned comparison table for {subset_label}",
                    path=_repo_relative(tuned_run["comparison_path"], repo_root),
                ),
                ArtifactReference(
                    label=f"Tuned subset metadata for {subset_label}",
                    path=_repo_relative(tuned_run["run_dir"] / "subset_metadata.json", repo_root),
                ),
            ]
        )

    for shap_run in shap_runs:
        references.extend(
            [
                ArtifactReference(
                    label=f"{shap_run.scope} feature importance",
                    path=shap_run.source_csv,
                ),
                ArtifactReference(
                    label=f"{shap_run.scope} metadata",
                    path=shap_run.source_metadata,
                ),
            ]
        )

    for artifact_path in extra_artifact_paths:
        references.append(
            ArtifactReference(
                label=f"Additional artifact: {artifact_path.name}",
                path=_repo_relative(artifact_path, repo_root),
            )
        )

    deduped: list[ArtifactReference] = []
    seen_pairs: set[tuple[str, str]] = set()
    for reference in references:
        pair = (reference.label, reference.path)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            deduped.append(reference)
    return deduped


def build_milestone_report(
    config: MilestoneReportConfig,
    *,
    config_path: Path,
    repo_root: Path,
) -> MilestoneReport:
    """Build the structured report object without writing files."""

    global_run = None
    if config.artifacts.global_classification_run is not None:
        global_run = _load_classification_run(
            _resolve_config_path(config.artifacts.global_classification_run, repo_root),
            repo_root,
        )

    stratified_run = None
    if config.artifacts.stratified_classification_run is not None:
        stratified_run = _load_stratified_run(
            _resolve_config_path(config.artifacts.stratified_classification_run, repo_root),
            repo_root,
        )

    regression_run = None
    if config.artifacts.regression_run is not None:
        regression_run = _load_regression_run(
            _resolve_config_path(config.artifacts.regression_run, repo_root),
            repo_root,
        )

    tuned_runs = [
        _load_tuned_run(_resolve_config_path(path_value, repo_root), repo_root)
        for path_value in config.artifacts.tuned_runs
    ]
    shap_runs = [
        _load_shap_run(_resolve_config_path(path_value, repo_root), repo_root)
        for path_value in config.artifacts.shap_runs
    ]
    extra_artifact_paths = [
        _resolve_config_path(path_value, repo_root)
        for path_value in config.artifacts.extra_artifact_paths
    ]

    models_compared = list(config.what_was_compared.models_compared)
    feature_sets_compared = list(config.what_was_compared.feature_sets_compared)
    prediction_contexts: list[str] = []
    excluded_feature_columns: list[str] = []
    notes = list(config.what_was_compared.notes)

    for run_payload in [global_run, stratified_run]:
        if run_payload is None:
            continue
        training_config = run_payload["run_config"].get("training", {})
        models_compared.extend(
            _pretty_model_name(str(model_name))
            for model_name in training_config.get("model_names", [])
        )
        feature_set = training_config.get("feature_set_name")
        if feature_set:
            feature_sets_compared.append(str(feature_set))
        prediction_context = training_config.get("prediction_context")
        if prediction_context:
            prediction_contexts.append(str(prediction_context))

        feature_policy = run_payload.get("feature_policy", {})
        if not feature_policy and run_payload.get("feature_policy_by_qubit"):
            first_policy = next(iter(run_payload["feature_policy_by_qubit"].values()), {})
            feature_policy = first_policy if isinstance(first_policy, dict) else {}
        excluded_feature_columns.extend(
            str(column) for column in feature_policy.get("excluded_feature_columns", [])
        )

    if regression_run is not None:
        training_config = regression_run["run_config"].get("training", {})
        models_compared.extend(
            _pretty_model_name(str(model_name))
            for model_name in training_config.get("model_names", [])
        )
        feature_set = training_config.get("feature_set_name")
        if feature_set:
            feature_sets_compared.append(str(feature_set))

    tuning_comparisons = [
        _build_tuning_comparison(tuned_run, stratified_run, repo_root) for tuned_run in tuned_runs
    ]

    return MilestoneReport(
        metadata=ReportMetadata(
            report_title=config.title,
            report_slug=config.report_slug,
            report_timestamp=datetime.now().isoformat(timespec="seconds"),
            run_group_name=config.run_group_name,
            dataset_used=config.dataset_used,
            split_strategy=config.split_strategy,
            experiment_scope=config.experiment_scope,
            subgroup=config.subgroup,
            report_config_path=_repo_relative(config_path.resolve(), repo_root),
        ),
        what_was_compared=ComparisonOverview(
            models_compared=_dedupe_preserve_order(models_compared),
            feature_sets_compared=_dedupe_preserve_order(feature_sets_compared),
            tuning_was_run=bool(tuned_runs),
            shap_was_run=bool(shap_runs),
            prediction_contexts=_dedupe_preserve_order(prediction_contexts),
            excluded_feature_columns=_dedupe_preserve_order(excluded_feature_columns),
            notes=notes,
        ),
        best_raw_results=BestRawResults(
            global_validation_winner=(
                global_run["validation_winner"] if global_run is not None else None
            ),
            global_test_winner=global_run["test_winner"] if global_run is not None else None,
            stratified_validation_winner=(
                stratified_run["validation_winner"] if stratified_run is not None else None
            ),
            stratified_test_winner=(
                stratified_run["test_winner"] if stratified_run is not None else None
            ),
            subgroup_winners=(
                stratified_run["subgroup_winners"] if stratified_run is not None else []
            ),
            regression_reference=regression_run["winner"] if regression_run is not None else None,
            tuning_comparisons=tuning_comparisons,
            shap_highlights=shap_runs,
        ),
        main_scientific_takeaway=MainScientificTakeaway(
            plain_language_conclusion=config.manual_interpretation.plain_language_conclusion,
            scientific_meaning=config.manual_interpretation.scientific_meaning,
            negative_results_to_preserve=config.manual_interpretation.negative_results_to_preserve,
        ),
        important_caveats=config.manual_interpretation.caveats,
        methodological_warnings=config.manual_interpretation.methodological_warnings,
        thesis_framing_recommendation=ThesisFramingRecommendation(
            headline_result=config.manual_interpretation.thesis_framing.headline_result,
            more_trustworthy_result=config.manual_interpretation.thesis_framing.more_trustworthy_result,
            held_out_test_comparator=(
                config.manual_interpretation.thesis_framing.held_out_test_comparator
            ),
            presentation_notes=config.manual_interpretation.thesis_framing.presentation_notes,
        ),
        recommended_next_steps=config.manual_interpretation.recommended_next_steps,
        thesis_reuse_sentences=config.manual_interpretation.thesis_reuse_sentences,
        artifact_references=_collect_artifact_references(
            global_run=global_run,
            stratified_run=stratified_run,
            regression_run=regression_run,
            tuned_runs=tuned_runs,
            shap_runs=shap_runs,
            extra_artifact_paths=[path for path in extra_artifact_paths if path is not None],
            repo_root=repo_root,
        ),
    )


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_optional_line(label: str, value: str | None) -> str:
    return f"- {label}: {value}" if value else ""


def _render_metadata_section(report: MilestoneReport) -> str:
    lines = [
        f"- Report timestamp: {report.metadata.report_timestamp}",
        f"- Dataset used: {report.metadata.dataset_used or 'not recorded'}",
        f"- Split strategy: {report.metadata.split_strategy}",
        f"- Experiment scope: {report.metadata.experiment_scope}",
        f"- Subgroup: {report.metadata.subgroup or 'not subgroup-specific'}",
    ]
    if report.metadata.run_group_name:
        lines.append(f"- Run group: {report.metadata.run_group_name}")
    lines.append(f"- Report config: `{report.metadata.report_config_path}`")
    return "\n".join(lines)


def _render_what_was_compared(report: MilestoneReport) -> str:
    models_text = ", ".join(report.what_was_compared.models_compared) or "not recorded"
    feature_sets_text = ", ".join(report.what_was_compared.feature_sets_compared) or "not recorded"
    lines = [
        f"- Models compared: {models_text}",
        f"- Feature sets compared: {feature_sets_text}",
        f"- Tuning was run: {'yes' if report.what_was_compared.tuning_was_run else 'no'}",
        f"- SHAP was run: {'yes' if report.what_was_compared.shap_was_run else 'no'}",
    ]
    if report.what_was_compared.prediction_contexts:
        lines.append(
            "- Prediction contexts: "
            + ", ".join(report.what_was_compared.prediction_contexts)
        )
    if report.what_was_compared.excluded_feature_columns:
        lines.append(
            "- Excluded feature columns: "
            + ", ".join(report.what_was_compared.excluded_feature_columns)
        )
    lines.extend(f"- {note}" for note in report.what_was_compared.notes)
    return "\n".join(lines)


def _render_classification_winner(
    label: str,
    winner: ClassificationResultSummary | None,
) -> str:
    if winner is None:
        return f"- {label}: not included"
    subgroup = f", subgroup {winner.subgroup}" if winner.subgroup else ""
    return (
        f"- {label}: {winner.model_display_name}{subgroup} "
        f"(validation macro-F1 {_format_float(winner.validation_macro_f1)}, "
        f"test macro-F1 {_format_float(winner.test_macro_f1)}, "
        f"source `{winner.source_table}`)"
    )


def _render_regression_reference(reference: RegressionResultSummary | None) -> str:
    if reference is None:
        return "- Regression reference: not included"
    return (
        f"- Regression reference: {reference.model_display_name} "
        f"(validation R2 {_format_float(reference.validation_r2)}, "
        f"test R2 {_format_float(reference.test_r2)}, "
        f"test MAE {_format_float(reference.test_mae)}, "
        f"source `{reference.source_table}`)"
    )


def _render_subgroup_winners(winners: list[ClassificationResultSummary]) -> str:
    if not winners:
        return "_No subgroup winners recorded._"
    lines = [
        "| Subgroup | Model | Validation macro-F1 | Test macro-F1 |",
        "| --- | --- | ---: | ---: |",
    ]
    for winner in winners:
        lines.append(
            f"| {winner.subgroup or 'n/a'} | {winner.model_display_name} | "
            f"{_format_float(winner.validation_macro_f1)} | {_format_float(winner.test_macro_f1)} |"
        )
    return "\n".join(lines)


def _render_tuning_comparisons(comparisons: list[TuningComparison]) -> str:
    if not comparisons:
        return "_No tuning runs recorded._"
    lines = []
    for comparison in comparisons:
        lines.append(
            f"- {comparison.subgroup}: untuned test macro-F1 "
            f"{_format_float(comparison.untuned_test_macro_f1)} vs tuned "
            f"{_format_float(comparison.tuned_test_macro_f1)}; "
            f"validation delta {_format_float(comparison.validation_delta)}, "
            f"test delta {_format_float(comparison.test_delta)}. {comparison.interpretation}"
        )
    return "\n".join(lines)


def _render_shap_highlights(shap_runs: list[ShapRunSummary]) -> str:
    if not shap_runs:
        return "_No SHAP runs recorded._"
    lines = []
    for shap_run in shap_runs:
        top_features = ", ".join(
            f"{feature.feature} ({_format_float(feature.mean_abs_shap, digits=3)})"
            for feature in shap_run.top_features
        )
        lines.append(
            f"- {shap_run.scope}: {shap_run.selected_model_display_name} on the "
            f"{shap_run.explained_split} split. Top features: {top_features}. "
            f"Source `{shap_run.source_csv}`"
        )
    return "\n".join(lines)


def _render_best_raw_results(report: MilestoneReport) -> str:
    sections = [
        _render_classification_winner(
            "Best global validation result", report.best_raw_results.global_validation_winner
        ),
        _render_classification_winner(
            "Best global held-out test result", report.best_raw_results.global_test_winner
        ),
        _render_classification_winner(
            "Best stratified validation result",
            report.best_raw_results.stratified_validation_winner,
        ),
        _render_classification_winner(
            "Best stratified held-out test result", report.best_raw_results.stratified_test_winner
        ),
        _render_regression_reference(report.best_raw_results.regression_reference),
        "",
        "Subgroup winners",
        _render_subgroup_winners(report.best_raw_results.subgroup_winners),
        "",
        "Tuning comparisons",
        _render_tuning_comparisons(report.best_raw_results.tuning_comparisons),
        "",
        "SHAP highlights",
        _render_shap_highlights(report.best_raw_results.shap_highlights),
    ]
    return "\n".join(sections)


def _render_list_or_placeholder(values: list[str], placeholder: str) -> str:
    if not values:
        return placeholder
    return "\n".join(f"- {value}" for value in values)


def _render_takeaway_section(report: MilestoneReport) -> str:
    lines = [
        f"- Plain-language conclusion: {report.main_scientific_takeaway.plain_language_conclusion}",
        f"- Scientific meaning: {report.main_scientific_takeaway.scientific_meaning}",
    ]
    if report.main_scientific_takeaway.negative_results_to_preserve:
        lines.append("")
        lines.append("Negative results to preserve")
        lines.extend(
            f"- {value}"
            for value in report.main_scientific_takeaway.negative_results_to_preserve
        )
    return "\n".join(lines)


def _render_thesis_framing(report: MilestoneReport) -> str:
    lines = [
        f"- Recommended headline result: {report.thesis_framing_recommendation.headline_result}",
        "- More trustworthy result: "
        f"{report.thesis_framing_recommendation.more_trustworthy_result}",
    ]
    comparator_line = _format_optional_line(
        "Held-out test comparator", report.thesis_framing_recommendation.held_out_test_comparator
    )
    if comparator_line:
        lines.append(comparator_line)
    lines.extend(
        f"- {note}" for note in report.thesis_framing_recommendation.presentation_notes
    )
    return "\n".join(lines)


def _render_artifact_references(report: MilestoneReport) -> str:
    return "\n".join(
        f"- {reference.label}: `{reference.path}`" for reference in report.artifact_references
    )


def render_markdown_report(report: MilestoneReport) -> str:
    """Render the structured report into the durable Markdown template."""

    template = Template(REPORT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        title=report.metadata.report_title,
        metadata_section=_render_metadata_section(report),
        what_was_compared_section=_render_what_was_compared(report),
        best_raw_results_section=_render_best_raw_results(report),
        scientific_takeaway_section=_render_takeaway_section(report),
        caveats_section=_render_list_or_placeholder(
            report.important_caveats,
            "_No caveats were recorded._",
        ),
        warnings_section=_render_list_or_placeholder(
            report.methodological_warnings,
            "_No additional methodological warnings were recorded._",
        ),
        thesis_framing_section=_render_thesis_framing(report),
        next_steps_section=_render_list_or_placeholder(
            report.recommended_next_steps,
            "_No next steps were recorded._",
        ),
        thesis_reuse_section=_render_list_or_placeholder(
            report.thesis_reuse_sentences,
            "_No reusable thesis sentences were recorded yet._",
        ),
        artifact_references_section=_render_artifact_references(report),
    )


def generate_milestone_report(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> GeneratedReportArtifacts:
    """Build and write the Markdown report, JSON summary, and JSON schema."""

    config_file = Path(config_path).resolve()
    resolved_repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    config = load_milestone_report_config(config_file)
    report = build_milestone_report(config, config_path=config_file, repo_root=resolved_repo_root)
    output_dir = _resolve_config_path(config.output_dir, resolved_repo_root)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"{config.report_slug}.md"
    json_path = output_dir / f"{config.report_slug}.json"
    schema_path = output_dir / f"{config.report_slug}.schema.json"

    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(MilestoneReport.model_json_schema(), indent=2),
        encoding="utf-8",
    )

    return GeneratedReportArtifacts(
        markdown_path=markdown_path,
        json_path=json_path,
        schema_path=schema_path,
    )
