"""Build a master experiment matrix across saved benchmark runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.io import load_config
from src.evaluation.metrics import save_json_report
from src.models.model_suite import get_display_name

SUMMARY_FILENAME = "master_experiment_matrix.csv"
INVENTORY_FILENAME = "master_experiment_inventory.json"


def _relative_path(path: Path) -> str:
    root = Path.cwd().resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Experiment summary source table is empty: {path}")
    return frame


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _base_row(
    *,
    family: str,
    run_root: Path,
    run_dir: Path,
    source_table: Path,
    config,
    target_kind: str,
    selection_scope: str,
    tuning_applied: bool,
    subset_column: str | None = None,
    subset_value: str | int | None = None,
) -> dict[str, Any]:
    return {
        "experiment_family": family,
        "run_root": run_root.name,
        "run_name": run_dir.name,
        "run_directory": _relative_path(run_dir),
        "dataset_path": config.data.dataset_path.as_posix(),
        "dataset_name": config.data.dataset_path.name,
        "cleaned_dataset_path": config.data.cleaned_dataset_path.as_posix(),
        "feature_set_name": config.training.feature_set_name,
        "prediction_context": getattr(config.training, "prediction_context", None),
        "target_kind": target_kind,
        "selection_scope": selection_scope,
        "subset_column": subset_column,
        "subset_value": subset_value,
        "tuning_applied": tuning_applied,
        "source_table": _relative_path(source_table),
    }


def _scan_model_benchmark_runs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_root.exists():
        return rows

    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        comparison_path = run_dir / "model_comparison.csv"
        run_config_path = run_dir / "run_config.yaml"
        if not comparison_path.exists() or not run_config_path.exists():
            continue

        config = load_config(run_config_path)
        comparison_frame = _read_csv(comparison_path)
        has_validation_metrics = {
            "validation_macro_f1",
            "validation_accuracy",
        }.issubset(comparison_frame.columns)
        best_validation_index = (
            comparison_frame.sort_values(
                by=["validation_macro_f1", "validation_accuracy"],
                ascending=[False, False],
            ).index[0]
            if has_validation_metrics
            else None
        )
        if {"test_macro_f1", "test_accuracy"}.issubset(comparison_frame.columns):
            best_test_index = comparison_frame.sort_values(
                by=["test_macro_f1", "test_accuracy"],
                ascending=[False, False],
            ).index[0]
        else:
            best_test_index = comparison_frame.sort_values(
                by=["macro_f1", "accuracy"],
                ascending=[False, False],
            ).index[0]

        for row_index, row in comparison_frame.iterrows():
            validation_feature_count = (
                int(row["validation_feature_columns_before_encoding"])
                if "validation_feature_columns_before_encoding" in comparison_frame.columns
                else None
            )
            test_feature_count = (
                int(row["test_feature_columns_before_encoding"])
                if "test_feature_columns_before_encoding" in comparison_frame.columns
                else (
                    int(row["feature_columns_before_encoding"])
                    if "feature_columns_before_encoding" in comparison_frame.columns
                    else None
                )
            )
            rows.append(
                {
                    **_base_row(
                        family="classification_global",
                        run_root=run_root,
                        run_dir=run_dir,
                        source_table=comparison_path,
                        config=config,
                        target_kind="fault_type_classification",
                        selection_scope="run",
                        tuning_applied=False,
                    ),
                    "model_name": str(row["model_name"]),
                    "model_display_name": str(row["model_display_name"]),
                    "validation_macro_f1": (
                        float(row["validation_macro_f1"])
                        if has_validation_metrics
                        else None
                    ),
                    "validation_accuracy": (
                        float(row["validation_accuracy"])
                        if has_validation_metrics
                        else None
                    ),
                    "test_macro_f1": (
                        float(row["test_macro_f1"])
                        if "test_macro_f1" in comparison_frame.columns
                        else float(row["macro_f1"])
                    ),
                    "test_accuracy": (
                        float(row["test_accuracy"])
                        if "test_accuracy" in comparison_frame.columns
                        else float(row["accuracy"])
                    ),
                    "validation_r2": None,
                    "test_r2": None,
                    "validation_mae": None,
                    "test_mae": None,
                    "validation_feature_columns_before_encoding": validation_feature_count,
                    "test_feature_columns_before_encoding": test_feature_count,
                    "best_params": None,
                    "is_best_validation_in_scope": (
                        row_index == best_validation_index
                        if best_validation_index is not None
                        else False
                    ),
                    "is_best_test_in_scope": row_index == best_test_index,
                }
            )
    return rows


def _scan_qubit_stratified_runs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_root.exists():
        return rows

    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        comparison_path = run_dir / "qubit_model_comparison.csv"
        run_config_path = run_dir / "run_config.yaml"
        if not comparison_path.exists() or not run_config_path.exists():
            continue

        config = load_config(run_config_path)
        comparison_frame = _read_csv(comparison_path)
        best_validation_indices = set(
            comparison_frame.sort_values(
                by=["qubit_count", "validation_macro_f1", "validation_accuracy"],
                ascending=[True, False, False],
            )
            .groupby("qubit_count")
            .head(1)
            .index
            .tolist()
        )
        best_test_indices = set(
            comparison_frame.sort_values(
                by=["qubit_count", "test_macro_f1", "test_accuracy"],
                ascending=[True, False, False],
            )
            .groupby("qubit_count")
            .head(1)
            .index
            .tolist()
        )

        for row_index, row in comparison_frame.iterrows():
            qubit_count = int(row["qubit_count"])
            rows.append(
                {
                    **_base_row(
                        family="classification_stratified",
                        run_root=run_root,
                        run_dir=run_dir,
                        source_table=comparison_path,
                        config=config,
                        target_kind="fault_type_classification",
                        selection_scope="qubit_count subgroup",
                        tuning_applied=False,
                        subset_column="qubit_count",
                        subset_value=qubit_count,
                    ),
                    "model_name": str(row["model_name"]),
                    "model_display_name": str(row["model_display_name"]),
                    "validation_macro_f1": float(row["validation_macro_f1"]),
                    "validation_accuracy": float(row["validation_accuracy"]),
                    "test_macro_f1": float(row["test_macro_f1"]),
                    "test_accuracy": float(row["test_accuracy"]),
                    "validation_r2": None,
                    "test_r2": None,
                    "validation_mae": None,
                    "test_mae": None,
                    "validation_feature_columns_before_encoding": int(
                        row["validation_feature_columns_before_encoding"]
                    ),
                    "test_feature_columns_before_encoding": int(
                        row["test_feature_columns_before_encoding"]
                    ),
                    "best_params": None,
                    "is_best_validation_in_scope": row_index in best_validation_indices,
                    "is_best_test_in_scope": row_index in best_test_indices,
                }
            )
    return rows


def _scan_regression_runs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_root.exists():
        return rows

    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        comparison_path = run_dir / "model_comparison.csv"
        run_config_path = run_dir / "run_config.yaml"
        if not comparison_path.exists() or not run_config_path.exists():
            continue

        config = load_config(run_config_path)
        comparison_frame = _read_csv(comparison_path)
        best_validation_index = comparison_frame.sort_values(
            by=["validation_r2", "validation_mae"],
            ascending=[False, True],
        ).index[0]
        best_test_index = comparison_frame.sort_values(
            by=["test_r2", "test_mae"],
            ascending=[False, True],
        ).index[0]

        for row_index, row in comparison_frame.iterrows():
            rows.append(
                {
                    **_base_row(
                        family="fidelity_regression",
                        run_root=run_root,
                        run_dir=run_dir,
                        source_table=comparison_path,
                        config=config,
                        target_kind="fidelity_regression",
                        selection_scope="run",
                        tuning_applied=False,
                    ),
                    "model_name": str(row["model_name"]),
                    "model_display_name": str(row["model_display_name"]),
                    "validation_macro_f1": None,
                    "validation_accuracy": None,
                    "test_macro_f1": None,
                    "test_accuracy": None,
                    "validation_r2": float(row["validation_r2"]),
                    "test_r2": float(row["test_r2"]),
                    "validation_mae": float(row["validation_mae"]),
                    "test_mae": float(row["test_mae"]),
                    "validation_feature_columns_before_encoding": int(
                        row["validation_feature_columns_before_encoding"]
                    ),
                    "test_feature_columns_before_encoding": int(
                        row["test_feature_columns_before_encoding"]
                    ),
                    "best_params": None,
                    "is_best_validation_in_scope": row_index == best_validation_index,
                    "is_best_test_in_scope": row_index == best_test_index,
                }
            )
    return rows


def _scan_tuned_runs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_root.exists():
        return rows

    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        comparison_path = run_dir / "tuned_model_comparison.csv"
        subset_path = run_dir / "subset_metadata.json"
        run_config_path = run_dir / "run_config.yaml"
        if not comparison_path.exists() or not subset_path.exists() or not run_config_path.exists():
            continue

        config = load_config(run_config_path)
        comparison_frame = _read_csv(comparison_path)
        subset_metadata = pd.read_json(subset_path, typ="series")
        subset_column = str(subset_metadata["filter_column"])
        subset_value = _safe_value(subset_metadata["filter_value"])
        best_validation_index = comparison_frame.sort_values(
            by=["best_validation_macro_f1", "test_accuracy"],
            ascending=[False, False],
        ).index[0]
        best_test_index = comparison_frame.sort_values(
            by=["test_macro_f1", "test_accuracy"],
            ascending=[False, False],
        ).index[0]

        for row_index, row in comparison_frame.iterrows():
            rows.append(
                {
                    **_base_row(
                        family="classification_tuned",
                        run_root=run_root,
                        run_dir=run_dir,
                        source_table=comparison_path,
                        config=config,
                        target_kind="fault_type_classification",
                        selection_scope="run",
                        tuning_applied=True,
                        subset_column=subset_column,
                        subset_value=subset_value,
                    ),
                    "model_name": str(row["model_name"]),
                    "model_display_name": str(row["model_display_name"]),
                    "validation_macro_f1": float(row["best_validation_macro_f1"]),
                    "validation_accuracy": None,
                    "test_macro_f1": float(row["test_macro_f1"]),
                    "test_accuracy": float(row["test_accuracy"]),
                    "validation_r2": None,
                    "test_r2": None,
                    "validation_mae": None,
                    "test_mae": None,
                    "validation_feature_columns_before_encoding": None,
                    "test_feature_columns_before_encoding": None,
                    "best_params": _safe_value(row.get("best_params")),
                    "is_best_validation_in_scope": row_index == best_validation_index,
                    "is_best_test_in_scope": row_index == best_test_index,
                }
            )
    return rows


def _scan_rf_baseline_runs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_root.exists():
        return rows

    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.json"
        run_config_path = run_dir / "run_config.yaml"
        if not metrics_path.exists() or not run_config_path.exists():
            continue

        config = load_config(run_config_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_name = (
            config.training.model_names[0]
            if config.training.model_names
            else "random_forest"
        )
        legacy_test_metrics = metrics
        validation_metrics = metrics.get("validation_metrics", {})
        test_metrics = metrics.get("test_metrics", legacy_test_metrics)
        if not isinstance(validation_metrics, dict):
            validation_metrics = {}
        if not isinstance(test_metrics, dict):
            test_metrics = {}

        rows.append(
            {
                **_base_row(
                    family="rf_baseline_single_model",
                    run_root=run_root,
                    run_dir=run_dir,
                    source_table=metrics_path,
                    config=config,
                    target_kind="fault_type_classification",
                    selection_scope="run",
                    tuning_applied=False,
                ),
                "model_name": str(metrics.get("model_name", model_name)),
                "model_display_name": str(
                    metrics.get("model_display_name", get_display_name(model_name))
                ),
                "validation_macro_f1": _safe_value(validation_metrics.get("macro_f1")),
                "validation_accuracy": _safe_value(validation_metrics.get("accuracy")),
                "test_macro_f1": _safe_value(test_metrics.get("macro_f1")),
                "test_accuracy": _safe_value(test_metrics.get("accuracy")),
                "validation_r2": None,
                "test_r2": None,
                "validation_mae": None,
                "test_mae": None,
                "validation_feature_columns_before_encoding": _safe_value(
                    validation_metrics.get("feature_columns_before_encoding")
                ),
                "test_feature_columns_before_encoding": _safe_value(
                    test_metrics.get(
                        "feature_columns_before_encoding",
                        metrics.get("feature_columns_before_encoding"),
                    )
                ),
                "best_params": None,
                "is_best_validation_in_scope": bool(validation_metrics),
                "is_best_test_in_scope": True,
            }
        )
    return rows


def build_experiment_matrix(experiments_root: str | Path = Path("experiments")) -> pd.DataFrame:
    """Collect standardized experiment rows across saved run directories."""

    root = Path(experiments_root)
    rows = [
        *_scan_rf_baseline_runs(root / "rf_baseline"),
        *_scan_model_benchmark_runs(root / "model_benchmark"),
        *_scan_qubit_stratified_runs(root / "qubit_stratified"),
        *_scan_regression_runs(root / "fidelity_regression"),
        *_scan_tuned_runs(root / "tuned_classification"),
    ]
    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "dataset_name",
                "experiment_family",
                "run_name",
                "subset_value",
                "model_display_name",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_experiment_inventory(
    matrix: pd.DataFrame,
    *,
    experiments_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a compact JSON inventory for the master experiment matrix."""

    if matrix.empty:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "experiments_root": Path(experiments_root).as_posix(),
            "output_dir": Path(output_dir).as_posix(),
            "total_rows": 0,
            "total_runs": 0,
            "run_counts_by_family": {},
            "datasets": [],
            "feature_sets": [],
            "prediction_contexts": [],
        }

    run_counts_by_family = {
        family: int(count)
        for family, count in matrix.groupby("experiment_family")["run_directory"].nunique().items()
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiments_root": Path(experiments_root).as_posix(),
        "output_dir": Path(output_dir).as_posix(),
        "total_rows": int(len(matrix)),
        "total_runs": int(matrix["run_directory"].nunique()),
        "run_counts_by_family": run_counts_by_family,
        "datasets": sorted(matrix["dataset_name"].dropna().astype(str).unique().tolist()),
        "feature_sets": sorted(matrix["feature_set_name"].dropna().astype(str).unique().tolist()),
        "prediction_contexts": sorted(
            matrix["prediction_context"].dropna().astype(str).unique().tolist()
        ),
    }


def write_experiment_summary(
    *,
    experiments_root: str | Path = Path("experiments"),
    output_dir: str | Path = Path("reports/experiments"),
) -> tuple[Path, Path]:
    """Write the master experiment matrix CSV and inventory JSON."""

    matrix = build_experiment_matrix(experiments_root)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolved_output_dir / SUMMARY_FILENAME
    json_path = resolved_output_dir / INVENTORY_FILENAME
    matrix.to_csv(csv_path, index=False)
    save_json_report(
        build_experiment_inventory(
            matrix,
            experiments_root=experiments_root,
            output_dir=output_dir,
        ),
        json_path,
    )
    return csv_path, json_path
