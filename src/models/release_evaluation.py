"""Evaluation helpers for grouped thesis regression experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data.dataset import validate_required_columns
from src.evaluation.metrics import save_json_report
from src.visualization.plots import plot_feature_importance, plot_shap_summary

DEFAULT_SLICE_COLUMNS = ["family", "qubit_count", "compiler_variant", "difficulty_bucket"]


@dataclass(slots=True)
class DifficultyBucketer:
    """Store train-derived difficulty bucket boundaries."""

    reference_column: str
    labels: list[str]
    edges: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_column": self.reference_column,
            "labels": self.labels,
            "edges": self.edges,
        }


@dataclass(slots=True)
class ReleaseShapArtifacts:
    """Container for SHAP outputs derived from one saved regression model."""

    source_importance_frame: pd.DataFrame
    transformed_importance_frame: pd.DataFrame
    transformed_feature_frame: pd.DataFrame
    shap_values_frame: pd.DataFrame
    metadata: dict[str, object]
    shap_values: np.ndarray


def fit_difficulty_bucketer(
    train_frame: pd.DataFrame,
    *,
    reference_column: str,
    bucket_count: int,
) -> DifficultyBucketer:
    """Fit difficulty buckets from training data only."""

    validate_required_columns(train_frame, [reference_column])
    values = pd.to_numeric(train_frame[reference_column], errors="coerce")
    if values.isna().any():
        missing_count = int(values.isna().sum())
        raise ValueError(
            f"Difficulty reference column {reference_column!r} contains {missing_count} missing "
            "or non-numeric values."
        )

    quantile_positions = np.linspace(0.0, 1.0, bucket_count + 1)
    quantile_edges = values.quantile(quantile_positions).to_numpy(dtype=float).copy()
    quantile_edges[0] = float("-inf")
    quantile_edges[-1] = float("inf")
    for index in range(1, len(quantile_edges)):
        if quantile_edges[index] <= quantile_edges[index - 1]:
            quantile_edges[index] = quantile_edges[index - 1] + 1e-9

    labels = [f"q{index + 1}" for index in range(bucket_count)]
    return DifficultyBucketer(
        reference_column=reference_column,
        labels=labels,
        edges=[float(edge) for edge in quantile_edges],
    )


def apply_difficulty_bucketer(
    frame: pd.DataFrame,
    bucketer: DifficultyBucketer,
    *,
    output_column: str = "difficulty_bucket",
) -> pd.DataFrame:
    """Assign train-derived difficulty buckets to a frame."""

    validate_required_columns(frame, [bucketer.reference_column])
    difficulty_values = pd.to_numeric(frame[bucketer.reference_column], errors="coerce")
    if difficulty_values.isna().any():
        missing_count = int(difficulty_values.isna().sum())
        raise ValueError(
            f"Difficulty reference column {bucketer.reference_column!r} contains {missing_count} "
            "missing or non-numeric values."
        )

    table = frame.copy()
    table[output_column] = pd.cut(
        difficulty_values,
        bins=bucketer.edges,
        labels=bucketer.labels,
        include_lowest=True,
        duplicates="drop",
    ).astype(str)
    return table


def build_prediction_frame(
    source_frame: pd.DataFrame,
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    split_name: str,
    ablation_name: str,
    model_name: str,
    target_column: str,
) -> pd.DataFrame:
    """Build one artifact-friendly frame that combines metadata and predictions."""

    prediction_frame = source_frame.copy()
    prediction_frame["split_name"] = split_name
    prediction_frame["ablation_name"] = ablation_name
    prediction_frame["model_name"] = model_name
    prediction_frame["target_column"] = target_column
    prediction_frame["target_value"] = y_true.astype(float).to_numpy()
    prediction_frame["predicted_value"] = y_pred.astype(float).to_numpy()
    prediction_frame["residual"] = (
        prediction_frame["predicted_value"] - prediction_frame["target_value"]
    )
    prediction_frame["absolute_error"] = prediction_frame["residual"].abs()
    prediction_frame["squared_error"] = prediction_frame["residual"] ** 2
    return prediction_frame


def _safe_regression_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    y_true = frame["target_value"].astype(float)
    y_pred = frame["predicted_value"].astype(float)
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    if len(frame) < 2:
        r2 = None
    else:
        r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": r2,
    }


def compute_slice_metrics(
    prediction_frame: pd.DataFrame,
    *,
    slice_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute global and sliced regression metrics with per-slice sample counts."""

    validate_required_columns(
        prediction_frame,
        ["split_name", "ablation_name", "model_name", "target_value", "predicted_value"],
    )
    requested_slice_columns = slice_columns or DEFAULT_SLICE_COLUMNS
    available_slice_columns = [
        column for column in requested_slice_columns if column in prediction_frame.columns
    ]

    rows: list[dict[str, object]] = []
    global_metrics = _safe_regression_metrics(prediction_frame)
    rows.append(
        {
            "split_name": prediction_frame["split_name"].iloc[0],
            "ablation_name": prediction_frame["ablation_name"].iloc[0],
            "model_name": prediction_frame["model_name"].iloc[0],
            "slice_name": "overall",
            "slice_value": "overall",
            "sample_count": int(len(prediction_frame)),
            **global_metrics,
        }
    )

    for slice_column in available_slice_columns:
        grouped = prediction_frame.groupby(slice_column, dropna=False, sort=True)
        for slice_value, group_frame in grouped:
            rows.append(
                {
                    "split_name": prediction_frame["split_name"].iloc[0],
                    "ablation_name": prediction_frame["ablation_name"].iloc[0],
                    "model_name": prediction_frame["model_name"].iloc[0],
                    "slice_name": slice_column,
                    "slice_value": str(slice_value),
                    "sample_count": int(len(group_frame)),
                    **_safe_regression_metrics(group_frame),
                }
            )

    return pd.DataFrame(rows)


def compute_family_dominance_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Show which families dominate the error budget."""

    validate_required_columns(prediction_frame, ["family", "absolute_error", "squared_error"])
    grouped = prediction_frame.groupby("family", dropna=False, sort=True)
    summary = grouped.agg(
        sample_count=("family", "size"),
        total_absolute_error=("absolute_error", "sum"),
        total_squared_error=("squared_error", "sum"),
        mean_absolute_error=("absolute_error", "mean"),
    ).reset_index()
    summary["sample_fraction"] = summary["sample_count"] / float(summary["sample_count"].sum())
    summary["absolute_error_fraction"] = summary["total_absolute_error"] / float(
        summary["total_absolute_error"].sum()
    )
    summary["squared_error_fraction"] = summary["total_squared_error"] / float(
        summary["total_squared_error"].sum()
    )
    return summary.sort_values(
        by=["squared_error_fraction", "sample_fraction"],
        ascending=[False, False],
    ).reset_index(drop=True)


def compute_variant_gap_diagnostic(prediction_frame: pd.DataFrame) -> dict[str, object]:
    """Compare model quality between raw and transpiled rows."""

    if "compiler_variant" not in prediction_frame.columns:
        return {"available": False}

    variant_metrics = []
    for variant_name, group_frame in prediction_frame.groupby("compiler_variant", dropna=False):
        variant_metrics.append(
            {
                "compiler_variant": str(variant_name),
                **_safe_regression_metrics(group_frame),
            }
        )
    variant_frame = pd.DataFrame(variant_metrics)
    if variant_frame.shape[0] < 2:
        return {"available": True, "variant_metrics": variant_frame.to_dict(orient="records")}

    raw_row = variant_frame.loc[variant_frame["compiler_variant"] == "raw"]
    transpiled_row = variant_frame.loc[variant_frame["compiler_variant"] == "transpiled"]
    if raw_row.empty or transpiled_row.empty:
        return {"available": True, "variant_metrics": variant_frame.to_dict(orient="records")}

    raw_metrics = raw_row.iloc[0]
    transpiled_metrics = transpiled_row.iloc[0]
    delta_mae = float(transpiled_metrics["mae"] - raw_metrics["mae"])
    raw_r2 = raw_metrics["r2"] if pd.notna(raw_metrics["r2"]) else 0.0
    transpiled_r2 = transpiled_metrics["r2"] if pd.notna(transpiled_metrics["r2"]) else 0.0
    delta_r2 = float(transpiled_r2 - raw_r2)
    return {
        "available": True,
        "variant_metrics": variant_frame.to_dict(orient="records"),
        "delta_mae_transpiled_minus_raw": delta_mae,
        "delta_r2_transpiled_minus_raw": delta_r2,
        "sharp_difference_flag": abs(delta_mae) >= 0.02 or abs(delta_r2) >= 0.05,
    }


def compute_worst_slice_frame(
    slice_metrics_frame: pd.DataFrame,
    *,
    slice_names: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the worst slices by MAE and R2 for thesis discussion."""

    filtered = slice_metrics_frame.loc[
        slice_metrics_frame["slice_name"].isin(slice_names)
        & (slice_metrics_frame["slice_name"] != "overall")
    ].copy()
    if filtered.empty:
        return filtered

    filtered["r2_sort"] = filtered["r2"].fillna(float("-inf"))
    return (
        filtered.sort_values(by=["mae", "r2_sort"], ascending=[False, True])
        .head(top_n)
        .drop(columns=["r2_sort"])
        .reset_index(drop=True)
    )


def compute_permutation_importance_frame(
    pipeline,
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    *,
    max_rows: int,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Compute permutation importance on a bounded evaluation sample."""

    if len(X_evaluation) > max_rows:
        X_sample = X_evaluation.sample(n=max_rows, random_state=random_state)
        y_sample = y_evaluation.loc[X_sample.index]
    else:
        X_sample = X_evaluation
        y_sample = y_evaluation

    importance = permutation_importance(
        pipeline,
        X_sample,
        y_sample,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    if isinstance(X_sample, pd.DataFrame):
        feature_names = X_sample.columns.astype(str).tolist()
    else:
        feature_names = [f"feature_{index}" for index in range(len(importance.importances_mean))]

    if len(feature_names) != len(importance.importances_mean):
        transformed_feature_names = (
            pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        )
        if len(transformed_feature_names) == len(importance.importances_mean):
            feature_names = transformed_feature_names
        else:
            feature_names = [
                f"feature_{index}" for index in range(len(importance.importances_mean))
            ]

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    return frame.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)


def _sample_feature_frame(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if len(frame) > max_rows:
        return frame.sample(n=max_rows, random_state=random_state).copy()
    return frame.copy()


def _build_release_shap_explainer(
    pipeline: Pipeline,
    model_name: str,
    background_transformed: np.ndarray,
    transformed_feature_names: np.ndarray,
):
    import shap

    model = pipeline.named_steps["model"]
    if model_name in {"random_forest_regressor", "xgboost_regressor"}:
        return shap.TreeExplainer(model, feature_names=transformed_feature_names.tolist())
    if model_name == "ridge_regression":
        return shap.LinearExplainer(
            model,
            background_transformed,
            feature_names=transformed_feature_names.tolist(),
        )
    raise ValueError(
        "SHAP analysis is implemented for Ridge, Random Forest, and XGBoost regression, "
        f"not {model_name}."
    )


def _mean_absolute_shap(values: np.ndarray) -> np.ndarray:
    absolute_values = np.abs(values)
    if absolute_values.ndim == 2:
        return absolute_values.mean(axis=0)
    if absolute_values.ndim == 3:
        return absolute_values.mean(axis=(0, 2))
    raise ValueError(f"Unsupported SHAP value shape: {absolute_values.shape}")


def _build_transformed_source_mapping(
    preprocessor: ColumnTransformer,
    transformed_feature_names: list[str],
) -> dict[str, str]:
    mapping = {feature_name: feature_name for feature_name in transformed_feature_names}

    transformer_columns = {
        name: [str(column) for column in columns]
        for name, _, columns in preprocessor.transformers_
        if name != "remainder"
    }
    numeric_columns = transformer_columns.get("numeric", [])
    categorical_columns = transformer_columns.get("categorical", [])

    for feature_name in transformed_feature_names:
        if feature_name in numeric_columns:
            mapping[feature_name] = feature_name
            continue

        matched_column = next(
            (
                column
                for column in categorical_columns
                if feature_name == column or feature_name.startswith(f"{column}_")
            ),
            None,
        )
        if matched_column is not None:
            mapping[feature_name] = matched_column

    return mapping


def compute_release_shap_artifacts(
    pipeline: Pipeline,
    *,
    model_name: str,
    X_background: pd.DataFrame,
    X_explained: pd.DataFrame,
    explained_split: str,
    max_rows: int,
    background_max_rows: int,
    random_state: int,
) -> ReleaseShapArtifacts:
    """Compute SHAP artifacts on a bounded evaluation sample."""

    background_sample = _sample_feature_frame(
        X_background,
        max_rows=background_max_rows,
        random_state=random_state,
    )
    explained_sample = _sample_feature_frame(
        X_explained,
        max_rows=max_rows,
        random_state=random_state + 1,
    )

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    transformed_feature_names = preprocessor.get_feature_names_out().astype(str)
    background_transformed = np.asarray(preprocessor.transform(background_sample), dtype=float)
    explained_transformed = np.asarray(preprocessor.transform(explained_sample), dtype=float)

    explainer = _build_release_shap_explainer(
        pipeline,
        model_name,
        background_transformed,
        transformed_feature_names,
    )
    explanation = explainer(explained_transformed)
    shap_values = np.asarray(explanation.values, dtype=float)
    mean_abs_transformed = _mean_absolute_shap(shap_values)

    source_mapping = _build_transformed_source_mapping(
        preprocessor,
        transformed_feature_names.tolist(),
    )
    transformed_importance_frame = (
        pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "source_feature": [
                    source_mapping.get(feature_name, feature_name)
                    for feature_name in transformed_feature_names
                ],
                "mean_abs_shap": mean_abs_transformed,
            }
        )
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    source_importance_frame = (
        transformed_importance_frame.groupby("source_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .rename(columns={"source_feature": "feature"})
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    transformed_feature_frame = (
        pd.DataFrame(
            explained_transformed,
            columns=transformed_feature_names,
            index=explained_sample.index,
        )
        .reset_index()
        .rename(columns={"index": "source_row_index"})
    )
    shap_values_frame = (
        pd.DataFrame(
            shap_values,
            columns=transformed_feature_names,
            index=explained_sample.index,
        )
        .reset_index()
        .rename(columns={"index": "source_row_index"})
    )
    metadata = {
        "available": True,
        "model_name": model_name,
        "explained_split": explained_split,
        "explained_rows": int(len(explained_sample)),
        "background_rows": int(len(background_sample)),
        "transformed_feature_count": int(len(transformed_feature_names)),
        "source_feature_count": int(len(source_importance_frame)),
    }
    return ReleaseShapArtifacts(
        source_importance_frame=source_importance_frame,
        transformed_importance_frame=transformed_importance_frame,
        transformed_feature_frame=transformed_feature_frame,
        shap_values_frame=shap_values_frame,
        metadata=metadata,
        shap_values=shap_values,
    )


def save_release_shap_artifacts(
    pipeline: Pipeline,
    *,
    model_name: str,
    model_display_name: str,
    X_background: pd.DataFrame,
    X_explained: pd.DataFrame,
    explained_split: str,
    output_directory: str | Path,
    max_rows: int,
    background_max_rows: int,
    random_state: int,
) -> Path:
    """Write SHAP artifacts for one saved release regression model."""

    shap_directory = Path(output_directory) / "shap_analysis"
    shap_directory.mkdir(parents=True, exist_ok=True)
    artifacts = compute_release_shap_artifacts(
        pipeline,
        model_name=model_name,
        X_background=X_background,
        X_explained=X_explained,
        explained_split=explained_split,
        max_rows=max_rows,
        background_max_rows=background_max_rows,
        random_state=random_state,
    )
    artifacts.source_importance_frame.to_csv(
        shap_directory / "shap_feature_importance.csv",
        index=False,
    )
    artifacts.transformed_importance_frame.to_csv(
        shap_directory / "shap_transformed_feature_importance.csv",
        index=False,
    )
    artifacts.transformed_feature_frame.to_parquet(
        shap_directory / "shap_feature_matrix.parquet",
        index=False,
    )
    artifacts.shap_values_frame.to_parquet(
        shap_directory / "shap_values.parquet",
        index=False,
    )
    plot_feature_importance(
        artifacts.source_importance_frame.rename(columns={"mean_abs_shap": "importance"}),
        output_path=shap_directory / "shap_feature_importance.png",
        title=f"{model_display_name} mean absolute SHAP values ({explained_split})",
    )
    plot_shap_summary(
        shap_values=artifacts.shap_values,
        feature_frame=artifacts.transformed_feature_frame.drop(columns=["source_row_index"]),
        output_path=shap_directory / "shap_summary.png",
        title=f"{model_display_name} SHAP summary ({explained_split})",
    )
    save_json_report(artifacts.metadata, shap_directory / "shap_metadata.json")
    return shap_directory


def build_local_feature_gain_diagnostic(
    summary_frame: pd.DataFrame,
    *,
    family_name: str = "routing_sensitive",
) -> dict[str, object]:
    """Compare whether local features help the routing-sensitive family."""

    required_ablations = {"both_without_local_features", "both_with_local_features"}
    filtered = summary_frame.loc[
        (summary_frame["split_name"] == "test")
        & (summary_frame["slice_name"] == "family")
        & (summary_frame["slice_value"] == family_name)
        & (summary_frame["ablation_name"].isin(required_ablations))
    ].copy()
    if filtered["ablation_name"].nunique() < 2:
        return {"available": False, "family_name": family_name}

    best_rows = (
        filtered.sort_values(by=["r2", "mae"], ascending=[False, True])
        .groupby("ablation_name", as_index=False)
        .first()
    )
    without_local = best_rows.loc[best_rows["ablation_name"] == "both_without_local_features"].iloc[
        0
    ]
    with_local = best_rows.loc[best_rows["ablation_name"] == "both_with_local_features"].iloc[0]
    delta_mae = float(with_local["mae"] - without_local["mae"])
    without_r2 = without_local["r2"] if pd.notna(without_local["r2"]) else 0.0
    with_r2 = with_local["r2"] if pd.notna(with_local["r2"]) else 0.0
    delta_r2 = float(with_r2 - without_r2)
    return {
        "available": True,
        "family_name": family_name,
        "delta_mae_with_local_minus_without_local": delta_mae,
        "delta_r2_with_local_minus_without_local": delta_r2,
        "material_improvement_flag": delta_mae <= -0.01 or delta_r2 >= 0.03,
        "without_local_sample_count": int(without_local["sample_count"]),
        "with_local_sample_count": int(with_local["sample_count"]),
    }
