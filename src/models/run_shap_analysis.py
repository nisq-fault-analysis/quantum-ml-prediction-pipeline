"""Run SHAP analysis for the best model in a saved benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config.io import load_config
from src.data.dataset import read_tabular_file, validate_required_columns
from src.evaluation.metrics import save_json_report
from src.models.classification_features import (
    build_classification_feature_policy,
    build_classification_features,
)
from src.models.splitting import split_dataset
from src.visualization.plots import plot_feature_importance


def _load_best_model_row(run_directory: Path) -> pd.Series:
    comparison_path = run_directory / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Could not find model comparison table at {comparison_path}")

    comparison_frame = pd.read_csv(comparison_path)
    if comparison_frame.empty:
        raise ValueError(f"Model comparison table is empty: {comparison_path}")

    return (
        comparison_frame.sort_values(
            by=["validation_macro_f1", "validation_accuracy"],
            ascending=[False, False],
        )
        .iloc[0]
        .copy()
    )


def _select_feature_path(run_config_path: Path) -> tuple[Path, Any]:
    config = load_config(run_config_path)
    feature_paths = {
        "baseline_raw": config.features.baseline_feature_path,
        "topology_aware": config.features.topology_feature_path,
        "enhanced_topology": config.features.enhanced_feature_path,
    }
    return feature_paths[config.training.feature_set_name], config


def _split_name_to_frame(split, split_name: str) -> pd.DataFrame:
    if split_name == "train":
        return split.X_train
    if split_name == "validation":
        return split.X_validation
    if split_name == "test":
        return split.X_test
    raise ValueError(f"Unsupported split name: {split_name}")


def _apply_subset_filter(
    feature_frame: pd.DataFrame, run_path: Path
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    subset_metadata_path = run_path / "subset_metadata.json"
    if not subset_metadata_path.exists():
        return feature_frame, None

    subset_metadata = json.loads(subset_metadata_path.read_text(encoding="utf-8"))
    filter_column = str(subset_metadata["filter_column"])
    filter_value = subset_metadata["filter_value"]
    filtered_frame = feature_frame.loc[feature_frame[filter_column] == filter_value].copy()
    if filtered_frame.empty:
        raise ValueError(
            f"Subset filter {filter_column} == {filter_value!r} produced no rows for {run_path}"
        )
    return filtered_frame, subset_metadata


def _build_explainer(
    pipeline: Pipeline,
    model_name: str,
    X_train_transformed: np.ndarray,
    feature_names: np.ndarray,
):
    model = pipeline.named_steps["model"]
    if model_name in {"random_forest", "xgboost"}:
        return shap.TreeExplainer(model, feature_names=feature_names.tolist())
    if isinstance(model, LogisticRegression):
        background = X_train_transformed[: min(len(X_train_transformed), 500)]
        return shap.LinearExplainer(
            model,
            background,
            feature_names=feature_names.tolist(),
        )
    raise ValueError(
        "SHAP analysis is implemented for Random Forest, XGBoost, and "
        f"Logistic Regression, not {model_name}."
    )


def _mean_absolute_shap(values: np.ndarray) -> np.ndarray:
    absolute_values = np.abs(values)
    if absolute_values.ndim == 2:
        return absolute_values.mean(axis=0)
    if absolute_values.ndim == 3:
        return absolute_values.mean(axis=(0, 2))
    raise ValueError(f"Unsupported SHAP value shape: {absolute_values.shape}")


def run_shap_analysis(run_directory: str | Path, split_name: str = "test") -> None:
    """Explain the best saved classifier in a benchmark run using SHAP."""

    run_path = Path(run_directory)
    run_config_path = run_path / "run_config.yaml"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Could not find run_config.yaml in {run_path}")

    best_model_row = _load_best_model_row(run_path)
    model_name = str(best_model_row["model_name"])
    model_path = run_path / model_name / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find fitted model artifact at {model_path}")

    feature_path, config = _select_feature_path(run_config_path)
    feature_frame = read_tabular_file(feature_path, file_format="auto")
    validate_required_columns(feature_frame, [config.data.id_column, config.data.label_column])
    feature_frame, subset_metadata = _apply_subset_filter(feature_frame, run_path)
    X, labels = build_classification_features(feature_frame, config)
    split = split_dataset(X, labels, config)

    pipeline: Pipeline = joblib.load(model_path)
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    X_train_transformed = np.asarray(preprocessor.transform(split.X_train))
    explained_frame = _split_name_to_frame(split, split_name)
    X_explained_transformed = np.asarray(preprocessor.transform(explained_frame))

    explainer = _build_explainer(pipeline, model_name, X_train_transformed, feature_names)
    explanation = explainer(X_explained_transformed)
    shap_values = np.asarray(explanation.values)
    mean_abs_shap = _mean_absolute_shap(shap_values)

    importance_frame = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_directory = run_path / "shap_analysis"
    shap_directory.mkdir(parents=True, exist_ok=True)
    importance_frame.to_csv(shap_directory / "shap_feature_importance.csv", index=False)
    plot_feature_importance(
        importance_frame.rename(columns={"mean_abs_shap": "importance"}),
        output_path=shap_directory / "shap_feature_importance.png",
        title=f"{best_model_row['model_display_name']} mean absolute SHAP values ({split_name})",
    )
    save_json_report(
        {
            "selected_model": best_model_row.to_dict(),
            "explained_split": split_name,
            "explained_rows": int(len(explained_frame)),
            "subset_metadata": subset_metadata,
            "feature_policy": build_classification_feature_policy(feature_frame, config, X),
        },
        shap_directory / "shap_metadata.json",
    )
    print(f"SHAP analysis artifacts saved to: {shap_directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis on a saved benchmark run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a saved model benchmark run directory.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Which deterministic split to explain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_shap_analysis(run_directory=args.run_dir, split_name=args.split)


if __name__ == "__main__":
    main()
