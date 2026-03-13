"""Lightweight validation-driven tuning for RF and XGBoost classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.config.io import (
    build_run_directory,
    ensure_project_directories,
    load_config,
    save_resolved_config,
)
from src.config.schema import ProjectConfig
from src.data.dataset import read_tabular_file, validate_required_columns
from src.evaluation.metrics import compute_classification_metrics, save_json_report
from src.models.classification_features import (
    build_classification_feature_policy,
    build_classification_features,
)
from src.models.random_forest import save_model_artifact
from src.models.splitting import build_split_summary, split_dataset
from src.visualization.plots import plot_confusion_matrix, plot_feature_importance

CLASSIFIER_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def _select_feature_path(config_path: str | Path) -> tuple[Path, ProjectConfig]:
    config = load_config(config_path)
    feature_paths = {
        "baseline_raw": config.features.baseline_feature_path,
        "topology_aware": config.features.topology_feature_path,
        "enhanced_topology": config.features.enhanced_feature_path,
    }
    return feature_paths[config.training.feature_set_name], config


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _rf_grid() -> list[dict[str, Any]]:
    return list(
        ParameterGrid(
            {
                "n_estimators": [200, 400],
                "max_depth": [8, 12, None],
                "min_samples_leaf": [1, 2],
                "min_samples_split": [2, 5],
            }
        )
    )


def _xgb_grid() -> list[dict[str, Any]]:
    return list(
        ParameterGrid(
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1],
            }
        )
    )


def _build_classifier(
    model_name: str, config: ProjectConfig, params: dict[str, Any], class_count: int
):
    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=config.training.random_state,
            n_jobs=-1,
            class_weight=(
                None if config.training.class_weight == "none" else config.training.class_weight
            ),
            **params,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        objective = "binary:logistic" if class_count == 2 else "multi:softprob"
        extra_args = {} if class_count == 2 else {"num_class": class_count}
        return XGBClassifier(
            objective=objective,
            subsample=config.training.xgboost_subsample,
            colsample_bytree=config.training.xgboost_colsample_bytree,
            reg_lambda=config.training.xgboost_reg_lambda,
            random_state=config.training.random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
            **params,
            **extra_args,
        )

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def _predict_with_pipeline(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.Series, list[list[float]] | None]:
    if model_name == "xgboost":
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        predictions = label_encoder.inverse_transform(pipeline.predict(X).astype(int))
        y_pred = pd.Series(predictions, index=X.index, name="predicted_label").astype(str)
        y_score = pipeline.predict_proba(X).tolist()
        return y_pred, y_score

    y_pred = pd.Series(pipeline.predict(X), index=X.index, name="predicted_label").astype(str)
    y_score = pipeline.predict_proba(X).tolist() if hasattr(pipeline, "predict_proba") else None
    return y_pred, y_score


def _fit_pipeline(
    model_name: str,
    split,
    config: ProjectConfig,
    params: dict[str, Any],
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(split.X_train)),
            ("model", _build_classifier(model_name, config, params, len(split.labels))),
        ]
    )
    if model_name == "xgboost":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(split.y_train)
        pipeline.fit(split.X_train, y_train_encoded)
    else:
        pipeline.fit(split.X_train, split.y_train)
    return pipeline


def _fit_refit_pipeline(
    model_name: str,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    config: ProjectConfig,
    params: dict[str, Any],
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(X_train_full)),
            ("model", _build_classifier(model_name, config, params, y_train_full.nunique())),
        ]
    )
    if model_name == "xgboost":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_full)
        pipeline.fit(X_train_full, y_train_encoded)
    else:
        pipeline.fit(X_train_full, y_train_full)
    return pipeline


def _build_feature_importance_frame(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    importance_frame = pd.DataFrame(
        {"feature": preprocessor.get_feature_names_out(), "importance": model.feature_importances_}
    )
    return importance_frame.sort_values(by="importance", ascending=False).reset_index(drop=True)


def tune_models(config_path: str | Path) -> None:
    feature_path, config = _select_feature_path(config_path)
    ensure_project_directories(config)
    run_directory = build_run_directory(config)
    config.output.run_name = run_directory.name
    save_resolved_config(config, run_directory / "run_config.yaml")

    feature_frame = read_tabular_file(feature_path, file_format="auto")
    validate_required_columns(feature_frame, [config.data.id_column, config.data.label_column])

    X, labels = build_classification_features(feature_frame, config)
    save_json_report(
        build_classification_feature_policy(feature_frame, config, X),
        run_directory / "feature_policy.json",
    )
    split = split_dataset(X, labels, config)
    save_json_report(build_split_summary(split), run_directory / "split_summary.json")

    tuning_summaries: list[dict[str, Any]] = []
    for model_name, param_grid in [("random_forest", _rf_grid()), ("xgboost", _xgb_grid())]:
        search_rows: list[dict[str, Any]] = []
        best_score = float("-inf")
        best_params: dict[str, Any] | None = None

        for params in param_grid:
            started_at = perf_counter()
            pipeline = _fit_pipeline(model_name, split, config, params)
            y_pred_validation, y_score_validation = _predict_with_pipeline(
                model_name, pipeline, split.X_validation, split.y_train
            )
            validation_metrics = compute_classification_metrics(
                y_true=split.y_validation.astype(str),
                y_pred=y_pred_validation.astype(str),
                y_score=y_score_validation,
                labels=split.labels,
                compute_roc_auc=False,
            )
            elapsed = round(perf_counter() - started_at, 4)
            search_rows.append(
                {
                    "model_name": model_name,
                    **params,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "training_seconds": elapsed,
                }
            )
            if validation_metrics["macro_f1"] > best_score:
                best_score = validation_metrics["macro_f1"]
                best_params = params

        if best_params is None:
            raise RuntimeError(f"Unable to find best parameters for {model_name}")

        model_directory = run_directory / model_name
        model_directory.mkdir(parents=True, exist_ok=True)
        search_frame = pd.DataFrame(search_rows).sort_values(
            by=["validation_macro_f1", "validation_accuracy"],
            ascending=[False, False],
        )
        search_frame.to_csv(model_directory / "tuning_results.csv", index=False)

        X_train_full = pd.concat([split.X_train, split.X_validation], axis=0)
        y_train_full = pd.concat([split.y_train, split.y_validation], axis=0)
        refit_pipeline = _fit_refit_pipeline(
            model_name, X_train_full, y_train_full, config, best_params
        )
        y_pred_test, y_score_test = _predict_with_pipeline(
            model_name, refit_pipeline, split.X_test, y_train_full
        )
        test_metrics = compute_classification_metrics(
            y_true=split.y_test.astype(str),
            y_pred=y_pred_test.astype(str),
            y_score=y_score_test,
            labels=split.labels,
            compute_roc_auc=False,
        )
        test_metrics.update(
            {
                "train_plus_validation_rows": int(len(X_train_full)),
                "test_rows": int(len(split.X_test)),
            }
        )

        save_model_artifact(refit_pipeline, model_directory / "model.joblib")
        save_json_report(
            {
                "model_name": model_name,
                "model_display_name": CLASSIFIER_DISPLAY_NAMES[model_name],
                "best_params": best_params,
                "best_validation_macro_f1": best_score,
                "test_metrics": test_metrics,
            },
            model_directory / "best_result.json",
        )
        plot_confusion_matrix(
            y_true=split.y_test.tolist(),
            y_pred=y_pred_test.tolist(),
            labels=split.labels,
            output_path=model_directory / "test_confusion_matrix.png",
            title=f"Tuned {CLASSIFIER_DISPLAY_NAMES[model_name]} test confusion matrix",
        )
        plot_feature_importance(
            importance_frame=_build_feature_importance_frame(refit_pipeline, model_name),
            output_path=model_directory / "feature_importance.png",
            title=f"Tuned {CLASSIFIER_DISPLAY_NAMES[model_name]} feature importance",
        )

        tuning_summaries.append(
            {
                "model_name": model_name,
                "model_display_name": CLASSIFIER_DISPLAY_NAMES[model_name],
                "best_validation_macro_f1": best_score,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "best_params": str(best_params),
            }
        )

    pd.DataFrame(tuning_summaries).sort_values(
        by=["best_validation_macro_f1", "test_accuracy"],
        ascending=[False, False],
    ).to_csv(run_directory / "tuned_model_comparison.csv", index=False)
    print(f"Tuned classification artifacts saved to: {run_directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune RF and XGBoost on the validation split.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/tuned_classification.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tune_models(config_path=args.config)


if __name__ == "__main__":
    main()
