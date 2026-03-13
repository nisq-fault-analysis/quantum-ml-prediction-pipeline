"""Command-line entry point for the baseline experiment pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pickle

import pandas as pd

from src.config.io import ensure_project_directories, load_config, save_resolved_config
from src.data.dataset import prepare_research_table, read_tabular_dataset, validate_required_columns
from src.evaluation.metrics import save_json_report
from src.features.gate_sequence import engineer_gate_sequence_features
from src.models.baseline import ModelRun, train_and_evaluate_models
from src.visualization.plots import plot_confusion_matrix, plot_shap_summary


def _safe_group_name(value: object) -> str:
    return str(value).replace(" ", "_").replace(".", "_")


def build_group_indices(
    frame: pd.DataFrame, qubit_count_column: str, minimum_size: int
) -> dict[str, pd.Index]:
    """Build dataset slices for qubit-stratified experiments."""

    groups = {"global": frame.index}

    if qubit_count_column not in frame.columns:
        return groups

    for qubit_count, subset in frame.groupby(qubit_count_column):
        if len(subset) < minimum_size:
            continue

        groups[f"qubit_{_safe_group_name(qubit_count)}"] = subset.index

    return groups


def save_model_run(
    run: ModelRun,
    group_output_dir: Path,
    save_predictions: bool,
    enable_shap: bool,
    shap_model_name: str | None,
) -> None:
    """Persist everything needed to inspect or cite a single model run."""

    model_output_dir = group_output_dir / run.model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    with (model_output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(run.estimator, handle)

    save_json_report(run.metrics, model_output_dir / "metrics.json")
    run.report_frame.to_csv(model_output_dir / "classification_report.csv", index=False)
    if save_predictions:
        run.predictions.to_csv(model_output_dir / "predictions.csv", index=False)

    plot_confusion_matrix(
        y_true=run.y_true_labels.tolist(),
        y_pred=run.y_pred_labels.tolist(),
        labels=run.label_names,
        output_path=model_output_dir / "confusion_matrix.png",
        title=f"{run.model_name} confusion matrix",
    )

    if enable_shap and shap_model_name == run.model_name and run.model_name in {
        "random_forest",
        "xgboost",
    }:
        plot_shap_summary(
            model=run.estimator,
            features=run.X_test,
            output_path=model_output_dir / "shap_summary.png",
        )


def run_pipeline(config_path: str | Path) -> None:
    """Run the full baseline experiment workflow from config to saved artifacts."""

    config = load_config(config_path)
    ensure_project_directories(config)
    save_resolved_config(config, config.output.output_dir / "resolved_config.yaml")

    raw_frame = read_tabular_dataset(config.data)
    dataset = prepare_research_table(raw_frame, config.data)

    required_columns = [
        config.data.label_column,
        config.data.gate_sequence_column,
    ]
    if config.training.stratify_by_qubit_count:
        required_columns.append(config.data.qubit_count_column)

    validate_required_columns(dataset, required_columns)

    feature_table = engineer_gate_sequence_features(
        frame=dataset,
        sequence_column=config.data.gate_sequence_column,
        feature_config=config.features,
        qubit_count_column=config.data.qubit_count_column,
    )
    labels = dataset[config.data.label_column].astype(str)

    feature_export = feature_table.copy()
    feature_export.insert(0, "row_index", feature_table.index)
    feature_export["target_label"] = labels
    interim_feature_path = (
        Path("data/interim") / f"{config.output.experiment_name}_features.parquet"
    )
    feature_export.to_parquet(interim_feature_path, index=False)

    group_indices = build_group_indices(
        frame=dataset,
        qubit_count_column=config.data.qubit_count_column,
        minimum_size=config.training.minimum_samples_per_qubit_group,
    )

    comparison_rows: list[dict[str, object]] = []
    for group_name, group_index in group_indices.items():
        group_features = feature_table.loc[group_index]
        group_labels = labels.loc[group_index]

        if group_labels.nunique() < 2:
            print(f"Skipping {group_name}: fewer than two classes are available.")
            continue

        minimum_rows = max(10, group_labels.nunique() * 2)
        if len(group_features) < minimum_rows:
            print(f"Skipping {group_name}: not enough rows for a stable train/test split.")
            continue

        print(f"Running baseline models for group: {group_name}")
        group_output_dir = config.output.output_dir / group_name
        model_runs = train_and_evaluate_models(group_features, group_labels, config)

        for model_run in model_runs:
            save_model_run(
                run=model_run,
                group_output_dir=group_output_dir,
                save_predictions=config.output.save_predictions,
                enable_shap=config.output.save_shap,
                shap_model_name=config.training.generate_shap_for,
            )

            comparison_rows.append(
                {
                    "group": group_name,
                    "model": model_run.model_name,
                    "n_samples": int(len(group_features)),
                    "n_features": int(group_features.shape[1]),
                    "n_classes": int(group_labels.nunique()),
                    **model_run.metrics,
                }
            )

    if comparison_rows:
        comparison_frame = pd.DataFrame(comparison_rows).sort_values(
            by=["group", "macro_f1"], ascending=[True, False]
        )
        comparison_frame.to_csv(config.output.output_dir / "model_comparison.csv", index=False)

    save_json_report(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_name": config.output.experiment_name,
            "feature_table_path": str(interim_feature_path),
            "stratified_by_qubit_count": config.training.stratify_by_qubit_count,
            "group_count_requested": len(group_indices),
        },
        config.output.output_dir / "run_metadata.json",
    )

    print(f"Experiment outputs saved to: {config.output.output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse the minimal CLI arguments for the training script."""

    parser = argparse.ArgumentParser(
        description="Run the baseline NISQ fault classification pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/baseline.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by both `python -m` and console scripts."""

    args = parse_args()
    run_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
