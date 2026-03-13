from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from src.config.io import load_config


def test_load_config_reads_yaml_into_typed_model(tmp_path: Path) -> None:
    config_path = tmp_path / "example.yaml"
    config_path.write_text(
        dedent(
            """
            data:
              dataset_path: data/raw/example.csv
              file_format: csv
              label_column: fault_type
              gate_sequence_column: gate_sequence
              qubit_count_column: qubit_count
              drop_columns: []
              test_size: 0.2
              random_state: 42
              stratify_by_label: true
            features:
              gate_delimiters: [" ", ";"]
              top_gates: ["x", "h"]
              lowercase_tokens: true
            training:
              enable_models: ["logreg", "random_forest"]
              stratify_by_qubit_count: true
              minimum_samples_per_qubit_group: 25
              generate_shap_for: random_forest
              class_weight: balanced
              n_estimators: 100
              max_depth: 5
              learning_rate: 0.1
            output:
              experiment_name: demo
              output_dir: experiments/runs/demo
              figures_dir: reports/figures
              save_predictions: true
              save_shap: false
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.dataset_path == Path("data/raw/example.csv")
    assert config.training.enable_models == ["logreg", "random_forest"]
    assert config.output.output_dir == Path("experiments/runs/demo")
