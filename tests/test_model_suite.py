from __future__ import annotations

import pandas as pd

from src.config.schema import ProjectConfig, TrainingConfig
from src.models.model_suite import build_model_comparison_frame, train_model_suite
from src.models.splitting import split_dataset


def test_train_model_suite_returns_results_for_requested_models() -> None:
    feature_frame = pd.DataFrame(
        {
            "qubit_count": [2, 2, 3, 3, 4, 4, 5, 5],
            "gate_depth": [5, 6, 8, 9, 11, 12, 14, 15],
            "error_rate_gate": [0.01, 0.02, 0.03, 0.04, 0.12, 0.13, 0.14, 0.15],
            "readout_error": [0.01, 0.01, 0.02, 0.02, 0.11, 0.12, 0.13, 0.14],
            "fidelity": [0.99, 0.98, 0.97, 0.96, 0.88, 0.87, 0.86, 0.85],
            "device_type": [
                "superconducting",
                "superconducting",
                "superconducting",
                "superconducting",
                "trapped_ion",
                "trapped_ion",
                "trapped_ion",
                "trapped_ion",
            ],
        }
    )
    labels = pd.Series(
        [
            "readout",
            "readout",
            "readout",
            "readout",
            "depolarizing",
            "depolarizing",
            "depolarizing",
            "depolarizing",
        ],
        name="error_type",
    )
    config = ProjectConfig(
        training=TrainingConfig(
            model_names=["dummy_most_frequent", "logistic_regression"],
            test_size=0.25,
            random_state=11,
            logistic_max_iter=300,
        )
    )

    _, results = train_model_suite(feature_frame, labels, config)

    assert [result.model_name for result in results] == [
        "dummy_most_frequent",
        "logistic_regression",
    ]
    assert all("accuracy" in result.validation_metrics for result in results)
    assert all("accuracy" in result.test_metrics for result in results)
    assert results[0].feature_importance_frame is None
    assert results[1].feature_importance_frame is not None

    comparison_frame = build_model_comparison_frame(results)

    assert comparison_frame.shape[0] == 2
    assert set(comparison_frame["model_name"]) == {
        "dummy_most_frequent",
        "logistic_regression",
    }


def test_split_dataset_creates_80_15_5_partition() -> None:
    feature_frame = pd.DataFrame(
        {
            "value": list(range(100)),
            "device_type": ["superconducting"] * 50 + ["trapped_ion"] * 50,
        }
    )
    labels = pd.Series(["depolarizing"] * 50 + ["readout"] * 50, name="error_type")
    config = ProjectConfig(
        training=TrainingConfig(
            validation_size=0.15,
            test_size=0.05,
            random_state=7,
        )
    )

    split = split_dataset(feature_frame, labels, config)

    assert len(split.X_train) == 80
    assert len(split.X_validation) == 15
    assert len(split.X_test) == 5
