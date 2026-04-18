from __future__ import annotations

import pandas as pd

from src.config.schema import ProjectConfig
from src.models.regression_suite import (
    build_regression_features,
    build_regression_split_from_precomputed_frames,
    train_regression_suite_on_split,
)


def test_build_regression_features_drops_target_and_outcome_derived_columns() -> None:
    feature_frame = pd.DataFrame(
        {
            "circuit_id": ["circ_1"],
            "error_type": ["readout"],
            "qubit_count": [5],
            "gate_depth": [12],
            "fidelity": [0.55],
            "fidelity_loss": [0.45],
            "bit_errors": [2.0],
            "observed_error_rate": [0.4],
            "bit_error_density": [0.16],
            "device_type": ["superconducting"],
        }
    )

    X, y = build_regression_features(feature_frame, ProjectConfig())

    assert y.tolist() == [0.55]
    assert "fidelity" not in X.columns
    assert "fidelity_loss" not in X.columns
    assert "bit_errors" not in X.columns
    assert "observed_error_rate" not in X.columns
    assert "bit_error_density" not in X.columns


def test_train_regression_suite_on_precomputed_split_handles_missing_values() -> None:
    train_frame = pd.DataFrame(
        {
            "depth": [4, 5, 6, 7, 8, 9],
            "device_family": ["a", "a", "b", "b", None, "a"],
            "local_t1_mean": [10.0, 12.0, None, 15.0, None, 18.0],
            "all_missing": [None, None, None, None, None, None],
            "reliability": [0.91, 0.88, 0.73, 0.7, 0.65, 0.6],
        }
    )
    validation_frame = pd.DataFrame(
        {
            "depth": [10, 11],
            "device_family": ["b", None],
            "local_t1_mean": [19.0, None],
            "all_missing": [None, None],
            "reliability": [0.58, 0.55],
        }
    )
    test_frame = pd.DataFrame(
        {
            "depth": [12, 13],
            "device_family": ["a", "b"],
            "local_t1_mean": [20.0, None],
            "all_missing": [None, None],
            "reliability": [0.52, 0.49],
        }
    )

    split = build_regression_split_from_precomputed_frames(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        target_column="reliability",
        feature_columns=["depth", "device_family", "local_t1_mean", "all_missing"],
    )
    _, results = train_regression_suite_on_split(split, ProjectConfig())

    assert "all_missing" not in split.X_train.columns
    assert split.X_train.columns.tolist() == ["depth", "device_family", "local_t1_mean"]
    assert len(results) == 3
    assert all("r2" in result.validation_metrics for result in results)
