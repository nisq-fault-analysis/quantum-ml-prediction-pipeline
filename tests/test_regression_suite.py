from __future__ import annotations

import pandas as pd

from src.config.schema import ProjectConfig
from src.models.regression_suite import build_regression_features


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
