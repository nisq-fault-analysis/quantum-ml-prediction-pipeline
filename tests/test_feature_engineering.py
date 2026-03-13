from __future__ import annotations

import pandas as pd

from src.config.schema import FeatureConfig
from src.features.gate_sequence import engineer_gate_sequence_features, tokenize_gate_sequence


def test_tokenize_gate_sequence_handles_multiple_delimiters() -> None:
    tokens = tokenize_gate_sequence("H->CX;MEASURE", delimiters=["->", ";"], lowercase=True)

    assert tokens == ["h", "cx", "measure"]


def test_engineer_gate_sequence_features_builds_expected_counts() -> None:
    frame = pd.DataFrame(
        {
            "gate_sequence": ["H CX MEASURE", "X;X;RZ"],
            "qubit_count": [2, 3],
        }
    )
    feature_config = FeatureConfig(top_gates=["h", "cx", "measure", "x", "rz"])

    feature_table = engineer_gate_sequence_features(
        frame=frame,
        sequence_column="gate_sequence",
        feature_config=feature_config,
        qubit_count_column="qubit_count",
    )

    assert feature_table.loc[0, "gate_token_count"] == 3.0
    assert feature_table.loc[0, "gate_count__measure"] == 1.0
    assert feature_table.loc[1, "gate_count__x"] == 2.0
    assert feature_table.loc[1, "qubit_count"] == 3
