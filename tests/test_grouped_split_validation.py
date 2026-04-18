from __future__ import annotations

import pandas as pd
import pytest

from src.models.grouped_split_validation import (
    assert_grouped_split_integrity,
    build_split_membership_frame,
)


def test_assert_grouped_split_integrity_passes_for_disjoint_base_circuit_ids() -> None:
    split_frames = {
        "train": pd.DataFrame(
            {
                "base_circuit_id": ["g1", "g1", "g2"],
                "circuit_id": ["g1-raw", "g1-transpiled", "g2-raw"],
                "compiler_variant": ["raw", "transpiled", "raw"],
            }
        ),
        "validation": pd.DataFrame(
            {
                "base_circuit_id": ["g3"],
                "circuit_id": ["g3-raw"],
                "compiler_variant": ["raw"],
            }
        ),
        "test": pd.DataFrame(
            {
                "base_circuit_id": ["g4"],
                "circuit_id": ["g4-transpiled"],
                "compiler_variant": ["transpiled"],
            }
        ),
    }

    audit = assert_grouped_split_integrity(
        split_frames,
        group_column="base_circuit_id",
        row_id_columns=["circuit_id", "compiler_variant"],
    )
    membership = build_split_membership_frame(
        split_frames,
        group_column="base_circuit_id",
        row_id_columns=["circuit_id", "compiler_variant"],
    )

    assert audit.passed is True
    assert membership.shape[0] == 5
    assert sorted(membership["split_name"].unique().tolist()) == ["test", "train", "validation"]


def test_assert_grouped_split_integrity_raises_for_group_overlap() -> None:
    split_frames = {
        "train": pd.DataFrame(
            {
                "base_circuit_id": ["g1"],
                "circuit_id": ["g1-raw"],
                "compiler_variant": ["raw"],
            }
        ),
        "validation": pd.DataFrame(
            {
                "base_circuit_id": ["g1"],
                "circuit_id": ["g1-transpiled"],
                "compiler_variant": ["transpiled"],
            }
        ),
        "test": pd.DataFrame(
            {
                "base_circuit_id": ["g2"],
                "circuit_id": ["g2-raw"],
                "compiler_variant": ["raw"],
            }
        ),
    }

    with pytest.raises(ValueError, match="Grouped split integrity validation failed"):
        assert_grouped_split_integrity(
            split_frames,
            group_column="base_circuit_id",
            row_id_columns=["circuit_id", "compiler_variant"],
        )
