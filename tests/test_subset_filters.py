from __future__ import annotations

import pandas as pd
import pytest

from src.models.subset_filters import filter_frame_by_subset


def test_filter_frame_by_subset_returns_filtered_rows_and_metadata() -> None:
    frame = pd.DataFrame(
        {
            "qubit_count": [6, 6, 8, 8],
            "error_type": ["readout", "depolarizing", "readout", "depolarizing"],
            "value": [1, 2, 3, 4],
        }
    )

    filtered_frame, subset_metadata = filter_frame_by_subset(
        frame,
        subset_column="qubit_count",
        subset_value="6",
        label_column="error_type",
    )

    assert filtered_frame["qubit_count"].tolist() == [6, 6]
    assert subset_metadata is not None
    assert subset_metadata["filter_column"] == "qubit_count"
    assert subset_metadata["filter_value"] == 6
    assert subset_metadata["row_count"] == 2
    assert subset_metadata["label_distribution"] == {"readout": 1, "depolarizing": 1}


def test_filter_frame_by_subset_requires_both_filter_parts() -> None:
    frame = pd.DataFrame({"qubit_count": [6, 8]})

    with pytest.raises(ValueError):
        filter_frame_by_subset(frame, subset_column="qubit_count", subset_value=None)
