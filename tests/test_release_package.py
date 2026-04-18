from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config.schema import DataConfig
from src.data.release_package import load_release_split_bundle


def test_load_release_split_bundle_resolves_manifest_paths_inside_release_folder(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "release" / "splits"
    split_dir.mkdir(parents=True)

    train_frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": ["raw", "transpiled"],
            "reliability": [0.9, 0.8],
        }
    )
    validation_frame = pd.DataFrame(
        {
            "feature_a": [3.0],
            "feature_b": ["raw"],
            "reliability": [0.7],
        }
    )
    test_frame = pd.DataFrame(
        {
            "feature_a": [4.0],
            "feature_b": ["transpiled"],
            "reliability": [0.6],
        }
    )
    train_frame.to_parquet(split_dir / "train.parquet", index=False)
    validation_frame.to_parquet(split_dir / "validation.parquet", index=False)
    test_frame.to_parquet(split_dir / "test.parquet", index=False)

    feature_manifest = {
        "input_feature_columns": ["feature_a", "feature_b"],
        "target_columns": ["reliability"],
        "recommended_group_columns": ["base_circuit_id"],
    }
    (split_dir / "feature_manifest.json").write_text(
        json.dumps(feature_manifest),
        encoding="utf-8",
    )
    split_manifest = {
        "dataset_id": "demo_release_v1",
        "profile_name": "demo_release",
        "files": {
            "train": "data\\processed\\demo_release\\release\\splits\\train.parquet",
            "validation": "data\\processed\\demo_release\\release\\splits\\validation.parquet",
            "test": "data\\processed\\demo_release\\release\\splits\\test.parquet",
        },
        "feature_manifest_path": (
            "data\\processed\\demo_release\\release\\splits\\feature_manifest.json"
        ),
        "row_counts": {"train": 2, "validation": 1, "test": 1},
    }
    split_manifest_path = split_dir / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest), encoding="utf-8")

    bundle = load_release_split_bundle(DataConfig(split_manifest_path=split_manifest_path))

    assert bundle.train_frame.equals(train_frame)
    assert bundle.validation_frame.equals(validation_frame)
    assert bundle.test_frame.equals(test_frame)
    assert bundle.feature_manifest is not None
    assert bundle.feature_manifest["input_feature_columns"] == ["feature_a", "feature_b"]
    assert bundle.split_paths["train"] == split_dir / "train.parquet"
