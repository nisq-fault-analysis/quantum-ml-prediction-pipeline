"""Helpers for packaged release datasets with shipped manifests and split files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.schema import DataConfig
from src.data.dataset import read_tabular_file


@dataclass(slots=True)
class ReleaseSplitBundle:
    """In-memory representation of one packaged dataset release."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    split_paths: dict[str, Path]
    split_manifest: dict[str, Any] | None
    feature_manifest: dict[str, Any] | None


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_packaged_path(reference: str | Path | None, *, anchor_directory: Path) -> Path | None:
    """Resolve manifest references that may have been moved into a release folder."""

    if reference is None:
        return None

    raw_path = Path(reference)
    candidates = [
        raw_path,
        anchor_directory / raw_path,
        anchor_directory / raw_path.name,
        anchor_directory.parent / raw_path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve packaged dataset path " f"{reference!s}. Checked: {candidate_text}"
    )


def _resolve_split_paths(config: DataConfig) -> tuple[dict[str, Path], dict[str, Any] | None]:
    """Resolve the train/validation/test parquet paths from config or split manifest."""

    direct_paths = {
        "train": config.train_split_path,
        "validation": config.validation_split_path,
        "test": config.test_split_path,
    }
    if all(path is not None for path in direct_paths.values()):
        return {key: Path(path) for key, path in direct_paths.items()}, None

    if config.split_manifest_path is None:
        raise ValueError(
            "Packaged split loading requires either split_manifest_path or all direct split paths."
        )

    split_manifest_path = Path(config.split_manifest_path)
    split_manifest = _load_json_file(split_manifest_path)
    anchor_directory = split_manifest_path.parent
    resolved_paths = {
        split_name: _resolve_packaged_path(path_hint, anchor_directory=anchor_directory)
        for split_name, path_hint in split_manifest["files"].items()
    }
    return (
        {key: value for key, value in resolved_paths.items() if value is not None},
        split_manifest,
    )


def _load_feature_manifest(
    config: DataConfig,
    *,
    split_manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Load the feature manifest when the packaged dataset provides one."""

    manifest_reference = config.feature_manifest_path
    if manifest_reference is None and split_manifest is not None:
        manifest_reference = split_manifest.get("feature_manifest_path")
        anchor_directory = Path(config.split_manifest_path).parent
    elif manifest_reference is not None:
        anchor_directory = Path(manifest_reference).parent
    else:
        return None

    manifest_path = _resolve_packaged_path(manifest_reference, anchor_directory=anchor_directory)
    if manifest_path is None:
        return None
    return _load_json_file(manifest_path)


def load_release_split_bundle(config: DataConfig) -> ReleaseSplitBundle:
    """Read one packaged dataset release with precomputed train/validation/test splits."""

    split_paths, split_manifest = _resolve_split_paths(config)
    feature_manifest = _load_feature_manifest(config, split_manifest=split_manifest)

    train_frame = read_tabular_file(split_paths["train"], file_format="auto")
    validation_frame = read_tabular_file(split_paths["validation"], file_format="auto")
    test_frame = read_tabular_file(split_paths["test"], file_format="auto")

    return ReleaseSplitBundle(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        split_paths=split_paths,
        split_manifest=split_manifest,
        feature_manifest=feature_manifest,
    )
