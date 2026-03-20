"""NISQ reliability dataset loader for pre-split parquet files.

Reads the three pre-split parquet files (train.parquet, validation.parquet,
test.parquet) and their JSON sidecar files (split_manifest.json,
feature_manifest.json) produced by the dataset-generator companion repo.

Design constraints
------------------
- Column lists are read from feature_manifest.json; nothing is hardcoded here.
- Group integrity is preserved: no base_circuit_id appears in more than one split.
- Optional extended columns (noise_regime, noise_dominant_channel, etc.) are
  included only when present in the actual data.
- Raw JSON payload columns (counts_json, ideal_distribution_json,
  compiler_metadata_json) are never passed to a model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

log = logging.getLogger(__name__)

# ── well-known column sets ────────────────────────────────────────────────────

#: All recognised prediction target columns.
ALL_TARGET_COLUMNS: tuple[str, ...] = (
    "reliability",
    "fidelity",
    "error_rate",
    "algorithmic_success_probability",
    "exact_output_success_rate",
)

#: Provenance / metadata columns — kept in meta, never in X.
ALL_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "profile_name",
    "base_circuit_id",
    "circuit_id",
    "family",
    "execution_mode",
    "seed",
    "compiler_variant",
    "hardware_id",
)

#: Raw JSON payloads — dropped entirely (unless drop_payload_columns=False).
RAW_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "counts_json",
    "ideal_distribution_json",
    "compiler_metadata_json",
)

#: Categorical feature columns that require encoding before modelling.
CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "noise_regime",
    "noise_dominant_channel",
)


# ── configuration ─────────────────────────────────────────────────────────────


@dataclass
class NISQDatasetConfig:
    """Configuration for the NISQ reliability dataset loader.

    Parameters
    ----------
    dataset_dir:
        Directory that contains train.parquet, validation.parquet,
        test.parquet, split_manifest.json, and feature_manifest.json.
    target_column:
        Which regression target to use as ``y``.  Defaults to ``"reliability"``.
    group_column:
        Column used to check split integrity.  Overridden by the value found in
        split_manifest.json when present.
    drop_payload_columns:
        Drop raw JSON payload columns before returning data.
    """

    dataset_dir: Path
    target_column: str = "reliability"
    group_column: str = "base_circuit_id"
    drop_payload_columns: bool = True

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        if self.target_column not in ALL_TARGET_COLUMNS:
            raise ValueError(
                f"target_column must be one of {list(ALL_TARGET_COLUMNS)}, "
                f"got {self.target_column!r}"
            )

    @classmethod
    def from_dict(cls, mapping: dict) -> NISQDatasetConfig:
        """Construct from a plain dictionary (e.g. loaded from a YAML config)."""
        return cls(
            dataset_dir=Path(mapping["dataset_dir"]),
            target_column=mapping.get("target_column", "reliability"),
            group_column=mapping.get("group_column", "base_circuit_id"),
            drop_payload_columns=bool(mapping.get("drop_payload_columns", True)),
        )


# ── split result types ────────────────────────────────────────────────────────


class SplitData(NamedTuple):
    """Features, primary target, and metadata for one split."""

    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


@dataclass
class NISQDatasetSplits:
    """All three pre-split datasets in structured form.

    Attributes
    ----------
    train, validation, test:
        Each is a :class:`SplitData` named-tuple with ``X``, ``y``, and
        ``meta`` DataFrames.
    feature_columns:
        Ordered list of input feature column names (after manifest resolution,
        before encoding).  Encoding may expand this list with dummy columns.
    target_columns:
        All target columns present in the dataset.
    split_manifest:
        Raw dict loaded from split_manifest.json (empty dict if file absent).
    feature_manifest:
        Raw dict loaded from feature_manifest.json.
    """

    train: SplitData
    validation: SplitData
    test: SplitData
    feature_columns: list[str]
    target_columns: list[str]
    split_manifest: dict
    feature_manifest: dict


# ── manifest helpers ──────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_feature_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "feature_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"feature_manifest.json not found in {dataset_dir}. "
            "The dataset directory must contain feature_manifest.json."
        )
    manifest = _load_json(path)
    log.info("Loaded feature_manifest.json from %s", dataset_dir)
    return manifest


def _read_split_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "split_manifest.json"
    if not path.exists():
        log.warning(
            "split_manifest.json not found in %s; proceeding without provenance.",
            dataset_dir,
        )
        return {}
    manifest = _load_json(path)
    log.info("Loaded split_manifest.json from %s", dataset_dir)
    return manifest


# ── column resolution ─────────────────────────────────────────────────────────


def _resolve_feature_columns(feature_manifest: dict, frame: pd.DataFrame) -> list[str]:
    """Derive the ordered input-feature column list from the manifest.

    Tries the keys ``"input_features"``, ``"features"``, and
    ``"feature_columns"`` in that order.  Nested dicts (grouped by category)
    are flattened.  Always intersects with columns actually present in
    ``frame``.  Optional categorical columns are appended when found.
    """
    candidates: list[str] = []

    for key in ("input_features", "features", "feature_columns"):
        section = feature_manifest.get(key)
        if isinstance(section, list):
            candidates = section
            log.info(
                "Feature column list read from manifest key %r (%d columns).",
                key,
                len(candidates),
            )
            break
        if isinstance(section, dict):
            # Nested by category e.g. {"pre_compilation": [...], "hardware": [...]}
            for sub in section.values():
                if isinstance(sub, list):
                    candidates.extend(sub)
            log.info(
                "Feature columns assembled from nested manifest key %r (%d columns).",
                key,
                len(candidates),
            )
            break

    if not candidates:
        raise ValueError(
            "feature_manifest.json does not contain a recognised feature column "
            "list.  Expected key 'input_features', 'features', or 'feature_columns'."
        )

    present = [c for c in candidates if c in frame.columns]
    missing = [c for c in candidates if c not in frame.columns]
    if missing:
        log.info(
            "%d feature columns listed in manifest are absent from the frame "
            "(skipped): %s",
            len(missing),
            missing,
        )

    # Include optional categorical columns when present in the data.
    for cat_col in CATEGORICAL_FEATURE_COLUMNS:
        if cat_col in frame.columns and cat_col not in present:
            present.append(cat_col)
            log.info("Including optional categorical feature column: %s", cat_col)

    return present


def _resolve_target_columns(feature_manifest: dict, frame: pd.DataFrame) -> list[str]:
    """Return target columns from the manifest, filtered to those in ``frame``."""
    manifest_targets: list[str] = feature_manifest.get(
        "target_columns", list(ALL_TARGET_COLUMNS)
    )
    present = [c for c in manifest_targets if c in frame.columns]
    missing = [c for c in manifest_targets if c not in frame.columns]
    if missing:
        log.info(
            "Manifest target columns absent from frame (skipped): %s", missing
        )
    return present


# ── parquet reading ───────────────────────────────────────────────────────────


def _read_split_parquet(dataset_dir: Path, split_name: str) -> pd.DataFrame:
    path = dataset_dir / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_parquet(path)
    log.info(
        "Read %s split: %d rows, %d columns.", split_name, len(df), df.shape[1]
    )
    return df


# ── feature matrix builder ────────────────────────────────────────────────────


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    encode_categoricals: bool = True,
) -> pd.DataFrame:
    """Build the feature matrix X from a raw split frame.

    Steps
    -----
    1. Asserts no target column leaks into ``feature_columns``.
    2. Selects only columns in ``feature_columns`` that are present in ``frame``.
    3. Logs included and excluded column counts at INFO level.
    4. One-hot encodes recognised categorical string columns when
       ``encode_categoricals`` is True.

    Parameters
    ----------
    frame:
        Raw split DataFrame as loaded from parquet.
    feature_columns:
        Ordered list of desired input columns.
    target_columns:
        Prediction targets — must not overlap with ``feature_columns``.
    encode_categoricals:
        When True, apply ``pd.get_dummies`` to categorical columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for sklearn / torch ingestion.
    """
    # Hard leakage guard — targets must never appear in X.
    leaked = set(feature_columns) & set(target_columns)
    assert not leaked, (
        f"Target column(s) leaked into the feature matrix: {sorted(leaked)}. "
        "Remove them from the feature manifest's input_features list."
    )

    present = [c for c in feature_columns if c in frame.columns]
    excluded = sorted(set(feature_columns) - set(present))
    if excluded:
        log.info(
            "Feature columns requested but absent from frame (excluded): %s",
            excluded,
        )
    log.info(
        "Building feature matrix: %d columns included, %d excluded.",
        len(present),
        len(excluded),
    )

    X = frame[present].copy()

    if encode_categoricals:
        cat_cols = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in X.columns]
        if cat_cols:
            log.info("One-hot encoding categorical columns: %s", cat_cols)
            X = pd.get_dummies(X, columns=cat_cols, dtype=float)
        else:
            log.info("No categorical feature columns to encode.")

    return X


# ── integrity checks ──────────────────────────────────────────────────────────


def check_split_integrity(
    splits: NISQDatasetSplits,
    *,
    group_column: str = "base_circuit_id",
) -> None:
    """Assert structural invariants required for the dataset to be trustworthy.

    Checks
    ------
    1. No ``base_circuit_id`` group appears in more than one split (leakage).
    2. All target columns have values in [0, 1], or are null for optional targets.
    3. ``reliability`` has non-zero variance in the training split.
    4. No required input-feature column is entirely null in the training split.

    Raises
    ------
    AssertionError
        On any integrity violation, with a diagnostic message.
    """
    _check_no_group_leakage(splits, group_column=group_column)
    _check_target_ranges(splits)
    _check_reliability_variance(splits)
    _check_no_all_null_features(splits)
    log.info("All split integrity checks passed.")


def _check_no_group_leakage(
    splits: NISQDatasetSplits, *, group_column: str
) -> None:
    """Assert that no group key appears in more than one split."""
    split_groups: dict[str, set] = {}

    for name, split_data in [
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ]:
        # group_column lives in meta after the split is built.
        if group_column in split_data.meta.columns:
            split_groups[name] = set(split_data.meta[group_column].dropna().unique())
        else:
            log.warning(
                "group_column %r not found in %r split; skipping leakage check.",
                group_column,
                name,
            )
            return

    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if a not in split_groups or b not in split_groups:
            continue
        overlap = split_groups[a] & split_groups[b]
        assert not overlap, (
            f"Group leakage: {len(overlap)} {group_column!r} values appear in "
            f"both the {a!r} and {b!r} splits.  The dataset must have been split "
            "at the row level rather than the group level."
        )

    log.info(
        "Group leakage check passed — no %r overlap across splits.", group_column
    )


def _check_target_ranges(splits: NISQDatasetSplits) -> None:
    """Assert all target values are in [0, 1] (nulls allowed for optional cols)."""
    optional_targets = {"algorithmic_success_probability", "exact_output_success_rate"}

    for split_name, split_data in [
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ]:
        for col in splits.target_columns:
            # Primary target is in y; other targets may be in meta.
            if split_data.y.name == col:
                series = split_data.y
            elif col in split_data.meta.columns:
                series = split_data.meta[col]
            else:
                continue

            non_null = series.dropna()
            if non_null.empty:
                if col in optional_targets:
                    log.info(
                        "Optional target %r is entirely null in %r (permitted).",
                        col,
                        split_name,
                    )
                    continue
                raise AssertionError(
                    f"Required target {col!r} is entirely null in the "
                    f"{split_name!r} split."
                )

            out_of_range = non_null[(non_null < 0) | (non_null > 1)]
            assert out_of_range.empty, (
                f"Target column {col!r} in {split_name!r} has "
                f"{len(out_of_range)} values outside [0, 1]: "
                f"min={non_null.min():.6f}, max={non_null.max():.6f}."
            )

    log.info("Target range check passed — all values are in [0, 1].")


def _check_reliability_variance(splits: NISQDatasetSplits) -> None:
    """Assert that reliability has non-zero variance in the training split."""
    if splits.train.y.name != "reliability":
        return  # variance check is specific to the primary reliability target
    variance = splits.train.y.var()
    assert variance > 0, (
        "reliability has zero variance in the training split.  "
        "This indicates degenerate dataset generation."
    )
    log.info("Reliability variance check passed: var=%.6f.", variance)


def _check_no_all_null_features(splits: NISQDatasetSplits) -> None:
    """Assert that no input-feature column is entirely null in the train split."""
    all_null = [
        col for col in splits.train.X.columns if splits.train.X[col].isna().all()
    ]
    assert not all_null, (
        f"The following feature columns are entirely null in the training split: "
        f"{all_null}.  Remove them from the feature manifest or investigate "
        "dataset generation."
    )
    log.info("All-null feature check passed.")


# ── internal split builder ────────────────────────────────────────────────────


def _build_split_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    target_column: str,
    group_column: str,
    drop_payload: bool,
) -> SplitData:
    """Partition a raw parquet frame into X, y, and metadata."""

    if group_column not in frame.columns:
        raise ValueError(
            f"Required group column {group_column!r} is absent from the dataset.  "
            "This column is needed for split integrity verification."
        )

    if drop_payload:
        payload_present = [c for c in RAW_PAYLOAD_COLUMNS if c in frame.columns]
        if payload_present:
            log.info("Dropping raw payload columns: %s", payload_present)
            frame = frame.drop(columns=payload_present)

    if target_column not in frame.columns:
        raise ValueError(
            f"Target column {target_column!r} is not present in the dataset.  "
            f"Available columns: {sorted(frame.columns.tolist())}"
        )

    # Metadata: everything that is not a feature and not a target.
    non_feature_non_target = [
        c
        for c in frame.columns
        if c not in feature_columns and c not in target_columns
    ]
    meta = frame[non_feature_non_target].copy()

    y = frame[target_column].copy()
    y.name = target_column

    X = build_feature_matrix(
        frame, feature_columns, target_columns, encode_categoricals=True
    )

    return SplitData(X=X, y=y, meta=meta)


# ── public API ────────────────────────────────────────────────────────────────


def load_nisq_splits(config: NISQDatasetConfig) -> NISQDatasetSplits:
    """Load all three pre-split parquet files and return structured split data.

    Parameters
    ----------
    config:
        Dataset configuration specifying the directory, target column, etc.

    Returns
    -------
    NISQDatasetSplits
        Structured container with :class:`SplitData` for each of train,
        validation, and test, plus the loaded manifests.

    Raises
    ------
    FileNotFoundError
        If any required file is missing from ``config.dataset_dir``.
    ValueError
        If the target column or group column is absent from the data.
    AssertionError
        If the data fails integrity checks (call :func:`check_split_integrity`
        separately to surface these after loading).
    """
    dataset_dir = config.dataset_dir

    feature_manifest = _read_feature_manifest(dataset_dir)
    split_manifest = _read_split_manifest(dataset_dir)

    # Prefer the group column from the split manifest when available.
    group_column = config.group_column
    manifest_group_cols: list[str] = split_manifest.get("group_columns", [])
    if manifest_group_cols:
        manifest_group = manifest_group_cols[0]
        if manifest_group != group_column:
            log.info(
                "Using group_column %r from split_manifest (config said %r).",
                manifest_group,
                group_column,
            )
        group_column = manifest_group

    train_df = _read_split_parquet(dataset_dir, "train")
    validation_df = _read_split_parquet(dataset_dir, "validation")
    test_df = _read_split_parquet(dataset_dir, "test")

    feature_columns = _resolve_feature_columns(feature_manifest, train_df)
    target_columns = _resolve_target_columns(feature_manifest, train_df)

    log.info(
        "Resolved %d feature columns, %d target columns.  Primary target: %r.",
        len(feature_columns),
        len(target_columns),
        config.target_column,
    )

    train_split = _build_split_data(
        train_df,
        feature_columns,
        target_columns,
        config.target_column,
        group_column,
        config.drop_payload_columns,
    )
    validation_split = _build_split_data(
        validation_df,
        feature_columns,
        target_columns,
        config.target_column,
        group_column,
        config.drop_payload_columns,
    )
    test_split = _build_split_data(
        test_df,
        feature_columns,
        target_columns,
        config.target_column,
        group_column,
        config.drop_payload_columns,
    )

    return NISQDatasetSplits(
        train=train_split,
        validation=validation_split,
        test=test_split,
        feature_columns=feature_columns,
        target_columns=target_columns,
        split_manifest=split_manifest,
        feature_manifest=feature_manifest,
    )
