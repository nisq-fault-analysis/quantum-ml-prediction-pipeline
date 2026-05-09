"""Tests for the NISQ reliability dataset loader.

Covers:
- NISQDatasetConfig construction and validation
- NISQReliabilityConfig Pydantic validation
- Feature manifest column resolution (flat list, nested dict, missing key)
- build_feature_matrix: feature selection, categorical encoding, leakage guard
- load_nisq_splits: happy path, missing files, missing group/target columns
- check_split_integrity: group leakage, target range violations, zero variance,
  all-null feature columns
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config.schema import NISQReliabilityConfig, ProjectConfig
from src.data.nisq_dataset import (
    NISQDatasetConfig,
    NISQDatasetSplits,
    SplitData,
    _resolve_feature_columns,
    build_feature_matrix,
    check_split_integrity,
    load_nisq_splits,
)

# ── shared test fixtures ──────────────────────────────────────────────────────

FEATURE_COLS = [
    "original_circuit_depth",
    "original_num_cx",
    "qubit_count",
    "circuit_depth",
    "t1_mean",
    "t2_mean",
    "readout_error",
]

TARGET_COLS = [
    "reliability",
    "fidelity",
    "error_rate",
    "algorithmic_success_probability",
    "exact_output_success_rate",
]

PROVENANCE_COLS = [
    "dataset_id",
    "base_circuit_id",
    "circuit_id",
    "compiler_variant",
    "hardware_id",
    "family",
    "execution_mode",
    "seed",
    "profile_name",
]


def _make_row(
    base_circuit_id: str,
    variant: int,
    rng: np.random.Generator,
    *,
    reliability: float | None = None,
    include_noise_regime: bool = False,
    include_payload: bool = True,
) -> dict:
    row: dict = {
        # provenance
        "dataset_id": "test_v1",
        "profile_name": "p1",
        "base_circuit_id": base_circuit_id,
        "circuit_id": f"{base_circuit_id}_{variant}",
        "family": "qft",
        "execution_mode": "noisy",
        "seed": 42,
        "compiler_variant": variant,
        "hardware_id": "fake_hw",
        # targets
        "reliability": reliability if reliability is not None else float(rng.uniform(0.1, 0.9)),
        "fidelity": float(rng.uniform(0.1, 0.9)),
        "error_rate": float(rng.uniform(0.1, 0.9)),
        "algorithmic_success_probability": float(rng.uniform(0, 1)),
        "exact_output_success_rate": float(rng.uniform(0, 1)),
        # features
        "original_circuit_depth": int(rng.integers(5, 30)),
        "original_num_cx": int(rng.integers(0, 10)),
        "qubit_count": int(rng.integers(2, 10)),
        "circuit_depth": int(rng.integers(5, 50)),
        "t1_mean": float(rng.uniform(50, 200)),
        "t2_mean": float(rng.uniform(30, 150)),
        "readout_error": float(rng.uniform(0.01, 0.1)),
    }
    if include_noise_regime:
        row["noise_regime"] = rng.choice(["thermal", "depolarising", "mixed"])
    if include_payload:
        row["counts_json"] = '{"0": 5}'
        row["ideal_distribution_json"] = '{"0": 1.0}'
        row["compiler_metadata_json"] = "{}"
    return row


def _build_dataframes(
    rng: np.random.Generator,
    *,
    n_train_groups: int = 8,
    n_val_groups: int = 2,
    n_test_groups: int = 2,
    variants_per_group: int = 4,
    include_noise_regime: bool = False,
    include_payload: bool = True,
    train_reliability: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = n_train_groups + n_val_groups + n_test_groups
    all_ids = [f"circ_{i:03d}" for i in range(total)]
    perm = rng.permutation(all_ids).tolist()
    train_ids = perm[:n_train_groups]
    val_ids = perm[n_train_groups : n_train_groups + n_val_groups]
    test_ids = perm[n_train_groups + n_val_groups :]

    def make_df(ids: list[str], reliability: float | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _make_row(
                    gid,
                    v,
                    rng,
                    reliability=reliability,
                    include_noise_regime=include_noise_regime,
                    include_payload=include_payload,
                )
                for gid in ids
                for v in range(variants_per_group)
            ]
        )

    return make_df(train_ids, train_reliability), make_df(val_ids), make_df(test_ids)


def _write_dataset(
    tmp_path: Path,
    rng: np.random.Generator,
    feature_manifest: dict | None = None,
    split_manifest: dict | None = None,
    **kwargs,
) -> Path:
    train_df, val_df, test_df = _build_dataframes(rng, **kwargs)

    if feature_manifest is None:
        feature_manifest = {
            "input_features": FEATURE_COLS,
            "target_columns": TARGET_COLS,
        }
    if split_manifest is None:
        split_manifest = {
            "dataset_id": "test_v1",
            "schema_version": "1.2.0",
            "group_columns": ["base_circuit_id"],
            "seed": 42,
            "row_counts": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df),
            },
        }

    train_df.to_parquet(tmp_path / "train.parquet", index=False)
    val_df.to_parquet(tmp_path / "validation.parquet", index=False)
    test_df.to_parquet(tmp_path / "test.parquet", index=False)
    (tmp_path / "feature_manifest.json").write_text(json.dumps(feature_manifest), encoding="utf-8")
    (tmp_path / "split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def dataset_dir(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Standard valid dataset directory used by most tests."""
    return _write_dataset(tmp_path, rng)


# ── NISQDatasetConfig ─────────────────────────────────────────────────────────


def test_nisq_dataset_config_defaults() -> None:
    cfg = NISQDatasetConfig(dataset_dir="/tmp/test")
    assert cfg.target_column == "reliability"
    assert cfg.group_column == "base_circuit_id"
    assert cfg.drop_payload_columns is True


def test_nisq_dataset_config_accepts_valid_target_columns() -> None:
    for col in (
        "reliability",
        "fidelity",
        "error_rate",
        "algorithmic_success_probability",
        "exact_output_success_rate",
    ):
        cfg = NISQDatasetConfig(dataset_dir="/tmp", target_column=col)
        assert cfg.target_column == col


def test_nisq_dataset_config_rejects_unknown_target_column() -> None:
    with pytest.raises(ValueError, match="target_column"):
        NISQDatasetConfig(dataset_dir="/tmp", target_column="not_a_target")


def test_nisq_dataset_config_from_dict_round_trips() -> None:
    d = {
        "dataset_dir": "/tmp/data",
        "target_column": "fidelity",
        "group_column": "base_circuit_id",
        "drop_payload_columns": False,
    }
    cfg = NISQDatasetConfig.from_dict(d)
    assert cfg.target_column == "fidelity"
    assert cfg.drop_payload_columns is False
    assert cfg.dataset_dir == Path("/tmp/data")


def test_nisq_dataset_config_from_dict_applies_defaults() -> None:
    cfg = NISQDatasetConfig.from_dict({"dataset_dir": "/tmp"})
    assert cfg.target_column == "reliability"
    assert cfg.drop_payload_columns is True


# ── NISQReliabilityConfig (Pydantic) ─────────────────────────────────────────


def test_nisq_reliability_config_default_target_column() -> None:
    cfg = NISQReliabilityConfig()
    assert cfg.target_column == "reliability"


def test_nisq_reliability_config_rejects_bad_target_column() -> None:
    with pytest.raises(ValidationError):
        NISQReliabilityConfig(target_column="not_valid")


def test_project_config_nisq_reliability_is_none_by_default() -> None:
    pc = ProjectConfig()
    assert pc.nisq_reliability is None


def test_project_config_accepts_nisq_reliability_block() -> None:
    pc = ProjectConfig.model_validate(
        {"nisq_reliability": {"dataset_dir": "/tmp/data", "target_column": "fidelity"}}
    )
    assert pc.nisq_reliability is not None
    assert pc.nisq_reliability.target_column == "fidelity"
    assert pc.nisq_reliability.dataset_dir == Path("/tmp/data")


# ── feature manifest resolution ───────────────────────────────────────────────


def test_resolve_feature_columns_reads_flat_input_features_list() -> None:
    manifest = {"input_features": ["col_a", "col_b", "col_c"]}
    frame = pd.DataFrame({"col_a": [1], "col_b": [2], "col_c": [3]})
    cols = _resolve_feature_columns(manifest, frame)
    assert cols == ["col_a", "col_b", "col_c"]


def test_resolve_feature_columns_falls_back_to_features_key() -> None:
    manifest = {"features": ["col_a", "col_b"]}
    frame = pd.DataFrame({"col_a": [1], "col_b": [2]})
    cols = _resolve_feature_columns(manifest, frame)
    assert cols == ["col_a", "col_b"]


def test_resolve_feature_columns_handles_nested_category_dict() -> None:
    manifest = {
        "input_features": {
            "pre_compilation": ["original_depth"],
            "hardware": ["t1_mean", "t2_mean"],
        }
    }
    frame = pd.DataFrame({"original_depth": [1], "t1_mean": [2], "t2_mean": [3]})
    cols = _resolve_feature_columns(manifest, frame)
    assert set(cols) == {"original_depth", "t1_mean", "t2_mean"}


def test_resolve_feature_columns_skips_columns_absent_from_frame() -> None:
    manifest = {"input_features": ["present", "absent"]}
    frame = pd.DataFrame({"present": [1]})
    cols = _resolve_feature_columns(manifest, frame)
    assert cols == ["present"]


def test_resolve_feature_columns_includes_optional_categorical_when_present() -> None:
    manifest = {"input_features": ["col_a"]}
    frame = pd.DataFrame({"col_a": [1], "noise_regime": ["thermal"]})
    cols = _resolve_feature_columns(manifest, frame)
    assert "noise_regime" in cols


def test_resolve_feature_columns_raises_when_no_recognised_key() -> None:
    manifest = {"unknown_key": ["col_a"]}
    frame = pd.DataFrame({"col_a": [1]})
    with pytest.raises(ValueError, match="recognised feature column list"):
        _resolve_feature_columns(manifest, frame)


# ── build_feature_matrix ──────────────────────────────────────────────────────


def test_build_feature_matrix_selects_only_feature_columns() -> None:
    frame = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0], "reliability": [0.8], "meta": ["x"]})
    X = build_feature_matrix(
        frame,
        feature_columns=["feat_a", "feat_b"],
        target_columns=["reliability"],
    )
    assert list(X.columns) == ["feat_a", "feat_b"]


def test_build_feature_matrix_raises_on_target_leakage() -> None:
    frame = pd.DataFrame({"feat_a": [1.0], "reliability": [0.8]})
    with pytest.raises(AssertionError, match="leaked"):
        build_feature_matrix(
            frame,
            feature_columns=["feat_a", "reliability"],  # reliability is a target
            target_columns=["reliability"],
        )


def test_build_feature_matrix_one_hot_encodes_noise_regime() -> None:
    frame = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0],
            "noise_regime": ["thermal", "depolarising", "thermal"],
        }
    )
    X = build_feature_matrix(frame, feature_columns=["feat_a", "noise_regime"], target_columns=[])
    assert "noise_regime" not in X.columns
    ohe_cols = [c for c in X.columns if c.startswith("noise_regime_")]
    assert len(ohe_cols) == 2  # thermal, depolarising


def test_build_feature_matrix_skips_encoding_when_no_categoricals_present() -> None:
    frame = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0]})
    X = build_feature_matrix(frame, feature_columns=["feat_a", "feat_b"], target_columns=[])
    assert list(X.columns) == ["feat_a", "feat_b"]


def test_build_feature_matrix_skips_absent_feature_columns_gracefully() -> None:
    frame = pd.DataFrame({"feat_a": [1.0]})
    X = build_feature_matrix(frame, feature_columns=["feat_a", "missing_col"], target_columns=[])
    assert list(X.columns) == ["feat_a"]


# ── load_nisq_splits — happy path ─────────────────────────────────────────────


def test_load_nisq_splits_returns_correct_split_shapes(
    dataset_dir: Path, rng: np.random.Generator
) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)

    # 8 groups × 4 variants each
    assert len(splits.train.y) == 32
    # 2 groups × 4 variants each
    assert len(splits.validation.y) == 8
    assert len(splits.test.y) == 8


def test_load_nisq_splits_primary_target_exposed_as_y(dataset_dir: Path) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir, target_column="fidelity")
    splits = load_nisq_splits(cfg)
    assert splits.train.y.name == "fidelity"


def test_load_nisq_splits_no_target_column_in_x(dataset_dir: Path) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)
    for target in TARGET_COLS:
        assert target not in splits.train.X.columns, f"{target} leaked into X"


def test_load_nisq_splits_payload_columns_dropped_by_default(dataset_dir: Path) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)
    for col in ("counts_json", "ideal_distribution_json", "compiler_metadata_json"):
        assert col not in splits.train.X.columns
        assert col not in splits.train.meta.columns


def test_load_nisq_splits_group_column_in_meta(dataset_dir: Path) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)
    assert "base_circuit_id" in splits.train.meta.columns


def test_load_nisq_splits_feature_columns_from_manifest(dataset_dir: Path) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)
    # All FEATURE_COLS should appear (before any OHE expansion)
    for col in FEATURE_COLS:
        assert col in splits.feature_columns


def test_load_nisq_splits_manifests_returned(dataset_dir: Path) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)
    assert splits.feature_manifest.get("input_features") == FEATURE_COLS
    assert splits.split_manifest.get("dataset_id") == "test_v1"


def test_load_nisq_splits_optional_categorical_encoded(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    _write_dataset(tmp_path, rng, include_noise_regime=True)
    manifest = {
        "input_features": FEATURE_COLS + ["noise_regime"],
        "target_columns": TARGET_COLS,
    }
    (tmp_path / "feature_manifest.json").write_text(json.dumps(manifest))

    cfg = NISQDatasetConfig(dataset_dir=tmp_path)
    splits = load_nisq_splits(cfg)

    ohe_cols = [c for c in splits.train.X.columns if c.startswith("noise_regime_")]
    assert ohe_cols, "noise_regime was not one-hot encoded"
    assert "noise_regime" not in splits.train.X.columns


def test_load_nisq_splits_group_column_taken_from_split_manifest(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    # split_manifest specifies a different group column name
    _write_dataset(
        tmp_path,
        rng,
        split_manifest={
            "dataset_id": "test_v1",
            "group_columns": ["base_circuit_id"],
            "seed": 42,
        },
    )
    cfg = NISQDatasetConfig(dataset_dir=tmp_path, group_column="base_circuit_id")
    splits = load_nisq_splits(cfg)  # should not raise
    assert "base_circuit_id" in splits.train.meta.columns


# ── load_nisq_splits — error cases ───────────────────────────────────────────


def test_load_nisq_splits_raises_when_feature_manifest_missing(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    _write_dataset(tmp_path, rng)
    (tmp_path / "feature_manifest.json").unlink()

    cfg = NISQDatasetConfig(dataset_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="feature_manifest.json"):
        load_nisq_splits(cfg)


def test_load_nisq_splits_raises_when_parquet_file_missing(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    _write_dataset(tmp_path, rng)
    (tmp_path / "test.parquet").unlink()

    cfg = NISQDatasetConfig(dataset_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="test.parquet"):
        load_nisq_splits(cfg)


def test_load_nisq_splits_raises_when_target_column_absent(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    train_df, val_df, test_df = _build_dataframes(rng)
    train_df = train_df.drop(columns=["fidelity"])
    train_df.to_parquet(tmp_path / "train.parquet", index=False)
    val_df.to_parquet(tmp_path / "validation.parquet", index=False)
    test_df.to_parquet(tmp_path / "test.parquet", index=False)
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps({"input_features": FEATURE_COLS, "target_columns": TARGET_COLS})
    )
    (tmp_path / "split_manifest.json").write_text(
        json.dumps({"group_columns": ["base_circuit_id"]})
    )

    cfg = NISQDatasetConfig(dataset_dir=tmp_path, target_column="fidelity")
    with pytest.raises(ValueError, match="fidelity"):
        load_nisq_splits(cfg)


def test_load_nisq_splits_raises_when_group_column_absent(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    train_df, val_df, test_df = _build_dataframes(rng)
    for df, name in [
        (train_df, "train"),
        (val_df, "validation"),
        (test_df, "test"),
    ]:
        df.drop(columns=["base_circuit_id"]).to_parquet(tmp_path / f"{name}.parquet", index=False)
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps({"input_features": FEATURE_COLS, "target_columns": TARGET_COLS})
    )
    (tmp_path / "split_manifest.json").write_text(
        json.dumps({"group_columns": ["base_circuit_id"]})
    )

    cfg = NISQDatasetConfig(dataset_dir=tmp_path)
    with pytest.raises(ValueError, match="base_circuit_id"):
        load_nisq_splits(cfg)


# ── check_split_integrity — passing ──────────────────────────────────────────


def test_check_split_integrity_passes_for_valid_dataset(dataset_dir: Path) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)
    check_split_integrity(splits, group_column="base_circuit_id")  # must not raise


# ── check_split_integrity — group leakage ─────────────────────────────────────


def test_check_split_integrity_detects_group_leakage(
    dataset_dir: Path, rng: np.random.Generator
) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)

    # Manually inject a shared group ID across train and validation meta.
    shared_id = splits.train.meta["base_circuit_id"].iloc[0]
    contaminated_meta = splits.validation.meta.copy()
    contaminated_meta.iloc[0, contaminated_meta.columns.get_loc("base_circuit_id")] = shared_id

    bad_splits = NISQDatasetSplits(
        train=splits.train,
        validation=SplitData(X=splits.validation.X, y=splits.validation.y, meta=contaminated_meta),
        test=splits.test,
        feature_columns=splits.feature_columns,
        target_columns=splits.target_columns,
        split_manifest=splits.split_manifest,
        feature_manifest=splits.feature_manifest,
    )

    with pytest.raises(AssertionError, match="leakage"):
        check_split_integrity(bad_splits, group_column="base_circuit_id")


# ── check_split_integrity — target range ─────────────────────────────────────


def test_check_split_integrity_detects_out_of_range_target(dataset_dir: Path) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)

    bad_y = splits.train.y.copy()
    bad_y.iloc[0] = 1.5  # outside [0, 1]

    bad_splits = NISQDatasetSplits(
        train=SplitData(X=splits.train.X, y=bad_y, meta=splits.train.meta),
        validation=splits.validation,
        test=splits.test,
        feature_columns=splits.feature_columns,
        target_columns=splits.target_columns,
        split_manifest=splits.split_manifest,
        feature_manifest=splits.feature_manifest,
    )

    with pytest.raises(AssertionError, match=r"\[0, 1\]"):
        check_split_integrity(bad_splits, group_column="base_circuit_id")


# ── check_split_integrity — zero variance ─────────────────────────────────────


def test_check_split_integrity_detects_zero_variance_reliability(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    _write_dataset(tmp_path, rng, train_reliability=0.5)  # constant reliability in train
    cfg = NISQDatasetConfig(dataset_dir=tmp_path, target_column="reliability")
    splits = load_nisq_splits(cfg)

    with pytest.raises(AssertionError, match="zero variance"):
        check_split_integrity(splits, group_column="base_circuit_id")


# ── check_split_integrity — all-null feature ──────────────────────────────────


def test_check_split_integrity_detects_all_null_feature_column(
    dataset_dir: Path,
) -> None:
    cfg = NISQDatasetConfig(dataset_dir=dataset_dir)
    splits = load_nisq_splits(cfg)

    X_bad = splits.train.X.copy()
    first_col = X_bad.columns[0]
    X_bad[first_col] = np.nan

    bad_splits = NISQDatasetSplits(
        train=SplitData(X=X_bad, y=splits.train.y, meta=splits.train.meta),
        validation=splits.validation,
        test=splits.test,
        feature_columns=splits.feature_columns,
        target_columns=splits.target_columns,
        split_manifest=splits.split_manifest,
        feature_manifest=splits.feature_manifest,
    )

    with pytest.raises(AssertionError, match="entirely null"):
        check_split_integrity(bad_splits, group_column="base_circuit_id")
