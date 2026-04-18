"""Small MLflow helpers for reproducible thesis experiment logging."""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config.schema import ProjectConfig


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised only when mlflow missing
        raise RuntimeError(
            "MLflow logging is enabled but `mlflow` is not installed. "
            "Reinstall the project dependencies to enable experiment tracking."
        ) from exc

    return mlflow


def resolve_git_metadata(repo_root: Path) -> dict[str, str | None]:
    """Return the current git commit and dirty status when available."""

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=repo_root,
            ).stdout.strip()
            or None
        )
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        ).stdout
        is_dirty = "true" if dirty_output.strip() else "false"
        return {"git_commit": commit, "git_dirty": is_dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}


def _flatten_for_mlflow(payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        else:
            flattened[key] = str(value)
    return flattened


@contextmanager
def start_mlflow_run(
    config: ProjectConfig,
    *,
    run_name: str,
    tags: dict[str, Any] | None = None,
):
    """Start an MLflow run when tracking is enabled in the config."""

    if not config.training.enable_mlflow:
        yield None
        return

    mlflow = _import_mlflow()
    if config.training.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
    mlflow.set_experiment(config.training.mlflow_experiment_name)

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(_flatten_for_mlflow(tags))
        yield mlflow


def log_params(mlflow, params: dict[str, Any]) -> None:
    """Log MLflow params if tracking is active."""

    if mlflow is None:
        return
    mlflow.log_params(_flatten_for_mlflow(params))


def log_metrics(mlflow, metrics: dict[str, Any], *, prefix: str | None = None) -> None:
    """Log numeric metrics if tracking is active."""

    if mlflow is None:
        return

    payload: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            payload[f"{prefix}_{key}" if prefix else key] = float(value)
        elif isinstance(value, (int, float)):
            payload[f"{prefix}_{key}" if prefix else key] = float(value)
    if payload:
        mlflow.log_metrics(payload)


def log_artifact_path(mlflow, path: str | Path, *, artifact_path: str | None = None) -> None:
    """Log a file or directory as an MLflow artifact when tracking is active."""

    if mlflow is None:
        return

    artifact = Path(path)
    if artifact.is_dir():
        mlflow.log_artifacts(str(artifact), artifact_path=artifact_path)
    elif artifact.exists():
        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
