"""MLflow integration: experiment tracking and model registration.

Local default: ``sqlite:///mlruns/mlflow.db`` (no server needed). In Docker the
``MLFLOW_TRACKING_URI`` points at the MLflow server. Set
``FINGUARD_DISABLE_MLFLOW=1`` to skip tracking (used by fast unit tests).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.config import PROJECT_ROOT, get_settings
from common.logging import get_logger

log = get_logger(__name__)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def enabled() -> bool:
    return os.environ.get("FINGUARD_DISABLE_MLFLOW", "0") not in ("1", "true", "True")


def tracking_uri() -> str:
    uri = get_settings()["mlflow"]["tracking_uri"]
    if uri:
        return uri
    (PROJECT_ROOT / "mlruns").mkdir(exist_ok=True)
    return f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}"


def _setup() -> Any:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    name = get_settings()["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        location = None
        if tracking_uri().startswith("sqlite"):
            location = (PROJECT_ROOT / "mlruns" / "artifacts").as_uri()
        mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)
    return mlflow


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, bool) or isinstance(v, (int, float)) and v == v:
            out[key] = float(v)
    return out


class Run:
    """Thin wrapper so pipeline code works identically with tracking disabled."""

    def __init__(self, mlflow_mod: Any | None, run: Any | None) -> None:
        self.mlflow = mlflow_mod
        self.run = run

    @property
    def run_id(self) -> str | None:
        return self.run.info.run_id if self.run else None

    def log_params(self, params: dict[str, Any]) -> None:
        if self.mlflow:
            self.mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        if self.mlflow:
            flat = _flatten(metrics, prefix)
            if flat:
                self.mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in flat.items()})

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.mlflow:
            self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifacts(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.mlflow:
            self.mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.mlflow:
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def register_bundle(self, bundle_dir: str | Path, registered_name: str, description: str = "") -> int | None:
        """Log the bundle directory and create a new registry version (alias: development)."""
        if not self.mlflow or not self.run:
            return None
        from mlflow import MlflowClient

        self.mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        client = MlflowClient()
        try:
            client.get_registered_model(registered_name)
        except Exception:  # noqa: BLE001 - not found
            client.create_registered_model(registered_name, description=description)
        mv = client.create_model_version(
            registered_name, source=self.mlflow.get_artifact_uri("bundle"), run_id=self.run_id
        )
        client.set_registered_model_alias(registered_name, "development", mv.version)
        log.info("model_registered", extra={"model": registered_name, "version": mv.version})
        return int(mv.version)


@contextmanager
def tracked_run(run_name: str, tags: dict[str, Any] | None = None) -> Iterator[Run]:
    if not enabled():
        yield Run(None, None)
        return
    try:
        mlflow = _setup()
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        log.warning("mlflow_unavailable", extra={"error": str(exc)})
        yield Run(None, None)
        return
    with mlflow.start_run(run_name=run_name) as run:
        r = Run(mlflow, run)
        r.set_tags({"project": "finguard", **(tags or {})})
        yield r


def next_version(registered_name: str, bundle_dir: str | Path, prefix: str) -> str:
    """Version string for a new model: ``<prefix>-v<N>``.

    N is the next MLflow registry version when tracking is enabled (so the
    string in metadata.json matches the registry), otherwise a local counter.
    """
    if enabled():
        try:
            _setup()
            from mlflow import MlflowClient

            versions = MlflowClient().search_model_versions(f"name='{registered_name}'")
            n = max((int(v.version) for v in versions), default=0) + 1
            return f"{prefix}-v{n}"
        except Exception as exc:  # noqa: BLE001
            log.warning("registry_version_lookup_failed", extra={"error": str(exc)})
    return next_local_version(bundle_dir, prefix)


def next_local_version(bundle_dir: str | Path, prefix: str) -> str:
    """Fallback version string when the registry is disabled."""
    meta = Path(bundle_dir) / "metadata.json"
    n = 1
    if meta.exists():
        import json

        old = json.loads(meta.read_text()).get("version", "")
        try:
            n = int(str(old).rsplit("-v", 1)[1]) + 1
        except (IndexError, ValueError):
            n = 1
    return f"{prefix}-v{n}"
