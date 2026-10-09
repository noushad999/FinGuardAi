"""Model promotion workflow on the MLflow Model Registry (aliases).

    development ──(validation gates)──> validation ──(approval)──> production
                                                                     │
                                                           rollback ─┘ (previous production)

* ``development``: every training run registers a new version with this alias.
* ``validate``: automated gates - required metrics present, test PR-AUC above an
  absolute floor, not worse than the current production version by more than a
  tolerance, and the feature schema matches the running code.
* ``promote``: moves ``production`` to the validated version and remembers the
  previous one in the ``previous_production`` alias.
* ``rollback``: points ``production`` back to ``previous_production``.
* ``deploy``: downloads the ``production`` bundle into ``models/<dir>`` (what the
  API/risk engine load when ``MODEL_SOURCE=local``); with ``MODEL_SOURCE=mlflow``
  services pull ``models:/<name>@production`` themselves.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.logging import get_logger
from ml.common.tracking import _setup

log = get_logger(__name__)

# registered name -> (local bundle dir, metric path, floor)
MODELS: dict[str, dict[str, Any]] = {
    "finguard-fraud-xgb": {"dir": "fraud_xgb", "metric": ["test", "at_policy_review", "pr_auc"], "floor": 0.5},
    "finguard-ato-xgb": {"dir": "ato_xgb", "metric": ["test", "at_policy_review", "pr_auc"], "floor": 0.5},
    "finguard-fraud-iforest": {"dir": "fraud_iforest", "metric": ["test", "pr_auc"], "floor": 0.05},
    "finguard-behavioral-anomaly": {"dir": "behavioral_anomaly", "metric": ["test", "pr_auc"], "floor": 0.05},
    "finguard-churn-xgb": {"dir": "churn_xgb", "metric": ["test", "roc_auc"], "floor": 0.65},
    "finguard-segmentation": {"dir": "segmentation", "metric": ["kmeans", "silhouette"], "floor": 0.05},
}
TOLERANCE = 0.02


@dataclass
class GateResult:
    passed: bool
    checks: dict[str, Any]


def _client():
    _setup()
    from mlflow import MlflowClient

    return MlflowClient()


def _metadata(client, name: str, version: str) -> dict[str, Any]:
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{version}", dst_path=tmp)
        return json.loads((Path(path) / "metadata.json").read_text())


def _metric(meta: dict[str, Any], path: list[str]) -> float | None:
    node: Any = meta.get("metrics", {})
    for p in path:
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return float(node) if node is not None else None


def _alias_version(client, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except Exception:  # noqa: BLE001 - alias not set
        return None


def validate(name: str, version: str | None = None) -> GateResult:
    """Run validation gates on ``version`` (default: the development alias)."""
    from features.definitions import FEATURE_NAMES

    client = _client()
    cfg = MODELS[name]
    version = version or _alias_version(client, name, "development")
    if version is None:
        return GateResult(False, {"error": "no development version"})
    meta = _metadata(client, name, version)
    value = _metric(meta, cfg["metric"])
    checks: dict[str, Any] = {
        "version": version,
        "metric": ".".join(cfg["metric"]),
        "value": value,
        "floor": cfg["floor"],
    }
    checks["metric_present"] = value is not None
    checks["above_floor"] = value is not None and value >= cfg["floor"]
    prod = _alias_version(client, name, "production")
    if prod and prod != version:
        prod_value = _metric(_metadata(client, name, prod), cfg["metric"])
        checks["production_version"] = prod
        checks["production_value"] = prod_value
        checks["not_worse_than_production"] = (
            value is not None and prod_value is not None and value >= prod_value - TOLERANCE
        )
    else:
        checks["not_worse_than_production"] = True
    if meta.get("model_type", "").startswith("xgboost") and cfg["dir"] in ("fraud_xgb", "ato_xgb", "fraud_iforest"):
        checks["schema_ok"] = set(meta["feature_names"]) <= set(FEATURE_NAMES)
    else:
        checks["schema_ok"] = True
    passed = all(checks[k] for k in ("metric_present", "above_floor", "not_worse_than_production", "schema_ok"))
    if passed:
        client.set_registered_model_alias(name, "validation", version)
    client.set_model_version_tag(name, version, "validation_status", "passed" if passed else "failed")
    log.info("model_validation", extra={"model": name, "version": version, "passed": passed})
    return GateResult(passed, checks)


def promote(name: str, version: str | None = None) -> str:
    """Promote the validated version to production (keeps previous_production)."""
    client = _client()
    version = version or _alias_version(client, name, "validation")
    if version is None:
        raise RuntimeError(f"{name}: nothing validated to promote")
    current = _alias_version(client, name, "production")
    if current and current != version:
        client.set_registered_model_alias(name, "previous_production", current)
    client.set_registered_model_alias(name, "production", version)
    log.info("model_promoted", extra={"model": name, "version": version, "previous": current})
    return version


def rollback(name: str) -> str:
    client = _client()
    prev = _alias_version(client, name, "previous_production")
    if prev is None:
        raise RuntimeError(f"{name}: no previous production version to roll back to")
    current = _alias_version(client, name, "production")
    client.set_registered_model_alias(name, "production", prev)
    if current:
        client.set_registered_model_alias(name, "previous_production", current)
    log.info("model_rolled_back", extra={"model": name, "to": prev, "from": current})
    return prev


def deploy(name: str, models_dir: Path, alias: str = "production") -> Path:
    """Materialise ``models:/name@alias`` into the local models directory."""
    import mlflow

    _setup()
    target = models_dir / MODELS[name]["dir"]
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@{alias}", dst_path=tmp)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path, target)
    log.info("model_deployed", extra={"model": name, "alias": alias, "path": str(target)})
    return target


def status() -> dict[str, dict[str, str | None]]:
    client = _client()
    out = {}
    for name in MODELS:
        try:
            client.get_registered_model(name)
        except Exception:  # noqa: BLE001
            continue
        out[name] = {
            a: _alias_version(client, name, a)
            for a in ("development", "validation", "production", "previous_production")
        }
    return out
