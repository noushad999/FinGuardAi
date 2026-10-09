"""Central configuration loading.

Configuration comes from ``configs/finguard.yaml`` and can be overridden by
environment variables:

* ``FINGUARD__<SECTION>__<KEY>=value`` for any key (values parsed as YAML), e.g.
  ``FINGUARD__TRAINING__TRAIN_FRACTION=0.7``.
* A few short aliases for values that change per environment
  (``DATASET_SIZE``, ``KAFKA_BOOTSTRAP_SERVERS``, ``DATABASE_URL`` ...).

The loaded config is a plain nested ``dict`` wrapped in :class:`Settings`, which
offers attribute access. Keeping it a dict avoids a large mirrored class
hierarchy while still being validated at the edges where it is used.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "finguard.yaml"
POLICY_PATH = PROJECT_ROOT / "configs" / "policy.yaml"

SCALE_PRESETS: dict[str, int] = {
    "small": 100_000,
    "medium": 1_000_000,
    "large": 10_000_000,
}

# short env alias -> (section, key)
ENV_ALIASES: dict[str, tuple[str, str]] = {
    "DATASET_SIZE": ("simulation", "n_transactions"),
    "SIM_DAYS": ("simulation", "days"),
    "SIM_SEED": ("simulation", "seed"),
    "KAFKA_BOOTSTRAP_SERVERS": ("kafka", "bootstrap_servers"),
    "LAKE_URI": ("storage", "lake_uri"),
    "S3_ENDPOINT": ("storage", "s3_endpoint"),
    "S3_ACCESS_KEY": ("storage", "s3_access_key"),
    "S3_SECRET_KEY": ("storage", "s3_secret_key"),
    "DATABASE_URL": ("database", "url"),
    "REDIS_URL": ("redis", "url"),
    "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
    "MODEL_SOURCE": ("api", "model_source"),
    "MODELS_DIR": ("paths", "models_dir"),
    "DATA_DIR": ("paths", "data_dir"),
    "SYNTHETIC_DIR": ("paths", "synthetic_dir"),
    "PROCESSED_DIR": ("paths", "processed_dir"),
    "REPORTS_DIR": ("paths", "reports_dir"),
}


class Settings(dict):
    """A dict with attribute access for nested sections."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - programming error
            raise AttributeError(item) from exc
        return Settings(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Resolve a ``paths.<key>`` entry relative to the project root."""
        p = Path(self["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _parse_dataset_size(raw: Any) -> int:
    if isinstance(raw, str) and raw.lower() in SCALE_PRESETS:
        return SCALE_PRESETS[raw.lower()]
    value = int(float(raw))
    if value <= 0:
        raise ValueError(f"DATASET_SIZE must be positive, got {raw!r}")
    return value


def apply_env_overrides(cfg: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Return a copy of ``cfg`` with environment overrides applied."""
    env = dict(os.environ if env is None else env)
    cfg = copy.deepcopy(cfg)
    for alias, (section, key) in ENV_ALIASES.items():
        if alias in env and env[alias] != "":
            cfg.setdefault(section, {})[key] = _parse_scalar(env[alias])
    for name, raw in env.items():
        if not name.startswith("FINGUARD__"):
            continue
        parts = [p.lower() for p in name.split("__")[1:]]
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    sim = cfg.get("simulation", {})
    if "n_transactions" in sim:
        sim["n_transactions"] = _parse_dataset_size(sim["n_transactions"])
    return cfg


def load_settings(path: str | Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Load YAML config and apply environment overrides."""
    path = Path(path) if path else Path(os.environ.get("FINGUARD_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings(apply_env_overrides(raw, env))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings."""
    return load_settings()


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load the business decision policy (configs/policy.yaml or $POLICY_PATH)."""
    path = Path(path) if path else Path(os.environ.get("POLICY_PATH", POLICY_PATH))
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
