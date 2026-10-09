"""Shared fixtures.

``small_env`` builds a small but complete environment once per test session:
synthetic data -> point-in-time features -> fraud/ATO/unsupervised models ->
graph, segmentation, churn and customer profiles, all in a temp directory with
MLflow disabled. It takes ~20-40 s and powers the ML, API and pipeline tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("FINGUARD_DISABLE_MLFLOW", "1")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "text")

from common.config import load_settings  # noqa: E402


def make_settings(root: Path, n_transactions: int = 12000, days: int = 150, seed: int = 11):
    env = {
        "DATASET_SIZE": str(n_transactions),
        "SIM_DAYS": str(days),
        "SIM_SEED": str(seed),
        "SYNTHETIC_DIR": str(root / "synthetic"),
        "PROCESSED_DIR": str(root / "processed"),
        "MODELS_DIR": str(root / "models"),
        "REPORTS_DIR": str(root / "reports"),
        "LAKE_URI": str(root / "lake"),
        "REDIS_URL": "",
        "FINGUARD__TRAINING__XGB__N_ESTIMATORS": "150",
        "FINGUARD__SIMULATION__TARGET_FRAUD_RATE": "0.03",
    }
    for d in ("synthetic", "processed", "models", "reports"):
        (root / d).mkdir(parents=True, exist_ok=True)
    return load_settings(env=env)


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    """Synthetic dataset only (fast)."""
    from simulator.generator import generate, write_dataset

    root = tmp_path_factory.mktemp("fg_data")
    settings = make_settings(root)
    ds = generate(settings)
    write_dataset(ds, settings.path("synthetic_dir"))
    return settings, ds


@pytest.fixture(scope="session")
def small_env(small_data):
    """Dataset + features + every trained model and batch output."""
    from features.engine import EngineConfig
    from features.offline import build_from_dir
    from ml.account_takeover import train as ato
    from ml.churn_prediction import churn
    from ml.customer_profile import profile
    from ml.customer_segmentation import segment
    from ml.fraud_detection import thresholds
    from ml.fraud_detection import train as fraud
    from ml.graph_risk import graph

    settings, ds = small_data
    build_from_dir(
        settings.path("synthetic_dir"),
        settings.path("processed_dir") / "transaction_features.parquet",
        EngineConfig.from_settings(settings),
    )
    results = {"fraud": fraud.run(settings), "ato": ato.run(settings)}
    thresholds.analyse(settings)
    results["graph"] = graph.run(settings)
    results["segmentation"] = segment.run(settings)
    results["churn"] = churn.run(settings)
    profile.run(settings)
    return settings, ds, results
