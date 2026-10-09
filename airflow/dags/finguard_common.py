"""Shared helpers for FinGuard DAGs.

Design choices
--------------
* Airflow only *orchestrates*: every task shells out to the FinGuard CLI inside an
  isolated virtualenv (``$FINGUARD_PYTHON``), so Airflow's own dependencies never
  constrain the ML stack.
* DAGs are chained with **Datasets** (data-aware scheduling): a DAG runs when the
  data it consumes has been updated, instead of guessing cron offsets.
* Every task is **idempotent**: outputs are overwritten deterministically (fixed
  seed, full-refresh loads, TRUNCATE+COPY, dbt table rebuilds, MLflow versions
  that only move aliases), so retries and backfills are safe.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.datasets import Dataset
from airflow.operators.bash import BashOperator

HOME = os.environ.get("FINGUARD_HOME", "/opt/finguard")
PY = os.environ.get("FINGUARD_PYTHON", "/opt/finguard-venv/bin/python")
DBT = os.path.join(os.path.dirname(PY), "dbt")

SYNTHETIC = Dataset(f"file://{HOME}/data/synthetic/transactions.parquet")
WAREHOUSE_RAW = Dataset("postgres://postgres:5432/finguard/raw")
FEATURES = Dataset(f"file://{HOME}/data/processed/transaction_features.parquet")
MODELS_DEV = Dataset("mlflow://finguard/models@development")
MODELS_VALIDATED = Dataset("mlflow://finguard/models@validation")
MODELS_PROD = Dataset("mlflow://finguard/models@production")

DEFAULT_ARGS = {
    "owner": "finguard",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=2),
}
START = datetime(2026, 1, 1)
# Keep Airflow's own site-packages (~/.local) out of the FinGuard virtualenv's subprocesses.
TASK_ENV = {"PYTHONNOUSERSITE": "1", "PYTHONPATH": HOME}
# ...and enforce it inside the shell too: executors may inject their own PYTHONPATH/user site.
ISOLATE = f"unset PYTHONHOME; export PYTHONNOUSERSITE=1 PYTHONPATH={HOME}; "


def cli(task_id: str, args: str, **kwargs) -> BashOperator:
    """Run ``python -m <args>`` from the project root inside the FinGuard virtualenv."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"{ISOLATE}cd {HOME} && {PY} -m {args}",
        env=TASK_ENV,
        append_env=True,
        **kwargs,
    )


def dbt_build(task_id: str = "dbt_build", select: str | None = None, **kwargs) -> BashOperator:
    """``dbt build`` with the policy thresholds passed as dbt vars (single source of truth)."""
    sel = f"--select {select}" if select else ""
    vars_cmd = (
        f'$({PY} -c "import json; from common.config import load_policy; '
        f"t=load_policy()['thresholds']; print(json.dumps({{'review_threshold': t['review'], "
        f"'block_threshold': t['block']}}))\")"
    )
    return BashOperator(
        task_id=task_id,
        bash_command=f"{ISOLATE}cd {HOME} && VARS={vars_cmd} && cd dbt/finguard && "
        f'DBT_PROFILES_DIR=. {DBT} build {sel} --vars "$VARS"',
        env=TASK_ENV,
        append_env=True,
        **kwargs,
    )
