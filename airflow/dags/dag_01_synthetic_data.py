"""1. Synthetic data generation + data-quality gate."""

from airflow import DAG
from finguard_common import DEFAULT_ARGS, START, SYNTHETIC, cli

with DAG(
    dag_id="finguard_01_synthetic_data",
    description="Generate the synthetic MFS dataset (idempotent: fixed seed) and gate it on data quality",
    start_date=START,
    schedule=None,  # on demand; in production this would be the upstream extract arriving
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["finguard", "data"],
) as dag:
    generate = cli("generate_synthetic_data", "simulator")
    quality = cli("data_quality_gate", "data_engineering.validation --fail-on-error", outlets=[SYNTHETIC])
    generate >> quality
