"""3. Point-in-time feature generation (replays history through the online feature engine)."""

from airflow import DAG
from finguard_common import DEFAULT_ARGS, FEATURES, START, SYNTHETIC, cli

with DAG(
    dag_id="finguard_03_feature_generation",
    description="Build training features with the same code path as real-time scoring",
    start_date=START,
    schedule=[SYNTHETIC],
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["finguard", "features"],
) as dag:
    cli("build_transaction_features", "features build", outlets=[FEATURES])
