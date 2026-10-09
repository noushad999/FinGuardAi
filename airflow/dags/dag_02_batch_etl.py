"""2. Batch ETL: extracts -> data lake -> warehouse raw -> dbt star schema."""

from airflow import DAG
from finguard_common import DEFAULT_ARGS, START, SYNTHETIC, WAREHOUSE_RAW, cli, dbt_build

with DAG(
    dag_id="finguard_02_batch_etl",
    description="Land extracts in the lake (partitioned Parquet), load PostgreSQL raw, rebuild dbt models",
    start_date=START,
    schedule=[SYNTHETIC],
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["finguard", "etl"],
) as dag:
    init_db = cli("init_warehouse_schema", "data_engineering.pipelines init-db")
    lake = cli("ingest_lake_raw_zone", "data_engineering.pipelines ingest-lake")
    raw = cli("load_warehouse_raw", "data_engineering.pipelines load-raw", outlets=[WAREHOUSE_RAW])
    dbt = dbt_build()
    init_db >> lake >> raw >> dbt
