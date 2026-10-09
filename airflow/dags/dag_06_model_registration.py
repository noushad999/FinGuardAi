"""6. Model registration: promote validated models to production, deploy bundles, publish batch scores."""

from airflow import DAG
from finguard_common import DEFAULT_ARGS, MODELS_PROD, MODELS_VALIDATED, START, cli, dbt_build

with DAG(
    dag_id="finguard_06_model_registration",
    description="validation -> production alias (previous kept for rollback), deploy, load ML outputs, refresh marts",
    start_date=START,
    schedule=[MODELS_VALIDATED],
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["finguard", "mlops"],
) as dag:
    promote = cli("promote_to_production", "ml promote")
    deploy = cli("deploy_production_bundles", "ml deploy", outlets=[MODELS_PROD])
    load_ml = cli("load_ml_outputs", "data_engineering.pipelines load-ml")
    marts = dbt_build("refresh_marts")
    hydrate = cli("bootstrap_online_feature_store", "features hydrate --if-empty")
    promote >> deploy >> [load_ml, hydrate]
    load_ml >> marts
