"""4. Model training: fraud (supervised + unsupervised + behavioural), ATO, graph, segmentation, churn, profiles."""

from airflow import DAG
from finguard_common import DEFAULT_ARGS, FEATURES, MODELS_DEV, START, cli

with DAG(
    dag_id="finguard_04_model_training",
    description="Train all models on temporal splits; every run is tracked in MLflow and registered as @development",
    start_date=START,
    schedule=[FEATURES],
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["finguard", "ml"],
) as dag:
    fraud = cli("train_fraud_models", "ml train-fraud")
    ato = cli("train_account_takeover", "ml train-ato")
    graph = cli("graph_analytics", "ml graph")
    segment = cli("customer_segmentation", "ml segment")
    churn = cli("churn_model", "ml churn")
    profile = cli("customer_profiles", "ml profile", outlets=[MODELS_DEV])
    fraud >> ato >> graph  # graph uses fraud scores for cluster ranking
    [graph, segment, churn] >> profile
