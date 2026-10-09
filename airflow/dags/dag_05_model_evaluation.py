"""5. Model evaluation: threshold analysis, drift/performance monitoring, registry validation gates."""

from airflow import DAG
from finguard_common import DEFAULT_ARGS, MODELS_DEV, MODELS_VALIDATED, START, cli

with DAG(
    dag_id="finguard_05_model_evaluation",
    description="Decision-threshold analysis, monitoring report, and automated validation gates",
    start_date=START,
    schedule=[MODELS_DEV],
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["finguard", "ml", "mlops"],
) as dag:
    thresholds = cli("threshold_analysis", "ml.fraud_detection.thresholds")
    monitoring = cli("monitoring_report", "monitoring")
    gates = cli("registry_validation_gates", "ml validate", outlets=[MODELS_VALIDATED])
    thresholds >> monitoring >> gates
