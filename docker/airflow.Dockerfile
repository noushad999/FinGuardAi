# syntax=docker/dockerfile:1.7
# Airflow orchestrates; the FinGuard code runs in an isolated virtualenv so its
# dependencies (numpy 2, SQLAlchemy 2, MLflow 3, dbt) never conflict with Airflow's.
FROM apache/airflow:2.11.0-python3.11

USER root
RUN mkdir -p /opt/finguard-venv /opt/finguard && chown airflow:0 /opt/finguard-venv /opt/finguard
USER airflow

COPY --chown=airflow requirements/ /opt/finguard/requirements/
RUN --mount=type=secret,id=extra_ca,required=false,uid=50000 \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca SSL_CERT_FILE=/run/secrets/extra_ca REQUESTS_CA_BUNDLE=/run/secrets/extra_ca; fi; \
    python -m venv /opt/finguard-venv \
    && /opt/finguard-venv/bin/pip install --no-cache-dir -r /opt/finguard/requirements/base.txt \
       -r /opt/finguard/requirements/dbt.txt
COPY --chown=airflow . /opt/finguard/
ENV FINGUARD_HOME=/opt/finguard FINGUARD_PYTHON=/opt/finguard-venv/bin/python MLFLOW_DISABLE_AGENT_HINT=1
