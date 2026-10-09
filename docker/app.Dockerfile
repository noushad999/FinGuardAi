# syntax=docker/dockerfile:1.7
# FinGuard application image: API, streaming risk engine, pipelines, ML training, dbt, dashboard, MLflow server.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app
COPY requirements/ requirements/
# Optional build secret "extra_ca": a CA bundle for builds behind a TLS-intercepting corporate proxy.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca SSL_CERT_FILE=/run/secrets/extra_ca REQUESTS_CA_BUNDLE=/run/secrets/extra_ca; fi; \
    pip install -r requirements/base.txt -r requirements/dashboard.txt -r requirements/dbt.txt

RUN useradd --create-home --uid 1000 finguard
COPY --chown=finguard . .
USER finguard

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
