# FinGuard - common tasks.  `make help` lists them.
SHELL := /bin/bash
PY ?= python
COMPOSE ?= docker compose

.DEFAULT_GOAL := help
.PHONY: help install data features train pipeline-local test test-unit test-integration lint format \
        env build up pipeline apps replay airflow demo down clean logs ps smoke dbt dq monitor

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- local (no Docker)
install: ## Create .venv and install all Python dependencies
	python3.11 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements/dev.txt

data: ## Generate synthetic data (DATASET_SIZE=small|medium|large|<n>)
	$(PY) -m simulator

features: ## Build point-in-time training features
	$(PY) -m features build

train: ## Train, evaluate and register every model (MLflow: ./mlruns)
	$(PY) -m ml train-all

pipeline-local: data features train ## Data -> features -> models, without any infrastructure

dq: ## Data-quality report on the synthetic extracts
	$(PY) -m data_engineering.validation

monitor: ## Drift / performance monitoring report
	$(PY) -m monitoring

test: test-unit ## Run the unit test suite

test-unit: ## Unit + ML tests (no services needed)
	$(PY) -m pytest -m "not integration" -q

test-integration: ## Integration tests against the running Docker stack (make up pipeline apps first)
	FINGUARD_INTEGRATION=1 $(PY) -m pytest -m integration -q

lint: ## Ruff lint
	$(PY) -m ruff check .

format: ## Ruff format
	$(PY) -m ruff format .

# ---------------------------------------------------------------- Docker platform
env:
	@test -f .env || cp .env.example .env

build: env ## Build all images
	$(COMPOSE) --profile apps --profile pipeline --profile replay build

up: env ## Start infrastructure (Postgres, Redis, Kafka, S3, MLflow, Prometheus) and bootstrap it
	$(COMPOSE) up -d postgres redis kafka seaweedfs bootstrap mlflow prometheus
	$(COMPOSE) up bootstrap --exit-code-from bootstrap

pipeline: env ## Run the full batch pipeline in Docker (data, DQ, lake, warehouse, features, ML, registry, dbt)
	$(COMPOSE) --profile pipeline run --rm pipeline

apps: env ## Start API, streaming risk engine, Spark streaming and the dashboard
	$(COMPOSE) --profile apps up -d api risk-engine spark-streaming dashboard

replay: env ## Stream the test period through Kafka (REPLAY_RATE events/s)
	$(COMPOSE) --profile replay run --rm producer

airflow: env ## Start Airflow (http://localhost:8080, admin/admin)
	$(COMPOSE) --profile airflow up -d airflow

demo: up pipeline apps replay ## Everything: infra, pipeline, apps, then a streaming replay
	@echo "API http://localhost:8000/docs | Dashboard http://localhost:8501 | MLflow http://localhost:5000 | Prometheus http://localhost:9090"

smoke: ## Quick end-to-end check of a running stack
	bash scripts/smoke_test.sh

dbt: ## dbt build against the local warehouse
	cd dbt/finguard && DBT_PROFILES_DIR=. dbt build

ps: ## Service status
	$(COMPOSE) --profile apps --profile airflow ps

logs: ## Tail application logs
	$(COMPOSE) --profile apps logs -f --tail=100 api risk-engine spark-streaming

down: ## Stop everything (keeps volumes)
	$(COMPOSE) --profile apps --profile airflow --profile pipeline --profile replay down

clean: ## Stop everything and delete volumes + generated data
	$(COMPOSE) --profile apps --profile airflow --profile pipeline --profile replay down -v
	rm -rf data/synthetic/*.parquet data/processed/*.parquet data/lake mlruns models/*/
