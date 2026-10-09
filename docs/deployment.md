# Deployment & operations

## Local stack (Docker Compose)

Requirements: Docker with Compose v2, ~8 GB RAM for the full stack (≈4 GB without Airflow),
~12 GB disk for images.

```bash
make up          # Postgres, Redis, Kafka (KRaft), SeaweedFS (S3), MLflow, Prometheus + bootstrap
make pipeline    # data -> DQ gate -> lake -> warehouse -> features -> models -> registry -> dbt -> Redis
make apps        # API :8000, risk engine :9102, Spark streaming, dashboard :8501
make replay      # stream the test period through Kafka (REPLAY_RATE events/s)
make smoke       # end-to-end smoke test
make airflow     # optional: Airflow :8080 (admin/admin)
make down        # stop (keeps volumes) | make clean  # stop + delete volumes and generated data
```

| Service | URL |
|---|---|
| API (OpenAPI docs) | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Airflow | http://localhost:8080 |
| Kafka (host) | localhost:9094 |
| PostgreSQL | localhost:5432 (finguard/finguard) |
| S3 endpoint | http://localhost:8333 (finguard / finguard-secret) |

Compose profiles keep the default footprint small: infrastructure starts with `make up`;
`apps`, `pipeline`, `replay` and `airflow` are opt-in.

### Images
* `finguard-app` (python:3.11-slim + pinned requirements): API, risk engine, pipelines, ML,
  dbt, dashboard, MLflow server — one image, different commands.
* `finguard-spark` (apache/spark:3.5.6 + Kafka source + S3A jars): the streaming job.
* `finguard-airflow` (apache/airflow:2.11 + an isolated virtualenv with the FinGuard code).

Builds behind a TLS-intercepting corporate proxy: set `EXTRA_CA_CERT=/path/to/ca-bundle.crt`
in `.env`; it is passed as a BuildKit secret (never stored in an image layer).

### Object storage
The lake and MLflow artifacts use an S3 API. SeaweedFS is used because MinIO no longer publishes
community Docker images; any S3-compatible endpoint works by changing `S3_ENDPOINT`/credentials
(`storage.*` in `configs/finguard.yaml`). With `LAKE_URI=data/lake` the lake is a local folder.

## Configuration
`configs/finguard.yaml` + environment overrides (`FINGUARD__SECTION__KEY=value` or short aliases
such as `DATASET_SIZE`, `KAFKA_BOOTSTRAP_SERVERS`, `DATABASE_URL`, `REDIS_URL`,
`MLFLOW_TRACKING_URI`, `MODEL_SOURCE`). Business decision policy: `configs/policy.yaml`
(`POLICY_PATH` to override).

## Scale presets
| Preset | `DATASET_SIZE` | Customers | Generation | Feature replay | Peak RAM |
|---|---|---|---|---|---|
| SMALL (default) | `small` = 100,000 | 2,500 | 2.5 s | 6.1 s | 0.5 GB |
| MEDIUM | `medium` = 1,000,000 | 25,000 | 18.8 s | 85.4 s | 3.0 GB |
| LARGE | `large` = 10,000,000 | 250,000 | not run | not run | plan ≥ 16 GB |

Measured (wall clock, peak RSS) on a 4-vCPU container. LARGE is supported by the code
(vectorised generation, chunked replay, COPY loads) but was not executed here; expect roughly
linear scaling of generation and replay time (~15 min replay).

```bash
DATASET_SIZE=medium make pipeline        # Docker
DATASET_SIZE=1000000 python -m simulator # local
```

## Operations runbook

**Model promotion**
```bash
python -m ml validate                    # gates: metric floor, not worse than production, schema
python -m ml promote                     # @validation -> @production (previous kept)
python -m ml deploy                      # materialise @production bundles into models/
docker compose restart api risk-engine   # rolling restart picks up the new bundles
```
**Rollback**: `python -m ml rollback --model finguard-fraud-xgb && python -m ml deploy --model finguard-fraud-xgb`
then restart the services. Because every bundle carries its feature schema, a rollback to a
model trained on a different feature set fails fast at start-up instead of scoring garbage.

**Policy change** (no retraining): edit `configs/policy.yaml` (thresholds come from
`reports/threshold_analysis.md`), bump `version`, restart API/risk engine.

**Kafka outage**: producers buffer and retry (idempotent producer, `delivery.timeout.ms=120s`);
the risk engine stops consuming and resumes from its last committed offset; nothing is lost
while the retention window (7 days) covers the outage. Spark resumes from its checkpoint. See
the interview guide for the full answer.

**Replaying / backfilling the online store**: `python -m features hydrate --until <ts>` rebuilds
Redis state from history (same engine code as training).

**Alerts** (Prometheus rules in `docker/prometheus/alerts.yml`): consumer lag, API 5xx rate,
p95 scoring latency, DLQ spikes.

## Production hardening (not in the local stack)
Kafka with ≥3 brokers and replication factor 3; managed Postgres/warehouse (Snowflake/BigQuery/
Redshift — the dbt models are portable); Redis Cluster with Lua scripts for atomic multi-key
state updates; secrets from a vault instead of `.env`; TLS/SASL everywhere; the risk engine as a
Kubernetes Deployment scaled on consumer lag (KEDA) up to the partition count; Spark on
Kubernetes/EMR with checkpoints on S3; schema registry (Avro/Protobuf) once several teams
publish to the topic.
