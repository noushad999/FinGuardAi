# Architecture

FinGuard simulates the risk and analytics backbone of a Mobile Financial Services (MFS) wallet
platform. **All data is synthetic**; nothing is connected to a real provider.

The design goal is *architectural coherence*: every component exists because a requirement
needs it, and the online and offline paths share code wherever correctness depends on it
(features, policy, schemas).

## 1. System architecture

```mermaid
flowchart LR
    subgraph Sources
        SIM[Transaction simulator<br/>simulator/]
        CM[Case management<br/>late fraud labels]
    end

    subgraph Streaming["Real-time path"]
        K[(Kafka<br/>fg.transactions)]
        DLQ[(fg.transactions.dlq)]
        AL[(fg.alerts)]
        RE[Risk engine worker<br/>services/risk_engine]
        SP[Spark Structured Streaming<br/>streaming/spark_jobs]
        R[(Redis<br/>online feature state)]
    end

    subgraph Storage
        S3[(S3 data lake<br/>SeaweedFS: raw / bronze / quarantine)]
        PG[(PostgreSQL<br/>raw · ops · ml · analytics)]
        ML[(MLflow<br/>tracking + registry)]
    end

    subgraph Serving
        API[FastAPI<br/>api/]
        DASH[Dashboard<br/>Streamlit]
        PBI[Power BI / SQL clients]
        PROM[Prometheus]
    end

    SIM -- replay producer --> K
    K --> RE
    K --> SP
    RE -- invalid --> DLQ
    RE <--> R
    RE -- decisions + alerts --> PG
    RE -- REVIEW/BLOCK --> AL
    SP -- bronze Parquet --> S3
    SP -- windows / throughput --> PG
    SIM -- batch extracts --> S3
    CM --> S3
    S3 -- ETL --> PG
    PG -- dbt --> PG
    ML <--> AF[Airflow DAGs]
    AF --> PG
    API <--> R
    API --> PG
    API -- loads @production --> ML
    DASH --> PG
    DASH --> API
    DASH --> PROM
    PBI --> PG
    PROM -- scrape --> API
    PROM -- scrape --> RE
```

| Component | Why it exists |
|---|---|
| **Kafka** (KRaft, no ZooKeeper) | Durable, replayable, partitioned event log. Decouples producers from several independent consumers (risk engine, Spark) that progress at their own pace; keyed by `customer_id` so per-customer order is preserved. |
| **Risk engine worker** | Per-event, low-latency, *stateful* scoring (features need the customer's previous events). Runs the same `FeatureEngine` as training. |
| **Redis** | Shared online feature state (customer / device / merchant documents) so the API and N workers see the same state; also the de-duplication set. |
| **Spark Structured Streaming** | High-throughput stream ETL: validation, quarantine, append-only bronze lake, event-time windows with watermarks. Throughput-oriented, not per-event decisions. |
| **S3-compatible lake (SeaweedFS)** | Cheap immutable storage of raw extracts and the bronze stream for replay/backfills. SeaweedFS replaces MinIO (whose images are no longer published); any S3 endpoint works. |
| **PostgreSQL** | Warehouse (raw → dbt star schema) and operational store (decisions, alerts, monitoring). One engine keeps the local stack light; the SQL is warehouse-portable. |
| **dbt** | Versioned, tested SQL transformations (star schema + marts) with lineage and data tests. |
| **Airflow** | Orchestrates batch pipelines with Dataset-driven scheduling, retries and idempotent tasks. |
| **MLflow** | Experiment tracking, model registry, alias-based promotion (`development → validation → production`) and rollback. |
| **FastAPI** | Synchronous scoring for authorisation-time decisions and read APIs for customer/transaction/alert intelligence. |
| **Prometheus** | System + ML metrics (throughput, latency, lag, decision mix, score distribution) and alert rules. |
| **Streamlit dashboard** | Decision-oriented pages over the marts; Power BI can use the same `analytics` schema. |

Deliberately **not** included: ZooKeeper (KRaft), Grafana (Prometheus + the dashboard's operations
page cover it), a feature-store product (Redis + one Python engine is enough at this scale), a
GNN (see [ml_methodology.md](ml_methodology.md#graph-analytics)).

## 2. Data pipeline

```mermaid
flowchart TB
    subgraph Batch["Offline / analytics path (Airflow)"]
        A1[simulator: extracts<br/>customers, devices, merchants,<br/>transactions, labels] --> A2{Data-quality gate}
        A2 -- pass --> A3[Lake raw zone<br/>Parquet, event_date partitions]
        A2 -- fail --> X1[Stop + report]
        A3 --> A4[PostgreSQL raw.*<br/>TRUNCATE + COPY]
        A4 --> A5[dbt staging views]
        A5 --> A6[dbt core: dims + facts]
        A6 --> A7[dbt reporting marts]
        A7 --> A8[Dashboard / Power BI]
    end
    subgraph Stream["Real-time path"]
        B1[Kafka fg.transactions] --> B2[Spark: parse + validate]
        B2 -- invalid --> B3[lake quarantine/]
        B2 -- valid --> B4[lake bronze/ append-only]
        B2 -- watermark + dedup --> B5[event-time windows]
        B5 --> B6[ops.stream_customer_windows<br/>ops.stream_metrics_minute]
        B1 --> B7[Risk engine] --> B8[ops.risk_decisions<br/>ops.fraud_alerts]
    end
    B8 --> A5
```

Idempotency everywhere: lake partitions are overwritten (`delete_matching`), warehouse loads are
full refreshes inside a transaction, dbt rebuilds tables, streaming sinks upsert on natural keys,
decisions are `ON CONFLICT DO NOTHING`.

## 3. ML pipeline

```mermaid
flowchart LR
    T[transactions] --> F[Replay through FeatureEngine<br/>point-in-time features]
    L[labels + reported_at] --> J
    F --> J[Join labels<br/>known before cutoff]
    J --> S[Temporal split<br/>train · validation · test]
    S --> X[XGBoost fraud<br/>imbalance strategy on val]
    S --> ATO[XGBoost ATO]
    S --> IF[Isolation Forest<br/>no labels]
    S --> BH[Behavioural baseline<br/>no labels]
    X --> C[Platt calibration<br/>+ thresholds on val]
    C --> E[Test evaluation<br/>PR-AUC, P@k, per-scenario recall, SHAP]
    E --> REG[MLflow: register @development]
    REG --> G{Validation gates}
    G -- pass --> V[@validation] --> P[@production<br/>previous kept for rollback]
    P --> D[models/ bundle<br/>API + risk engine]
    T --> GR[Graph analytics] --> PROF[Customer profiles]
    T --> SEG[Segmentation] --> PROF
    T --> CH[Churn snapshots] --> PROF
    X --> PROF
```

## 4. Fraud decision flow

```mermaid
sequenceDiagram
    participant C as Client / Kafka
    participant FE as FeatureEngine
    participant ST as Redis state
    participant M as Models
    participant P as Policy (configs/policy.yaml)
    participant O as Sinks
    C->>FE: transaction
    FE->>ST: MGET customer, receiver, device, merchant
    ST-->>FE: state documents
    FE->>M: 61 features (history only, never the current event)
    M-->>P: P(fraud), P(ATO), anomaly pct, SHAP
    P-->>O: risk_score = 100·max(P(fraud), P(ATO)) → APPROVE / REVIEW / BLOCK (+ rule escalations)
    FE->>ST: commit event into state (stream / commit=true)
    O->>O: ops.risk_decisions, ops.fraud_alerts, fg.alerts
```

The model never decides; it produces calibrated probabilities and explanations. The policy file
decides and can only be *escalated* by deterministic rules.

## 5. Database schema (star schema, `analytics`)

```mermaid
erDiagram
    DIM_CUSTOMER ||--o{ FACT_TRANSACTIONS : "customer_id"
    DIM_DEVICE ||--o{ FACT_TRANSACTIONS : "device_id"
    DIM_MERCHANT ||--o{ FACT_TRANSACTIONS : "merchant_id"
    DIM_DATE ||--o{ FACT_TRANSACTIONS : "date_key"
    DIM_LOCATION ||--o{ FACT_TRANSACTIONS : "location_key"
    FACT_TRANSACTIONS ||--o{ FACT_FRAUD_PREDICTIONS : "transaction_id"
    DIM_CUSTOMER ||--o{ FACT_CUSTOMER_ACTIVITY : "customer_id"
    DIM_DATE ||--o{ FACT_CUSTOMER_ACTIVITY : "date_key"
    DIM_LOCATION ||--o{ DIM_CUSTOMER : "location_key"

    FACT_TRANSACTIONS {
        text transaction_id PK
        text customer_id FK
        text device_id FK
        text merchant_id FK
        int date_key FK
        text location_key FK
        numeric amount
        text transaction_type
        text status
        bool is_reported_fraud
    }
    FACT_FRAUD_PREDICTIONS {
        text prediction_key PK
        text transaction_id FK
        text source
        text model_version
        float fraud_probability
        int risk_score
        text decision
    }
    FACT_CUSTOMER_ACTIVITY {
        text customer_id PK
        int date_key PK
        int txn_count
        numeric amount_total
        int failed_count
    }
    DIM_CUSTOMER {
        text customer_id PK
        text segment_name
        float churn_probability
        text customer_risk_level
    }
    DIM_DEVICE {
        text device_id PK
        int accounts_seen
    }
    DIM_MERCHANT {
        text merchant_id PK
        text merchant_type
        text risk_category
    }
    DIM_DATE {
        int date_key PK
        date date
        bool is_weekend
    }
    DIM_LOCATION {
        text location_key PK
        text district
        text division
    }
```

Grain, keys and lineage per table: [data_dictionary.md](data_dictionary.md).

## 6. Transaction graph

```mermaid
flowchart LR
    V1((victim A)) -- sends_money_to --> M1((mule 1))
    V2((victim B)) -- sends_money_to --> M1
    V3((victim C)) -- sends_money_to --> M2((mule 2))
    M1 -- sends_money_to --> H((ring head))
    M2 -- sends_money_to --> H
    M1 -. uses_device .- D1[/device farm/]
    M2 -. uses_device .- D1
    H -. uses_device .- D1
    H -- pays_merchant / cash-out --> AG[agent]
    C1((customer)) -- pays_merchant --> SHOP[merchant]
    C1 -. uses_device .- D2[/own phone/]
```

Nodes: customers, devices, merchants/agents. Edges: `sends_money_to` (P2P, weighted by amount),
`uses_device`, `pays_merchant`. Communities are found on the P2P + *shared*-device projection,
which is what ties mule rings together (see [fraud_detection.md](fraud_detection.md#graph-signals)).

## 7. Streaming semantics

* **Ordering**: Kafka key = `customer_id` → one customer's events are in one partition, in order;
  the feature engine relies on it.
* **Delivery**: at-least-once. Offsets are committed after a batch is persisted. The risk engine
  claims each `transaction_id` once (Redis `SET NX`) before touching state; sinks are idempotent.
* **Invalid data**: schema violations go to the DLQ with error headers (risk engine) or the lake
  quarantine zone with a reason (Spark). Poison records are isolated after retries so one bad
  message never blocks a partition.
* **Event time vs processing time**: Spark windows use `event_time` with a 10-minute watermark.
  The watermark (max event time seen − 10 min) tells Spark when a window is final and when state
  can be dropped; events older than the watermark are excluded from aggregates (counted as
  `numRowsDroppedByWatermark`) but still land in bronze for batch reconciliation.
* **State**: Spark keeps window and dedup state in checkpointed state stores (HDFS-backed
  provider on the lake), so restarts resume exactly where they stopped. The risk engine's state
  lives in Redis.
