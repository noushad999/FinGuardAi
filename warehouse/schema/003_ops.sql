-- Operational tables written in (near) real time.

-- One row per scored transaction (idempotent on transaction_id: replays/duplicates are no-ops).
CREATE TABLE IF NOT EXISTS ops.risk_decisions (
    transaction_id                 TEXT PRIMARY KEY,
    customer_id                    TEXT NOT NULL,
    event_time                     TIMESTAMPTZ,
    scored_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
    source                         TEXT NOT NULL,            -- stream | api
    transaction_type               TEXT,
    amount                         NUMERIC(14, 2),
    fraud_probability              DOUBLE PRECISION NOT NULL,
    account_takeover_probability   DOUBLE PRECISION,
    anomaly_percentile             DOUBLE PRECISION,
    behavioral_anomaly_percentile  DOUBLE PRECISION,
    risk_score                     INTEGER NOT NULL,
    risk_level                     TEXT NOT NULL,
    decision                       TEXT NOT NULL,
    risk_reasons                   JSONB,
    model_version                  TEXT NOT NULL,
    model_versions                 JSONB,
    policy_version                 TEXT,
    latency_ms                     DOUBLE PRECISION,
    features                       JSONB
);
CREATE INDEX IF NOT EXISTS ix_decisions_scored_at ON ops.risk_decisions (scored_at);
CREATE INDEX IF NOT EXISTS ix_decisions_customer ON ops.risk_decisions (customer_id, event_time);

-- Alert queue for fraud analysts (REVIEW / BLOCK decisions).
CREATE TABLE IF NOT EXISTS ops.fraud_alerts (
    alert_id          BIGSERIAL PRIMARY KEY,
    transaction_id    TEXT UNIQUE NOT NULL,
    customer_id       TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_time        TIMESTAMPTZ,
    risk_score        INTEGER NOT NULL,
    risk_level        TEXT NOT NULL,
    decision          TEXT NOT NULL,
    reasons           JSONB,
    model_version     TEXT,
    status            TEXT NOT NULL DEFAULT 'OPEN'   -- OPEN | CONFIRMED_FRAUD | FALSE_POSITIVE
);
CREATE INDEX IF NOT EXISTS ix_alerts_created ON ops.fraud_alerts (created_at DESC);

-- Spark Structured Streaming: per-customer event-time window aggregates.
CREATE TABLE IF NOT EXISTS ops.stream_customer_windows (
    customer_id        TEXT NOT NULL,
    window_start       TIMESTAMPTZ NOT NULL,
    window_end         TIMESTAMPTZ NOT NULL,
    window_size        TEXT NOT NULL,              -- '5 minutes' | '1 hour'
    txn_count          INTEGER NOT NULL,
    amount_sum         DOUBLE PRECISION,
    amount_avg         DOUBLE PRECISION,
    unique_receivers   INTEGER,
    unique_devices     INTEGER,
    failed_count       INTEGER,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, window_size, window_start)
);

-- Spark Structured Streaming: platform-wide per-minute throughput.
CREATE TABLE IF NOT EXISTS ops.stream_metrics_minute (
    window_start       TIMESTAMPTZ PRIMARY KEY,
    txn_count          INTEGER NOT NULL,
    amount_sum         DOUBLE PRECISION,
    failed_count       INTEGER,
    unique_customers   INTEGER,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Batch/stream job audit log.
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    pipeline      TEXT NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL,
    finished_at   TIMESTAMPTZ,
    status        TEXT NOT NULL,
    rows_written  BIGINT,
    details       JSONB
);

CREATE TABLE IF NOT EXISTS ops.data_quality_results (
    run_at        TIMESTAMPTZ NOT NULL,
    dataset       TEXT NOT NULL,
    check_name    TEXT NOT NULL,
    severity      TEXT NOT NULL,
    passed        BOOLEAN NOT NULL,
    failing_rows  BIGINT,
    details       JSONB,
    PRIMARY KEY (run_at, dataset, check_name)
);

CREATE TABLE IF NOT EXISTS ops.model_monitoring (
    run_at        TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    metric_type   TEXT NOT NULL,     -- feature_psi | prediction_psi | performance
    name          TEXT NOT NULL,
    value         DOUBLE PRECISION,
    status        TEXT,
    details       JSONB,
    PRIMARY KEY (run_at, model_version, metric_type, name)
);
