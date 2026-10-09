-- Raw zone: 1:1 with the lake extracts. Loaded by data_engineering/pipelines/load_warehouse.py
-- (full refresh inside a transaction -> idempotent re-runs).

CREATE TABLE IF NOT EXISTS raw.customers (
    customer_id          TEXT PRIMARY KEY,
    age                  INTEGER,
    registration_date    TIMESTAMPTZ NOT NULL,
    occupation_category  TEXT,
    district             TEXT,
    upazila              TEXT,
    customer_segment     TEXT,
    kyc_level            TEXT,
    account_status       TEXT
);

CREATE TABLE IF NOT EXISTS raw.merchants (
    merchant_id          TEXT PRIMARY KEY,
    merchant_type        TEXT NOT NULL,
    merchant_category    TEXT,
    district             TEXT,
    risk_category        TEXT,
    registration_date    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS raw.devices (
    device_id            TEXT PRIMARY KEY,
    customer_id          TEXT,
    device_type          TEXT,
    os                   TEXT,
    device_age_days      INTEGER,
    first_seen           TIMESTAMPTZ,
    last_seen            TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS raw.transactions (
    transaction_id       TEXT PRIMARY KEY,
    event_time           TIMESTAMPTZ NOT NULL,
    sender_id            TEXT,
    receiver_id          TEXT,
    customer_id          TEXT NOT NULL,
    merchant_id          TEXT,
    amount               NUMERIC(14, 2) NOT NULL,
    fee                  NUMERIC(12, 2),
    currency             TEXT,
    transaction_type     TEXT NOT NULL,
    channel              TEXT,
    device_id            TEXT,
    district             TEXT,
    latitude             DOUBLE PRECISION,
    longitude            DOUBLE PRECISION,
    status               TEXT NOT NULL,
    failure_reason       TEXT,
    balance_before       NUMERIC(14, 2),
    balance_after        NUMERIC(14, 2),
    metadata             JSONB
);
CREATE INDEX IF NOT EXISTS ix_raw_tx_customer_time ON raw.transactions (customer_id, event_time);
CREATE INDEX IF NOT EXISTS ix_raw_tx_time ON raw.transactions (event_time);

-- Fraud case-management feed (labels arrive late; see fraud_reported_at).
CREATE TABLE IF NOT EXISTS raw.transaction_labels (
    transaction_id       TEXT PRIMARY KEY,
    is_fraud             BOOLEAN NOT NULL,
    is_ato               BOOLEAN NOT NULL,
    fraud_scenario       TEXT,
    fraud_case_id        TEXT,
    fraud_reported_at    TIMESTAMPTZ
);
