-- Batch ML outputs (full refresh by the ML pipelines / loader).

CREATE TABLE IF NOT EXISTS ml.transaction_scores (
    transaction_id          TEXT PRIMARY KEY,
    fraud_probability       DOUBLE PRECISION NOT NULL,
    ato_probability         DOUBLE PRECISION,
    anomaly_percentile      DOUBLE PRECISION,
    behavioral_percentile   DOUBLE PRECISION,
    split                   TEXT,
    fraud_model_version     TEXT,
    ato_model_version       TEXT
);

CREATE TABLE IF NOT EXISTS ml.customer_segments (
    customer_id            TEXT PRIMARY KEY,
    segment_id             INTEGER,
    segment_name           TEXT,
    hdbscan_cluster        INTEGER,
    n_txn                  INTEGER,
    monthly_spend          DOUBLE PRECISION,
    segmentation_version   TEXT
);

CREATE TABLE IF NOT EXISTS ml.churn_scores (
    customer_id               TEXT PRIMARY KEY,
    churn_probability         DOUBLE PRECISION,
    churn_model_version       TEXT,
    as_of                     TIMESTAMPTZ,
    days_since_last_success   DOUBLE PRECISION,
    activity_trend            DOUBLE PRECISION,
    n_success_lookback        DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS ml.graph_customer_metrics (
    customer_id                TEXT PRIMARY KEY,
    in_degree                  DOUBLE PRECISION,
    out_degree                 DOUBLE PRECISION,
    weighted_in                DOUBLE PRECISION,
    weighted_out               DOUBLE PRECISION,
    pagerank                   DOUBLE PRECISION,
    betweenness                DOUBLE PRECISION,
    community_id               INTEGER,
    community_size             INTEGER,
    device_share_max_accounts  INTEGER,
    device_component_size      INTEGER,
    mule_score                 DOUBLE PRECISION,
    mule_score_percentile      DOUBLE PRECISION,
    in_suspicious_cluster      BOOLEAN,
    cluster_risk_score         DOUBLE PRECISION,
    is_suspicious_hub          BOOLEAN
);

CREATE TABLE IF NOT EXISTS ml.graph_communities (
    community_id         INTEGER PRIMARY KEY,
    size                 INTEGER,
    mule_score_mean      DOUBLE PRECISION,
    mule_score_max       DOUBLE PRECISION,
    max_device_sharing   INTEGER,
    total_inflow         DOUBLE PRECISION,
    flagged_txn_count    DOUBLE PRECISION,
    flagged_per_member   DOUBLE PRECISION,
    risk_score           DOUBLE PRECISION,
    is_suspicious        BOOLEAN
);

CREATE TABLE IF NOT EXISTS ml.graph_suspicious_edges (
    source     TEXT,
    target     TEXT,
    weight     DOUBLE PRECISION,
    edge_type  TEXT
);

CREATE TABLE IF NOT EXISTS ml.customer_profiles (
    customer_id                  TEXT PRIMARY KEY,
    district                     TEXT,
    product_tier                 TEXT,
    account_status               TEXT,
    fraud_probability_30d_max    DOUBLE PRECISION,
    fraud_probability_30d_mean   DOUBLE PRECISION,
    ato_probability_30d_max      DOUBLE PRECISION,
    flagged_txn_30d              INTEGER,
    churn_probability            DOUBLE PRECISION,
    segment_name                 TEXT,
    mule_score_percentile        DOUBLE PRECISION,
    in_suspicious_cluster        BOOLEAN,
    community_id                 DOUBLE PRECISION,
    txn_last_7d                  DOUBLE PRECISION,
    weekly_baseline              DOUBLE PRECISION,
    weekly_cv                    DOUBLE PRECISION,
    recent_activity              TEXT,
    behavior                     TEXT,
    risk_score                   INTEGER,
    risk_level                   TEXT,
    risk_reasons                 JSONB,
    as_of                        TIMESTAMPTZ
);
