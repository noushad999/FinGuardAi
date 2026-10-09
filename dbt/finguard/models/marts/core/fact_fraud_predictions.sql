-- Grain: one row per (transaction, prediction source, model version).
-- Joins the label as known today so realised precision/recall can be tracked as labels arrive.
select
    md5(p.transaction_id || '|' || p.source || '|' || coalesce(p.model_version, '')) as prediction_key,
    p.transaction_id,
    t.customer_id,
    t.date_key,
    t.location_key,
    t.transaction_type,
    t.amount,
    p.source,
    p.model_version,
    p.fraud_probability,
    p.account_takeover_probability,
    p.anomaly_percentile,
    p.risk_score,
    p.risk_level,
    p.decision,
    p.scored_at,
    p.evaluation_split,
    t.is_reported_fraud,
    t.fraud_scenario
from {{ ref('stg_predictions') }} p
join {{ ref('fact_transactions') }} t using (transaction_id)
