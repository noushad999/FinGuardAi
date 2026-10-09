-- Grain: one row per transaction with its most authoritative prediction
-- (real-time decision if the transaction went through the risk engine, else the batch score).
with ranked as (
    select p.*, row_number() over (
        partition by transaction_id
        order by case when source = 'batch' then 1 else 0 end, scored_at desc nulls last) as rn
    from {{ ref('fact_fraud_predictions') }} p
)
select
    t.*,
    r.source          as prediction_source,
    r.model_version,
    r.fraud_probability,
    r.account_takeover_probability,
    r.risk_score,
    r.risk_level,
    r.decision,
    c.segment_name,
    c.age_band,
    d.device_type,
    d.os,
    l.division,
    l.district        as txn_district
from {{ ref('fact_transactions') }} t
left join ranked r on r.transaction_id = t.transaction_id and r.rn = 1
left join {{ ref('dim_customer') }} c on c.customer_id = t.customer_id
left join {{ ref('dim_device') }} d on d.device_id = t.device_id
left join {{ ref('dim_location') }} l on l.location_key = t.location_key
