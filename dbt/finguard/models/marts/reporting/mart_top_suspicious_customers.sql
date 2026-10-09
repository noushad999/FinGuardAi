-- Grain: one row per customer with Medium/High risk, ranked for analyst work queues.
select
    p.customer_id,
    p.risk_level,
    p.risk_score,
    p.fraud_probability_30d_max,
    p.ato_probability_30d_max,
    p.flagged_txn_30d,
    p.mule_score_percentile,
    p.in_suspicious_cluster,
    p.segment_name,
    p.district,
    p.recent_activity,
    p.behavior,
    p.risk_reasons
from {{ source('ml', 'customer_profiles') }} p
where p.risk_level in ('High', 'Medium')
order by p.risk_score desc, p.flagged_txn_30d desc
