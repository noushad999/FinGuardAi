-- Model scores from both paths: batch backfill (ml.transaction_scores) and real time (ops.risk_decisions).
with batch as (
    select
        s.transaction_id,
        'batch'                                                     as source,
        s.fraud_model_version                                       as model_version,
        s.fraud_probability,
        s.ato_probability                                           as account_takeover_probability,
        s.anomaly_percentile,
        round(100 * greatest(s.fraud_probability, coalesce(s.ato_probability, 0)))::int as risk_score,
        null::timestamptz                                           as scored_at,
        s.split                                                     as evaluation_split
    from {{ source('ml', 'transaction_scores') }} s
),
realtime as (
    select
        d.transaction_id,
        d.source,
        d.model_version,
        d.fraud_probability,
        d.account_takeover_probability,
        d.anomaly_percentile,
        d.risk_score,
        d.scored_at,
        null::text as evaluation_split
    from {{ source('ops', 'risk_decisions') }} d
)
select
    *,
    case when risk_score >= {{ var('block_threshold') }} then 'HIGH'
         when risk_score >= {{ var('review_threshold') }} then 'MEDIUM' else 'LOW' end as risk_level,
    case when risk_score >= {{ var('block_threshold') }} then 'BLOCK'
         when risk_score >= {{ var('review_threshold') }} then 'REVIEW' else 'APPROVE' end as decision
from (select * from batch union all select * from realtime) u
