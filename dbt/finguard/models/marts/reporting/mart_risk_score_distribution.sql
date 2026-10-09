-- Grain: one row per 5-point risk-score bucket.
select
    (risk_score / 5) * 5                              as risk_score_bucket,
    count(*)                                          as transactions,
    count(*) filter (where is_reported_fraud)         as reported_fraud,
    count(*) filter (where is_reported_fraud)::float / count(*) as observed_fraud_rate
from {{ ref('mart_transaction_risk') }}
where risk_score is not null
group by 1
order by 1
