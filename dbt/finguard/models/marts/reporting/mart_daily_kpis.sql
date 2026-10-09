-- Executive overview. Grain: one row per local day.
select
    dd.date,
    count(*)                                                   as transactions,
    count(*) filter (where is_success)                         as successful_transactions,
    sum(amount) filter (where is_success)                      as transaction_value,
    count(distinct customer_id)                                as active_customers,
    count(*) filter (where not is_success)::float / count(*)   as failed_rate,
    count(*) filter (where is_reported_fraud)                  as reported_fraud_transactions,
    coalesce(sum(amount) filter (where is_reported_fraud and is_success), 0) as reported_fraud_value,
    count(*) filter (where is_reported_fraud)::float / count(*) as reported_fraud_rate,
    count(*) filter (where decision = 'REVIEW')                as review_decisions,
    count(*) filter (where decision = 'BLOCK')                 as block_decisions,
    avg(fraud_probability)                                     as avg_fraud_probability
from {{ ref('mart_transaction_risk') }} t
join {{ ref('dim_date') }} dd on dd.date_key = t.date_key
group by 1
order by 1
