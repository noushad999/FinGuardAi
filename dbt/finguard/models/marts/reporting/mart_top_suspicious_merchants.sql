-- Grain: one row per retail merchant with risk indicators.
select
    m.merchant_id,
    m.merchant_category,
    m.risk_category,
    m.district,
    m.registration_date,
    count(*)                                               as transactions,
    count(distinct t.customer_id)                          as distinct_customers,
    sum(t.amount) filter (where t.is_success)              as value,
    count(*) filter (where t.is_reported_fraud)            as reported_fraud_transactions,
    count(*) filter (where t.is_reported_fraud)::float / count(*) as reported_fraud_rate,
    avg(t.fraud_probability)                               as avg_fraud_probability,
    count(*) filter (where t.decision in ('REVIEW', 'BLOCK')) as flagged_transactions
from {{ ref('mart_transaction_risk') }} t
join {{ ref('dim_merchant') }} m on m.merchant_id = t.merchant_id
where m.merchant_type = 'merchant'
group by 1, 2, 3, 4, 5
having count(*) >= 5
order by avg(t.fraud_probability) desc
