-- Customer intelligence. Grain: one row per segment.
select
    c.segment_name,
    count(*)                                         as customers,
    avg(c.churn_probability)                         as avg_churn_probability,
    count(*) filter (where c.churn_probability >= 0.5) as customers_high_churn_risk,
    count(*) filter (where c.customer_risk_level = 'High') as high_risk_customers,
    avg(a.txn_30d)                                   as avg_txn_last_30d,
    avg(a.value_30d)                                 as avg_value_last_30d
from {{ ref('dim_customer') }} c
left join (
    select f.customer_id,
           sum(f.success_count) as txn_30d,
           sum(f.amount_total) as value_30d
    from {{ ref('fact_customer_activity') }} f
    join {{ ref('dim_date') }} d on d.date_key = f.date_key
    where d.date > (select max(event_date_local) from {{ ref('stg_transactions') }}) - 30
    group by 1
) a using (customer_id)
group by 1
order by customers desc
