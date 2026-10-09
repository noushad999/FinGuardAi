-- Grain: one row per customer per local calendar day with any activity.
select
    customer_id,
    date_key,
    count(*)                                                        as txn_count,
    count(*) filter (where is_success)                              as success_count,
    count(*) filter (where not is_success)                          as failed_count,
    sum(amount) filter (where is_success)                           as amount_total,
    sum(amount) filter (where is_success and direction = 'DEBIT')   as debit_amount,
    sum(amount) filter (where is_success and direction = 'CREDIT')  as credit_amount,
    count(*) filter (where transaction_type = 'P2P')                as p2p_count,
    count(*) filter (where transaction_type = 'CASH_OUT')           as cash_out_count,
    count(*) filter (where transaction_type = 'MERCHANT_PAYMENT')   as merchant_payment_count,
    count(distinct coalesce(receiver_customer_id, merchant_id))     as distinct_counterparties,
    count(distinct device_id)                                       as distinct_devices,
    count(*) filter (where is_reported_fraud)                       as reported_fraud_count
from {{ ref('fact_transactions') }}
group by 1, 2
