-- Grain: one row per transaction attempt. FKs: customer, device, merchant, date, location.
select
    t.transaction_id,
    t.customer_id,
    t.device_id,
    t.merchant_id,
    t.receiver_customer_id,
    to_char(t.event_date_local, 'YYYYMMDD')::int as date_key,
    md5(t.district)                              as location_key,
    t.event_time,
    t.event_hour_local,
    t.transaction_type,
    t.channel,
    t.direction,
    t.status,
    t.is_success,
    t.failure_reason,
    t.amount,
    t.fee,
    t.balance_before,
    t.balance_after,
    t.pin_attempts,
    coalesce(l.is_reported_fraud, false) as is_reported_fraud,
    coalesce(l.is_reported_ato, false)   as is_reported_ato,
    l.fraud_scenario,
    l.fraud_reported_at
from {{ ref('stg_transactions') }} t
left join {{ ref('stg_fraud_labels') }} l using (transaction_id)
