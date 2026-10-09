-- Grain: one row per device. accounts_seen = distinct accounts observed on the handset.
select
    d.device_id,
    d.primary_customer_id,
    d.device_type,
    d.os,
    d.device_age_days,
    d.first_seen,
    d.last_seen,
    coalesce(a.accounts_seen, 0) as accounts_seen,
    coalesce(a.accounts_seen, 0) >= 3 as is_shared_device
from {{ ref('stg_devices') }} d
left join (
    select device_id, count(distinct customer_id) as accounts_seen
    from {{ ref('stg_transactions') }} where device_id is not null group by 1
) a using (device_id)
