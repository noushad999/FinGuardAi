-- Grain: one row per business counterparty (merchant, agent, biller, operator, bank).
select
    merchant_id,
    merchant_type,
    merchant_category,
    district,
    md5(district) as location_key,
    risk_category,
    registration_date
from {{ ref('stg_merchants') }}
