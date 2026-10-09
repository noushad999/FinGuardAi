select device_id, customer_id as primary_customer_id, device_type, os, device_age_days, first_seen, last_seen
from {{ source('raw', 'devices') }}
