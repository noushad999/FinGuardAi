select merchant_id, merchant_type, merchant_category, district, risk_category, registration_date
from {{ source('raw', 'merchants') }}
