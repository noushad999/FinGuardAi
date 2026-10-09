select
    customer_id,
    age,
    case when age < 25 then '18-24' when age < 35 then '25-34' when age < 45 then '35-44'
         when age < 60 then '45-59' else '60+' end as age_band,
    registration_date,
    occupation_category,
    district,
    upazila,
    customer_segment as product_tier,
    kyc_level,
    account_status
from {{ source('raw', 'customers') }}
