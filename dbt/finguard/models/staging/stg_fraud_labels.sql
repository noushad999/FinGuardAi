-- Only *reported* fraud is known to the business; unreported ground truth stays out of analytics.
select
    transaction_id,
    is_fraud and fraud_reported_at is not null  as is_reported_fraud,
    is_ato and fraud_reported_at is not null    as is_reported_ato,
    fraud_scenario,
    fraud_case_id,
    fraud_reported_at
from {{ source('raw', 'transaction_labels') }}
where is_fraud
