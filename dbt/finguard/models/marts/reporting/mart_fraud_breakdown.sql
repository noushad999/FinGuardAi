-- Fraud intelligence by dimension. Grain: one row per (dimension, value).
{% set dims = [('transaction_type', 'transaction_type'), ('district', 'txn_district'), ('division', 'division'),
               ('channel', 'channel'), ('device_type', "coalesce(device_type, 'agent-initiated')"),
               ('segment', 'segment_name'), ('hour_local', 'event_hour_local::text'), ('age_band', 'age_band')] %}
{% for name, expr in dims %}
select
    '{{ name }}'                                                  as dimension,
    {{ expr }}                                                    as value,
    count(*)                                                      as transactions,
    count(*) filter (where is_reported_fraud)                     as fraud_transactions,
    count(*) filter (where is_reported_fraud)::float / count(*)   as fraud_rate,
    coalesce(sum(amount) filter (where is_reported_fraud and is_success), 0) as fraud_value,
    count(*) filter (where decision in ('REVIEW', 'BLOCK'))       as flagged_transactions,
    avg(risk_score)                                               as avg_risk_score
from {{ ref('mart_transaction_risk') }}
group by 1, 2
{% if not loop.last %}union all{% endif %}
{% endfor %}
