-- Grain: one row per customer (SCD type 1: current attributes + latest ML enrichments).
select
    c.customer_id,
    c.age,
    c.age_band,
    c.registration_date,
    c.occupation_category,
    c.district,
    md5(c.district)                        as location_key,
    c.upazila,
    c.product_tier,
    c.kyc_level,
    c.account_status,
    coalesce(s.segment_name, 'Unsegmented (low activity)') as segment_name,
    ch.churn_probability,
    g.mule_score_percentile,
    coalesce(g.in_suspicious_cluster, false) as in_suspicious_cluster,
    p.risk_level                           as customer_risk_level,
    p.risk_score                           as customer_risk_score
from {{ ref('stg_customers') }} c
left join {{ source('ml', 'customer_segments') }} s using (customer_id)
left join {{ source('ml', 'churn_scores') }} ch using (customer_id)
left join {{ source('ml', 'graph_customer_metrics') }} g using (customer_id)
left join {{ source('ml', 'customer_profiles') }} p using (customer_id)
