-- Grain: one row per calendar day covering the transaction history (+30 days for forecasts).
with bounds as (
    select min(event_date_local) as d0, max(event_date_local) + 30 as d1 from {{ ref('stg_transactions') }}
)
select
    to_char(d, 'YYYYMMDD')::int                     as date_key,
    d::date                                         as date,
    extract(year from d)::int                       as year,
    extract(quarter from d)::int                    as quarter,
    extract(month from d)::int                      as month,
    to_char(d, 'Mon YYYY')                          as month_label,
    extract(isodow from d)::int                     as iso_day_of_week,
    to_char(d, 'Dy')                                as day_name,
    extract(isodow from d) in (5, 6)                as is_weekend,     -- Friday/Saturday in Bangladesh
    extract(day from d) <= 7                        as is_month_start_week
from bounds, generate_series(bounds.d0, bounds.d1, interval '1 day') as d
