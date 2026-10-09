-- Grain: one row per district. location_key is the stable district code (natural key).
select
    md5(district)            as location_key,
    district,
    division,
    latitude,
    longitude
from {{ ref('districts') }}
