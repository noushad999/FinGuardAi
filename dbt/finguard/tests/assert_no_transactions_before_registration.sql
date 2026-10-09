-- Singular test: a customer cannot transact before registering (timestamp integrity).
select t.transaction_id
from {{ ref('fact_transactions') }} t
join {{ ref('dim_customer') }} c using (customer_id)
where t.event_time < c.registration_date
