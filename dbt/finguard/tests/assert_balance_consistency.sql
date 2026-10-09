-- Singular test: ledger arithmetic must hold for every transaction.
--   successful debit : balance_after = balance_before - amount - fee
--   successful credit: balance_after = balance_before + amount
--   failed           : balance unchanged
select transaction_id
from {{ ref('fact_transactions') }}
where balance_after < 0
   or (is_success and direction = 'DEBIT'  and abs(balance_after - (balance_before - amount - fee)) > 0.05)
   or (is_success and direction = 'CREDIT' and abs(balance_after - (balance_before + amount)) > 0.05)
   or (not is_success and abs(balance_after - balance_before) > 0.05)
