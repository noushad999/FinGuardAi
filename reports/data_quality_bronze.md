# Data-quality report: bronze

Run at 2026-09-27T09:40:10.812638+00:00 - 21,399 rows - overall: **FAIL** (1 error checks failed, 0 warnings)

| Check | Severity | Result | Failing rows | Description | Sample |
|---|---|---|---|---|---|
| schema_drift | error | PASS | 0 | Columns and types match the transaction contract |  |
| required_not_null | error | PASS | 0 | No nulls in transaction_id, event_time, customer_id, amount, transaction_type, status |  |
| merchant_id_present_for_merchant_payments | error | PASS | 0 | Merchant payments reference a merchant |  |
| failure_reason_iff_failed | error | PASS | 0 | failure_reason is set exactly for FAILED transactions |  |
| duplicate_transaction_id | error | FAIL | 217 | transaction_id is unique | TXN00000080043, TXN00000080093, TXN00000080602 |
| suspected_duplicate_submission | warning | PASS | 0 | Same customer/time/amount/receiver/type under different transaction ids |  |
| invalid_amount | error | PASS | 0 | 0 < amount <= 10,000,000 BDT |  |
| invalid_timestamp | error | PASS | 0 | event_time parseable, not in the future, not before platform launch |  |
| invalid_enumeration | error | PASS | 0 | transaction_type / channel / status / failure_reason in allowed sets |  |
| invalid_customer_id_format | error | PASS | 0 | customer_id matches CUS_####### |  |
| geo_out_of_bounds | warning | PASS | 0 | Coordinates inside Bangladesh bounding box |  |
| negative_balance | error | PASS | 0 | Balances are never negative |  |
| ledger_arithmetic | error | PASS | 0 | balance_after = balance_before -/+ amount (- fee); unchanged when failed |  |
| unknown_customer | error | PASS | 0 | customer_id exists in customers |  |
| unknown_p2p_receiver | error | PASS | 0 | P2P receivers exist in customers |  |
| transaction_before_registration | error | PASS | 0 | No activity before customer registration |  |
| unknown_device | error | PASS | 0 | device_id exists in devices |  |
| unknown_merchant | error | PASS | 0 | merchant_id exists in merchants |  |
