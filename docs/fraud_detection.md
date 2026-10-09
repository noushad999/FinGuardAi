# Fraud detection

## 1. Synthetic fraud scenarios

The fraud engine (`simulator/fraud.py`) injects *cases*: a coherent story (victim, perpetrator,
device, place, sequence) whose every transaction carries `is_fraud`, `is_ato`, `fraud_scenario`
and `fraud_case_id`. ATO = someone other than the owner controls the account.

| # | Scenario | Mechanism | ATO | Legitimate behaviour it overlaps with |
|---|---|---|---|---|
| 1 | velocity_attack | 4–11 rapid P2P/cash-outs from an attacker handset, each draining 25–60% of the balance | ✔ | follow-up bursts (pay then recharge), small businesses |
| 2 | sudden_high_value | One P2P 4–18× the victim's median to a mule, from the victim's own phone (coerced) | | legitimate large purchases / family support (0.4% of events) |
| 3 | new_device_high_value | Fresh handset, 1–2 transactions draining 60–95% | ✔ | 12% of customers change phones during the period |
| 4 | impossible_travel | 8–50 min after a legit transaction, activity from a district >200 km away | ✔ | legitimate travel (trips last days, realistic speed) |
| 5 | device_sharing_ring | One device-farm handset operates 4–10 mule accounts that consolidate funds | | 3% of customers share a phone with family |
| 6 | suspicious_merchant | Short-lived scam shops collect payments from 3–9 victims within weeks of opening | | 10% of legitimate merchants are also new |
| 7 | rapid_fund_movement | Victim → mule transfer, forwarded (90–99%) or cashed out within 2–25 min | | people spending money they just received |
| 8 | structuring | 3–5 cash-ins and 2–3 transfers at 88–99.5% of the BDT 25,000 threshold | | small businesses moving near-threshold amounts |
| 9 | mule_collection | 5–14 victims each send modest amounts to one mule, who cashes out days later | | popular recipients (family members) |
| 10 | failed_auth_then_success | 2–5 WRONG_PIN failures from an attacker device, then a successful drain | ✔ | normal users mistype PINs (0.6%) |
| 11 | unusual_recipient_scam | Victim sends 0.8–4× their usual amount to a never-seen recipient, own phone and place | | 4% of legit P2P goes to a new recipient |
| 12 | abnormal_nighttime | Stolen-phone drains between 00:30 and 04:30 for customers with no night history | ✔ | night-shift workers, young night owls |

Mules are regular customer accounts (recently registered, low activity, some camouflage
transactions) organised in rings of 3–8 that share 1–2 device-farm handsets. Drains are expressed
as a fraction of the balance *at that moment* (resolved by the ledger), so attacks on empty wallets
fail naturally (`INSUFFICIENT_BALANCE`).

**Labels arrive late**: `fraud_reported_at` = end of case + lognormal delay (median 6 days); 5% of
cases are never reported.

Default dataset (seed 42, SMALL): 101,159 transactions, 2,500 customers, **1,132 fraud
transactions (1.12%) in 272 cases**, 407 ATO transactions, 30 mule accounts
(`data/synthetic/generation_summary.json`).

## 2. Three detection approaches (test period: last 20% of time)

`reports/fraud_model_comparison.md` (185 fraud among 21,152 test transactions; equal alert
budget of the top 1% = 212 alerts per model):

| Model | PR-AUC | ROC-AUC | Precision@212 | Recall@212 |
|---|---|---|---|---|
| A. XGBoost (supervised) | **0.804** | 0.996 | 0.665 | 0.762 |
| B. Isolation Forest (no labels) | 0.370 | 0.911 | 0.349 | 0.400 |
| C. Behavioural baseline (no labels) | 0.300 | 0.915 | 0.311 | 0.357 |

Reading the comparison:
* The supervised model dominates overall, as expected when labelled history exists.
* The unsupervised models are not useless: at the same budget Isolation Forest catches more
  `device_sharing_ring` (84% vs 78%) and ties on `failed_auth_then_success`; together the three
  alert sets reach **87.6% recall** vs 76.2% for XGBoost alone. Isolation Forest caught 11 frauds
  and the behavioural baseline 17 that XGBoost missed at that budget.
* Hardest for everyone: `mule_collection` victims and `unusual_recipient_scam` — genuine customers
  on their own device paying moderate amounts. This is where graph signals (receiver in-degree,
  mule score) and the network page of the dashboard matter.

Deployed usage: XGBoost (and ATO) drive the risk score; the Isolation Forest percentile is a
policy **escalation rule** (top 0.1% of training traffic → REVIEW) so unseen patterns reach
analysts without the unsupervised model ever blocking anyone by itself. The behavioural
baseline provides customer-relative explanations.

## 3. Supervised model details

* 61 point-in-time features; `scale_pos_weight` chosen on validation PR-AUC:
  none → 0.847, √ratio (10) → 0.776, full ratio (99.5) → 0.694 — **no re-weighting won**.
* Platt calibration on validation: test Brier 0.0036, ECE 0.0012.
* Operating points on the test period (policy review = 0.30, block = 0.90):

| Operating point | precision | recall | F1 | alerts | FP | TP |
|---|---|---|---|---|---|---|
| REVIEW (≥0.30) | 0.734 | 0.670 | 0.701 | 0.80% | 45 | 124 |
| BLOCK (≥0.90) | 0.983 | 0.314 | 0.475 | 0.28% | 1 | 58 |

* Precision@50 = 1.00, precision@100 = 0.88, recall@500 = 0.97.
* Recall per scenario at the review threshold: impossible travel 1.00, suspicious merchant 1.00,
  velocity 0.88, abnormal nighttime 0.73, structuring 0.71, mule collection 0.44, unusual
  recipient 0.50.
* Top global SHAP features (`reports/fraud_xgb_shap_importance.json`): merchant age, receiver
  account age, amount-to-balance ratio, device transaction count, new recipient, receiver
  in-degree.

## 4. Account takeover model

Separate label (`is_ato`) and a restricted feature set (device/location/time novelty, PIN
failures, velocity, amount vs. own baseline). Test: **PR-AUC 0.939**, ROC-AUC 0.999; at the review
threshold precision 0.851 / recall 0.826; per-scenario recall: impossible travel 1.00, velocity
0.88, failed-auth 0.82, night 0.80, new-device 0.62 (`reports/ato_xgb_metrics.json`).

Output: `account_takeover_probability`, `account_takeover_risk_score` (0–100) and reasons, e.g.

```json
{"account_takeover_risk_score": 62,
 "account_takeover_reasons": ["New device for this customer", "High transaction amount (BDT 18,500)",
                              "Transaction uses 88% of the wallet balance"]}
```

## 5. Decision thresholds

`reports/threshold_analysis.md` tabulates, for each candidate `risk_score` threshold, validation
(mature labels) and test precision/recall/alert volume. Policy targets: REVIEW ≥80% validation
precision, BLOCK ≥92.5% → review = **30**, block = **90**. The scores are combined as
`100·max(P(fraud), P(ATO))`.

## 6. Real-time results (streaming replay of the test period)

Replaying the 21,152 test-period transactions through Kafka (with 1% duplicates, 0.5% invalid
records, 0.5% late events) into the live risk engine:

| Decision | Transactions | of which fraud |
|---|---|---|
| APPROVE | 20,961 | 49 |
| REVIEW | 120 | 66 |
| BLOCK | 71 | 70 |

* Exactly one decision per unique transaction (21,152); 212 duplicates skipped; 107 invalid → DLQ.
* 136 of 185 frauds (73.5%) received REVIEW or BLOCK; BLOCK precision 98.6%; REVIEW includes
  anomaly-rule escalations, hence lower precision (55%) than the model-only operating point.
* Scoring latency (feature engine + 4 models + policy): p50 10.4 ms, p95 13.4 ms per transaction.

## 7. Graph signals

`reports/graph_analysis.json` (2,500 customers, 30 hidden mules):
* Unsupervised mule score: PR-AUC 0.698, ROC-AUC 0.991, recall@60 = 0.83. **Honest note:** at the
  top-30 cut the combined score (precision 0.60) is *worse* than ranking by in-degree alone (0.73);
  its value is in recall deeper in the list and in explainable components (pass-through, shared
  devices, account age). Weighting the components with labelled mules would be the next step.
* Louvain communities on the P2P + shared-device graph: 10 flagged clusters (269 customers)
  contain **all 30 mules**; mule share 11.2% vs 1.2% base rate (9.3× lift).

## 8. Explainability

Every decision includes `risk_reasons` generated from exact TreeSHAP contributions of the model
that drives the score, mapped to templates ("Transaction amount 7.2x the customer's typical
amount", "New device for this customer", "Recipient has received money from 12 different
senders"). The response carries an explicit note that reasons describe *the model's* reasoning and
contributed to the prediction — not proof or cause of fraud.
