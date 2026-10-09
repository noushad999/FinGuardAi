# ML methodology

Principles: temporal validation everywhere, point-in-time features, labels only as known at the
time, metrics suited to imbalance, calibrated probabilities, business policy outside the model,
and every number computed by the pipeline (reports in `reports/`).

## 1. Feature engineering (shared online/offline)

`features/engine.py` maintains small state documents per customer (24 h event buffer, Welford
amount statistics, known devices/recipients/districts, hour histogram, last location, inflows),
per device (accounts seen), and per merchant (customers seen). For each transaction it computes
61 features **before** folding the event into state:

| Group | Examples |
|---|---|
| Transaction | amount, log_amount, type/channel one-hots, merchant risk level & age, near-threshold flag, amount/balance, PIN attempts |
| Behavioural | counts in 1 min / 5 min / 1 h / 24 h, amount sums, historical mean/std, z-score, amount vs. average, unique recipients/devices 24 h, failure ratio, wrong-PIN count, velocity vs. own daily rate |
| Temporal | local hour, weekday, weekend (Fri/Sat), night, log seconds since previous, share of own history at this hour |
| Device | new device, device transaction count, accounts per device, devices per customer |
| Geographic | new district, district frequency, distance and implied speed from previous transaction |
| Network | sender out-degree, new recipient, receiver in-degree and 24 h inflows, receiver account age, minutes since last inflow, outflow/inflow ratio |

*Training*: `python -m features build` replays the whole log through the engine (≈5 s for 100K
events, 85 s for 1M). *Serving*: the API and risk engine call the same code with Redis-backed
state. Tests assert (a) offline replay == online event-by-event features, (b) Redis and in-memory
stores are identical, (c) changing future events never changes past features.

## 2. Splits and labels

Event-time split: train 60% (to 2026-04-18), validation 20% (to 2026-05-24), test 20%.
The model is "built" at the end of validation and sees only labels reported by then; validation
rows from the last 21 days (label-maturity window) are excluded from early stopping,
calibration and threshold choice. On the default data 591 of 625 training frauds are known at
the cutoff. Test metrics use full ground truth.

## 3. Fraud models
* **A – XGBoost**: `binary:logistic`, hist trees, depth 5, lr 0.05, early stopping on validation
  PR-AUC. Class imbalance: `scale_pos_weight` ∈ {1, √r, r} chosen on validation (1 won:
  0.847 vs 0.776 vs 0.694). Platt calibration on validation margins.
* **B – Isolation Forest**: 100 trees, 4,096-sample subsampling, trained on training-period
  features without labels; raw scores mapped to percentiles of the training distribution.
* **C – Behavioural baseline**: nine customer-relative deviation components (amount above own
  range, rare hour, velocity, new device/location/recipient, balance drain, auth failures,
  pass-through). Each gets an upper-tail probability under the training population; the score is
  Fisher's Σ −log p. Customers with < 3 prior transactions only get history-free components.

Evaluation: PR-AUC, ROC-AUC, precision/recall/F1 and confusion matrices at policy thresholds,
precision@k/recall@k, per-scenario recall, calibration (Brier, ECE, reliability table) and an
equal-alert-budget comparison with alert-set overlap. Results: [fraud_detection.md](fraud_detection.md).

## 4. Account takeover
Same training recipe on label `is_ato` with 27 change-of-controller features
(`features.definitions.ATO_FEATURES`). Test PR-AUC 0.939.

## 5. Graph analytics
NetworkX over all history: P2P money-flow digraph (degree, weighted degree, PageRank, sampled
betweenness), customer–device bipartite graph (device-sharing components), Louvain communities
(resolution 3) on the P2P + shared-device graph. An unsupervised mule score averages percentile
ranks of six explainable signals. Clusters are ranked by mean mule score, flagged-transaction
density and device sharing. Evaluated against hidden mule labels (PR-AUC 0.698; flagged
clusters: 100% of mules at 9.3× lift).

**Why no GNN?** 30 labelled mules in the default data is far too few to train one, and the
hand-crafted graph features are cheap, explainable and already expose the rings. A GraphSAGE
prototype becomes worthwhile with many labelled rings and relational patterns that summary
statistics miss.

## 6. Customer segmentation
Behaviour over the last 90 days (frequency, active-day ratio, average debit, monthly spend,
merchant diversity, mix of P2P / cash-out / cash-in / bill / recharge / merchant / bank, share of
P2P received) for customers with ≥3 transactions, standardised. K-Means with k chosen by
silhouette inside the business range 5–8 → **k = 7**, silhouette 0.184. HDBSCAN finds only 2
dense clusters with 29% noise (ARI 0.055) — useful to show the space is a continuum rather than
well-separated blobs, less useful as an operational segmentation. Clusters are named from
centroid profiles:

| Segment | Customers | Txns / 90 d | Monthly spend (BDT) | Signature |
|---|---|---|---|---|
| High-Value Trader | 452 | 56.1 | 7,867 | high volume, cash-in/out, P2P |
| Frequent Digital User | 470 | 26.7 | 3,878 | 34% merchant payments |
| Bill-Paying Salaried | 225 | 15.2 | 3,598 | 22% bill payments |
| Bank-Linked Saver | 104 | 17.7 | 2,575 | 14% bank transfers |
| Mobile-First Low-Value | 236 | 9.5 | 402 | 43% mobile recharge |
| Occasional User | 214 | 10.8 | 567 | 46% P2P, low frequency |
| Cash-Out Remittance Receiver | 232 | 5.7 | 260 | 64% cash-out |

Against the hidden personas (never used in fitting) the adjusted Rand index is 0.344; e.g. 91% of
"Cash-Out Remittance Receiver" customers really are remittance receivers.

## 7. Churn
*Definition*: eligible if ≥3 successful transactions in the 60 days before snapshot T; churned if
no successful transaction in [T, T+30 d). Snapshots every 15 days; train snapshots
2026-03-01/16/31, validation 2026-04-30, test 2026-05-30 — no label window overlaps a later
split. 21 features (recency, activity trend last 15 d vs. prior 45 d, failures incl. network
timeouts, product mix, counterparties, tenure, devices).

Test (1,752 eligible customers, 9.4% churn): **ROC-AUC 0.846**, PR-AUC 0.316, Brier 0.072,
ECE 0.018; recency-only baseline ROC-AUC 0.752 / PR-AUC 0.264. At the max-F1 validation
threshold: precision 0.31, recall 0.60 — useful for ranking retention campaigns, not for
deterministic decisions.

## 8. Customer risk profile
Per customer: max/mean fraud probability and max ATO probability over the last 30 days, flagged
transactions, churn probability, segment, mule-score percentile, suspicious-cluster membership,
behaviour (Stable / Variable / Erratic / Declining / Dormant from weekly activity CV and churn)
and recent activity (Normal / Elevated / Low / Inactive vs. 8-week baseline). `risk_level` uses
the same policy thresholds as real-time decisions. Served by `GET /customer/{id}` together with
live Redis state.

## 9. MLOps
* MLflow run per model: parameters, metrics, artifacts (bundle, metadata), dataset version
  (content hash of the transaction extract), code-independent model version string.
* Registry aliases: `development` (every training run) → `validation` (automated gates: metric
  present, above floor, not worse than production by > 0.02, feature schema compatible) →
  `production` (promotion keeps `previous_production` for one-command rollback).
* Serving loads bundles from `models/` or pulls `models:/<name>@production`.
* Monitoring: feature PSI vs. training reference, prediction PSI, alert rate, weekly realised
  precision/recall on matured labels (`reports/monitoring_report.json`). On the streamed test
  period: prediction PSI 0.0001 (stable); 2 non-cumulative features with significant PSI
  (`amount_std_hist`, `district_frequency` — both stabilise as customer history grows), and the
  expected shift of tenure counters, reported separately.
