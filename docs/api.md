# FinGuard Risk API

FastAPI service (`api/main.py`). Interactive OpenAPI docs: <http://localhost:8000/docs>.
All request/response bodies are Pydantic models (`api/schemas.py`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + dependency status (models, Redis state store, PostgreSQL) |
| POST | `/risk/score` | Score a transaction in real time (ML probabilities + policy decision + reasons) |
| GET | `/customer/{customer_id}` | Unified customer intelligence profile + live online state |
| GET | `/transaction/{transaction_id}` | Transaction details + every prediction made for it |
| GET | `/fraud/alerts` | Analyst alert queue (`limit`, `decision=REVIEW|BLOCK`, `customer_id`) |
| GET | `/model/info` | Deployed model versions, test metrics, data lineage, thresholds, active policy |
| GET | `/metrics` | Prometheus metrics |

## POST /risk/score

```bash
curl -s -X POST localhost:8000/risk/score -H 'content-type: application/json' -d '{
  "transaction_id": "TXN_DEMO_0001",
  "event_time": "2026-06-29T21:40:00Z",
  "customer_id": "CUS_0000100", "sender_id": "CUS_0000100", "receiver_id": "CUS_0001999",
  "amount": 18500, "transaction_type": "P2P", "channel": "APP",
  "device_id": "DEV_NEVER_SEEN", "district": "Sylhet", "latitude": 24.89, "longitude": 91.87,
  "balance_before": 21000, "metadata": {"pin_attempts": 3}
}'
```

Response captured from the local stack (`reason_details` truncated to two entries):

```json
{
  "transaction_id": "TXN_DEMO_0001",
  "customer_id": "CUS_0000100",
  "fraud_probability": 0.008905,
  "account_takeover_probability": 0.21449,
  "account_takeover_risk_score": 21,
  "account_takeover_reasons": ["New device for this customer", "High transaction amount (BDT 18,500)",
                               "Device has little or no transaction history", "New recipient never paid before"],
  "anomaly_percentile": 97.649,
  "behavioral_anomaly_percentile": 75.857,
  "risk_score": 21,
  "risk_level": "LOW",
  "decision": "APPROVE",
  "risk_reasons": ["New device for this customer", "High transaction amount (BDT 18,500)",
                   "Device has little or no transaction history", "New recipient never paid before"],
  "reason_details": [
    {"feature": "is_new_device", "value": 1.0, "contribution": 0.9237, "reason": "New device for this customer"},
    {"feature": "log_amount", "value": 9.8256, "contribution": 0.6965, "reason": "High transaction amount (BDT 18,500)"}
  ],
  "score_driver": "account_takeover",
  "explanation_note": "Reasons list the factors that contributed most strongly to the model's prediction (SHAP attributions). They describe the model's reasoning, not proof or cause of fraud.",
  "model_version": "fraud-xgb-v1",
  "model_versions": {"fraud": "fraud-xgb-v1", "account_takeover": "ato-xgb-v1",
                     "isolation_forest": "iforest-v1", "behavioral_anomaly": "behavioral-v1"},
  "policy_version": "policy-v1",
  "scored_at": "2026-09-27T09:41:20.640525Z",
  "latency_ms": 28.411,
  "state_committed": false
}
```
The new device, 3 PIN attempts and a large amount raise the ATO probability to 0.21, below the
review threshold (30) for this customer; the same pattern on an account with a stable device
history, at night, or with a burst of activity crosses it. (`latency_ms` here includes the cold
first request; steady-state scoring is ~10 ms.)

* `risk_score = round(100 · max(P(fraud), P(ATO)))`; `decision` comes from `configs/policy.yaml`.
* `?commit=true` folds the transaction into the online state (use when the API, not Kafka, is the
  system of record). Default is read-only scoring (`api.commit_state_on_score`).
* `transaction_id` is optional; an `API_…` id is generated when missing.
* Every API decision is persisted to `ops.risk_decisions` (source=`api`) and, when not approved,
  to `ops.fraud_alerts`.

### Validation & errors
Invalid input returns **422** with a structured body:
```json
{"error": "validation_error", "detail": [{"loc": ["body", "amount"], "msg": "Input should be greater than 0", "type": "greater_than"}]}
```
Rules include: `amount > 0`, `customer_id` = `CUS_<digits>`, known `transaction_type`/`channel`,
coordinates inside Bangladesh, `merchant_id` required for merchant payments, `receiver_id`
required for P2P, supported `schema_version`. Unknown customers/transactions return **404**;
unexpected failures return **500** `{"error": "internal_error"}` (details only in logs).

## GET /customer/{id}
```json
{
  "customer_id": "CUS_0000100",
  "risk_level": "Low",
  "risk_score": 4,
  "fraud_probability": 0.0353,
  "account_takeover_probability": 0.0063,
  "churn_probability": 0.273,
  "customer_segment": "Cash-Out Remittance Receiver",
  "behavior": "Erratic",
  "recent_activity": "Elevated",
  "mule_score_percentile": 95.72,
  "in_suspicious_cluster": false,
  "flagged_transactions_30d": 0,
  "risk_reasons": ["Activity this week well above the customer's baseline"],
  "district": "Mymensingh",
  "product_tier": "mass_affluent",
  "profile_as_of": "2026-06-30T00:00:00Z",
  "live_state": {"prior_transactions": 3, "last_transaction_at": "2026-06-28T10:47:38Z", "known_devices": 1,
                 "known_counterparties": 2, "known_districts": 1, "transactions_last_24h": 2},
  "recent_decisions": ["..."]
}
```
The batch part (profile) is refreshed by the ML pipeline; `live_state` comes from Redis.

## Versioning & observability
* Model versions are MLflow registry versions (`fraud-xgb-v<N>`); the policy has its own version.
* The service loads bundles from `models/` (`MODEL_SOURCE=local`) or pulls
  `models:/<name>@production` from MLflow at start-up (`MODEL_SOURCE=mlflow`). A feature-schema
  mismatch between a bundle and the running code fails start-up.
* Structured JSON logs (`transaction_scored` with decision, score, latency).
* Prometheus: `finguard_api_requests_total`, `finguard_api_request_seconds`,
  `finguard_scoring_seconds`, `finguard_decisions_total`, `finguard_risk_score`,
  `finguard_fraud_probability`.
