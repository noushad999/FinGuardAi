#!/usr/bin/env bash
# Quick end-to-end check of a running stack: API, scoring, customer profile, alerts, Prometheus, warehouse.
set -euo pipefail
API=${API_URL:-http://localhost:8000}
pass() { echo "  [OK] $*"; }
fail() { echo "  [FAIL] $*"; exit 1; }

echo "FinGuard smoke test against $API"
curl -fs "$API/health" | grep -q '"status":"ok"' && pass "API healthy" || fail "API health"
RESP=$(curl -fs -X POST "$API/risk/score" -H 'content-type: application/json' -d '{
  "event_time": "2026-06-29T21:40:00Z", "customer_id": "CUS_0000042", "sender_id": "CUS_0000042",
  "receiver_id": "CUS_0001999", "amount": 18500, "transaction_type": "P2P", "channel": "APP",
  "device_id": "DEV_SMOKE_TEST", "district": "Sylhet", "latitude": 24.89, "longitude": 91.87,
  "balance_before": 21000, "metadata": {"pin_attempts": 3}}')
echo "$RESP" | python3 -c 'import json,sys; r=json.load(sys.stdin); print("     ->", r["decision"], r["risk_score"], r["risk_reasons"][:2])'
pass "risk scoring"
curl -fs "$API/customer/CUS_0000042" >/dev/null && pass "customer profile"
curl -fs "$API/fraud/alerts?limit=5" >/dev/null && pass "alerts"
curl -fs "$API/model/info" >/dev/null && pass "model info"
curl -fs "$API/metrics" | grep -q finguard_decisions_total && pass "API metrics"
curl -fs "http://localhost:9090/-/ready" >/dev/null && pass "Prometheus"
curl -fs "http://localhost:5000/health" >/dev/null && pass "MLflow"
curl -fs "http://localhost:8501/_stcore/health" >/dev/null && pass "dashboard"
docker compose exec -T postgres psql -U finguard -d finguard -Atc \
  "select 'fact_transactions=' || count(*) from analytics.fact_transactions" && pass "warehouse marts"
echo "Smoke test passed."
