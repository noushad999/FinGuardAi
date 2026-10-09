"""Prometheus metrics shared by the API and the streaming risk engine.

System metrics: throughput, latency, error rate, Kafka consumer lag.
ML metrics: decision mix, risk-score distribution, fraud-probability distribution.
Both processes expose them on ``/metrics`` for Prometheus to scrape.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
SCORE_BUCKETS = tuple(range(0, 101, 10))
PROB_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)

# ---- API ----
API_REQUESTS = Counter("finguard_api_requests_total", "API requests", ["method", "route", "status"])
API_LATENCY = Histogram(
    "finguard_api_request_seconds", "API request latency", ["method", "route"], buckets=LATENCY_BUCKETS
)

# ---- scoring (API and stream) ----
SCORING_LATENCY = Histogram(
    "finguard_scoring_seconds", "Feature + model + policy latency per transaction", ["source"], buckets=LATENCY_BUCKETS
)
DECISIONS = Counter("finguard_decisions_total", "Risk decisions", ["source", "decision"])
RISK_SCORE = Histogram("finguard_risk_score", "Distribution of risk scores", ["source"], buckets=SCORE_BUCKETS)
FRAUD_PROBABILITY = Histogram(
    "finguard_fraud_probability", "Distribution of fraud probabilities", ["source"], buckets=PROB_BUCKETS
)

# ---- stream processing ----
EVENTS_CONSUMED = Counter("finguard_stream_events_total", "Kafka messages consumed", ["outcome"])
DLQ_MESSAGES = Counter("finguard_dlq_messages_total", "Messages sent to the dead-letter topic", ["reason"])
DUPLICATES = Counter("finguard_duplicate_events_total", "Duplicate transaction events skipped")
PROCESSING_ERRORS = Counter("finguard_processing_errors_total", "Handler failures (before retry)")
CONSUMER_LAG = Gauge("finguard_kafka_consumer_lag", "Consumer lag (messages) per partition", ["partition"])
CONSUMER_LAG_TOTAL = Gauge("finguard_kafka_consumer_lag_total", "Total consumer lag (messages)")
LAST_EVENT_TS = Gauge("finguard_last_event_timestamp_seconds", "Event time of the last processed transaction")
BATCH_SECONDS = Histogram(
    "finguard_stream_batch_seconds",
    "End-to-end processing time per consumed batch",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
