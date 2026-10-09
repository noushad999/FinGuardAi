"""Kafka topic management (idempotent)."""

from __future__ import annotations

from confluent_kafka.admin import AdminClient, NewTopic

from common.config import Settings, get_settings
from common.logging import get_logger

log = get_logger(__name__)

# topic key -> (partitions, config)
TOPIC_SPECS: dict[str, tuple[int, dict[str, str]]] = {
    # keyed by customer_id -> per-customer ordering; 6 partitions = up to 6 parallel risk workers
    "transactions": (6, {"retention.ms": str(7 * 24 * 3600 * 1000), "cleanup.policy": "delete"}),
    "customers": (3, {"cleanup.policy": "compact"}),
    "devices": (3, {"cleanup.policy": "compact"}),
    "alerts": (3, {"retention.ms": str(30 * 24 * 3600 * 1000)}),
    "dead_letter": (1, {"retention.ms": str(30 * 24 * 3600 * 1000)}),
}


def ensure_topics(settings: Settings | None = None, replication_factor: int = 1) -> list[str]:
    settings = settings or get_settings()
    kafka = settings["kafka"]
    admin = AdminClient({"bootstrap.servers": kafka["bootstrap_servers"]})
    existing = set(admin.list_topics(timeout=15).topics)
    wanted = []
    for key, (partitions, cfg) in TOPIC_SPECS.items():
        name = kafka["topics"][key]
        if name not in existing:
            wanted.append(NewTopic(name, num_partitions=partitions, replication_factor=replication_factor, config=cfg))
    if wanted:
        for name, fut in admin.create_topics(wanted).items():
            try:
                fut.result()
                log.info("topic_created", extra={"topic": name})
            except Exception as exc:  # noqa: BLE001 - already exists in a race
                if "TOPIC_ALREADY_EXISTS" not in str(exc):
                    raise
    return [kafka["topics"][k] for k in TOPIC_SPECS]


if __name__ == "__main__":  # pragma: no cover
    print(ensure_topics())
