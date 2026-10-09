"""Duplicate-event protection for the stateful risk engine.

Kafka delivery is at-least-once and upstream systems occasionally resend a
transaction. Scoring is harmless to repeat, but *committing* the same event
twice would corrupt velocity/amount state. Each transaction_id is therefore
claimed once (Redis ``SET NX EX`` shared across workers, or an in-process LRU).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class InMemoryDeduplicator:
    def __init__(self, capacity: int = 500_000) -> None:
        self.capacity = capacity
        self._seen: OrderedDict[str, None] = OrderedDict()

    def claim(self, transaction_id: str) -> bool:
        """True if this id is new (and is now claimed), False if already seen."""
        if transaction_id in self._seen:
            self._seen.move_to_end(transaction_id)
            return False
        self._seen[transaction_id] = None
        if len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        return True


class RedisDeduplicator:
    def __init__(self, client: Any, ttl_seconds: int = 3 * 24 * 3600, prefix: str = "fg:seen:") -> None:
        self.client = client
        self.ttl = ttl_seconds
        self.prefix = prefix

    def claim(self, transaction_id: str) -> bool:
        return bool(self.client.set(self.prefix + transaction_id, 1, nx=True, ex=self.ttl))
