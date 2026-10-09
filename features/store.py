"""Online state stores for the feature engine.

The feature engine keeps small per-entity state documents (customer, device,
merchant). Two interchangeable implementations:

* :class:`InMemoryStateStore` - offline training replay, tests, single-process demo.
* :class:`RedisStateStore` - shared online store for the API and the streaming
  risk engine (JSON documents under ``fg:<kind>:<id>``).

Both expose the same two batch operations so the engine code is identical
online and offline (the core defence against training/serving skew).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Protocol

State = dict[str, Any]


class StateStore(Protocol):
    def get_many(self, keys: list[str]) -> dict[str, State | None]: ...

    def put_many(self, items: dict[str, State]) -> None: ...


class InMemoryStateStore:
    """Dict-backed store. Values are kept as live Python objects (no copy)."""

    def __init__(self) -> None:
        self.data: dict[str, State] = {}

    def get_many(self, keys: list[str]) -> dict[str, State | None]:
        return {k: self.data.get(k) for k in keys}

    def put_many(self, items: dict[str, State]) -> None:
        self.data.update(items)

    def items(self) -> Iterable[tuple[str, State]]:
        return self.data.items()

    def __len__(self) -> int:
        return len(self.data)


class RedisStateStore:
    """Redis-backed store (JSON serialised)."""

    def __init__(self, url: str, prefix: str = "fg:", client: Any | None = None) -> None:
        if client is None:
            import redis  # imported lazily so offline code does not need redis

            client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)
        self.client = client
        self.prefix = prefix

    def get_many(self, keys: list[str]) -> dict[str, State | None]:
        if not keys:
            return {}
        values = self.client.mget([self.prefix + k for k in keys])
        return {k: (json.loads(v) if v else None) for k, v in zip(keys, values)}

    def put_many(self, items: dict[str, State]) -> None:
        if not items:
            return
        pipe = self.client.pipeline(transaction=False)
        for k, v in items.items():
            pipe.set(self.prefix + k, json.dumps(v, separators=(",", ":")))
        pipe.execute()

    def ping(self) -> bool:
        return bool(self.client.ping())

    def count(self) -> int:
        return sum(1 for _ in self.client.scan_iter(self.prefix + "*", count=1000))


def build_store(redis_url: str | None) -> StateStore:
    """Redis if a URL is configured, else in-process memory."""
    if redis_url:
        return RedisStateStore(redis_url)
    return InMemoryStateStore()
