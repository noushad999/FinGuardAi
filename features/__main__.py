"""Feature pipeline CLI.

``python -m features build``            -> data/processed/transaction_features.parquet
``python -m features hydrate [--until TS] [--if-empty]`` -> bootstrap the online store (Redis) from history

Hydration *replaces* entity state with a replay of history up to ``--until``.
On a store that live traffic has already advanced this would roll state back,
so pipelines call it with ``--if-empty`` (bootstrap only).
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.config import get_settings
from features.engine import EngineConfig
from features.offline import build_from_dir, hydrate_store
from features.reference import ReferenceData
from features.store import RedisStateStore


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="FinGuard feature pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="build point-in-time transaction features")
    h = sub.add_parser("hydrate", help="warm the online (Redis) state store")
    h.add_argument("--until", help="ISO timestamp; default = end of the training+validation period")
    h.add_argument("--if-empty", action="store_true", help="skip if the store already holds entity state")
    args = parser.parse_args()
    cfg = EngineConfig.from_settings(settings)
    synth = settings.path("synthetic_dir")
    if args.cmd == "build":
        build_from_dir(synth, settings.path("processed_dir") / "transaction_features.parquet", cfg)
    elif args.cmd == "hydrate":
        redis_url = settings["redis"]["url"]
        if not redis_url:
            raise SystemExit("REDIS_URL is not configured")
        store = RedisStateStore(redis_url)
        if args.if_empty:
            existing = sum(1 for _ in store.client.scan_iter(store.prefix + "c:*", count=1000))
            if existing:
                print(f"online store already holds {existing} customer states - not overwriting (live state)")
                return
        tx = pd.read_parquet(synth / "transactions.parquet")
        if args.until:
            until = (
                pd.Timestamp(args.until, tz="UTC")
                if pd.Timestamp(args.until).tzinfo is None
                else pd.Timestamp(args.until)
            )
        else:
            from ml.common.splits import temporal_boundaries

            until = temporal_boundaries(tx["event_time"], settings)[1]
        hydrate_store(tx, ReferenceData.from_parquet(synth), store, until, cfg)


if __name__ == "__main__":
    main()
