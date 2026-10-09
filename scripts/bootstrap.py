"""Idempotent platform bootstrap: wait for dependencies, create buckets, Kafka topics and DB schemas.

python -m scripts.bootstrap
"""

from __future__ import annotations

import time
from collections.abc import Callable

from common.config import get_settings
from common.logging import get_logger

log = get_logger("bootstrap")


def _retry(name: str, fn: Callable[[], object], attempts: int = 60, delay: float = 3.0) -> object:
    for i in range(1, attempts + 1):
        try:
            out = fn()
            log.info("ready", extra={"dependency": name})
            return out
        except Exception as exc:  # noqa: BLE001
            if i == attempts:
                raise
            log.info("waiting", extra={"dependency": name, "attempt": i, "error": str(exc)[:120]})
            time.sleep(delay)
    return None


def main() -> None:
    settings = get_settings()

    from common.db import get_engine, init_schema

    engine = get_engine()
    _retry("postgres", lambda: engine.connect().close())
    init_schema(engine)

    from ingestion.kafka_admin import ensure_topics

    _retry("kafka", lambda: ensure_topics(settings))

    if settings["storage"]["lake_uri"].startswith("s3://"):
        import pyarrow.fs as pafs

        from common.storage import Lake

        lake = Lake(settings)
        _retry("object-storage", lake.ensure_bucket)
        _retry("mlflow-bucket", lambda: lake.fs.create_dir("mlflow", recursive=False))
        assert isinstance(lake.fs, pafs.S3FileSystem)
    log.info("bootstrap_complete")


if __name__ == "__main__":
    main()
