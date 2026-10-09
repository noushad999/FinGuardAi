"""Structured (JSON) logging used by every FinGuard component.

Usage::

    from common.logging import get_logger
    log = get_logger(__name__)
    log.info("model_loaded", extra={"model_version": "fraud-xgb-v3"})

Anything passed in ``extra`` is emitted as top-level JSON fields, which keeps
logs machine-queryable (Loki/ELK/CloudWatch) without a heavy dependency.
Set ``LOG_FORMAT=text`` for human-friendly local output.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        base = f"{datetime.now().strftime('%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    fmt = os.environ.get("LOG_FORMAT", "json").lower()
    handler.setFormatter(TextFormatter() if fmt == "text" else JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel((level or os.environ.get("LOG_LEVEL", "INFO")).upper())
    # Quieten chatty libraries.
    for noisy in ("urllib3", "botocore", "kafka", "py4j", "matplotlib", "alembic", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
