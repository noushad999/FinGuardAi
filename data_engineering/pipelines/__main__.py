"""Batch pipeline CLI: ``python -m data_engineering.pipelines <init-db|ingest-lake|load-raw|load-ml|all>``."""

from __future__ import annotations

import argparse
import json

from common.db import get_engine, init_schema
from data_engineering.pipelines.batch import ingest_lake, load_ml, load_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="FinGuard batch pipelines")
    parser.add_argument("command", choices=["init-db", "ingest-lake", "load-raw", "load-ml", "all"])
    args = parser.parse_args()
    out: dict = {}
    if args.command in ("init-db",):
        out["schema"] = init_schema(get_engine())
    if args.command in ("ingest-lake", "all"):
        out["lake"] = ingest_lake()
    if args.command in ("load-raw", "all"):
        out["raw"] = load_raw()
    if args.command in ("load-ml", "all"):
        out["ml"] = load_ml()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
