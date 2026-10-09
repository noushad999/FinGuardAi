"""CLI: ``python -m simulator [--size small|medium|large|<int>] [--out DIR] [--seed N]``."""

from __future__ import annotations

import argparse
import json
import os

from common.config import get_settings, load_settings
from simulator.generator import generate, write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the FinGuard synthetic MFS dataset")
    parser.add_argument("--size", help="small | medium | large | <number of transactions>")
    parser.add_argument("--days", type=int, help="simulated days")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", help="output directory (default: paths.synthetic_dir)")
    args = parser.parse_args()
    if args.size:
        os.environ["DATASET_SIZE"] = args.size
    if args.days:
        os.environ["SIM_DAYS"] = str(args.days)
    if args.seed is not None:
        os.environ["SIM_SEED"] = str(args.seed)
    settings = load_settings() if (args.size or args.days or args.seed is not None) else get_settings()
    ds = generate(settings)
    out = write_dataset(ds, args.out or settings.path("synthetic_dir"))
    print(json.dumps({"output": str(out), **ds.summary}, indent=2))


if __name__ == "__main__":
    main()
