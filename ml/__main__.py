"""ML pipeline CLI.

python -m ml train-all          # fraud (A/B/C) + ATO + graph + segmentation + churn + profiles
python -m ml train-fraud | train-ato | graph | segment | churn | profile
python -m ml validate  [--model NAME]   # registry gates (development -> validation)
python -m ml promote   [--model NAME]   # validation -> production
python -m ml rollback   --model NAME
python -m ml registry-status
"""

from __future__ import annotations

import argparse
import json
import time

from common.config import get_settings
from common.logging import get_logger

log = get_logger("ml")


def _train_all() -> None:
    from ml.account_takeover import train as ato
    from ml.churn_prediction import churn
    from ml.customer_profile import profile
    from ml.customer_segmentation import segment
    from ml.fraud_detection import thresholds
    from ml.fraud_detection import train as fraud
    from ml.graph_risk import graph

    steps = [
        ("fraud", fraud.run),
        ("ato", ato.run),
        ("thresholds", thresholds.analyse),
        ("graph", graph.run),
        ("segmentation", segment.run),
        ("churn", churn.run),
        ("profile", profile.run),
    ]
    for name, fn in steps:
        t0 = time.time()
        fn()
        log.info("pipeline_step_done", extra={"step": name, "seconds": round(time.time() - t0, 1)})


def main() -> None:
    parser = argparse.ArgumentParser(description="FinGuard ML pipelines")
    parser.add_argument(
        "command",
        choices=[
            "train-all",
            "train-fraud",
            "train-ato",
            "graph",
            "segment",
            "churn",
            "profile",
            "validate",
            "promote",
            "rollback",
            "deploy",
            "registry-status",
        ],
    )
    parser.add_argument("--model", help="registered model name (default: all)")
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "train-all":
        _train_all()
    elif args.command == "train-fraud":
        from ml.fraud_detection import train as fraud

        fraud.run()
    elif args.command == "train-ato":
        from ml.account_takeover import train as ato

        ato.run()
    elif args.command == "graph":
        from ml.graph_risk import graph

        graph.run()
    elif args.command == "segment":
        from ml.customer_segmentation import segment

        segment.run()
    elif args.command == "churn":
        from ml.churn_prediction import churn

        churn.run()
    elif args.command == "profile":
        from ml.customer_profile import profile

        profile.run()
    else:
        from ml.common import registry

        names = [args.model] if args.model else list(registry.MODELS)
        if args.command == "registry-status":
            print(json.dumps(registry.status(), indent=2))
            return
        results = {}
        for name in names:
            if args.command == "validate":
                r = registry.validate(name)
                results[name] = {"passed": r.passed, **r.checks}
            elif args.command == "promote":
                results[name] = registry.promote(name)
            elif args.command == "rollback":
                results[name] = registry.rollback(name)
            elif args.command == "deploy":
                results[name] = str(registry.deploy(name, settings.path("models_dir")))
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
