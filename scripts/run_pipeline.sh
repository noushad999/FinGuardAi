#!/usr/bin/env bash
# End-to-end batch pipeline (same steps the Airflow DAGs run, in one go).
# Used by `make pipeline` inside Docker and by developers locally.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

step() { echo; echo "==> $*"; }

step "1/10 Generate synthetic data";            $PY -m simulator
step "2/10 Data-quality gate";                  $PY -m data_engineering.validation --fail-on-error
step "3/10 Land extracts in the data lake";     $PY -m data_engineering.pipelines ingest-lake
step "4/10 Load warehouse raw schema";          $PY -m data_engineering.pipelines load-raw
step "5/10 Point-in-time features";             $PY -m features build
step "6/10 Train & evaluate all models";        $PY -m ml train-all
step "7/10 Registry: validate -> promote";      $PY -m ml validate && $PY -m ml promote
step "8/10 Load ML outputs";                    $PY -m data_engineering.pipelines load-ml
step "9/10 dbt build (star schema + tests)"
VARS=$($PY -c "import json; from common.config import load_policy; t=load_policy()['thresholds']; print(json.dumps({'review_threshold': t['review'], 'block_threshold': t['block']}))")
(cd dbt/finguard && DBT_PROFILES_DIR=. dbt build --vars "$VARS")
step "10/10 Monitoring report";                 $PY -m monitoring
if [ -n "${REDIS_URL:-}" ]; then
  step "Bootstrap the online feature store (Redis) up to the end of the validation period (only if empty)"
  $PY -m features hydrate --if-empty
fi
echo; echo "Pipeline complete."
