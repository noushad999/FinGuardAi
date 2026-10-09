#!/usr/bin/env bash
# Regenerate data, features and all models locally (no Docker required).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY -m simulator
$PY -m features build
$PY -m ml train-all
