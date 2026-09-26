#!/usr/bin/env bash
# ~20-40 min. Runs Nov-Dec 2022 (9 weekly models) and compares against the Mac's
# output. MUST PASS before launching run_all.sh -- proves the env reproduces exactly.
set -euo pipefail
cd "$(dirname "$0")"; source common_args.sh
mkdir -p smoke
PYTHONUNBUFFERED=1 ./_py.sh forecast_returns_ml_weekly_pit.py --input-file input_to_20221231.parquet \
  $ARGS --train-start 2022-11-01 --output smoke/smoke_vps.parquet > smoke/smoke_vps.log 2>&1
./_py.sh verify_smoke.py
