#!/usr/bin/env bash
# Full job, one chunk per calendar year 2014..2022, sequential. RESUMABLE: a year
# whose output already exists is skipped, so after any crash/reboot just rerun.
# Launch detached:   nohup ./run_all.sh > run_all.log 2>&1 &
# Progress:          tail -f run_all.log ; ls out/
set -uo pipefail
cd "$(dirname "$0")"; source common_args.sh
mkdir -p out logs
for Y in 2014 2015 2016 2017 2018 2019 2020 2021 2022; do
  if [ -f "out/weekly_pit_${Y}.parquet" ]; then echo "$(date) $Y already done -- skip"; continue; fi
  echo "$(date) $Y: slicing input to ${Y}-12-31 ..."
  ./_py.sh -c "import pandas as pd; d=pd.read_parquet('input_to_20221231.parquet'); d['Date']=pd.to_datetime(d['Date']); d[d['Date']<='${Y}-12-31'].to_parquet('logs/input_${Y}.parquet', index=False)"
  echo "$(date) $Y: training weekly models (predictions ${Y}-01-01 .. ${Y}-12-31) ..."
  PYTHONUNBUFFERED=1 ./_py.sh forecast_returns_ml_weekly_pit.py --input-file "logs/input_${Y}.parquet" \
      $ARGS --train-start "${Y}-01-01" --output "logs/weekly_pit_${Y}.partial.parquet" > "logs/run_${Y}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    mv "logs/weekly_pit_${Y}.partial.parquet" "out/weekly_pit_${Y}.parquet"   # atomic: only complete years land in out/
    rm -f "logs/input_${Y}.parquet"
    echo "$(date) $Y: DONE ($(grep -c 'Trained on' logs/run_${Y}.log) weekly models)"
  else
    echo "$(date) $Y: FAILED rc=$rc -- see logs/run_${Y}.log ; stopping"; exit $rc
  fi
done
echo "$(date) ALL YEARS DONE -> ./_py.sh collect.py"
./_py.sh collect.py
