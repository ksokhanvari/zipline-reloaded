# EXACT production-equivalent PIT config (same as PIT_BASE_WEEKLY_90d_from2023_LSEG,
# the 2023+ half this run completes). Do not change.
ARGS="--forecast-days 1 --target-return-days 90 --lookback-months 12 \
      --walk-frequency weekly --fundamental-only --ffill-target \
      --no-lag --num-leaves 127 --n-estimators 1000"
