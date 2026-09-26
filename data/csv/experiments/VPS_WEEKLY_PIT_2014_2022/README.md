# Weekly PIT mlf1, 2014–2022 — VPS job

Produces the honest point-in-time **weekly** production-equivalent forecast for
2014-01 → 2022-12. Joined to the existing 2023+ weekly file
(`PIT_BASE_WEEKLY_90d_from2023_LSEG`, identical config) it gives a weekly `mlf1`
covering 2014 → 2026, so it can be backtested head-to-head with the max-rank file
over its full span (2015 → 2026).

## Config (do not change)

`forecast_returns_ml_weekly_pit.py`, production-equivalent:
90-day target, 1-day forecast, 12-month lookback, weekly Tuesday walk,
`--fundamental-only` (269 features = production's set), `--ffill-target`
(per-period as-of ffill, no leak), `--no-lag --num-leaves 127 --n-estimators 1000`.
Input: production parquet 20260922 cut at 2022-12-31 (history from 2003 kept for
features/training), prior-run `predicted_return`/`forward_return` dropped.

## Why yearly chunks

The script writes output only at the very end, so one ~20-40 h run loses everything
on any interruption. Each weekly model uses only data up to its own date, so a chunk
for year Y (input cut at Y-12-31, predictions from Y-01-01) is **identical** to what a
single full run would produce for Y. `run_all.sh` does 2014..2022 in sequence and
**skips any year already in `out/`** — after a crash/reboot just rerun it.

## Steps

```bash
tar -xzf VPS_WEEKLY_PIT_2014_2022.tar.gz && cd VPS_WEEKLY_PIT_2014_2022

./setup_env.sh        # 1. pinned env: py3.8.18 pandas2.0.3 numpy1.24.4 sklearn1.3.2 pyarrow17 (~5 min)
./smoke_test.sh       # 2. ~20-40 min: Nov-Dec 2022 vs the Mac's reference. MUST print PASS.
nohup ./run_all.sh > run_all.log 2>&1 &   # 3. the full job, detached
tail -f run_all.log   #    progress; each year prints DONE with its weekly-model count
```

When it finishes it runs `collect.py` automatically →
**`WEEKLY_PIT_2014_2022_forecast_only.csv`** — the only file to bring back.

If you'd rather run it manually: `./_py.sh collect.py` after all 9 years are in `out/`.

## Sizing

- Machine: the more cores the better (HistGradientBoosting uses all of them).
  Reference: 16-core Mac ≈ 160 s per weekly model ≈ 2.3 h per year ≈ **~21 h** total.
  An 8-core VPS: expect roughly 35–45 h. RAM: ≥ 16 GB (32 GB comfortable).
- Disk: ~5 GB free (input 0.76 GB + per-year outputs ~0.4 GB each + env).

## If the smoke test FAILS

Don't run the full job — results would not be comparable to the Mac's. Send back
`smoke/smoke_vps.log` and the printed diff. Usual cause: a different
numpy/sklearn build; `setup_env.sh` pins versions to avoid exactly this.
