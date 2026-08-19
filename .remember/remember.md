# Handoff

## State
Branch `claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA`. **Production `data/csv/forecast_returns_ml_walk_forward.py` = UNTOUCHED** (only committed change all session is v3.3.26 one-liner `23398a1e`, not pushed). ALL experimental work is in the copy **`data/csv/forecast_returns_ml_walk_forward_pit.py`** (untracked). Uncommitted v3.3.26 doc edits still pending (CHANGELOG/README/USAGE/weekly_command_guide/Docs INDEX+FORECAST_STABILITY).

## PIT script flags added this session (all in _pit.py only)
- `--point-in-time`: drop training rows whose 90d target realizes on/after the prediction month (honest backtest).
- `--ffill-target`: per-month MASK-and-FFILL — keep all rows, give unrealized-target rows the stock's last-known target (verified non-leaky; reproduces the live "keep all X, ffill y" process). Takes precedence over --point-in-time.
- `--feature-whitelist "f1,f2,..."`, `--technicals-only`, `--fundamental-only` (drops only the new t_ batch).
- Added features: `momentum_30/60/90d`, `momentum_90d_zscore`, ~73 `t_` technical indicators (RSI/MACD/stoch/BB/ATR/downside-vol/drawdown/OBV-flow/MFI/sector-relative). All backward-looking, look-ahead test = 0.00e+00.
- Also FIXED in _pit.py (not production): the ffill-of-target bug (target now protected from global ffill).

## Data files created (data/csv/): 2022-2026, top-N mcap per month
`20220101_20260728_top{200,300,500}mcap_with_metadata_with_fmpdata.csv` (200/300/500 stocks/month exactly). Source = `20091231_20260728_with_metadata_with_fmpdata.csv` (mcap col = lowercase `companymarketcap`, date col = `date`).

## Key findings (the crux of the whole session)
- **Leaky vs honest:** production backtest IC ~0.5 (top-400) is LEAKAGE, not skill. Honest (PIT) IC ~0.05 full-universe, ~0.01–0.05 large-cap. Top-45 book honest alpha ~+3%/yr (long-run), ~+15% recent — modest, concentrated in the TAIL (within-45 IC ≈ 0 by range restriction).
- **The ffill bug:** production forward-fills `forward_return`, defeating `valid_idx`, so backtest rebuilds train on future-completing targets = leaky. BUT **live weekly runs are NON-leaky** (frontier: no future exists to leak). User's live process is honest; only backtest *rebuilds* leak.
- **Momentum:** leaky signal ≈ 90d momentum (corr 0.81 in 2026); but momentum's own IC in 2026 ≈ 0 (choppy/non-trending year). Live +74%/7mo was leverage + concentration + real tail-selection, NOT momentum working.
- **Shifting forecast 90d does NOT de-leak** (residual ~1.5mo forward peek; can't un-train the model).
- **cash_return = FOCFExDividends_Discrete / EnterpriseValue_DailyTimeSeries_** is a real large-cap factor (IC ~0.036, t=5, WORKS where ML is ~0); user's cash-return weighting adds ~+1.5%/yr.

## Recommendations given
- Flags: `--forecast-days 1` (not 10 — keeps fast features fresh, matches next-day entry) and `--target-return-days 42` (~2mo sweet spot; 90 too long/can't validate live for a quarter).
- Judge performance on LIVE/frontier or PIT script — never the leaky production backtest.
- Don't add model complexity (267 factors, 1000 trees) — edge is data-limited, not capacity-limited; ~50 trees held IC.

## Next / open
1. User was running top-200/300/500 PIT runs + fundamental-only/technicals-only/ffill-target experiments — compute IC on outputs when they land.
2. Commit the pending v3.3.26 doc edits if user wants; push only if asked.
3. Wrote 3-bullet investor blurb (adaptive approach + honest limits) — flagged needs compliance review.

## Preferences
- NEVER modify production `forecast_returns_ml_walk_forward.py` — all experiments go in `_pit.py`. Verify "production UNMODIFIED" after edits.
- User is a sharp quant — be honest, show the math, don't fit narratives (they caught me over-claiming "all leverage/beta"). Don't push/commit without explicit ask.
