# Handoff

## State (as of 2026-08-19)
Branch `claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA` — **clean working tree, fully pushed to origin.** Production script is at **v3.3.27**.
Last 3 commits: `dd457051` (chore: LS strategy scripts + .remember + notebooks), `4412a295` (feat: v3.3.27 last-12-months reporting), `23398a1e` (fix: v3.3.26 current-month detection).

## ⚠️ PIT script is GONE
`data/csv/forecast_returns_ml_walk_forward_pit.py` was **deleted and was never committed** (untracked) → not recoverable from git. Lost with it: `--point-in-time`, `--ffill-target` (per-month mask-and-ffill), `--technicals-only`, `--fundamental-only`, `--feature-whitelist`, `momentum_30/60/90d`, `momentum_90d_zscore`, ~73 `t_` technical indicators, and the target-ffill fix. User said "that's fine" — rebuild only if asked. Spec for rebuilding is in the Findings/Flags sections below.

## v3.3.27 (this session) — production change
Added a **LAST 12 MONTHS** block to the run summary, printed right after `🎯 MODEL PERFORMANCE` (forecast_returns_ml_walk_forward.py ~line 2736). Reports correlation/MAE/RMSE/direction accuracy over trailing 12 months + row count + date range + **cross-sectional rank IC** (mean per-date Spearman, % days positive).
**Diff was 39 insertions, 0 deletions** — purely additive reporting; reads only the finished `df_predictions`. Forecasts/output files bit-identical to before. Docs all bumped to v3.3.27 (CHANGELOG/README/USAGE/weekly_command_guide/Docs INDEX+FORECAST_STABILITY/FILES.md).
Observed on real data: all-history corr **0.6955** / dir 73.6% vs **last-12-mo corr 0.3425** / dir 64.0% / rank IC **+0.3208**.

## Key findings (carry forward — hard-won)
- **Leaky vs honest:** production backtest IC ~0.5 is LEAKAGE (the ffill-of-`forward_return` bug defeats `valid_idx`, so rebuilds train on future-completing targets). Honest IC ~0.05 full-universe, ~0.01–0.05 large-cap. Top-45 book honest alpha ~+3%/yr long-run, ~+15% recent; within-45 IC ≈ 0 (range restriction) → **equal-weight the 45**.
- **Live runs are NON-leaky** (at the frontier no future exists to leak). Only *backtest rebuilds* leak. Never judge performance from a production backtest.
- **Shifting the forecast 90d does NOT de-leak** (residual ~1.5mo peek; can't un-train the model).
- **Momentum:** leaky signal ≈ 90d momentum (corr 0.81 in 2026), but momentum's own IC in 2026 ≈ 0 (choppy year). Honest model is short-term reversal + long-term trend, NOT a momentum proxy.
- **cash_return = FOCFExDividends_Discrete / EnterpriseValue_DailyTimeSeries_** is a genuine large-cap factor (IC ~0.036, t=5 — works where the ML is ~0). User's cash-return weighting adds ~+1.5%/yr. Keep it.
- **Feature count matters:** top-20 features gave only ~55% of the full ~265-feature IC → the long tail collectively earns its keep. Don't simplify. But ~50 trees ≈ 1000 trees → don't add capacity either. Edge is data-limited, not model-limited.
- **Flags:** recommend `--forecast-days 1` (not 10 — matches next-day entry, keeps fast features fresh) and `--target-return-days 42` (~2mo; 90 overshoots the ~63td earnings cycle and can't be validated live for a quarter). `forecast_days` = entry lag; `target_return_days` = holding length; window = `[T+fd, T+fd+trd]`, all **trading** days.

## Data files available (data/csv/)
`20220101_20260728_top{200,300,500}mcap_with_metadata_with_fmpdata.csv` — 2022-2026, exactly N stocks/month.
Source CSVs `20091231_2026MMDD_with_metadata_with_fmpdata.csv` (mcap col = lowercase `companymarketcap`, date col = `date`). Weekly prediction parquets in `MLData/`.

## Next / open
- Nothing pending. Possible: rebuild the PIT script; run top-200/300/500 experiments; monitor the new last-12-months rank IC weekly as a live health check.

## Preferences
- **Never modify production `forecast_returns_ml_walk_forward.py` without asking** — experiments belong in a separate copy. Always verify + show the diff afterward.
- User is a sharp quant: be honest, show the math, run the test rather than assert. Don't fit narratives (they correctly caught an over-claim that returns were "all leverage/beta" — the arithmetic didn't support it; concentrated tail selection was the missing term).
- Don't commit/push without an explicit ask.
