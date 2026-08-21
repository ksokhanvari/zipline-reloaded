# Handoff

## State (as of 2026-08-20)
Branch `claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA` — **clean working tree, fully pushed to origin.** Production script is at **v3.3.27**.
Last commits: `b8e056b3` (docs: handoff notes), `dd457051` (chore: LS strategy scripts + .remember + notebooks), `4412a295` (feat: v3.3.27 last-12-months reporting), `23398a1e` (fix: v3.3.26 current-month detection).
The riskbrakes strategy file is **already committed** inside `dd457051` — nothing about it is uncommitted.

## ✅ PIT script RECOVERED (2026-08-20) — supersedes the earlier "it's gone" note
`data/csv/forecast_returns_ml_walk_forward_pit.py` is **back**, restored from a stale Google-Drive upload-staging copy (`~/Library/CloudStorage/GoogleDrive-.../.tmp/443909/`, dated Jul 30, sha256 `67c4f9eb…`). Verified: parses, `--help` runs, byte-identical to the recovered original. It was never in git and was never gitignored either — it was simply never `git add`ed, which is exactly how it got lost. **Commit it.**

**All the expensive correctness fixes are present:**
- `exclude_from_ffill = ['Date','Symbol','forward_return']` (line ~740) — *the* leakage fix.
- PIT guard applied at `train_positions_fit` (lines ~1891, 1912–1915), **after** the ranking block — this is the fix for the bug where guarding `train_positions` too early shrank the cross-sectional ranking universe and decorrelated every month.
- `--ffill-target` as per-month mask-then-ffill (lines ~1893–1904) — the non-leaky design.
- `target_date` computed (line ~389) and excluded from features (line ~775).
- Flags `--point-in-time`, `--ffill-target`, `--technicals-only`, `--fundamental-only`, `--feature-whitelist` all present.

**It is an INTERMEDIATE build, not the final one.** Missing vs. the last known state: only **20** `t_` technical indicators (not ~73 — this is the *earlier* batch that `--fundamental-only` was defined to keep), and only `momentum_90d` + `momentum_90d_zscore` (no `momentum_30d`/`momentum_60d`). Re-adding those is the remaining gap; the hard part (leak-free plumbing) is done.

## v3.3.27 (this session) — production change
Added a **LAST 12 MONTHS** block to the run summary, printed right after `🎯 MODEL PERFORMANCE` (forecast_returns_ml_walk_forward.py ~line 2736). Reports correlation/MAE/RMSE/direction accuracy over trailing 12 months + row count + date range + **cross-sectional rank IC** (mean per-date Spearman, % days positive).
**Diff was 39 insertions, 0 deletions** — purely additive reporting; reads only the finished `df_predictions`. Forecasts/output files bit-identical to before. Docs all bumped to v3.3.27 (CHANGELOG/README/USAGE/weekly_command_guide/Docs INDEX+FORECAST_STABILITY/FILES.md).
Observed on real data: all-history corr **0.6955** / dir 73.6% vs **last-12-mo corr 0.3425** / dir 64.0% / rank IC **+0.3208**.

## Risk brakes — `data/csv/LS-prod-claud-func-refactor-MLf1-lowvol-riskbrakes.py`
The risk-overlay fork of the live Zipline/QuantRocket strategy (186 KB, committed in `dd457051`). Built to defend the 2.3-beta momentum book after the July factor unwind (SPY −0.5% while the book ran ≈ −20%). Every site is greppable by `RISKBRAKE`; each brake prints when it acts and each is individually switchable, so setting all switches off reproduces the lowvol base exactly.

| # | Switch | What it does | Default |
|---|--------|--------------|---------|
| 1 | `USE_VIXDATA_REGIME` | Restores the vixdata regime lookup + fixes flip-detection (`vixflag_prev` captured before update) | `False` (pending verification) |
| 2 | `USE_GARCH_VOL_BRAKE` | GJR-GARCH conditional-vol scaler on gross exposure | `True` |
| 3 | `USE_MLF1_HEALTH_BRAKE` | Promotes the MLF1 health guard from passive check to an acting brake that scales gross | `True` |
| 4 | `USE_TWO_SIDED_DD` | Two-sided `dd_factor` — the capped dip-buy leg no longer presses into drawdowns | `True` |
| 5 | `USE_INTRADAY_HEDGE_ESCALATION` | Symmetric intraday hedge trigger (previously the hedge could only shrink) | `True` |
| 6 | `QQQ_HEDGE_FRACTION` | Splits the hedge between IWM and QQQ so it actually matches the book's factor loading | `0.30` |
| 7 | `USE_VALUE_TILT_SLIDER` | Drawdown-gated value/momentum sleeve dial (slides the MLF1 exponent, nominally 1.8); composes with 2/4 rather than duplicating them | `True` |

Original gaps this fixed in `LS-prod-claud-func-refactor-MLf1.py`: vixflag hardcoded off; `mlf1` could not reduce gross; `dd_factor` *pressed* drawdowns; intraday hedge only shrank; IWM-only hedge missed the factor; 2.3β left undefended.

## Measurement work from the same session (2026-08-20)
- **Backtest leak quantified:** ≈ **+13.9%/mo phantom** alpha pre-Mar-2026 in the rebuilt backtest vs ≈ **+1.3%/mo** attributable live. The 2022 distribution break (+203% backtest vs −20% live) and the Mar-2026 boundary are the tells.
- **Honest signal measured:** IC ≈ **0.117** on the top-200-by-momentum slice (2025); ≈ 0.066 on the 10-day full-universe cut (t = 3.25, p < 0.01). Beta-neutral the signal earns ≈ **46 bps / 10 days** vs ≈ −1 bp naked.
- **Calibrated forward expectation: +35–45% annual at ~35% vol** (β-neutral basis) — the realistic number to quote, not the backtest.
- `cash_return_zsoft` measured at **+8.4%/yr alpha (t = 2.3)** on this cut — stronger than the earlier +1.5%/yr estimate; both agree it is a genuine, keepable large-cap value signal.
- Recommendation on file: re-anchor the production **MLF1 exponent 1.8 → 1.0**.
- Watch item: a broker-reported IR of 23 was a **units bug** (missing √252) — true ≈ 1.5. Don't quote broker IR without checking annualization.

## Key findings (carry forward — hard-won)
- **Leaky vs honest:** production backtest IC ~0.5 is LEAKAGE (the ffill-of-`forward_return` bug defeats `valid_idx`, so rebuilds train on future-completing targets). Honest IC ~0.05 full-universe, ~0.01–0.05 large-cap. Top-45 book honest alpha ~+3%/yr long-run, ~+15% recent; within-45 IC ≈ 0 (range restriction) → **equal-weight the 45**.
- **Live runs are NON-leaky** (at the frontier no future exists to leak). Only *backtest rebuilds* leak. Never judge performance from a production backtest.
- **Shifting the forecast 90d does NOT de-leak** (residual ~1.5mo peek; can't un-train the model).
- **Momentum:** leaky signal ≈ 90d momentum (corr 0.81 in 2026), but momentum's own IC in 2026 ≈ 0 (choppy year). Honest model is short-term reversal + long-term trend, NOT a momentum proxy.
- **cash_return = FOCFExDividends_Discrete / EnterpriseValue_DailyTimeSeries_** is a genuine large-cap factor (IC ~0.036, t=5 — works where the ML is ~0). User's cash-return weighting adds ~+1.5%/yr. Keep it.
- **Feature count matters:** top-20 features gave only ~55% of the full ~265-feature IC → the long tail collectively earns its keep. Don't simplify. But ~50 trees ≈ 1000 trees → don't add capacity either. Edge is data-limited, not model-limited.
- **Flags:** recommend `--forecast-days 1` (not 10 — matches next-day entry, keeps fast features fresh). `forecast_days` = entry lag; `target_return_days` = holding length; window = `[T+fd, T+fd+trd]`, all **trading** days.
  - ⚠️ **`--target-return-days` is unsettled — two live recommendations exist.** `20` is the *validated* one (PIT script, IC 0.06 over 16 yr, 80% of periods positive, t = 10.3, and 20d beat 90d head-to-head). `42` was the *reasoned* compromise (~2 mo; 90 overshoots the ~63-td earnings cycle and can't be validated live for a quarter). 42 has never been measured. **Re-test before committing to either.**

## Data files available (data/csv/)
`20220101_20260728_top{200,300,500}mcap_with_metadata_with_fmpdata.csv` — 2022-2026, exactly N stocks/month.
Source CSVs `20091231_2026MMDD_with_metadata_with_fmpdata.csv` (mcap col = lowercase `companymarketcap`, date col = `date`). Weekly prediction parquets in `MLData/`.

## Next / open
Nothing is mid-flight. Candidates, roughly in priority order:
- **Decide `--target-return-days` (20 vs 42) with a measurement**, not a judgement call — see the flags caveat above.
- **Verify and enable `USE_VIXDATA_REGIME` (RISKBRAKE-1)** — it is the one brake still defaulted off pending verification.
- Act on (or reject) the **MLF1 exponent 1.8 → 1.0** production re-anchor.
- Monitor the new v3.3.27 last-12-months rank IC weekly as a live health check.
- **Re-add the missing PIT features** — the later ~53 `t_` technical indicators and `momentum_30d`/`momentum_60d`. The recovered script has everything else.
- Run the top-200/300/500 experiments on the prepared CSVs (now unblocked — the PIT script is back).
- Deliberately deferred: the production ffill-of-target bug ("let's leave the production code alone for now").

## Preferences
- **Never modify production `forecast_returns_ml_walk_forward.py` without asking** — experiments belong in a separate copy. Always verify + show the diff afterward.
- User is a sharp quant: be honest, show the math, run the test rather than assert. Don't fit narratives (they correctly caught an over-claim that returns were "all leverage/beta" — the arithmetic didn't support it; concentrated tail selection was the missing term).
- Don't commit/push without an explicit ask.
