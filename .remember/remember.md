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

## Objective-alignment experiments (2026-08-31) — RESULTS IN
**Motivation:** measured on real data, squared-error loss on raw % returns is spent almost entirely on stocks that are never traded. By market-cap decile: smallest decile carries **80.8%** of total squared error, bottom 20% of names carry **88%**, top 20% carry **1.3%** — a **65× loss-per-row** imbalance. The model optimizes ~99% for names outside the tradeable universe.

**New flags added to the PIT script only** (production untouched):
`--train-universe-top N`, `--target-normalize {none,zscore,rank}`, `--target-winsor PCT`.
Applied AFTER the PIT guard and AFTER the ranking block, so the ranking universe stays the full cross-section. Target normalization transforms the LABEL only (every value in date D's transform is itself a date-D label ⇒ no look-ahead). Unit-tested: per-date mean 0 / std 1, within-date ordering preserved exactly (rank corr 1.0000), imbalance cut 34×, and `none` byte-identical to the old default.

**Setup:** 2023-01-01→2026-08-25 full-universe slice (2.79M rows, all symbols), `--forecast-days 1 --target-return-days 90 --lookback-months 12 --point-in-time`. Scored 2024-01→2026-04-21 on independently recomputed outcomes. Identical row counts (1,675,078) across runs. ~52 min/run.

| run | IC full | IC top-400 | excess/90d | train rows/mo |
|---|---|---|---|---|
| A baseline | +0.0985 | +0.0566 | +1.64% | ~455,000 |
| B `--train-universe-top 400` | **+0.0458** | **+0.0546** | +1.60% | ~65,000 |
| C `--target-normalize zscore` | **+0.1020** | **+0.0619** | **+1.89%** | ~455,000 |

- **B is decisively wrong — breadth matters.** It loses even on the top-400 evaluation (0.0546 vs A 0.0566, C 0.0619): training only on large caps doesn't even help predict large caps. Beats A in **1 of 28 months**. Microcap rows carry signal that transfers upward. **Never restrict the training universe.**
- **C wins, and its edge is entirely regime-conditional:** strong months C−A = **+0.0003** (nothing); 2025-06..12 drawdown C−A = **+0.0131** (roughly doubles IC, flips two negative months positive). Beats A 5/7 in the drawdown, 11/21 otherwise. Mechanism: turbulent markets blow up cross-sectional dispersion, so MSE gets even more outlier-dominated; per-date normalization caps that. **Z-scoring the target is insurance, not alpha** — but it is free (no added training time) and never materially lost. Adopt it.
- **Answer to "IC as objective vs returns":** returns, but *normalized*. Not raw returns, not a ranking objective — LambdaRank would discard the magnitude the softplus tilt depends on, and this result says the payoff wouldn't justify the migration.
- **Biggest takeaway:** correcting a 65× loss misallocation bought 3.6% relative IC. Combined with 50 trees ≈ 1000 trees, **the edge is data-limited, not model- or objective-limited.** Spend effort on features/data, not on objectives or capacity.
- Both objectives collapse identically 2025-06..11 (A +0.0105, C +0.0236, B negative). That is a regime the signal does not work in; no objective fixes it.

**Artifacts:** `data/csv/experiments/exp{A_baseline,B_top400,C_zscore}.parquet`; slice `data/csv/20230101_20260825_fulluniv_with_metadata_with_fmpdata.csv` (1.93 GB). Both untracked — same exposure that lost the PIT script before.

## ✅ LSEG IS FULLY REPLACEABLE (2026-09-18) — verified end-to-end
Question: can the 38 LSEG columns in `20091231_YYYYMMDD.csv` be sourced from Sharadar/FMP instead?
**Answer: yes, all 38. Zero LSEG columns still required, and the signal got BETTER.**

| group | n | resolution |
|---|---|---|
| keys, price/volume, identity | 8 | unchanged / `sharadar_*` already merged |
| quarterly fundamentals | 8 | **already in the file** as `*_fmp` — no download |
| daily EV, market cap, 3 EV ratios | 5 | one `SHARADAR/DAILY` pull |
| analyst estimates + StarMine ranks | 20 | **drop** — removing them RAISED IC |

**Measured, live window 2026-01-26..04-21 (top-400 rank IC):**
- FULL (LSEG, 383 feat): **+0.1672**
- TIER-A (LSEG, 20 cols nulled, 278 feat): **+0.2177**  ← dropping the 20 gained ~30%
- SUBST (FMP+Sharadar, 278 feat): **+0.2196**  ← substitution is FREE (+0.9%, inside noise)

SUBST matched or beat TIER-A in all 3 live months and led on the top-50 book (+20.69% vs +19.35%).

**Why the 20 are droppable rather than a loss:** FMP exposes only *current* estimate snapshots, so backfilling them would stamp 2026 views onto 2009-2025 history. They are also mostly collinear (forward P/S, P/CF, EV/OCF, PEG all reduce to price ÷ an estimate) and stale between sparse revisions. Do NOT buy the TipRanks point-in-time add-on — the evidence says these fields do not earn their place.

**The quarterly half needed no work at all.** The `*_fmp` columns already land on `accepteddate_fmp` (the filing's publication timestamp), 97.8% of symbols, ~5 filings/symbol/yr. Sparse (~2% of rows) by design; the forecast script's per-symbol ffill propagates each forward, which is correct PIT behaviour. Agreement with LSEG on filing rows (Spearman): debt +0.971, cash +0.964, FCF +0.952, interest +0.909, EPS +0.861.

**Scripts (committed):**
- `data/csv/fetch_sharadar_replacements.py` — bulk-export pull of SHARADAR/DAILY (+SF1 if ever needed). No `nasdaqdatalink` dependency.
- `data/csv/substitute_lseg_columns.py` — does the replacement.

**FOUR bugs found during this work — all silent, none would have raised an error:**
1. **Units.** SHARADAR/DAILY reports `ev`/`marketcap` in MILLIONS (AAPL = 4,522,736.4); LSEG in actual USD. Scaled ×1e6. Harmless for a single tree feature, but any downstream ratio mixing it with a dollar denominator is off by 1e6.
2. **Ratio staleness.** LSEG's `*_DailyTimeSeriesRatio_` update DAILY. Computing `ev / raw_quarterly_denominator` only yields values on ~2% of rows (filing dates); ffilling that freezes the ratio between quarters. Fix: ffill the denominator FIRST, then divide daily EV into it.
3. **Quarter×4 ≠ TTM.** Annualising one quarter gave EV/EBITDA rank corr 0.475, EV/EBIT 0.362. A true rolling 4-filing sum plus a top-400 split showed the residual disagreement is microcaps only: **top-400 EV/EBITDA 0.816, EV/EBIT 0.716** (acceptable); beyond rank 1000 it is 0.43/0.28.
4. **Dropping vs nulling.** Feature engineering references several of the 20 by name (`ReturnOnAssets_SmartEstimate`) and raises KeyError if absent. They must be **nulled in place**, not removed.

**⚠️ `.env` defines `NASDAQ_DATA_LINK_API_KEY` TWICE** (line 18 is a placeholder ending in `#`, line 27 is real). A naive first-match regex silently returns HTTP 200 with **0 rows** — no error. Always take the LAST definition.

**Coverage caveat:** SHARADAR/DAILY covers 98.0% of top-400 rows (matching LSEG) but only 82% beyond rank 1000 (SPACs, OTC, closed-end funds, renames like ABC→COR). Full-universe IC is therefore NOT comparable between LSEG-sourced and Sharadar-sourced files; top-400 metrics are.

## 🔴 OPEN / HIGHEST-VALUE: the production ffill is costing LIVE performance
Six controlled runs on the same slice, same flags, one variable each. Scored 2024-01→2026-04-21 on independently recomputed outcomes.

| run | config | IC full | IC top-400 | excess/90d |
|---|---|---|---|---|
| A | PIT guard (drop unrealized rows) | +0.0985 | +0.0566 | +1.64% |
| C | + per-date z-score target | +0.1020 | +0.0619 | +1.89% |
| D | ffill bug **+ PIT guard** | +0.0899 | +0.0585 | +1.79% |
| **E** | **ffill bug, NO guard (rebuild)** | **+0.3282** | **+0.3101** | **+7.90%** |
| **F** | **`--ffill-target` (live emulation)** | **+0.0573** | **+0.0194** | **+1.19%** |

**1. The leak is now reproduced from code.** E hits IC +0.3282 with a **100% daily win rate** (+21.9%/yr basket excess) on data where a guarded model gets +0.0985. That is the mechanism behind the live parquet's fake 0.38–0.72 pre-2026 IC. Not inferred any more — generated on demand.

**2. The ffill is not just a backtest illusion — it degrades the LIVE model.** F (what production actually does: keep every row, fill unknown targets with last-known) drops IC to +0.0573 full and **+0.0194 top-400** — a two-thirds cut exactly where the book trades. F also reproduces the live *shape*, which A does not:
`F 2026: +0.178 → +0.141 → +0.078 → +0.016` vs `live: +0.164 → +0.116 → +0.022 → −0.060` (A stays flat ~+0.10–0.14 and never fades).

**3. Non-leaky and harmful are compatible.** The user's earlier intuition ("why does it leak if you mask+ffill?") was correct — mask-and-ffill does NOT leak. But it trains the model against labels that are simply wrong for those rows, and the model learns worse relationships. Dropping beats filling.

**4. D shows the damage is conditional:** with the PIT guard present the ffill bug costs almost nothing (+0.0899 vs +0.0985), because the guard removes precisely the rows the bug corrupts. The bug is only devastating when nothing else drops unrealized rows — which is production's situation.

**THE FIX (production, one line + its consequence):**
```python
exclude_from_ffill = ['Date', 'Symbol', 'forward_return']   # add the target
```
then let `valid_idx` drop the resulting NaN rows. Expected: full IC ~+73%, top-400 IC ~3×. Optionally stack per-date z-score normalization (run C) for a further small, regime-conditional gain.
**Not yet applied — production remains untouched pending the user's decision.**

**Caveats:** 28 months of overlapping 90-day windows ≈ 6 independent observations; F approximates production rather than being it. But A-vs-F is internally controlled (same script/data/pipeline, one variable), so the direction is solid even if the magnitude is not pinned.

Artifacts: `data/csv/experiments/exp{A,B,C,D,E,F}*.parquet`. New PIT flag `--emulate-prod-ffill` (diagnostic only, never trade it).

## ✅ CLOSED: PIT rerun vs live weekly discrepancy — explained by the above
Same claimed point-in-time basis, very different recent IC:

| month | PIT rerun (expA) | live weekly parquet |
|---|---|---|
| 2026-01 | +0.0975 | +0.164 |
| 2026-02 | +0.1411 | +0.116 |
| 2026-03 | **+0.1416** | **+0.022** |
| 2026-04 | **+0.0988** | **−0.060** |

Leading hypothesis: the production script's ffill bug corrupts **training labels** (rows whose target has not realized get a stale carried value instead of NaN, so they train with wrong y). The PIT script excludes them properly. If that is the cause, fixing the production ffill could roughly double recent live IC — which would make it the highest-value open item in the project. Not yet verified. Alternative: the 2023+ slice vs full-history training data differ in some way that matters despite both using a 12-month rolling window.

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

## ⚠️ LIVE SIGNAL IS NEGATIVE RIGHT NOW (measured 2026-09-17)
Scored against realized 30-trading-day outcomes (the 90d target only realizes through 2026-04-21, so this is the freshest honest read available):

| month | IC full | IC top-400 | days | % days +ve |
|---|---|---|---|---|
| 2026-02 | +0.056 | +0.026 | 21 | 76% |
| 2026-03 | +0.110 | +0.168 | 22 | 95% |
| 2026-04 | +0.073 | +0.155 | 21 | 95% |
| 2026-05 | −0.039 | −0.035 | 22 | 32% |
| **2026-06** | **−0.169** | **−0.270** | 21 | **0%** |
| **2026-07** | **−0.129** | **−0.209** | 14 | **0%** |

June and July had **zero positive days**. Top-50 basket returned −11.61% in June while the pool returned +1.31% and the BOTTOM-50 returned +5.39% — a −17.0% long/short spread, i.e. the ranking inverted. Two independent non-overlapping observations in that stretch agree (−0.057, −0.150).

Same shape as the 2025-06..11 collapse, which appeared identically in ALL SIX objective-function experiments — so it is regime, not code. No objective, feature set, or capacity setting fixed it then. This is what the riskbrakes overlay exists for. **Treat as the top priority over any modelling work.**

## Next / open
Nothing is mid-flight. Candidates, roughly in priority order:
- **Investigate the June-July signal inversion** (above) — live capital is exposed.
- **Port the two fixes into production** (`forecast_returns_ml_walk_forward_FIXED.py` is committed and validated at ~2x live IC). Requires a from-scratch rerun afterward: `--preserve-existing` would otherwise freeze the degraded predictions.
- **Adopt the LSEG replacement** — scripts committed, verified free. Removes the vendor dependency.
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
