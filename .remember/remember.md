# Handoff

## State (2026-09-25)
Branch `claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA`. **Production is UNCHANGED and should stay that way** — v3.3.27, only the additive 9/6/12-month reporting block was ever touched.
Commits: `688a702f` composite leak fix + merge tooling · `b7a26935` technical factors · `d34ce024` weekly PIT script · `17b5ee10` LSEG replacement · `49029640` ffill/PIT fixes.

## 🆕 FACTOR STUDY + FORECAST CANDIDATES (2026-09-24/25) — READ FIRST

### Current best: MAX-RANK blend  (not yet live — pending the 2018-26 backtest)
`data/csv/experiments/FACTOR19_FORECAST/MAXRANK_F19_MLF1_forecast_only.csv` (2015-01 → 2026-09, mlf1 units)
score = max(pct-rank 19-factor GBM, pct-rank honest mlf1) per day → top-N = union of both models' strongest picks.
Algo backtests, mom=30 + regime-aware beta-dual, 2023-02→2026-09:
| file | CAGR | vol | Sharpe | maxDD | beta | vol-matched CAGR |
|---|---|---|---|---|---|---|
| **maxrank** | +52.0% | 23.4% | **1.91** | **−19.6%** | 1.08 | **+71.3%** |
| blend 50/50 | +53.8% | 25.0% | 1.85 | −21.4% | 1.11 | +68.2% |
| 19factor GBM | +41.8% | 21.4% | 1.74 | −24.7% | 0.90 | +62.6% |
| direct EQ19 (no ML) | +38.0% | 22.3% | 1.56 | −20.9% | 0.83 | +53.8% |
| baseline weekly PIT mlf1 | +59.8% | 30.7% | 1.68 | −24.3% | 1.26 | +59.8% |
Max-rank beats baseline 3 of 4 years (loses 2025), 22/44 months — edge is LOWER RISK, not higher return.
**2018-26 RESULT (done):** maxrank Sharpe 1.49 vs monthly PIT mlf1 1.32, DD −22.6% vs −24.1%, vol-matched CAGR
+50.9% vs +43.6% → passes the rule. BUT 2018-22 slight loss (1.27 vs 1.34, lost 2018/2021/2022); 2023-26 sweep
(1.88 vs 1.30). vs WEEKLY mlf1 2023-26: 1.91 vs 1.68, DD −19.6 vs −24.3. Recommendation: switch, monitor monthly.
**TODO (on hold, user 2026-09-25):** build WEEKLY PIT mlf1 from 2018 (~6-8h run: PIT script, --walk-frequency weekly,
--train-start 2018-01-01, --fundamental-only --ffill-target, same input as PIT_BASE_WEEKLY) → backtest vs maxrank 2018-26.
**(superseded)** backtest maxrank vs monthly PIT mlf1 (`PIT_BASE_MONTHLY_90d_FULLHIST_LSEG_forecast_only.csv`) over
2018-01→2026-09 (weekly file doesn't go back). Switch production only if maxrank holds there.

### The 19 factors (selected on 2010-17 ONLY, tested 2018-26)
Scripts: `build_factor_panel.py`, `experiments/FACTOR_STUDY/{score_factors,select_and_validate}.py`;
list: `experiments/FACTOR_STUDY/selected_factors_2010_2017.csv`. Families: earnings surprise (FMP sue sign, LSEG
EPS surprise/px, GPM surprise) · quality (StarMine EarningsQuality, fscore, GP/assets, goodwill/assets) · cash-flow
value (FOCF/mcap, FCF/EV, GP/EV, trailing E/P) · issuance (share growth, −) · analyst (StarMine Alpha sector rank,
LTG) · growth-tilt regime bets (B/P −, div yield −, R&D/rev, SBC/rev, dollar vol). 7 of 19 are LSEG-only.
OOS 2018-26 top-400: GBM-19 IC +0.079 (t 3.96, 70% months +) vs GBM all-85 +0.059 vs production mlf1 +0.020.
But top-30 basket ties mlf1 (+3.4% vs +3.7%/90d) — mlf1 picks the top well, GBM ranks the list well → hence blend.

### Blending — measured (OOS top-30 excess, top-400)
max-rank +5.21% > 50/50 avg +3.49% ≈ GBM +3.43% > min-rank(AND) +3.01% ≈ screen-then-pick ≈ adaptive-IC.
LEARNED blends (stacked ranker, tail classifier, regime-aware) do NOT beat max-rank: they overfit 2023-26 and
fail 2018-22 (regime version: +8.29% recent, +0.03% 2018-22). With ~1 independent obs/quarter, fixed rules win.
Direct (non-ML) equal-weight score: IC +0.046 OOS; adaptive IC-weighting WORSE (0.027); dropping growth-tilt WORSE.

### Nulls — don't retry
- Daily refresh: 21-day-old factor values keep 80-120% of IC (EV ratios ~100%). Monthly carry is fine.
  Only untested daily idea: event-triggered re-score on earnings dates (surprises used ~2 weeks late now).
- Technicals: 0 of 18 full-history close-based technicals pass the 2010-17 gate; most flip sign 2010-13 vs 14-17.
  Third independent null for technicals at 90d in large caps.

### DATA DEFECTS in the production input (affect production too)
1. ~10% of FMP filings stamped ON the quarter-end (date-only) — 1-2 months before public (real median lag 38d).
   LOOK-AHEAD in production features. Study pushes them +60d. Fix at source: re-export statements with filingDate.
2. RefPriceClose NOT split-adjusted (AMZN 2022 −94.9%); 192 artefacts. Study uses mcap return on those days.
3. 52 of 213 "month-ends" were stray weekend rows of junk symbols (mcap 0). Any month-end sampling must require a
   real cross-section (≥1500 rows). This produced a fake IC +0.15 / +70% baskets before it was caught.
4. EV / debt metrics meaningless for Financials — null them; raw $ levels (EV, Debt_Total, PT) are size/price proxies.

### NEW FMP DATA — downloaded & tested 2026-09-25: ALL NULL (don't retry)
Script `data/csv/fetch_fmp_extras.py` (key FMP_API_KEY in .env; resumable cache experiments/FMP_EXTRAS/raw/).
Factors `data/csv/build_fmp_extra_factors.py`, gate `experiments/FACTOR_STUDY/test_fmp_extras.py`.
- earnings (EPS+revenue surprise, beat streaks): pass 2010-17, FADE 2018+ (rev_beat 0.032→0.005); adding 5 of them
  HURT OOS (IC 0.079→0.073, top-30 3.44%→2.63%). Post-earnings drift has decayed.
- analyst grade changes (net upgrades 90d): null, sign flips. insider open-market buys 180d: slightly NEGATIVE, n.s.
  (insider effect lives in small caps; top-1000 mostly zero buys).
- Gotchas: all-company earnings-calendar CAPS at 4,000 rows/call (a month hits it) → use per-symbol /earnings.
  Unfiltered insider feed ~8h (award/tax filings) → filter transactionType=P-Purchase (~25 min).
- FILING-DATE LEAK ROOT CAUSE (found in user's collector repo /Users/kamran/Documents/Code/qtrader/fmp-data):
  collector keeps all columns; consolidate_fmp_data.py:846 correctly uses filingDate — but FMP's OWN filingDate
  equals the period end for 8-13% of rows (acceptedDate no better, per-symbol endpoint identical) → no clean source.
  Fix = rule at line 846: filingDate < period_end+10d → SEC deadline (10-Q +45d, Q4/10-K +90d). Patch drafted
  (see session), NOT applied — user said leave it for now (2026-09-25). Applying needs consolidator rerun + full retrain.

### FMP export recommendation (probed live via FMP connector)
✅ earnings history (EPS + REVENUE actual vs est, 2009+; treat est==actual as missing — backfilled placeholders)
✅ insider trades (Form 4, filingDate) · ✅ analyst grade actions (2012+) · ✅ statements re-export WITH filingDate
❌ TTM/bulk snapshots, forward estimates (look-ahead), shares float, historical consensus (2019+ only, noisy)

## 🆕 SESSION 2026-09-23/24 — honest PIT baseline + algo sleeve tests (READ FIRST)

### The honest history exists now
`experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/` — production config exactly (production parquet 20260922, LSEG intact,
`--fundamental-only` = 269 feats = production's set, monthly, 90d/1d/12m, `--ffill-target`, predict 2010-06→2026-05).
`experiments/PIT_BASE_WEEKLY_90d_from2023_LSEG/` — same, weekly, from 2023-01.
NOTE: the PIT script adds 71 native `t_` technicals that production does NOT build → always use `--fundamental-only` for a production twin.

**The production parquet's history is fiction:** top-400 IC +0.61 / ICIR 3.29 / 99% days+ (rebuild leak). Honest 16-yr: IC +0.013, ICIR 0.078, 54%.
`PRODUCTION_SUBSTITUTED_*` (production script, rebuild) carries the same leak (IC +0.55, 100% days+) — don't evaluate on it.
**Use ICIR as a leak detector:** real equity signals run ~0.3–1.0; >1.5 or >85% days positive = contamination.
90d overlap: lag-1 rho 0.886, Newey-West VIF ~40 → 640 daily obs ≈ 16 independent. Never annualize ICIR by √252.

### The signal selects, it doesn't rank (algo universe = top-400 by mcap, FILTERED_UNIVERSE_SIZE=400)
Honest PIT, top-N by forecast within top-400: 150 +0.31% / 50 +1.23% / 30 +2.02% / 10 +4.55% per 90d, ~60% dates.
Pool gate (150 of 400) is near-worthless (+0.06% in 2010s). Value is in TIGHT selection = the 30-name momentum sleeve.
Top picks are high-beta (top-30 β 1.20, vol 1.31×) but ~60–70% of excess survives beta adjustment (top-30 alpha +1.25%/90d).

### Cadence must match horizon
90d: MONTHLY beats weekly (IC +0.049 vs +0.031, ICIR 0.358 vs 0.217, wins every year 2023–26). 20d: weekly won.
Production walks monthly at 90d = correct.

### Algo backtests (all on honest weekly PIT forecast, 2021-02→2026-09)
| config | weekly era 2023+ CAGR / Sharpe / maxDD |
|---|---|
| **mom=30 (keep)** | **+57.5% / 1.54 / −25.4%** |
| mom=0 | +35.1% / 1.48 / −15.2% |
Pre-2023 the weekly file has NO coverage (algo ran on older mlf1) — that's where mom=0's apparent edge came from.
The earlier "+766% vs +381%, Sharpe 4.0" mom=0 win was on LEAKED predictions. **Recommendation: keep TOP_MOMENTUM_STOCKS=30.**

### Dead ends this session (all null — weight-formula tuning is exhausted)
- CONSENSUS_BOOST 0.5 (weight names both sleeves picked): corr 0.996 vs base, Sharpe 1.30 vs 1.28 → delete the multiplier.
- Overlap as quality signal: high overlap predicts WORSE fwd returns (rank corr −0.125 @21d), regime-confounded. Book holds ~41 names
  (median 9 overlap), effective N 26.7.
- Beta-dual (extra /winsorize(beta60IWM) on the long weight; needs clip(0.3,3)+NaN→1 guard): 2023-26 Sharpe 1.62 vs 1.52 base,
  CAGR +59.7% vs +57.0%, but book beta unchanged at 1.35 and maxDD −26.3%. Small gain, not a drawdown fix. **Keep.**
- Posfill / top-up to 50 names with next value picks: beta 1.35→1.24 but CAGR −4.5pp, Sharpe 1.59. Flat. **Drop.**
- Beta-dual + posfill combined: Sharpe 1.49, CAGR +48.5% — WORSE than either alone. **Drop.**
- **Weight/selection tuning is exhausted** (consensus, overlap, top-up, beta penalty all within noise). Stop proposing them.
  FINAL: mom=30 + beta-dual REGIME-AWARE SPY/IWM (guarded; Sharpe 1.68 vs 1.62 IWM-only), CONSENSUS_BOOST=0, no top-up.
  Forecast input: weekly PIT today; candidate replacement = MAXRANK file (see top).
  Remaining levers: forecast quality, and book-level risk (hedge sizing / gross / riskbrakes) for the ~−26% DD.
- Universe narrowing, mom=0 — withdrawn (my errors: assumed 1500 universe, and pool ranks by ML not mcap).

### Next
1. Backtest the MONTHLY PIT forecast (2010→) — one honest long track record, better IC than weekly.
2. Upside is in forecast quality, not weights: switching mom=30 from old→weekly PIT forecast took Sharpe 0.76→1.54.
3. Still open: signal-quality monitor (weekly rolling ICIR/hit-rate), factor attribution.

### Process lesson
I drew conclusions from the wrong universe twice before reading `process_universe`. Grep the algo's constants
(FILTERED_UNIVERSE_SIZE, POOL_ENTER_N, LONG_PORTFOLIO_SIZE) and trace selection BEFORE modelling it. Snippets must
anchor on code, not line numbers (a mis-anchored insert caused an UnboundLocalError).

## 🟢 BOTTOM LINE: KEEP PRODUCTION AS-IS
Eleven attempts to improve the signal. **Every one failed or was overturned by better measurement.** The user's instinct to resist changes was correct each time.

| change | verdict |
|---|---|
| apply the ffill "fix" | ❌ **user's theory won** — see below |
| +77 technical indicators | ❌ null in prod config (rank corr **0.972**, IC gap +0.0005) |
| swap in raw momentum | ❌ 2026 backtest +61.9% vs live +81.3%, Sharpe 1.94 vs 2.50, beta 1.58 vs 1.31 |
| momentum-only model | ❌ catastrophic: IC −0.026, exc50 +1.19% vs raw momentum's +7.68% |
| 3-month lookback | ❌ IC +0.0036 vs 12-month's +0.0239 (6.6× worse) |
| residual (beta/vol) target | ❌ basket excess worse |
| restrict training universe | ❌ worse even on its own top-400 evaluation |
| two-stage selection | ❌ −8% to −19%; tighter screen = worse |
| post-hoc z-blend | ❌ −5%; joint training beats blending |
| composite multi-horizon target | ~ no gain |
| per-date z-score target | ~ +3.6% relative, marginal |
| **weekly over monthly cadence** | ✅ **+0.0323 vs +0.0175 — already what production does** |
| **drop 20 LSEG analyst/StarMine cols** | ✅ +30% IC — but see LSEG note |

## 🔴 I WAS WRONG ABOUT THE FFILL — the user's theory is correct
I pushed the ffill fix on a 2× IC gain (+0.0992 → +0.1672, 626 days, identical rows). **That measured the wrong objective.** IC is on the stated 90d target; the book cares about basket excess, and it moves the other way:

| top-400 | IC | exc50 | exc30 | win% |
|---|---|---|---|---|
| FFILL_ON (stale labels kept) | +0.0272 | **+6.87%** | **+11.75%** | **72%** |
| FFILL_OFF (unrealized dropped) | **+0.0397** | +6.29% | +7.46% | 58% |

**User's mechanism, verified:** a ffilled label IS the symbol's last realized 90d return = trailing momentum. On the rows actually ffilled it correlates **+0.411** with trailing 90d momentum. ~36% of training rows carry one. Removing them raises IC and lowers book performance.
➡️ **Do NOT apply the ffill fix.** `forecast_returns_ml_walk_forward_FIXED.py` stays a reference artifact only.

## 🔑 WHY NOTHING IMPROVES THE SIGNAL — the structural answer
Decile analysis of trailing 90d momentum vs forward 90d return (top-400, 2024-06..2026-05):
```
decile 0-8:  +3.25% to +4.42% mean   <- FLAT. 90% of the distribution has NO signal.
decile 9  : +10.69% mean, but median only +3.20%, skew 2.79
   top  1% of decile-9 names ->  22.3% of its total return
   top  5%                   ->  67.7%
   top 10%                   -> 100.2%   (the other 90% net to zero)
   decile-9 names beating the rest's median: 50.8%  <- a coin flip
```
**The tradeable content is ONE threshold ("top decile or not") with the payoff in a right tail nobody can forecast.** A rank captures that with zero parameters. A 127-leaf/1000-tree learner spends its capacity on the flat 90% and fits noise — which is exactly why the momentum-only model *inverted* (predictions corr **−0.159** monthly, **−0.126** weekly with the momentum it trained on). Weekly retraining did not fix it: the problem isn't stale parameters, it's that there is almost nothing to learn.

This single fact explains all eleven failures: more features have nothing to add, better objectives have nothing to optimize, more capacity fits more noise, shorter lookback gives fewer tail observations.

**Implication:** returns come from *reliably holding enough top-decile names to be present when the tail fires* — so breadth, rebalance discipline and drawdown survival matter more than signal refinement. The riskbrakes work has a stronger case than another model iteration.

## ⚠️ OPEN RISK: signal negative since April 2026
Every configuration, including the live file: top-400 IC +0.32 (Jan) → +0.18 → +0.07 → **−0.07 (Apr) → −0.13 (May)**. Basket excess follows: +31.7% → +18.2% → +4.6% → +1.5% → −4.4%. Same shape as the 2025-06..11 collapse that appeared in all six objective experiments ⇒ regime, not code. **Live capital is exposed and nothing in this session addresses it.**

## Backtest reality check
+77% CAGR = **+24.4%/yr beta (β 1.22) + ~38%/yr alpha at 2.06× gross**. Real max DD **−30.8%** (monthly returns understate it). Turnover **88×/yr** → −8.8%/yr at 10bps. Removing the 10 best days turns 2024 AND 2025 negative — fat-tailed in every year.
**Returns are not ML-driven:** the 20d signal (IC +0.046) and 90d signal (IC +0.002) produced +366.8% and +359.0%. In 2026Q2 the ML measured *negative* IC while the book made +65.6%. The earners are `cash_return`/`fcf`/`bc`/`30mom` + leverage.

## Momentum vs ML — leadership rotates by regime
```
top-50 basket excess, top-400:
  2025 only     momentum +6.49%   ML  +5.03%   -> momentum
  2026 only     momentum +10.90%  ML +13.47%   -> ML
  full period   momentum  +7.61%  ML  +7.17%   -> momentum (barely)
```
Neither dominates. My earlier "momentum beats the ML" claim was a full-period average that inverts in 2026 — I should have split by year before stating it.

## LSEG replacement — built, free, NOT yet A/B'd at production horizon
Scripts committed and working. Measured free at 20d weekly (+0.2196 vs +0.2177). **Caveat: never A/B'd at 90d weekly with LSEG intact as the control**, so the "free" claim is horizon-transferred, not proven at production settings. The substituted weekly 90d forecast exists if wanted:
`experiments/PIT_WEEKLY_90d_SUBSTITUTED_20230101_20260915_predfrom_20240102/` (2.09M rows, 2024-01-02→2026-09-15, uses non-leaky `--ffill-target`).
Its prediction distribution differs from production (mean +6.69 std **26.10** vs live +9.43 std 15.38) — the `**1.2` exponent is scale-sensitive, so sizing would differ.

## ⚠️ THREE LEAKS FOUND — all caught by SCORING, none by reading code
1. **production ffill in REBUILDS** — rebuild IC +0.47 vs honest ~+0.10. (Live runs are fine: no future exists to leak.)
2. **`trend_visual_ichimoku_a/b`** — Ichimoku cloud spans are deliberately displaced FORWARD. Caught by the `--verify` scramble (3.6e+00 vs 0.00e+00 for the other 84). Now in `NON_CAUSAL`.
3. **`_cmp_*` composite columns became FEATURES** — handed the model the forward return. IC +0.82, 100% of days positive. **Cost 6.5h.** Now excluded by prefix.

➡️ **`data/csv/tools/preflight.py` gates every long run in ~2 min** (committed — it used to live in the scratchpad, which rotates) — fails the launch if a feature is the label, correlates >0.95 with the target, or the mode filter misfires. Use it. Always.

## Gotchas that cost real time
- **Feature starvation**: 200-day windows need ~9.5 months. Slicing an input to the first training date leaves early features 0% valid. **Always build the slice ≥12 months before the first prediction date.** This invalidated one whole live-vs-PIT comparison.
- **`.env` defines `NASDAQ_DATA_LINK_API_KEY` twice** (line 18 placeholder ending `#`, line 27 real). First-match regex → HTTP 200 with **0 rows**, no error. Take the LAST.
- **Symbol case is meaningful**: LSEG marks share classes lowercase (BRKa≠BRKA). `.str.upper()` merges them. Fold only for the Sharadar join.
- **SHARADAR/DAILY is in MILLIONS** (ev, marketcap) — scale ×1e6.
- **Quarter×4 ≠ TTM** (EV/EBITDA rank corr 0.475 → 0.816 in top-400).
- **Dropped LSEG columns must be NULLED IN PLACE**, not removed — feature engineering references them by name.
- **`pandas-ta` needs Python ≥3.12** — use `ta` 0.11.0.
- **`predicted_return` is NOT in `exclude_cols`** — building an input from a production OUTPUT parquet requires dropping it, or it becomes a feature.
- **Untracked scripts keep getting deleted mid-session** (PIT script in Aug; weekly + substitute scripts on Sep 20). Commit immediately.

## Scripts (all committed)
`forecast_returns_ml_walk_forward.py` (production, untouched) · `..._FIXED.py` (ffill/PIT fixes — reference only, do NOT deploy) · `forecast_returns_ml_weekly_pit.py` (weekly walk, `--train-start`, `--composite-target`, `--target-residual`, `--feature-whitelist`) · `fetch_sharadar_replacements.py` (DAILY/SF1/SEP) · `substitute_lseg_columns.py` · `build_technical_features.py` (77 indicators, `--verify`) · `merge_technical_features.py` · `build_prod2025_tech_input.py`

## Artifact inventory — `data/csv/experiments/` (all gitignored, on disk only)
Naming: `TAG_<config>_<daterange>_predfrom_<first prediction>`. Each dir holds
`<TAG>.parquet` (full) and `<TAG>_forecast_only.csv` (Symbol,Date,predicted_return).

**The runs that produced the conclusions:**
| dir | what it settles |
|---|---|
| `FFILL_THEORY_MONTHLY_2024_predfrom_2025/` | **FFILL_ON vs FFILL_OFF** — the user's stale-label theory. The decisive pair. |
| `MOMONLY_MONTHLY_COMPOSITE_2024_predfrom_2025/` | momentum-only model inverts (IC −0.034) |
| `WEEKLY_MOMTEST_2024_predfrom_2025/WK_MOMONLY` | weekly retraining does NOT fix the inversion (−0.126 vs −0.159) |
| `LB3_…` vs `LB12_WEEKLY_COMPOSITE_TECH_2024_predfrom_2025/` | lookback 3 vs 12, one variable. 12 wins 6.6× |
| `PROD2025_TECH146_WEEKLY_90d_predfrom_2026/` | 77 technicals in production config — null (rank corr 0.972) |
| `PIT_WEEKLY_20d_{,TECHONLY,FUNDONLY}_SUBSTITUTED_…/` | the three-way feature split |
| `PIT_WEEKLY_90d_SUBSTITUTED_…/` | **the weekly LSEG-replacement forecast**, if a backtest is wanted |
| `PRODUCTION_SUBSTITUTED_20091231_20260915/` | full-history (2010-2026) production-script run on substituted data |

**Reusable inputs already built** (rebuilding costs hours):
- `experiments/20091231_20260915_SUBSTITUTED.parquet` — full history, LSEG replaced
- `experiments/20230101_20260915_SUBST_TECH146.parquet` — 2023+, substituted + 77 technicals
- `experiments/LB3_…/input_2024_WITH_TECH.parquet` — 2024+, **LSEG INTACT** + 77 technicals
- `technical_features_2022_2026.parquet` — 77 indicators, float32, causality-verified
- `sharadar_raw/sharadar_{daily,sep}_*.parquet` — DAILY (22.4M rows) and SEP OHLCV (7.8M)

## Tooling kept in-repo (`data/csv/tools/`)
`preflight.py` — **run before every multi-hour job.** ~2 min; fails the launch on
label leakage, any feature >0.95 correlated with the target, a mis-applied mode
filter, or a pipeline that cannot train. `score_runs.py` — compares finished runs
on rank IC + basket excess, always recomputing outcomes from RefPriceClose.
See `tools/README.md`.

## Next / open — in priority order
1. **The negative signal since April 2026.** Live exposure; nothing here addresses it. Risk management, not modelling.
2. **Factor-attribute the book** (`cash_return` vs `fcf` vs `bc` vs `30mom` vs `mlf`). The ML contributes modestly; knowing which sleeve earns would direct effort far better than more model work.
3. LSEG replacement: operational/cost decision (removes a vendor), not a performance one. A/B at 90d weekly if you want it proven.
4. Everything else in the modelling pipeline is measured and closed.

## Preferences
- **Never modify production without asking.** Show the diff.
- **Run `preflight.py` before any multi-hour job.**
- **Score the output immediately** — reviewing code caught none of the three leaks; scoring caught all three.
- **Split results by period before making a claim.** "Momentum beats the ML" was a full-period average that inverted in the period that mattered.
- **IC is not the objective — basket excess is.** The ffill mistake came from optimizing the wrong metric.
- Don't commit/push unprompted. Name artefacts tag + date range + config.
- User is a sharp quant with good instincts: they were right on the ffill, right to resist production changes, right that something looked off. When a result looks too good, it's a bug until proven otherwise.
