# Handoff

## State (2026-09-22)
Branch `claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA`. **Production is UNCHANGED and should stay that way** — v3.3.27, only the additive 9/6/12-month reporting block was ever touched.
Commits: `688a702f` composite leak fix + merge tooling · `b7a26935` technical factors · `d34ce024` weekly PIT script · `17b5ee10` LSEG replacement · `49029640` ffill/PIT fixes.

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

➡️ **`scratchpad/preflight.py` gates every long run in ~2 min** — fails the launch if a feature is the label, correlates >0.95 with the target, or the mode filter misfires. Use it. Always.

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
