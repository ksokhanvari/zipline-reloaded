# Handoff

## State (2026-09-21)
Branch `claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA`. Production script **v3.3.27, behaviour unchanged all session** (only the additive 9/6/12-month reporting block).
Commits: `b7a26935` technical factors · `d34ce024` weekly PIT script · `17b5ee10` LSEG replacement · `49029640` ffill/PIT fixes.

## 🔴 THE ONE UNAPPLIED WIN: production ffill fix
`data/csv/forecast_returns_ml_walk_forward_FIXED.py` = production + 4 edits (record `target_date`; exclude `forward_return` from ffill; keep `target_date` out of features; PIT guard on training rows placed AFTER the ranking block).
**Measured on the live window, production's own flags: top-400 rank IC +0.1672 vs the live file's +0.0992; basket excess +7.65% vs +4.22%/90d.** Strongest evidence in the project (626 days, identical rows). Still NOT ported into production.
Porting requires a from-scratch rerun: `--preserve-existing` would otherwise freeze the degraded predictions.

## What the whole experiment programme measured
All on the weekly Tuesday walk, 20d target, PIT-guarded, scored on independently recomputed returns (never the stored `forward_return`).

| change | top-400 IC | verdict |
|---|---|---|
| **fix ffill + PIT guard** | +0.0992 → **+0.1672** | ✅ biggest win, unapplied |
| **drop 20 LSEG analyst/StarMine cols** | +0.1672 → **+0.2177** | ✅ they were *hurting* |
| **replace LSEG with FMP+Sharadar** | +0.2196 vs +0.2177 | ✅ free — vendor removable |
| **weekly vs monthly cadence** | +0.0323 vs +0.0175 | ✅ weekly ~2x, costs nothing |
| 20d vs 90d horizon | +0.0459 vs +0.0022 | ✅ 90d is ~zero at top-400 |
| per-date z-score target | +3.6% relative | marginal, drawdown insurance |
| +77 library indicators (monthly) | +0.0175 → +0.0216 | ~ within noise, direction consistent |
| residual (beta/vol-adjusted) target | basket excess **worse** | ❌ rejected |
| restrict training universe to top-400 | decisively worse | ❌ breadth matters |
| two-stage selection (screen→finalise) | −8% to −19% | ❌ screening destroys info |
| post-hoc z-blend of two models | −5% | ❌ joint training beats blending |

**Pattern: both real wins came from REMOVING bad information. Every attempt to add or reformulate was neutral-to-negative.** Combined with "50 trees ≈ 1000 trees" and "top-20 features = 55% of full IC", the edge is **data-limited**, not model/objective-limited.

## Feature-set split (weekly, 20d, top-400)
| run | features | IC | exc50 |
|---|---|---|---|
| ALL | 340 | +0.0323 | +0.98% |
| FUND only | 269 | +0.0260 | +0.81% |
| TECH only | 91 | +0.0161 | **+0.92%** |

Fundamentals rank the cross-section better; **technicals build better top-50 baskets**. Dropping technicals costs −17% of basket excess, dropping the entire 269-feature fundamental stack costs only −6%. On identical rows TECH alone (+0.926%) ≈ ALL (+0.917%). Combined still beats both ⇒ complementary, keep both.
**~78% of inputs are quarterly fundamentals updating 4x/yr and ffilled between filings** — that frequency mismatch is the structural ceiling on a 20-day forecast.

## 🏃 RUNNING NOW (started 13:20, ETA ~16:50)
`experiments/PROD2025_TECH146_WEEKLY_90d_predfrom_2026/`
Built from the **production** parquet `MLData/20091231_20260915_...perdict-90-1-rolling-12.parquet`, sliced 2025-01-01+, prior-run `predicted_return`/`forward_return` dropped, **LSEG intact (not substituted)**. Production config: 90d/1d/12m, weekly Tuesday walk, predict 2026 only (53 weeks of 2025 = lookback), 37 periods.
- `PROD2025_WITH_TECH` — 314 input cols → 417 trained features
- `PROD2025_NO_TECH`  — 237 input cols → 340 trained features (matched control)
Purpose: WITH vs NO isolates the technicals cleanly; WITH vs the user's live backtest answers "would this help the book" but is confounded (live also has the ffill bug and trains on 2009+).

## ⚠️ THREE LEAKS FOUND THIS SESSION — all caught by SCORING, none by reading code
1. **production ffill** — ffilling `forward_return` defeats `valid_idx`; rebuild IC +0.47 vs honest ~+0.10.
2. **`trend_visual_ichimoku_a/b`** — Ichimoku cloud spans are deliberately displaced FORWARD. Caught by the `--verify` scramble (max diff 3.6e+00 vs 0.00e+00 for the other 84). Now in a `NON_CAUSAL` blocklist.
3. **`_cmp_*` composite columns became FEATURES** — handed the model the forward return itself. IC +0.82, 100% of days positive. **Cost 6.5h of compute.** Now excluded by prefix.

➡️ **`scratchpad/preflight.py` exists to stop #3 recurring.** Runs the real pipeline on a small slice in ~2 min and fails the launch if any feature is the label, correlates >0.95 with the target, or if the mode filter misfires. **Run it before every long job.**

## Scripts (all committed)
| file | purpose |
|---|---|
| `forecast_returns_ml_walk_forward.py` | production, v3.3.27, untouched |
| `forecast_returns_ml_walk_forward_FIXED.py` | production + the 4 ffill/PIT fixes |
| `forecast_returns_ml_weekly_pit.py` | weekly walk + `--train-start`, `--composite-target`, `--target-residual` |
| `fetch_sharadar_replacements.py` | SHARADAR DAILY / SF1 / SEP bulk export |
| `substitute_lseg_columns.py` | LSEG → FMP+Sharadar |
| `build_technical_features.py` | 77 indicators from SEP OHLCV, `--verify` causality |
| `merge_technical_features.py` | join indicators into a training input |
| `build_prod2025_tech_input.py` | build the running experiment's inputs |

## Gotchas that cost real time
- **`.env` defines `NASDAQ_DATA_LINK_API_KEY` twice** (line 18 is a placeholder ending `#`, line 27 is real). A first-match regex returns HTTP 200 with **0 rows** — no error. Take the LAST definition.
- **Symbol case is meaningful**: LSEG marks share classes with a lowercase suffix (BRKa≠BRKA). `.str.upper()` merged them and duplicated rows. Fold case only for the Sharadar join.
- **SHARADAR/DAILY is denominated in MILLIONS** (ev, marketcap). Scale ×1e6.
- **LSEG daily ratios need the denominator ffilled BEFORE dividing**, else the ratio freezes between filings.
- **Quarter×4 ≠ TTM** — EV/EBITDA rank corr 0.475 vs 0.816 within top-400.
- **Feature engineering references dropped columns by name** (`ReturnOnAssets_SmartEstimate`) — the 20 LSEG columns must be **nulled in place**, never removed.
- **`pandas-ta` has no Python 3.8 build** (needs ≥3.12). Use `ta` 0.11.0.
- **Untracked scripts keep getting deleted** — the PIT script (Aug), then the weekly script + substitute script (Sep 20, mid-run). Commit immediately.

## Backtest reality check (from the user's zipline runs)
+77% CAGR decomposes to **+24.4%/yr beta (β 1.22) + ~38%/yr alpha at 2.06x gross**; real max DD **−30.8%** (monthly returns understate it); turnover **88x/yr** (→ 8.8%/yr drag at 10bps; the 90d signal's 52x is cheaper). **Returns are NOT driven by the ML signal** — the 20d (IC +0.046) and 90d (IC +0.002) signals produced +366.8% and +359.0%. In 2026Q2 the ML measured **negative** IC while the book made +65.6%. The earners are `cash_return`/`fcf`/`bc`/`30mom` + leverage. Removing the 10 best days turns 2024 and 2025 negative.
**Live signal has been negative since June 2026** (top-400 IC −0.27 Jun, −0.21 Jul, zero positive days), confirmed independently in the live parquet and the PIT runs. Same shape as the 2025-06..11 collapse, which appeared in all six objective experiments ⇒ regime, not code.

## Next / open
1. **Port the ffill fix to production** + a from-scratch rerun — the one validated, unapplied win.
2. Read the running experiment (~16:50) — does the technical set help in production config.
3. Investigate the June-2026-onward signal inversion; live capital is exposed.
4. Factor-attribute the book (`cash_return` vs `fcf` vs `bc` vs `30mom` vs `mlf`) — likely worth more than further feature work.
5. Optional: weekly TECH146 to see if the cadence and feature gains compound.

## Preferences
- **Never modify production without asking.** Show the diff afterward.
- **Run `preflight.py` before any multi-hour job.** Three leaks, 6.5h lost.
- Score the output immediately — reviewing code has caught none of the bugs.
- Don't commit/push unprompted. Name every artefact with tag + date range + config.
- User is a sharp quant: show the math, run the test, don't fit narratives. When a result looks too good, it is a bug until proven otherwise.
