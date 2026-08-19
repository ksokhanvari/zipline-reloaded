"""
Long-Short Equity Trading Algorithm
====================================

Overview
--------
This algorithm implements a systematic long-short equity strategy designed for
the US equity market. It combines fundamental analysis, technical indicators,
and machine learning predictions (mlf1) to construct a market-neutral
portfolio that aims to generate alpha in various market conditions.
(Sentiment factors exist in the pipeline but are currently commented out.)

Strategy Components
-------------------
1. **Long Portfolio Selection**:
   - Value stocks: Selected based on cash return (FCF - Interest) / Enterprise Value,
     from the top names ranked by the mlf1 ML prediction (estrank)
   - Momentum stocks: Currently ranked by mlf1 (myrs); the earlier relative
     strength vs QQQ / price-slope ranking is kept as commented code
   - Weighting: Cash return (z-scored softplus) normalized by beta, scaled
     by mlf1 strength (floored positive -- see BUGFIX-1)

2. **Short Portfolio Selection**:
   - Stocks with lowest cash return from sectors not in top momentum
   - Hedged primarily through IWM (Russell 2000 ETF)
   - IWM hedge sized by beta-targeting: the short is anchored to the long
     book's measured IWM dollar-beta (per-stock beta60IWM), scaled by a
     regime-dependent hedge ratio and tactical tilts, blended with the
     legacy alpha-based short signal, and bounded by a floor/cap so the
     hedge is always on while the book is invested but never becomes a
     net-short directional bet

3. **Risk Management**:
   - Dynamic position sizing based on VIX regime signals
   - Drawdown protection: capped dip-buy on shallow drawdowns, genuine
     de-risking beyond DD_DERISK_THRESHOLD (RISKBRAKE-4)
   - Beta-adjusted weighting to control market exposure
   - Sector rotation to avoid shorting momentum sectors
   - Correlated-unwind risk brakes (RISKBRAKE-1..6): GARCH conditional-vol
     gross scaler, acting MLF1 health brake, symmetric intraday hedge
     escalation, split-instrument hedge

4. **Market Regime Detection**:
   - VIX-based trend indicator for bullish/bearish regime classification
     (data lookup restorable via USE_VIXDATA_REGIME; flip detection wiring
     fixed -- see RISKBRAKE-1)
   - SPY moving average crossovers for tactical adjustments
   - Barchart trend data for additional confirmation
   - GJR-GARCH(1,1) conditional volatility of SPY (level, not just flag)

Key Parameters
--------------
- Universe: Top 1500 stocks by market cap, filtered to 500
- Long positions: 50 stocks (30 value + 20 momentum)
- Short positions: 50 stocks (hedged via IWM, optionally IWM+QQQ)
- Rebalancing: Weekly (Tuesdays)
- Maximum single position: 6% (long), 2% (short)

Bugfixes (Jun-2026, merged after passing the A/B backtest)
----------------------------------------------------------
Four code-review fixes, validated side-by-side in
LS-prod-claud-func-refactor-MLf1-bugfix-test.py before merging. Grep
"BUGFIX" for every fix site; fixes 1 and 3 print "[BUGFIX-n]" lines
whenever a guard actually fires.

- BUGFIX-1 (select_long_portfolio): negative mlf1 multipliers floored at
  MLF1_WEIGHT_FLOOR -- previously the abs() in get_normalized_weights gave
  the most ML-bearish stocks LARGE long weights (sign inversion).
- BUGFIX-2 (DRAWDOWN_FACTOR_MULTIPLIER docstring): corrected to state that
  dd_factor INCREASES exposure in drawdowns (capped dip-buy). Doc only.
  (Superseded in this version by RISKBRAKE-4, which makes the response
  two-sided behind USE_TWO_SIDED_DD.)
- BUGFIX-3 (process_universe): beta_ratio NaN fallback (1.3) and ceiling
  (BETA_RATIO_CAP) -- legacy could propagate NaN into orders or inflate
  short weights on a near-zero short-basket beta.
- BUGFIX-4 (get_algo_logger): single Flightlog handler for the algo's
  lifetime instead of one per call (duplicate lines, memory growth).
  Logging only.
- MLF1 health guard (process_universe): daily NaN-fraction / dispersion /
  uniqueness checks on the external mlf1 column, printing "[MLF1-GUARD]"
  warnings to the console when the collection looks stale or broken.
  (Upgraded in this version by RISKBRAKE-3: optionally an acting brake.)

Changes in this Version (iwmhedge-betatarget)
---------------------------------------------
The IWM short-weight computation was rebuilt. Everything else (universe,
long selection/weighting, regime detection, schedules) is unchanged.

Legacy sizing problems addressed:
1. The hedge size was derived from the alpha scores of 50 short candidates
   that are never actually traded, so it had no link to the long book it
   exists to offset (a high-beta and a low-beta long book got the same short).
2. The short side did not scale with dd_factor, so net market exposure
   drifted long during drawdowns exactly when protection mattered most.
3. shortfact was applied twice (once inside total_ws, again at order time in
   place_short_orders), and context.iwm_w stored the pre-scaling number, so
   the intraday adjustment logic operated on a weight that was never held.

The new sizing is OPT-IN via the USE_BETA_TARGETED_HEDGE master switch
(default False = exact legacy sizing, AST-verified identical to the
pre-iwmhedge production code at every order-placing site). Flip it to True
to activate the model.

New sizing (see compute_iwm_hedge_weight for the full model):
- The hedge is anchored to the long book's measured IWM dollar-beta
  (per-stock beta60IWM from the pipeline + the SPY sleeve), so it
  automatically inherits dd_factor and all long-side multipliers.
- A regime-dependent hedge ratio carries only a fraction of beta-neutral:
  light in confirmed uptrends (maximize return), heavier when SPY is below
  its 80-day MA or the VIX regime is bearish (protection first).
- Tactical tilts add return when shorting IWM is favorable (IWM Hull-MA
  downtrend, bc1 bear confirmation) and trim the drag when it is not.
- The legacy alpha-based signal (total_ws) is kept as a 35% blend so the
  relative-weakness information in the short candidate list still matters.
- Hard floor/cap bounds guarantee the hedge never vanishes while the book
  is invested and never becomes a net-short directional bet. When the long
  book is empty (bearish below MA80) the legacy directional short is used
  unchanged, clamped at IWM_BEAR_MAX_SHORT.
- New helper place_hedge_orders orders the exact computed weight (no double
  shortfact), and context.iwm_w now always equals the weight actually held.

All tunables live in the "IWM Hedge Sizing Parameters" constants block
below; each constant documents its purpose, the effect of raising/lowering
it, and a suggested backtest sweep range.

Changes in this Version (lowvol)
--------------------------------
Adds a GROSS-PRESERVING low-volatility overlay to the final long weights.
Goal: raise the Sharpe ratio by removing uncompensated cross-sectional risk
while leaving the return engine untouched. Everything the returns are driven
by is deliberately NOT modified:

  - stock SELECTION (mlf1 estrank/myrs, cash_return value screen) unchanged
  - the alpha WEIGHT signal (cash_return_zsoft / beta * mlf1^1.8) unchanged
  - regime logic, longfact/dd_factor/vix/bc multiplier chain unchanged
  - the IWM hedge model, schedules, and short-signal path unchanged
  - the book's GROSS LONG WEIGHT at every rebalance is bit-for-bit identical
    to the base file: the overlay only REDISTRIBUTES weight between the 50
    long names it was already going to hold. total_wl, the SPY sleeve, and
    the hedge's beta-neutral anchor are therefore the same size as before --
    average market exposure (the beta-driven return component) is preserved
    by construction.

The overlay applies four bounded, sum-preserving levers to the final long
weights (regular_allocation, after ALL alpha/regime multipliers, before the
order loop), each behind its own switch in the "Low-Volatility Overlay
Parameters" block, with a LOWVOL_MASTER kill switch that reproduces the base
file exactly:

  1. Inverse-vol temper  - w_i scaled by (vol_i / median vol)^-eta, clipped
     to [LOWVOL_TEMPER_MIN, LOWVOL_TEMPER_MAX]. Same-alpha names shift
     weight from the highest-vol to the lowest-vol, directly cutting the
     sum(w^2 * sigma^2) term of portfolio variance. A bounded tilt -- the
     alpha ranking is never inverted.
  2. Overbought trim     - names extended > LOWVOL_OB_EXTENSION above their
     own LOWVOL_OB_MA_WINDOW-day MA (parabolic blowoffs) get their weight
     scaled by LOWVOL_OB_TRIM_FACTOR; freed weight goes pro-rata to the
     UNTRIMMED names (exact redistribution -- fixed in riskbrakes; the
     earlier global renorm leaked a sliver of each trim back to its target).
  3. Sector cap          - no GICS sector may exceed LOWVOL_SECTOR_MAX_FRACTION
     of the long book; excess redistributed pro-rata to other sectors
     (i.e. to the next-best alpha names, since weights are alpha-sized).
  4. True per-name cap   - water-filled cap at LOWVOL_NAME_CAP_FRACTION of
     the book. The legacy 6% cap in get_normalized_weights is applied
     BEFORE the final renormalization, so realized weights can drift above
     it; this one holds exactly. (Re-enabled in riskbrakes -- it had been
     switched off while the docstring still advertised it.)

Honest caveat on the "no return reduction" claim: gross preservation
guarantees the same average exposure, and bounded tilts keep the book's
alpha loading close to the base file's. What is NOT guaranteed is the fat
right tail: if a specific year's return is dominated by one high-vol,
overbought name (2021 meme regime), tempering/trimming it costs some of
that year's return -- that IS the volatility being removed. Across a full
cycle the low-vol tilt historically prices flat-to-positive, but A/B the
overlay against the base file on the same day's data (see the baseline
note below) and sweep LOWVOL_VOL_TEMPER_ETA / LOWVOL_OB_TRIM_FACTOR before
trusting it. Every lever prints a "[LOWVOL]" line when it binds, and every
rebalance prints the naive vol-proxy reduction achieved.

Changes in this Version (riskbrakes)
------------------------------------
Adds the CORRELATED-UNWIND RISK BRAKES: six independent, individually
switchable defenses against the one scenario the base algorithm was
structurally unable to respond to -- a market-wide momentum unwind hitting
a ~2.3-beta book of mlf1-selected names.

Motivation (Jul-2026 review of the mlf1 forecasting pipeline): mlf1 was
shown to be, at the data edge, substantially a trailing-90-day-momentum
proxy (rank correlation up to 0.81 with plain trailing return). The book's
existing defenses (weekly mlf1 re-rank, mlf1^1.8 weight scaling,
exit_positions, vol-outlier drop, position caps) all operate PER NAME and
handle idiosyncratic breakdowns well -- measured live: a >10% one-week
dropper loses ~14 percentile points of signal rank and exits the top decile
59% of the time. But every one of those defenses is cross-sectional:
percentiles do not move when all names fall together, weight normalization
restores gross no matter how weak the signal level gets, and rank-based
selection always buys 50 names. Meanwhile the VIX regime system -- the one
designed portfolio-level brake -- was hardcoded off, dd_factor PRESSED INTO
drawdowns, and the intraday triggers could only ever REDUCE the hedge.
Momentum crashes (Daniel-Moskowitz 2016; 1932, 2009) occur after market
declines, during vol spikes, on the rebound -- and vol-MANAGED momentum
(Barroso & Santa-Clara 2015) historically RAISES momentum's return while
halving its crash risk, so the GARCH brake is expected to be return-
positive for this specific book, not a Sharpe-for-CAGR trade.

The six brakes (grep "RISKBRAKE" for every site; every brake prints a
"[BRAKE]" or ALERT line whenever it changes a value):

  RISKBRAKE-1  VIX regime wiring fix + restorable lookup.
               context.vixflag_prev is now captured before each daily
               update (the legacy code NEVER updated it, so the intraday
               VIX-flip handler could not fire even with live data). The
               vixdata lookup itself returns behind USE_VIXDATA_REGIME
               (default False = the existing hardcode, preserved until the
               vixdata collection is re-verified).
  RISKBRAKE-2  GARCH conditional-vol gross scaler (USE_GARCH_VOL_BRAKE).
               The GJR-GARCH estimator that previously only gated the
               bcfactor cut now returns the vol LEVEL (spy_garch_vol) and
               scales the whole long book by ON_VOL/vol (floored) whenever
               conditional vol exceeds GARCH_BRAKE_ON_VOL. Idle (exactly
               1.0x) in calm markets. The hedge anchor scales with the
               book automatically via beta-targeting.
  RISKBRAKE-3  Acting MLF1 health brake (USE_MLF1_HEALTH_BRAKE). The
               logging-only [MLF1-GUARD] checks now also scale gross by
               MLF1_BRAKE_SCALE while the signal column is broken/stale --
               a broken collection previously traded a random book at full
               size.
  RISKBRAKE-4  Two-sided dd_factor (USE_TWO_SIDED_DD). The capped dip-buy
               is kept for shallow drawdowns (unchanged behavior up to
               DD_DERISK_THRESHOLD); beyond it dd_factor declines linearly
               to DD_DERISK_FLOOR instead of pressing at the clip cap.
  RISKBRAKE-5  Symmetric intraday hedge escalation
               (USE_INTRADAY_HEDGE_ESCALATION). SPY crossing BELOW MA80
               now escalates the hedge to the BULL_WEAK ratio the same
               day, mirroring the existing above-crossing reduction --
               previously protection waited up to a week for Tuesday.
  RISKBRAKE-6  Split-instrument hedge (QQQ_HEDGE_FRACTION). The hedge
               order is optionally split IWM/QQQ so part of it covers the
               large-cap growth/momentum leg where the mlf1 book actually
               concentrates, not just small-cap market beta. Sizing is
               unchanged (still anchored to the book's IWM dollar-beta);
               only execution is split. 0 restores IWM-only.
  RISKBRAKE-7  Value-tilt slider (USE_VALUE_TILT_SLIDER). Unlike brakes
               2-4, which reduce exposure, this one RE-AIMS it: as the
               account draws down, the long book slides along its own
               value/momentum axis -- the momentum sleeve shrinks in favor
               of the cash-return (value) sleeve, and the mlf1 weighting
               exponent softens so cash_return_zsoft reasserts control of
               the weight ordering. The book stays fully invested and
               fully aligned with the signal pipeline; only the internal
               mix moves. Evidence (Jul-2026 measurement): within the
               momentum pool, the top-30-by-cash-return slice returned
               +2.3% in July 2026 while the pool fell -4.9% (+7.2pt
               protective spread), and the cash-return selection carries
               IC t=2.6 over 67 months. Cost side is symmetric: the same
               tilt lost -4.2%/-7.6% vs the pool in the Jan/Apr-2026
               momentum-mania months -- which is why the slider is
               drawdown-gated instead of always-on.

A/B guarantee: with USE_VIXDATA_REGIME=False, USE_GARCH_VOL_BRAKE=False,
USE_MLF1_HEALTH_BRAKE=False, USE_TWO_SIDED_DD=False,
USE_INTRADAY_HEDGE_ESCALATION=False, USE_VALUE_TILT_SLIDER=False and
QQQ_HEDGE_FRACTION=0.0 this file
trades identically to the lowvol base (the RISKBRAKE-1 vixflag_prev capture
is inert while vixdata stays hardcoded to 0, and the lowvol trim/name-cap
corrections are part of the lowvol A/B leg here). Recommended validation:
run the full-off configuration against the lowvol file first, then enable
brakes one at a time.

Tuning summary (most impactful first):
  GARCH_BRAKE_ON_VOL     - when the vol brake engages; sweep 0.18 - 0.28
  DD_DERISK_THRESHOLD    - where dip-buy hands over to de-risk; 0.06 - 0.12
  QQQ_HEDGE_FRACTION     - factor-vs-market hedge mix; sweep 0.0 - 0.5
  VALUE_TILT_DD_START/_FULL - where the value tilt engages/saturates
  VALUE_TILT_MAX_SHIFT   - how many momentum names hand over to value; 8 - 20
  MLF1_BRAKE_SCALE       - broken-signal gross haircut; 0.3 - 0.7
  DD_DERISK_FLOOR/SLOPE  - depth/speed of the deep-drawdown unwind

Baseline reproducibility note (Jun-2026 debugging session): yearly returns
recorded as baselines are only comparable if they were produced by the same
code AND data vintage. Differences observed against pre-Jun-2026 baselines
(e.g. 2021 ~350% -> ~246%) were traced to earlier commits in this file --
notably the top-2 volatility outlier drop in select_long_portfolio (hits
2021 meme-stock winners hardest), the fcf switch from Sharadar to LSEG
(changes the core cash_return alpha), and the bc1 seasonal short adjustment
-- plus day-to-day re-collection of the custom databases, which can shift
even closed historical years between runs.
USE_BETA_TARGETED_HEDGE = False reproduces the pre-iwmhedge legacy behavior
on the same day's data, not baselines recorded under older code/data.

Dependencies
------------
- Zipline/QuantRocket for backtesting and live trading
- Sharadar fundamentals database
- Custom databases: refe-fundamentals, refe-fundamentals-mlf1, vixdata,
  bcdata (refe-fundamentals-sent is defined but currently unused -- its
  sentiment columns are commented out of the pipeline)
- scikit-learn for ML factor

Author: Kamran Sokhanvari
Version: claud-noeps-cr-slope120-t-aug-sept-noesttemp-spy-.1-lw-.9-noshift-pricemc-mlzscore-MLC15-mlsent-iwmhedge-betatarget-bugfix-lowvol-riskbrakes
Last Updated: Jul-2026
"""

import logging
from pytz import timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
from sklearn import impute, linear_model, preprocessing
from arch import arch_model

import zipline.api as algo
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.data.db import Database, Column
from zipline.pipeline.factors import Returns, SimpleMovingAverage, SimpleBeta
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline import sharadar

from quantrocket.master import get_securities
from quantrocket.flightlog import FlightlogHandler


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# These parameters control the algorithm's behavior and can be tuned for
# optimization. Changes here affect portfolio construction, risk management,
# and trading execution.

# -----------------------------------------------------------------------------
# Portfolio Construction Parameters
# -----------------------------------------------------------------------------
UNIVERSE_SIZE = 1500 
"""int: Initial universe size - top N stocks by market cap to consider.
Larger values increase computational cost but may find better opportunities."""

FILTERED_UNIVERSE_SIZE = 500
"""int: Final universe size after applying fundamental and liquidity filters.
This is the pool from which long and short candidates are selected."""

TOP_MOMENTUM_STOCKS = 30
"""int: Number of momentum-based stocks to include in the long portfolio.
These are selected based on relative strength and slope indicators."""

LONG_PORTFOLIO_SIZE = 50
"""int: Total number of long positions in the portfolio.
Split between value stocks (LONG_PORTFOLIO_SIZE - TOP_MOMENTUM_STOCKS) and momentum stocks."""

SHORT_PORTFOLIO_SIZE = 50
"""int: Number of short candidates identified for hedging calculations.
Actual shorting is done via IWM ETF rather than individual stocks."""

MAX_POSITION_SIZE_LONG = 0.06
"""float: Maximum weight for any single long position (6%).
Prevents concentration risk in individual names."""

MAX_POSITION_SIZE_SHORT = 0.02
"""float: Maximum weight for any single short position (2%).
More conservative than longs due to unlimited loss potential."""

# -----------------------------------------------------------------------------
# Correlated-Unwind Risk Brake Parameters (RISKBRAKE-1..6)
# -----------------------------------------------------------------------------
# Six independent defenses against a market-wide momentum unwind -- the
# scenario every existing per-name control (mlf1 re-rank, weight scaling,
# exit_positions, vol-outlier drop) is structurally blind to, because
# cross-sectional ranks and normalized weights are invariant to shocks
# shared by the whole book. See the module docstring "Changes in this
# Version (riskbrakes)" for the full motivation and the A/B guarantee.
#
# Every brake prints a "[BRAKE]" (or ALERT) line whenever it changes a
# value, so the log always shows which defenses fired on any given day.

USE_VIXDATA_REGIME = False
"""bool: RISKBRAKE-1 -- restore the vixdata regime lookup.

False (default): context.vixflag stays hardcoded to 0 exactly as the
legacy line did, so the algorithm remains permanently in the bullish
regime branch. This preserves current live behavior until the vixdata
collection has been re-verified (it was commented out at some point and
the reason is not recorded -- do NOT flip this on without first checking
that CustomFundamentals4.pred is populated and sane in the pipeline
output, e.g. print df.loc[context.ibm_sid].vixflag for a few sessions).

True: the pipeline's vixflag column (vixdata.pred, shifted) drives
compute_trend again -- bearish regime (pred > 0) sets longfact 0.0 below
MA80, shortfact 0.5, clip 1.6, and the intraday VIX-flip de-risking in
initial_allocation becomes reachable.

Independent of this switch, RISKBRAKE-1 also fixes the flip-detection
wiring: vixflag_prev is now captured before each daily update. The legacy
code initialized it once and never updated it, so the intraday VIX-flip
handler could never fire even with live vixdata."""

USE_GARCH_VOL_BRAKE = True
"""bool: RISKBRAKE-2 -- GJR-GARCH conditional-vol gross scaler.

When SPY's annualized conditional vol (spy_garch_vol, the same estimator
that gates the bcfactor cut) exceeds GARCH_BRAKE_ON_VOL at a weekly
rebalance, every long weight is scaled by max(GARCH_BRAKE_MIN_SCALE,
GARCH_BRAKE_ON_VOL / vol). Below the threshold the scale is exactly 1.0 --
the brake costs NOTHING in calm markets, which is where the track record
was built. Because the IWM hedge is beta-targeted off the realized long
weights, the hedge anchor shrinks proportionally and the book stays
internally balanced while de-levered.

Why this is expected to be return-POSITIVE for this book, not a
Sharpe-for-CAGR trade: mlf1 is substantially a momentum factor, and
momentum's catastrophic months cluster in high-vol states after market
declines (Daniel & Moskowitz). Scaling momentum by inverse forecast vol
historically raised its absolute return while halving crash risk
(Barroso & Santa-Clara 2015). Applied weekly at rebalance; the intraday
gap between rebalances is covered by RISKBRAKE-5."""

GARCH_BRAKE_ON_VOL = 0.22
"""float: Annualized conditional-vol level at which the GARCH brake
engages (0.22 = 22%). Below it the brake is idle (scale 1.0). SPY
conditional vol sits under 20% in normal tapes, so 22% keeps the brake
asleep through ordinary corrections and wakes it in genuine vol regimes
(2018Q4, 2020, 2022). The same value serves as the scaling numerator:
at 30% vol the book runs 0.73x, at 40% it hits the floor.
Lower -> earlier, gentler de-levering (more protection, some carry cost).
Raise -> only true crises trigger it. Sweep 0.18 - 0.28."""

GARCH_BRAKE_MIN_SCALE = 0.55
"""float: Floor on the GARCH brake's gross scale. 0.55 means even in a
2020-grade vol spike the book keeps 55% of its normal gross -- de-risked,
not liquidated, so rebound participation (which the dd dip-buy exists for)
is retained. Lower -> deeper crisis de-lever; 1.0 disables the brake's
effect entirely. Sweep 0.40 - 0.70."""

USE_MLF1_HEALTH_BRAKE = True
"""bool: RISKBRAKE-3 -- make the MLF1 health guard an acting brake.

The [MLF1-GUARD] checks in process_universe (NaN fraction, dispersion
collapse, constant column) previously only printed warnings while the
algorithm went on trading a possibly-random 50-name book at full size.
With this switch on, any tripped check also scales the long book by
MLF1_BRAKE_SCALE at the next rebalance (and every rebalance until the
column is healthy again). The guard thresholds themselves are unchanged
(MLF1_MAX_NAN_FRACTION / MLF1_MIN_CROSS_STD below) -- they fire on broken
DATA, not on a weak-but-working signal, so false positives are rare and
the brake should essentially never bind in normal operation."""

MLF1_BRAKE_SCALE = 0.50
"""float: Gross-long multiplier applied while the mlf1 column looks
broken/stale. 0.50 halves exposure to a book whose selection ranks are
unreliable. Lower is more conservative; 1.0 disables the effect (guard
reverts to logging-only). Sweep 0.3 - 0.7."""

USE_TWO_SIDED_DD = True
"""bool: RISKBRAKE-4 -- two-sided drawdown response.

The legacy dd_factor = min(clip, 1 + dd * 5) only ever INCREASES exposure
in drawdowns (capped dip-buy, saturating at clip=1.2 by a 4% drawdown --
see the corrected BUGFIX-2 docstring). That is the right response to the
shallow pullbacks that dominated the live record, and it is PRESERVED
unchanged up to DD_DERISK_THRESHOLD. Beyond the threshold the response
inverts: dd_factor declines linearly from clip at the threshold down to
DD_DERISK_FLOOR, so deep drawdowns -- the regime where a momentum book
pressing 1.2x is the classic crash amplifier -- de-risk instead.

With the defaults (threshold 0.08, slope 5.0, floor 0.5, clip 1.2):
  dd  4% -> 1.20 (dip-buy, unchanged)   dd 12% -> 1.00 (flat)
  dd  8% -> 1.20 (handover point)       dd 16% -> 0.80
  dd 22%+ -> 0.50 (floor)
The hedge inherits the reduction automatically via beta-targeting."""

DD_DERISK_THRESHOLD = 0.08
"""float: Drawdown depth where the dip-buy hands over to de-risking.
Below it, behavior is bit-identical to the legacy dd_factor. 0.08 sits
well beyond the 4% level where the dip-buy saturates, so the entire
historical dip-buy benefit is retained; drawdowns past 8% on a 2.3-beta
book signal something bigger than a dip. Sweep 0.06 - 0.12."""

DD_DERISK_SLOPE = 5.0
"""float: Linear rate at which dd_factor falls past the threshold
(dd_factor = clip - (dd - threshold) * slope). 5.0 mirrors the dip-buy
multiplier for symmetry: exposure unwinds as fast as it was pressed.
Raise -> faster de-risk in deep drawdowns; lower -> gentler glide."""

DD_DERISK_FLOOR = 0.50
"""float: Minimum dd_factor in the deepest drawdowns. 0.50 keeps half of
normal sizing on -- the book never abandons the position entirely, so a
V-recovery is still participated in (at half size) and the strategy
cannot be flat-lined by its own brake. Sweep 0.35 - 0.65."""

USE_INTRADAY_HEDGE_ESCALATION = True
"""bool: RISKBRAKE-5 -- symmetric intraday hedge trigger.

The legacy intraday events in initial_allocation only ever REDUCE the IWM
short (VIX flip bullish; SPY crossing ABOVE MA80). SPY crossing BELOW
MA80 did nothing until the next Tuesday rebalance -- in a fast unwind, up
to a week at the thin bull-regime hedge. This brake mirrors the existing
above-crossing block: on the day after SPY closes below its 80-day MA
(having been above), the hedge is escalated to the IWM_HEDGE_RATIO_BULL_WEAK
sizing immediately, using the beta-neutral anchor published by the last
rebalance (context.iwm_hedge_floor / IWM_HEDGE_FLOOR_RATIO). The weekly
rebalance then re-sizes it properly. Requires USE_BETA_TARGETED_HEDGE.
Never REDUCES the hedge -- if the current short is already larger than
the escalation target, nothing happens."""

QQQ_HEDGE_FRACTION = 0.30
"""float: RISKBRAKE-6 -- fraction of the hedge executed in QQQ instead of
IWM (0.0 = IWM-only, exactly the prior behavior).

The mlf1 book concentrates in large-cap growth/momentum names, but the
hedge instrument is IWM -- small-cap, value-tilted. In a momentum unwind
the longs fall much harder than IWM, leaving the factor leg of the risk
unhedged (basis risk). Splitting execution 70/30 IWM/QQQ puts part of the
short on the index that actually tracks the book's concentration. Sizing
is UNCHANGED -- the total hedge weight is still computed against the long
book's IWM dollar-beta by compute_iwm_hedge_weight -- only the order
execution is split across the two ETFs (both already in the protected
list of exit_positions). Applies to the beta-targeted path and all
intraday hedge adjustments; the legacy (USE_BETA_TARGETED_HEDGE=False)
path is untouched. Sweep 0.0 - 0.5."""

USE_VALUE_TILT_SLIDER = True
"""bool: RISKBRAKE-7 -- drawdown-gated value/momentum tilt slider.

The long book contains two return engines with opposite factor loadings:
the mlf1 momentum ranking (the primary earner in trending regimes; its
within-pool IC inverts in unwinds) and the cash-return value screen
((FCF - interest) / EV; IC t=2.6 over 67 months, and the ONLY component
that made money in the Jul-2026 factor unwind: +2.3% for the top-30 value
slice vs -4.9% for the momentum pool). This slider moves the book along
that internal axis as the account draws down, via two coupled dials in
select_long_portfolio:

  1. Sleeve mix: the momentum sleeve gives up to VALUE_TILT_MAX_SHIFT
     names to the value sleeve (30/20 momentum/value at rest ->
     16/34 at full tilt with the default shift of 14).
  2. Weight ordering: the mlf1 exponent slides from MLF1_WEIGHT_EXPONENT
     (momentum dominates weights) toward VALUE_TILT_MLF1_EXP_MIN
     (cash_return_zsoft reasserts control).

The tilt factor lambda ramps linearly from 0 at VALUE_TILT_DD_START to 1
at VALUE_TILT_DD_FULL and unwinds automatically as the account recovers
(the high-water mark resets the drawdown). Unlike the gross brakes this
keeps the book FULLY INVESTED and inside the same signal pipeline -- it
manages drawdown by changing WHICH conviction gets sized, not how much
market is held, so it composes with (not duplicates) RISKBRAKE-2/4.

Cost awareness: the same tilt that paid +5.7%/+7.2pt in the Mar/Jul-2026
reversal months cost -4.2%/-7.6% vs the pool in the Jan/Apr-2026 mania
months. The drawdown gate exists precisely so the insurance is bought
only once the account is already paying for momentum concentration.
False = slider off, sleeve counts and exponent identical to the base
file at every drawdown level."""

VALUE_TILT_DD_START = 0.06
"""float: Account drawdown at which the value tilt begins to engage.
Below this the book is 100% its normal momentum-dominant configuration.
0.06 sits above routine pullbacks for a ~2-beta book (a 2.5% market dip)
but early enough to start re-aiming well before the two-sided dd brake
(DD_DERISK_THRESHOLD 0.08) begins cutting gross -- tilt first, de-lever
second. Sweep 0.04 - 0.10; keep < DD_DERISK_THRESHOLD so the re-aim
leads the de-risk."""

VALUE_TILT_DD_FULL = 0.18
"""float: Drawdown at which the tilt saturates (lambda = 1). Between
START and FULL the tilt ramps linearly -- e.g. with defaults, a 12%
drawdown runs lambda = 0.5: sleeves 23/27 momentum/value and mlf1
exponent ~1.35. 0.18 means the book reaches its maximum-value posture
in the depth range where momentum unwinds historically feed on
themselves. Sweep 0.14 - 0.25."""

VALUE_TILT_MAX_SHIFT = 14
"""int: Maximum number of long slots the momentum sleeve surrenders to
the value sleeve at full tilt. 14 takes the split from 30/20 to 16/34 --
a decisive re-aim that still keeps a meaningful momentum core (16 names)
for the rebound, so the book is never fully out of its primary signal.
Raise -> deeper insurance, more mania-month drag if a drawdown overlaps
a resumed rally; lower -> gentler. Must be < TOP_MOMENTUM_STOCKS.
Sweep 8 - 20."""

MLF1_WEIGHT_EXPONENT = 1.8
"""float: Baseline exponent of the mlf1 weight multiplier
(mlf1 ** exponent for positive mlf1) -- the legacy hardcoded 1.8,
promoted to a constant so RISKBRAKE-7 can slide it. At 1.8 the
winsorized mlf1 values produce ~100:1 cross-sectional weight dispersion,
which is why momentum dominates the weight ordering at rest. This value
is the exact legacy behavior; change it only as a deliberate re-tuning
of the base strategy, not as part of the slider."""

VALUE_TILT_MLF1_EXP_MIN = 0.9
"""float: mlf1 exponent at full value tilt (lambda = 1). Halving the
exponent (1.8 -> 0.9) compresses the mlf1 multiplier's dispersion
roughly to its square root, letting the cash_return_zsoft component --
whose own dispersion is unchanged -- reassert control of relative
sizing among the 50 selected names. 1.0 would make the multiplier
linear in mlf1; going below ~0.7 makes mlf1 nearly rank-flat and is
not recommended (the momentum sleeve's names would be sized almost
purely by value, fighting their selection rationale). Sweep 0.7 - 1.2."""

# -----------------------------------------------------------------------------
# Low-Volatility Overlay Parameters
# -----------------------------------------------------------------------------
# GROSS-PRESERVING risk reshaping of the final long weights (see the module
# docstring "Changes in this Version (lowvol)"). Applied by
# apply_lowvol_overlay() in regular_allocation AFTER all alpha/regime
# multipliers and BEFORE the order loop. Invariant: the sum of long weights
# after the overlay equals the sum before it, to float precision -- the
# overlay can only move weight BETWEEN the selected names, never remove any.
# total_wl / SPY sleeve / hedge anchor are therefore unchanged in size.
#
# Tuning summary (most Sharpe-sensitive first):
#   LOWVOL_VOL_TEMPER_ETA      - main lever; sweep 0.25 - 0.75
#   LOWVOL_OB_TRIM_FACTOR      - blowoff-top management; sweep 0.50 - 0.85
#   LOWVOL_SECTOR_MAX_FRACTION - binds rarely at 0.40; sweep 0.30 - 0.45
#   LOWVOL_NAME_CAP_FRACTION   - safety rail; rarely needs tuning

LOWVOL_MASTER = True
"""bool: Master switch for the entire low-volatility overlay.

False: apply_lowvol_overlay returns its input untouched -- the file trades
IDENTICALLY to the base LS-prod-claud-func-refactor-MLf1.py on the same
day's data (the overlay is the only functional difference between the two
files). Use False for the A/B baseline leg.

True: the four levers below run, each still individually switchable."""

LOWVOL_USE_VOL_TEMPER = True
"""bool: Lever 1 -- inverse-volatility weight temper.

Each long weight is scaled by (vol_i / median_vol)^-LOWVOL_VOL_TEMPER_ETA
(realized LOWVOL_VOL_WINDOW-day daily-return std), the factor clipped to
[LOWVOL_TEMPER_MIN, LOWVOL_TEMPER_MAX], then the book is renormalized to
its prior gross. Names at the cross-sectional median vol are untouched;
high-vol names fund low-vol names. Because portfolio variance carries a
sum(w_i^2 sigma_i^2) term, shifting weight down the vol spectrum lowers
realized vol even when expected returns are identical -- this is the
overlay's main Sharpe lever.

NOTE (riskbrakes review): long weights are already divided by winsorized
beta60IWM in select_long_portfolio, and vol correlates strongly with beta
cross-sectionally, so the temper is a SECOND dose of a low-vol/low-beta
tilt. Keep that in mind when sweeping eta -- the combined tilt is stronger
than eta alone suggests."""

LOWVOL_VOL_WINDOW = 60
"""int: Lookback (trading days) for the realized vol used by the temper.
60d is slow enough to keep the temper factors stable week to week (limits
overlay-induced turnover) while adapting within a quarter. The pipeline's
10-day 'vol' column is deliberately NOT used -- too noisy, would churn the
book. Falls back to the pipeline column only if the history fetch fails.
Sweep 40 - 90."""

LOWVOL_VOL_TEMPER_ETA = 0.5
"""float: Exponent of the inverse-vol temper. 0 disables (all factors 1);
1.0 approaches full inverse-vol weighting WITHIN the clip bounds. 0.5 is a
half-strength tilt: a name at 2x median vol gets ~0.71x its alpha weight, a
name at half median vol gets ~1.41x (both then clipped). Raise -> lower
realized vol, larger drift from pure alpha sizing (and more of the caveat
about fat-right-tail years applies). Sweep 0.25 - 0.75."""

LOWVOL_TEMPER_MIN = 0.65
"""float: Lower clip on the temper factor. Guarantees no name loses more
than 35% of its alpha-assigned weight to the vol tilt -- the temper is a
tilt, never an exclusion, so the mlf1/cash_return conviction ranking always
survives. Lowering this deepens the vol cut but weakens that guarantee."""

LOWVOL_TEMPER_MAX = 1.35
"""float: Upper clip on the temper factor. Caps how much extra weight a
very-low-vol name can attract, so the overlay cannot concentrate the book
into a handful of sleepy names (which would raise idiosyncratic risk from
the other direction and dilute alpha loading)."""

LOWVOL_USE_OVERBOUGHT_TRIM = True
"""bool: Lever 2 -- parabolic-extension (overbought) trim.

At each rebalance, any long whose price sits more than LOWVOL_OB_EXTENSION
above its own LOWVOL_OB_MA_WINDOW-day simple MA gets its weight multiplied
by LOWVOL_OB_TRIM_FACTOR; the freed weight is redistributed pro-rata across
the UNTRIMMED names (exact redistribution as of riskbrakes). This is
sizing-level profit management on blowoff tops -- the names most exposed
to sharp mean-reversion gaps -- while the name stays in the book and keeps
earning if the trend continues. Gentler than the existing selection-level
drop_top_vol_outliers, which removes names entirely (and is left
unchanged)."""

LOWVOL_OB_MA_WINDOW = 50
"""int: MA window (trading days) defining the overbought reference trend.
50d matches the intermediate-term trend horizon of the book's momentum
sleeve. Shorter windows fire on ordinary breakouts; longer ones only on
multi-month manias. Sweep 40 - 80."""

LOWVOL_OB_EXTENSION = 0.25
"""float: Extension above the MA that counts as overbought (0.25 = price
25% above its 50d MA). Healthy breakouts typically run 10-15% above; 25%
targets genuine parabolic states, so the trim should touch only a few
names per year. Lower -> trims more often (more vol removed, more upside
momentum surrendered). Sweep 0.20 - 0.35."""

LOWVOL_OB_TRIM_FACTOR = 0.65
"""float: Weight multiplier applied to overbought names (0.65 = trim 35%
of the position's weight, redistributed to the rest of the book). 1.0
disables the trim's effect. Sweep 0.50 - 0.85."""

LOWVOL_USE_SECTOR_CAP = True
"""bool: Lever 3 -- long-book sector concentration cap.

Caps every GICS sector at LOWVOL_SECTOR_MAX_FRACTION of the long book's
gross weight; excess is redistributed pro-rata to names in uncapped
sectors. Because weights are alpha-sized, the redistribution flows to the
next-highest-conviction names automatically. Complements the existing
universe-level controls (Financials excluded, Energy/RealEstate count
limits, shorts avoid the top momentum sector), which do not constrain the
LONG book's realized sector mix at all."""

LOWVOL_SECTOR_MAX_FRACTION = 0.42 #0.40
"""float: Maximum fraction of long-book gross weight in one GICS sector.
0.42 is deliberately generous -- mlf1 clusters by sector, and that
clustering is part of the alpha, so the cap should bind only in extreme
concentrations (e.g. a 55% Tech book), not reshape normal ones. Expressed
as a fraction OF THE BOOK, so it is invariant to regime multipliers /
gross level. Sweep 0.30 - 0.45; below ~0.30 it starts fighting the signal."""

LOWVOL_USE_NAME_CAP = True
"""bool: Lever 4 -- true per-name cap with water-filling redistribution.

RE-ENABLED in riskbrakes: the lowvol file shipped with this switched off
(False #True) while the module docstring still advertised the lever as
active -- code and documentation now agree.

The legacy MAX_POSITION_SIZE_LONG cap in get_normalized_weights is applied
BEFORE the final renormalization, so realized weights can drift above 6% of
the book. This lever enforces LOWVOL_NAME_CAP_FRACTION exactly, AFTER all
other levers: weight above the cap is clipped and redistributed pro-rata to
uncapped names, iterating until no name exceeds the cap. Reduces
single-name event risk (earnings gaps, halts) at the very top of the book."""

LOWVOL_NAME_CAP_FRACTION = 0.05
"""float: Hard per-name ceiling as a fraction of long-book gross weight.
0.05 tightens the legacy ~6%-with-leakage to a true 5%. With 50 names the
cap is far from the equal-weight floor (2%), so it touches only the top
handful of positions. Raise to 0.06 to keep legacy-like concentration;
lower toward 0.04 for stronger event-risk control at some alpha-sizing
cost. Safety rail -- tune the temper first."""

# -----------------------------------------------------------------------------
# IWM Hedge Sizing Parameters
# -----------------------------------------------------------------------------
# The IWM short is sized against the long book's actual IWM dollar-beta
# (beta-targeting) rather than the alpha scores of untraded short candidates.
# See compute_iwm_hedge_weight() for the full sizing model.
#
# How the pieces fit together:
#
#   beta_neutral_w = -(long book's IWM dollar-beta)        <- structural anchor
#   beta_hedge_w   = beta_neutral_w * RATIO_* * tilt       <- regime + tactics
#   iwm_w          = BLEND * beta_hedge_w
#                    + (1 - BLEND) * legacy alpha signal   <- final blend
#   iwm_w bounded to [CAP, FLOOR] * beta_neutral_w         <- hedge-role guarantee
#
# Tuning summary (most return-sensitive first):
#   IWM_HEDGE_RATIO_BULL  - biggest lever on bull-market carry; sweep 0.25-0.40
#   IWM_HEDGE_BETA_BLEND  - structural vs alpha-driven sizing; sweep 0.50-0.80
#   IWM_HEDGE_RATIO_BULL_WEAK / _BEAR - drawdown protection depth
#   IWM_HEDGE_FLOOR_RATIO / _CAP_RATIO - safety rails, rarely need tuning

USE_BETA_TARGETED_HEDGE = True
"""bool: Master switch for the IWM short sizing model.

False (default): EXACT legacy sizing. The IWM weight is computed and ordered
precisely as in the pre-iwmhedge production code (see git history) -- the
alpha-based total_ws chain with shortfact re-applied at order time by
place_short_orders, the min(-0.384 * total_wl, total_ws) branch when SPY is
below MA80 in an uptrend, and the original intraday de-risking formulas
(including NOT updating context.iwm_w after intraday orders). With False,
the algo trades exactly as it did before the hedge model was added.
Caveat: that means "identical to the legacy code run today" -- it does NOT
resurrect baseline numbers recorded under an older code/data vintage (see
the "Baseline reproducibility note" in the module docstring).

True: beta-targeted hedge via compute_iwm_hedge_weight(); the IWM_HEDGE_*
constants below take effect and place_hedge_orders orders the exact weight.

IMPORTANT when tuning with True: the hedge floor is
IWM_HEDGE_FLOOR_RATIO * beta-neutral (~ -0.18 at typical book size) and it
BINDS whenever ratio * tilt asks for a smaller short. Example: with
IWM_HEDGE_RATIO_BULL = 0.15 the seasonal bull hedge computes to ~ -0.11, so
the floor overrides it to ~ -0.18 -- LARGER than the legacy seasonal short
(~ -0.12), which drags on bull-market years (this is what depressed
2021/2022 returns). If you want IWM_HEDGE_RATIO_BULL below ~0.20 to take
effect, lower IWM_HEDGE_FLOOR_RATIO to ~0.05 as well."""

IWM_HEDGE_RATIO_BULL = 0.15 #0.3
"""float: Fraction of the long book's beta-neutral IWM weight carried as a
hedge in the bullish regime (VIX signal <= 0) with SPY at/above its 80-day MA.

Purpose: this is the steady-state cost of the hedge in good markets, and
therefore the single biggest lever on total return. 0.30 means the book
keeps ~70% of its net long exposure in confirmed uptrends.

Raise it  -> smoother equity curve, smaller drawdowns, lower CAGR.
Lower it  -> more upside capture, but thinner protection between the weekly
             rebalance and the VIX/MA80 regime triggers firing.
Note: in bull regimes this ratio is further scaled by shortfact/0.9, so the
seasonal short calendar (SHORT_RESTRICTED_MONTHS + bc1) halves it in
restricted months exactly as the legacy code did.
Suggested backtest sweep: 0.25 - 0.40."""

IWM_HEDGE_RATIO_BULL_WEAK = 0.5
"""float: Hedge ratio in the bullish regime when SPY is below its 80-day MA
(uptrend intact but tape deteriorating).

Purpose: beta-aware replacement for the legacy
min(-0.384 * total_long, total_ws) hedge floor used in that state. With a
typical long-book IWM beta near 0.95, 0.50 of beta-neutral lands close to
the old 0.384-of-gross-long floor while now adapting to what the book
actually holds. RISKBRAKE-5 uses this same ratio for the intraday
below-MA80 escalation, so intraday and weekly protection depths agree.

Raise it  -> faster de-risking when SPY slips under MA80 (better in choppy
             corrections, costs return on quick V-shaped recoveries).
Lower it  -> stays closer to the bull setting; relies more on the VIX flag
             to catch real downturns.
Suggested backtest sweep: 0.40 - 0.60. Keep > IWM_HEDGE_RATIO_BULL."""

IWM_HEDGE_RATIO_BEAR = 0.50
"""float: Hedge ratio in the bearish (VIX signal > 0) regime while the long
book is still invested (SPY above MA80, longfact carried over from iwm_w).

Purpose: protection-first sizing once the VIX model turns negative. At 0.70
the residual net long exposure is ~30% of the book's IWM beta, so a market
leg down is mostly absorbed while some long alpha is retained.

Raise it  -> closer to market-neutral in bear regimes (1.0 = fully
             beta-neutral); safest, but gives up rebound participation.
Lower it  -> more rebound participation, more bear-market drawdown.
Suggested backtest sweep: 0.55 - 0.85. Keep > IWM_HEDGE_RATIO_BULL_WEAK."""

IWM_HEDGE_FLOOR_RATIO = 0.12
"""float: Minimum hedge as a fraction of the beta-neutral weight, enforced
whenever the long book is invested (total_wl > 0.2).

Purpose: the hedge-role guarantee. No combination of bullish regime,
seasonal shortfact, and tactical tilts can shrink the IWM short below this
level, so overnight gaps and signal failures always meet some protection.
This floor also seeds context.iwm_hedge_floor, which the intraday
de-risking events in initial_allocation refuse to cross (and which
RISKBRAKE-5 uses to recover the beta-neutral anchor intraday).

Raise it  -> stronger always-on protection, more drag in long bull runs.
Lower it  -> cheaper hedge, but approaches "effectively unhedged" in the
             most bullish states; 0 would disable the guarantee entirely.
Suggested range: 0.08 - 0.20. This is a safety rail -- tune the RATIO_*
constants for return, not this."""

IWM_HEDGE_CAP_RATIO = 1.05
"""float: Maximum hedge as a fraction of the beta-neutral weight, enforced
whenever the long book is invested.

Purpose: stops stacked multipliers (regime ratio x tilt x legacy blend)
from compounding into a short larger than the exposure being hedged --
beyond beta-neutral the "hedge" becomes a net-short directional bet, which
is not its role. 1.05 allows a 5% overshoot for the tactical tilts.

Raise it  -> permits deliberate net-short positioning in weak tapes (changes
             the character of the strategy; not recommended).
Lower it  -> hard ceiling closer to/below beta-neutral.
Suggested range: 1.00 - 1.15. Safety rail; rarely needs tuning."""

IWM_HEDGE_BETA_BLEND = 0.65
"""float: Weight of the beta-targeted component in the final hedge blend.
The remaining (1 - blend) keeps the legacy alpha-based short signal
(total_ws: the aggregated weakness of the 50 short candidates, with all of
its regime multipliers) as a return-seeking tilt.

Purpose: controls how much the hedge is structural (sized to measured
exposure) versus signal-driven (sized to how weak the shortable universe
looks). The blend also makes the transition from the legacy algo gradual
and directly A/B-testable: 1.0 = pure beta-targeting, 0.0 = legacy sizing
(minus the double-shortfact bug).

Raise it  -> more stable, exposure-true hedge; less week-to-week turnover.
Lower it  -> hedge swings more with the alpha signal's multiplier chain.
Suggested backtest sweep: 0.50 - 0.80."""

IWM_BEAR_MAX_SHORT = 0.60
"""float: Cap on the absolute IWM short when the long book is (near) empty
(bearish regime below MA80, total_wl <= 0.2).

Purpose: in that state the IWM short is the portfolio's directional
position, not a hedge, so the beta-based bounds above do not apply. The
legacy directional signal is used at full strength and this constant is the
only limit on it. 0.60 sits just above the historical bear-regime sizing
(~0.47-0.55) so behavior is unchanged unless the signal chain misfires.

Raise it  -> bigger bear-market bet, bigger squeeze risk on reversals.
Lower it  -> caps bear-market profit potential.
Suggested range: 0.50 - 0.75."""

# -----------------------------------------------------------------------------
# Bugfix Parameters (BUGFIX-1 / BUGFIX-3, merged Jun-2026 after A/B test)
# -----------------------------------------------------------------------------
# Four code-review fixes, validated in a side-by-side backtest before being
# merged here. Grep "BUGFIX" for every fix site; fixes 1 and 3 print a
# "[BUGFIX-n]" line whenever they actually alter a value, so the log always
# shows when a guard fires in live trading or backtests.

MLF1_WEIGHT_FLOOR = 1e-4
"""float: BUGFIX-1 -- minimum value of the mlf1 weight multiplier in
select_long_portfolio. The legacy code multiplied cash_return_zsoft by an
mlf1 factor that could be NEGATIVE; the abs() in get_normalized_weights then
turned the most ML-bearish stock into a LARGE long weight (sign inversion).
Flooring the multiplier at a tiny positive value means a negative ML
prediction yields a near-zero weight instead -- the intended monotonic
'scale weight by ML strength' behavior. Must be well below typical positive
mlf1**1.8 values (~1e-3 to 1e-1) so it never distorts normal sizing."""

BETA_RATIO_CAP = 2.5
"""float: BUGFIX-3 -- upper bound on beta_ratio (long_beta / short_beta).
The legacy max(1.3, ...) had no ceiling and no NaN guard: a short basket
with near-zero beta makes the ratio explode (inflating every short weight),
and a NaN propagates max(1.3, nan) = nan straight into the orders. The ratio
is now clipped to [1.3, BETA_RATIO_CAP] and any non-finite value falls back
to the 1.3 floor. Typical observed values run ~1.3-2.0, so 2.5 only clips
genuine degeneracies."""

# -----------------------------------------------------------------------------
# MLF1 Health Guard (RISKBRAKE-3: acting brake when USE_MLF1_HEALTH_BRAKE)
# -----------------------------------------------------------------------------
# Selection (estrank), momentum rank (myrs), and long-weight scaling all key
# off the single external mlf1 column (refe-fundamentals-mlf1 database). A
# stale or broken collection would silently corrupt the whole book, so
# process_universe checks the column's health every day and prints loud
# "[MLF1-GUARD]" warnings to the console when it looks wrong. With
# USE_MLF1_HEALTH_BRAKE = True (RISKBRAKE-3) a tripped check ALSO scales
# gross long by MLF1_BRAKE_SCALE at the next rebalance; with it False the
# guard is logging-only as before. Grep the log for "[MLF1-GUARD]" after
# any suspicious run, and check the mlf1 data collection if warnings appear.

MLF1_MAX_NAN_FRACTION = 0.20
"""float: Warn when more than this fraction of the filtered universe has a
NaN mlf1 value. NaNs rank as missing, so a mostly-NaN column makes
estrank/myrs effectively random. Typical healthy collections are near 0%
NaN after the universe filters; 20% is well outside normal jitter while
still tolerating partial-coverage days."""

MLF1_MIN_CROSS_STD = 1e-6
"""float: Warn when the cross-sectional standard deviation of mlf1 falls
below this value (or is non-finite). A (near-)constant column -- e.g. a
collection that wrote zeros, or repeated the same value for every stock --
produces meaningless ranks while still looking "populated". Typical healthy
dispersion is orders of magnitude above this, so the threshold only fires
on genuine signal collapse."""

# -----------------------------------------------------------------------------
# Risk Management Parameters
# -----------------------------------------------------------------------------
DRAWDOWN_FACTOR_MULTIPLIER = 5.0
"""float: BUGFIX-2 (documentation corrected) / RISKBRAKE-4 (behavior now
optionally two-sided).

Multiplier applied to drawdown in:
    dd_factor = min(context.clip, 1 + draw_down * DRAWDOWN_FACTOR_MULTIPLIER)

Since draw_down >= 0, dd_factor >= 1 on this leg: it INCREASES long
exposure as the account draws down (a capped dip-buying / pressing
response), up to context.clip (1.2 in bullish regime, 1.6 in bearish).
Higher values reach the clip cap on smaller drawdowns (5.0 saturates the
1.2 cap at a 4% drawdown).

With USE_TWO_SIDED_DD = True (RISKBRAKE-4), this dip-buy leg applies only
up to DD_DERISK_THRESHOLD; beyond it dd_factor declines linearly to
DD_DERISK_FLOOR (see those constants). With it False, the legacy one-sided
press-into-drawdown behavior is preserved at every depth."""

SLIPPAGE_SPREAD = 0.05
"""float: Fixed slippage assumption in dollars per share.
Used for realistic backtest cost modeling."""

COMMISSION_COST = 0.01
"""float: Per-share commission cost in dollars.
Combined with MIN_TRADE_COST for total transaction costs."""

MIN_TRADE_COST = 1.00
"""float: Minimum commission per trade in dollars.
Ensures small trades still incur realistic costs."""

# -----------------------------------------------------------------------------
# Seasonal and Regime Parameters
# -----------------------------------------------------------------------------
GROWTH_SEASON_MONTHS = {4, 5, 6, 7, 8, 9, 10, 11, 12}
"""set: Months when growth factor receives higher weight in ranking.
During these months, eps_gr_mean gets 4x weight vs 1x in other months."""

SHORT_RESTRICTED_MONTHS = {1, 2, 3, 5, 7, 8, 9, 10, 11, 12}
"""set: Months with reduced short exposure (shortfact = 0.45 vs 0.9).
Accounts for historical seasonal patterns in short selling effectiveness."""

# -----------------------------------------------------------------------------
# Machine Learning Configuration
# -----------------------------------------------------------------------------
ML_GLOBAL_COUNTER = 0
"""int: Global counter for ML model reuse tracking.
Resets to 0 after ML_MODEL_REUSE_LIMIT iterations."""

ML_MODEL_REUSE_LIMIT = 1
"""int: Number of days to reuse fitted ML model before refitting.
Value of 1 means refit daily; higher values reduce computation."""

ML_CLASSIFIER_GLOBAL = 0
"""object: Global storage for fitted ML classifier.
Allows model persistence across multiple factor computations."""

# -----------------------------------------------------------------------------
# Symbol Cache
# -----------------------------------------------------------------------------
SYM_SID_CACHE_DICT = {}
"""dict: Cache for symbol-to-SID lookups.
Improves performance by avoiding repeated database queries."""


# =============================================================================
# DATABASE DEFINITIONS
# =============================================================================
# These classes define connections to external data sources used by the
# algorithm. Each database provides specific fundamental, sentiment, or
# signal data that feeds into the stock selection pipeline.

class CustomFundamentals(Database):
    """
    Primary fundamentals database containing core company financial data.

    This database provides fundamental metrics from the 'refe-fundamentals'
    data source, including valuation ratios, earnings data, cash flow metrics,
    and proprietary alpha model rankings.

    Attributes
    ----------
    CODE : str
        Database identifier: 'refe-fundamentals'
    LOOKBACK_WINDOW : int
        Number of trading days of historical data to load (240 ~ 1 year)

    Columns
    -------
    Symbol : object
        Ticker symbol for the security
    CompanyCommonName : object
        Full company name
    GICSSectorName : object
        GICS sector classification (e.g., 'Technology', 'Healthcare')
    RefPriceClose : float
        Reference closing price from fundamental data source
    RefVolume : float
        Reference trading volume
    EnterpriseValue_DailyTimeSeries_ : float
        Enterprise value (market cap + debt - cash)
    CompanyMarketCap : float
        Market capitalization
    FOCFExDividends_Discrete : float
        Free operating cash flow excluding dividends
    InterestExpense_NetofCapitalizedInterest : float
        Net interest expense
    EarningsPerShare_ActualSurprise : float
        EPS surprise vs consensus estimates (prev quarter)
    LongTermGrowth_Mean : float
        Mean analyst long-term growth estimate
    CombinedAlphaModelSectorRank : float
        Proprietary sector-relative alpha ranking
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_ : float
        EV/EBITDA valuation multiple
    ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_ : float
        Forward EV/OCF ratio
    """
    CODE = "refe-fundamentals"
    LOOKBACK_WINDOW = 240

    Symbol = Column(object)
    CompanyCommonName = Column(object)
    GICSSectorName = Column(object)
    RefPriceClose = Column(float)
    RefVolume = Column(float)
    EnterpriseValue_DailyTimeSeries_ = Column(float)
    CompanyMarketCap = Column(float)
    FOCFExDividends_Discrete = Column(float)
    InterestExpense_NetofCapitalizedInterest = Column(float)
    EarningsPerShare_ActualSurprise = Column(float)
    LongTermGrowth_Mean = Column(float)
    CombinedAlphaModelSectorRank = Column(float)
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_ = Column(float)
    ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_ = Column(float)


class CustomFundamentals2(Database):
    """
    Sentiment data database containing news and social sentiment indicators.

    CURRENTLY UNUSED (Jun-2026 dead-weight cleanup): the sentcomb/sentest
    pipeline columns that consumed this database are commented out in
    make_pipeline, so this DB is not queried. The class is kept so reviving
    those columns is a one-line change.

    This database provides sentiment metrics derived from news articles,
    social media, and other text sources. Used to gauge market sentiment
    toward individual securities.

    Attributes
    ----------
    CODE : str
        Database identifier: 'refe-fundamentals-sent'
    LOOKBACK_WINDOW : int
        Historical lookback period (200 days)

    Columns
    -------
    sent2pol : float
        Sentiment polarity score (-1 to +1, negative to positive)
    sent2sub : float
        Sentiment subjectivity score (0 to 1, objective to subjective)
    sentvad_neg : float
        Negative sentiment intensity from VAD (Valence-Arousal-Dominance) model
    """
    CODE = "refe-fundamentals-sent"
    LOOKBACK_WINDOW = 200

    sent2pol = Column(float)
    sent2sub = Column(float)
    sentvad_neg = Column(float)


class CustomFundamentals4(Database):
    """
    VIX prediction/signal database for market regime detection.

    This database contains a proprietary VIX-based signal used to determine
    market regime (bullish vs bearish). The signal drives major allocation
    decisions including long/short exposure levels.

    RISKBRAKE-1: the daily lookup of this signal is restorable via
    USE_VIXDATA_REGIME (default False = legacy hardcode of 0). Verify the
    collection is live and sane before enabling.

    Attributes
    ----------
    CODE : str
        Database identifier: 'vixdata'
    LOOKBACK_WINDOW : int
        Historical lookback period (200 days)

    Columns
    -------
    pred : float
        VIX regime prediction signal
        - pred <= 0: Bullish regime (increase long exposure)
        - pred > 0: Bearish regime (reduce long exposure, increase hedging)
    """
    CODE = "vixdata"
    LOOKBACK_WINDOW = 200
    pred = Column(float)


class CustomFundamentals9(Database):
    """
    Barchart trend data for additional market confirmation signals.

    This database provides trend-following signals from Barchart's
    technical analysis service, used as confirmation for position sizing.

    Attributes
    ----------
    CODE : str
        Database identifier: 'bcdata'
    LOOKBACK_WINDOW : int
        Historical lookback period (200 days)

    Columns
    -------
    bc1 : float
        Barchart trend signal
        - bc1 == 1: Trend confirmed (may reduce exposure if SPY < MA150)
        - bc1 != 1: No trend confirmation
    """
    CODE = "bcdata"
    LOOKBACK_WINDOW = 200
    bc1 = Column(float)

class CustomFundamentals10(Database):
    """ MLF1 data database """

    CODE = "refe-fundamentals-mlf1"
    LOOKBACK_WINDOW = 200

    predicted_return = Column(float)


# =============================================================================
# CUSTOM FACTORS
# =============================================================================
# These classes define computed factors used in the stock selection pipeline.
# Each factor transforms raw price/volume/fundamental data into signals
# that contribute to the final ranking and selection of securities.

class Above200DMA(CustomFactor):
    """
    Binary indicator for price position relative to 200-day moving average.

    This factor identifies stocks trading above their long-term trend,
    which is often associated with positive momentum and relative strength.

    Calculation
    -----------
    1 if current_price > 200-day SMA, else 0

    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    window_length : int
        200 trading days

    Returns
    -------
    float
        Binary indicator: 1.0 (above MA) or 0.0 (below MA)

    Usage
    -----
    Used as a quality filter and for identifying stocks in uptrends.
    """
    inputs = [USEquityPricing.close]
    window_length = 200

    def compute(self, today, assets, out, close):
        latest_close = close[-1]
        ma_200 = np.mean(close, axis=0)
        out[:] = (latest_close > ma_200).astype(int)


class StochasticOscillatorWeekly(CustomFactor):
    """
    20-week stochastic oscillator computed from daily data.

    The stochastic oscillator measures where the current price sits within
    the recent trading range. Weekly timeframe smooths out daily noise
    while capturing intermediate-term momentum.

    Calculation
    -----------
    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100

    Where Highest High and Lowest Low are computed over 20 weekly periods.

    Parameters
    ----------
    inputs : list
        [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length : int
        100 trading days (20 weeks * 5 days/week)

    Returns
    -------
    float
        Oscillator value between 0 and 100
        - > 80: Potentially overbought
        - < 20: Potentially oversold

    Notes
    -----
    Daily data is aggregated into weekly bars before calculation to match
    the intended weekly timeframe of the indicator.
    """
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 20 * 5  # 20 weeks of daily data

    def compute(self, today, assets, out, high, low, close):
        # Reshape daily data into weekly bars
        high_weekly = high.reshape(-1, 5, high.shape[1]).max(axis=1)
        low_weekly = low.reshape(-1, 5, low.shape[1]).min(axis=1)
        close_weekly = close.reshape(-1, 5, close.shape[1])[:, -1]

        # Calculate stochastic
        highest_high = np.max(high_weekly, axis=0)
        lowest_low = np.min(low_weekly, axis=0)
        out[:] = ((close_weekly[-1] - lowest_low) / (highest_high - lowest_low)) * 100


class Slope(CustomFactor):
    """
    Linear regression slope of price data over the window period.

    Measures the trend direction and strength by fitting a linear regression
    to closing prices. Positive slope indicates uptrend, negative indicates
    downtrend. Magnitude indicates trend strength.

    Calculation
    -----------
    Fits OLS regression: price = alpha + beta * time
    Returns beta (slope coefficient)

    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    outputs : list
        ['slope', 'rsq'] - slope coefficient and R-squared (unused)
    window_length : int
        Configurable (commonly 30, 90, 120, 220 days)

    Returns
    -------
    slope : float
        Regression slope coefficient (price change per day)
    rsq : float
        R-squared of regression fit (currently set to 0, unused)

    Notes
    -----
    NaN values in price data are interpolated before regression to ensure
    stable computation. The slope is typically z-scored across the universe
    for cross-sectional comparison.
    """
    inputs = [USEquityPricing.close]
    outputs = ['slope', 'rsq']

    def compute(self, today, assets, out, closes):
        # Handle NaN values through interpolation
        try:
            mask = np.isnan(closes)
            closes[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), closes[~mask])
        except:
            pass

        # Fit linear regression
        lr = sm.OLS(closes, sm.add_constant(range(-len(closes) + 1, 1))).fit()
        out.slope[:] = lr.params[-1]
        out.rsq[:] = 0  # R-squared not currently used


class RelativeStrength(CustomFactor):
    """
    Relative strength of a security versus a benchmark index.

    Measures how much a stock has outperformed or underperformed a benchmark
    over the lookback period. Used to identify momentum leaders and laggards.

    Calculation
    -----------
    RS = ((1 + stock_return) / (1 + benchmark_return) - 1) * 100

    Where returns are calculated over the window period (close[-22] to close[0]).

    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    params : tuple
        (market_sid,) - SID of the benchmark security (e.g., QQQ)
    window_length : int
        Configurable (commonly 140, 160, 180 days)
    window_safe : bool
        True - factor can be used as input to other factors

    Returns
    -------
    float
        Relative strength percentage
        - Positive: Outperformed benchmark
        - Negative: Underperformed benchmark

    Example
    -------
    RS of 10 means the stock returned 10% more than the benchmark.
    RS of -5 means the stock returned 5% less than the benchmark.
    """
    params = ('market_sid',)
    inputs = [USEquityPricing.close]
    window_safe = True

    def compute(self, today, assets, out, close, market_sid):
        rsRankTable = pd.DataFrame(index=assets)

        # Calculate returns over approximately 1 month (22 trading days)
        returns = (close[-22] - close[0]) / close[0]

        # Find benchmark and compute relative performance
        market_idx = assets.get_loc(market_sid)
        rsRankTable["RS"] = (((returns + 1) / (returns[market_idx] + 1)) - 1) * 100

        out[:] = rsRankTable["RS"]


class Volatility(CustomFactor):
    """
    Historical volatility measured as standard deviation of daily returns.

    Provides a measure of price variability used for risk assessment
    and position sizing adjustments.

    Calculation
    -----------
    volatility = std(daily_returns) over window
    where daily_return = (close[t] - close[t-1]) / close[t-1]

    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    window_length : int
        Configurable (commonly 10, 20, 60 days)
    window_safe : bool
        True - factor can be used as input to other factors

    Returns
    -------
    float
        Standard deviation of daily returns (not annualized)

    Notes
    -----
    To annualize, multiply by sqrt(252). A value of 0.02 daily volatility
    corresponds to approximately 32% annualized volatility.
    """
    inputs = [USEquityPricing.close]
    window_safe = True

    def compute(self, today, assets, out, close_prices):
        daily_returns = np.diff(close_prices, axis=0) / close_prices[:-1]
        volatility = np.std(daily_returns, axis=0)
        out[:] = volatility


class PublicSince(CustomFactor):
    """
    Proxy for how long a security has been publicly traded.

    Uses the sum of early price data points as a heuristic for identifying
    established vs. newly listed securities. Securities with longer trading
    histories will have more non-zero early prices.

    Calculation
    -----------
    Sum of first 8 closing prices in the window

    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    window_length : int
        Configurable (commonly 121 days)

    Returns
    -------
    float
        Sum of early prices (higher = more established)

    Notes
    -----
    This is a simple heuristic. Newly listed stocks will have NaN or zero
    values for early dates, resulting in lower sums. Used to filter out
    very new listings that may have unreliable data.
    """
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, prices):
        prices = np.nan_to_num(prices)
        out[:] = ((prices[0] + prices[1] + prices[2] + prices[3] +
                   prices[4] + prices[5] + prices[6] + prices[7]))


class SumFactor(CustomFactor):
    """
    Sums a factor's values over the window period.

    Generic utility factor for accumulating time-series data, commonly
    used with sentiment indicators to aggregate signals over time.

    Parameters
    ----------
    inputs : list
        Single factor to sum (e.g., sentiment score)
    window_length : int
        Number of days to sum over
    window_safe : bool
        True - factor can be used as input to other factors

    Returns
    -------
    float
        Sum of factor values over the window

    Example
    -------
    SumFactor(sentiment_score, window_length=18) gives the cumulative
    sentiment over the past 18 days.
    """
    window_safe = True

    def compute(self, today, assets, out, factordata):
        out[:] = np.sum(factordata, axis=0)


class SumVolume(CustomFactor):
    """
    Sums trading volume over the window period.

    Used to identify recent trading activity levels, which can indicate
    liquidity and investor interest.

    Parameters
    ----------
    inputs : list
        [USEquityPricing.volume]
    window_length : int
        Number of days to sum (commonly 3-5 days)

    Returns
    -------
    float
        Total shares traded over the window
    """
    inputs = [USEquityPricing.volume]

    def compute(self, today, assets, out, volume):
        out[:] = np.sum(volume, axis=0)


class WeightedAlpha(CustomFactor):
    """
    Weighted excess return (alpha) versus SPY benchmark.

    Combines short, medium, and long-term alpha into a single score,
    with heavier weight on medium-term performance. This balances
    recent momentum with longer-term trend strength.

    Calculation
    -----------
    weighted_alpha = 0.15 * alpha_30 + 0.50 * alpha_90 + 0.35 * alpha_252

    Where alpha_N = stock_return_N - SPY_return_N

    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    window_length : int
        252 trading days (1 year)

    Returns
    -------
    float
        Weighted alpha score
        - Positive: Outperformed SPY on weighted basis
        - Negative: Underperformed SPY on weighted basis

    Notes
    -----
    The 50% weight on 90-day alpha emphasizes quarterly momentum,
    which often captures fundamental catalyst reactions while avoiding
    short-term noise and long-term mean reversion.
    """
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):
        # Get SPY index
        spy_sid = symbol("SPY").sid
        spy_idx = assets.get_loc(spy_sid)
        spy_close = close[:, spy_idx]

        # Separate SPY from other assets
        is_asset = np.arange(len(assets)) != spy_idx
        asset_close = close[:, is_asset]

        # Calculate asset returns at different horizons
        ret_30 = asset_close[-1] / asset_close[-22] - 1   # ~1 month
        ret_90 = asset_close[-1] / asset_close[-66] - 1   # ~3 months
        ret_252 = asset_close[-1] / asset_close[0] - 1    # ~1 year

        # Calculate SPY returns
        spy_ret_30 = spy_close[-1] / spy_close[-22] - 1
        spy_ret_90 = spy_close[-1] / spy_close[-66] - 1
        spy_ret_252 = spy_close[-1] / spy_close[0] - 1

        # Calculate alpha (excess return) at each horizon
        alpha_30 = ret_30 - spy_ret_30
        alpha_90 = ret_90 - spy_ret_90
        alpha_252 = ret_252 - spy_ret_252

        # Weighted combination
        weighted_alpha = 0.15 * alpha_30 + 0.5 * alpha_90 + 0.35 * alpha_252

        # Map back to full asset list
        out_vals = np.empty(len(assets))
        out_vals[:] = np.nan
        out_vals[is_asset] = weighted_alpha
        out[:] = out_vals


class MLFactorC(CustomFactor):
    """
    Machine learning factor predicting future returns using Linear Regression.

    This factor trains a linear regression model on fundamental features
    to predict forward returns. The model is periodically retrained to
    adapt to changing market conditions.

    Model Architecture
    ------------------
    - Algorithm: Linear Regression (sklearn)
    - Features: Fundamental ratios and rankings
    - Target: Forward returns over shift_target days
    - Preprocessing: Imputation (constant=0) + Robust Scaling

    Parameters
    ----------
    inputs : list
        [Returns, Feature1, Feature2, ...] - First input is target returns
    params : tuple
        (shift_target,) - Number of days forward to predict
    window_length : int
        Training window size (commonly 180 days)
    window_safe : bool
        True - factor can be used as input to other factors

    Returns
    -------
    float
        Predicted return score (higher = more bullish prediction)

    Training Process
    ----------------
    1. Stack features into 3D array (time x stocks x features)
    2. Create forward return labels shifted by shift_target days
    3. Flatten and clean training data (remove NaN/Inf)
    4. Apply imputation and scaling
    5. Fit linear regression
    6. Predict on most recent data

    Notes
    -----
    The model is reused for ML_MODEL_REUSE_LIMIT days before refitting
    to reduce computational overhead while maintaining adaptability.
    """
    params = ('shift_target',)
    window_safe = True

    def compute(self, today, assets, out, target, *features, shift_target):
        global ML_GLOBAL_COUNTER, ML_MODEL_REUSE_LIMIT, ML_CLASSIFIER_GLOBAL

        # Initialize persistent state on first call
        if not hasattr(self, "fitted"):
            self.fitted = False
        if not hasattr(self, "reuse_counter"):
            self.reuse_counter = 0
        if not hasattr(self, "reuse_limit"):
            self.reuse_limit = 1
        if not hasattr(self, "imputer"):
            self.imputer = impute.SimpleImputer(strategy="constant", fill_value=0)
        if not hasattr(self, "scaler"):
            self.scaler = preprocessing.RobustScaler()
        if not hasattr(self, "model"):
            self.model = linear_model.LinearRegression()

        # Prepare data
        X = np.dstack(features)  # Shape: (time, stocks, features)
        Y = target                # Shape: (time, stocks)
        n_time, n_stocks, n_factors = X.shape

        # Guard: insufficient data for forward prediction
        if shift_target >= n_time:
            out[:] = 0
            return

        # Create training data with forward-shifted labels
        X_train = X[:-shift_target].reshape(-1, n_factors)
        Y_train = Y[shift_target:].reshape(-1)

        # Clean training data
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train)
        X_train = X_train[mask]
        Y_train = Y_train[mask]

        # Guard: insufficient clean samples
        if len(X_train) < 5:
            out[:] = 0
            return

        # Fit model if needed (based on reuse counter)
        if not self.fitted or self.reuse_counter == 0:
            Xt = self.imputer.fit_transform(X_train)
            Xt = self.scaler.fit_transform(Xt)
            self.model.fit(Xt, Y_train)
            self.fitted = True
            self.reuse_counter = self.reuse_limit

        # Generate predictions for current day
        X_test = X[-1]  # Most recent data
        X_test = self.imputer.transform(X_test)
        X_test = self.scaler.transform(X_test)
        preds = self.model.predict(X_test)

        out[:] = preds
        self.reuse_counter -= 1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# Helper functions for symbol lookup, technical indicators, and data
# transformation. These provide reusable functionality across the algorithm.

def symbol(sym):
    """
    Get security ID (SID) for a ticker symbol with caching.

    Performs a lookup in the QuantRocket securities master database and
    caches results to avoid repeated database queries.

    Parameters
    ----------
    sym : str
        Ticker symbol (e.g., 'SPY', 'AAPL')

    Returns
    -------
    zipline.assets.Equity or None
        Security object if found, None if lookup fails

    Notes
    -----
    Results are cached in SYM_SID_CACHE_DICT for performance.
    First lookup for a symbol queries the database; subsequent
    lookups return cached value.
    """
    global SYM_SID_CACHE_DICT
    if SYM_SID_CACHE_DICT.get(sym) is None:
        try:
            securities = get_securities(vendors="usstock", fields=["Sid", "Symbol"])
            sid_val = algo.sid(sid=securities[securities.Symbol == sym].index.values[0])
            SYM_SID_CACHE_DICT.update({sym: sid_val})
        except Exception as e:
            print(f"Error getting symbol {sym}: {e}")
            sid_val = None
    else:
        sid_val = SYM_SID_CACHE_DICT[sym]
    return sid_val


def symbols(syms):
    """
    Get security IDs for multiple ticker symbols.

    Parameters
    ----------
    syms : list of str
        List of ticker symbols

    Returns
    -------
    list
        List of SID values (may contain None for failed lookups)
    """
    securities = get_securities(vendors="usstock", fields=["Sid", "Symbol"])
    securities = securities.reset_index()
    sidlist = []
    for sym in syms:
        sidlist.append(securities[securities.Symbol == sym].Sid)
    return sidlist


ALGO_LOGGER = None
"""Module-level cache for the Flightlog logger (BUGFIX-4)."""


def get_algo_logger():
    """
    Return the shared 'LS-Prod-Algo' Flightlog logger.

    BUGFIX-4: the legacy code created a new FlightlogHandler and attached it
    to the same named logger on EVERY call of initial_allocation (daily) and
    place_short_orders (weekly). Handlers accumulate on the logger object,
    so each log line was emitted once per attached handler -- duplicated log
    output and slow memory growth over a long live session. This helper
    attaches exactly one handler for the algorithm's lifetime. No effect on
    trading behavior.

    Returns
    -------
    logging.Logger
        Configured logger with a single FlightlogHandler attached.
    """
    global ALGO_LOGGER
    if ALGO_LOGGER is None:
        logger = logging.getLogger('LS-Prod-Algo')
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            logger.addHandler(FlightlogHandler())
        ALGO_LOGGER = logger
    return ALGO_LOGGER


def weighted_moving_average(prices, period):
    """
    Calculate Weighted Moving Average (WMA).

    WMA gives more weight to recent prices, with weights increasing
    linearly from 1 to period.

    Parameters
    ----------
    prices : array-like
        Price series
    period : int
        Lookback period

    Returns
    -------
    numpy.ndarray
        WMA values (length = len(prices) - period + 1)

    Formula
    -------
    WMA = sum(price[i] * weight[i]) / sum(weights)
    where weight[i] = i + 1 for i in range(period)
    """
    weights = np.arange(1, period + 1)
    wma = np.convolve(prices, weights / weights.sum(), mode='valid')
    return wma


def hull_moving_average(prices, period):
    """
    Calculate Hull Moving Average (HMA).

    HMA reduces lag while maintaining smoothness by using weighted
    moving averages of different periods combined in a specific way.

    Parameters
    ----------
    prices : array-like
        Price series
    period : int
        Base lookback period

    Returns
    -------
    numpy.ndarray
        HMA values

    Formula
    -------
    HMA = WMA(2 * WMA(price, period/2) - WMA(price, period), sqrt(period))

    Notes
    -----
    Developed by Alan Hull. More responsive than SMA/EMA while
    filtering out more noise. Good for trend identification.
    """
    wma_n = weighted_moving_average(prices, period)
    wma_half_n = weighted_moving_average(prices, period // 2)
    raw_hma = 2 * wma_half_n[-len(wma_n):] - wma_n
    sqrt_n = int(np.sqrt(period))
    hma = weighted_moving_average(raw_hma, sqrt_n)
    return hma


def hull_ma_trend(prices, period, lookback=3):
    """
    Determine trend direction based on Hull Moving Average.

    Parameters
    ----------
    prices : array-like
        Price series
    period : int
        HMA period
    lookback : int, optional
        Number of HMA values to compare (default: 3)

    Returns
    -------
    str
        'positive' if HMA is rising, 'negative' if falling
    """
    hma = hull_moving_average(prices, period)
    recent_hma = hma[-lookback:]
    trend = "positive" if recent_hma[-1] > recent_hma[0] else "negative"
    return trend


def compute_weekly_stochastic(df, lookback_weeks=14):
    """
    Compute weekly stochastic oscillator from daily OHLC data.

    Resamples daily data to weekly frequency before calculating
    the stochastic oscillator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'high', 'low', 'price' columns and DatetimeIndex
    lookback_weeks : int, optional
        Number of weeks for stochastic calculation (default: 14)

    Returns
    -------
    float
        Stochastic oscillator value (0-100)

    Notes
    -----
    Weekly bars are created ending on Fridays. The stochastic
    measures where the current close is within the N-week range.
    """
    # Resample to weekly frequency
    weekly_high = df['high'].resample('W-FRI').max()
    weekly_low = df['low'].resample('W-FRI').min()
    weekly_close = df['price'].resample('W-FRI').last()

    # Calculate rolling high/low
    highest_high = weekly_high.rolling(lookback_weeks).max()
    lowest_low = weekly_low.rolling(lookback_weeks).min()

    # Compute stochastic
    stochastic = ((weekly_close.iloc[-1] - lowest_low.iloc[-1]) /
                  (highest_high.iloc[-1] - lowest_low.iloc[-1])) * 100
    return stochastic


# =============================================================================
# MAIN ALGORITHM FUNCTIONS
# =============================================================================
# Core algorithm lifecycle functions called by the Zipline framework.
# These implement the trading logic from initialization through execution.

def initialize(context):
    """
    Initialize the trading algorithm.

    Called once at the start of the algorithm. Sets up the pipeline,
    benchmark, trading costs, scheduled functions, and initializes
    all context variables used throughout the algorithm.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context object for storing state

    Context Variables Initialized
    -----------------------------
    Portfolio State:
        longfact, shortfact : float
            Exposure multipliers for long/short positions
        max_liquid : float
            High-water mark for drawdown calculation
        dd_factor : float
            Current drawdown adjustment factor
        draw_down : float
            Current drawdown from high-water mark

    Trading Control:
        days_offset : int
            Day of week for weekly rebalancing (1 = Tuesday)
        initialized : int
            Flag indicating if first allocation completed
        verbose : int
            Logging verbosity level

    Market State:
        vixflag, vixflag_prev : float
            Current and previous VIX regime signals (RISKBRAKE-1: prev is
            now genuinely updated each day in before_trading_start)
        spy_below80ma : bool
            Flag for SPY below 80-day moving average
        iwm_w : float
            Current hedge weight actually held (total across hedge
            instruments when QQQ_HEDGE_FRACTION > 0)
        iwm_hedge_floor : float
            Most-recent hedge floor from compute_iwm_hedge_weight. Intraday
            de-risking events may reduce the hedge but never above this
            level; RISKBRAKE-5 also derives its beta-neutral anchor from it.
        mlf1_health_scale, risk_brake_scale : float
            RISKBRAKE-2/3 state (1.0 = no brake active)

    Scheduled Functions
    -------------------
    - initial_allocation: Daily, handles VIX signal changes
    - regular_allocation: Weekly (Tuesday), main rebalancing
    - exit_positions: Weekly (Tuesday), close unwanted positions
    """
    # Attach stock selection pipeline
    algo.attach_pipeline(make_pipeline(), 'my_pipeline')

    # Set benchmark for performance comparison
    algo.set_benchmark(algo.sid(symbol('SPY').real_sid))

    # Initialize portfolio state variables
    context.longfact = 1.0           # Long exposure multiplier
    context.shortfact = 1.0          # Short exposure multiplier
    context.order_id = {}            # Track pending orders
    context.topMom = 9               # Number of top momentum ETFs to track
    context.max_liquid = context.portfolio.starting_cash  # High-water mark
    context.cash_adjustment = 0      # External cash flow adjustment
    context.dd_factor = 1.0          # Drawdown adjustment factor
    context.draw_down = 0.0          # Current drawdown
    context.print_set_delta = False  # Debug flag
    context.days_offset = 1          # Rebalance day (1 = Tuesday)
    context.initialized = 0          # First allocation flag
    context.sids_initialized = 0     # SID lookup flag
    context.verbose = 1              # Logging verbosity
    context.qqq_ratio_prev = 0       # Previous QQQ allocation
    context.spy_ratio_prev = 0       # Previous SPY allocation
    context.total_ws = 0             # Total short weight (legacy alpha signal)
    context.iwm_w = 0                # Hedge weight actually held
    # Minimum IWM short the intraday de-risking events may reduce to before
    # the first weekly rebalance computes a beta-based floor. Conservative
    # default; overwritten by compute_iwm_hedge_weight on every rebalance.
    context.iwm_hedge_floor = -0.20
    context.spy_below80ma = False    # SPY below MA flag
    context.vixflag = 0              # Current VIX signal
    context.vixflag_prev = 0         # Previous VIX signal
    context.clip = 1.0               # Max DD factor clip
    # RISKBRAKE state (1.0 = inactive); refreshed daily / per rebalance
    context.mlf1_health_scale = 1.0  # RISKBRAKE-3 (set in process_universe)
    context.risk_brake_scale = 1.0   # RISKBRAKE-2/3 combined (set at rebalance)

    # Set realistic trading costs
    algo.set_slippage(algo.slippage.FixedSlippage(spread=SLIPPAGE_SPREAD))
    algo.set_commission(algo.commission.PerShare(cost=COMMISSION_COST, min_trade_cost=MIN_TRADE_COST))

    # Schedule trading functions
    algo.schedule_function(initial_allocation, date_rule=algo.date_rules.every_day())
    algo.schedule_function(regular_allocation, date_rule=algo.date_rules.week_start(days_offset=context.days_offset))
    algo.schedule_function(exit_positions, date_rule=algo.date_rules.week_start(days_offset=context.days_offset))


def make_pipeline():
    """
    Create the stock selection pipeline.

    Constructs a Zipline Pipeline that computes all factors needed for
    stock selection. The pipeline filters the universe, calculates
    fundamental and technical factors, and prepares data for ranking.

    Returns
    -------
    zipline.pipeline.Pipeline
        Configured pipeline with screen and factor columns

    Pipeline Structure
    ------------------
    Screen:
        Top 1500 stocks by market cap OR IBM (for VIX signal lookup)

    Factor Categories (active):
        - Identification: symbol, company name, sector
        - Valuation: market cap, enterprise value
        - Price/Volume: close, volume, moving averages
        - Fundamentals: FCF, interest expense, EPS surprise, growth
        - Risk: beta to SPY and IWM
        - Technical: slope90 (slope120/220/30, stochastic, 200DMA commented out)
        - Momentum: RS140_QQQ (RS160/180 and Ret60/120/220 commented out)
        - Quality: public trading history, volatility
        - Signals: VIX flag, Barchart trend
        - ML: mlf1 external return prediction (drives estrank/myrs/weighting)

    Disabled Categories (Jun-2026 dead-weight cleanup; uncomment to revive):
        - Sentiment: sentcomb / sentest (refe-fundamentals-sent DB unused)
        - MLfactor: in-file MLFactorC linear model (long since disabled)
    """
    # Define tradable universe
    tradable_filter = (
        CustomFundamentals.CompanyMarketCap.latest.shift().top(UNIVERSE_SIZE) |
        StaticAssets([symbol('IBM')])  # IBM used as proxy for signal lookup
    )

    # Get Sharadar fundamentals for FCF
    s_fundamentals = sharadar.Fundamentals.slice('ARQ', period_offset=0)

    pipe = Pipeline(
        screen=tradable_filter,
        columns={
            # === Identification ===
            'name': CustomFundamentals.Symbol.latest,
            'compname': CustomFundamentals.CompanyCommonName.latest,
            'sector': CustomFundamentals.GICSSectorName.latest,

            # === Valuation Metrics ===
            'market_cap': CustomFundamentals.CompanyMarketCap.latest,
            'entval': CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest,

            # === Price and Volume ===
            'price': USEquityPricing.close.latest,
            'volume': USEquityPricing.volume.latest,
            'fs_price': CustomFundamentals.RefPriceClose.latest,
            'fs_volume': CustomFundamentals.RefVolume.latest,
            'sumvolume': SumVolume(window_length=3),
            'smav': SimpleMovingAverage(inputs=[USEquityPricing.volume], window_length=10),

            # === Fundamental Metrics ===
            'eps_ActualSurprise_prev_Q_percent': CustomFundamentals.EarningsPerShare_ActualSurprise.latest.shift(),
            'eps_gr_mean': CustomFundamentals.LongTermGrowth_Mean.latest.shift(),
            'fcf_sharadar': s_fundamentals.FCF.latest,
            'fcf' :CustomFundamentals.FOCFExDividends_Discrete.latest,
            'int': CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest.shift(),

            # === Risk Metrics ===
            'beta60SPY': SimpleBeta(target=symbol('SPY'), regression_length=60).shift(),
            'beta60IWM': SimpleBeta(target=symbol('IWM'), regression_length=60).shift(),

            # === Technical Indicators ===
            # Dead-weight cleanup (Jun-2026): columns below that are computed
            # but never used downstream were commented out to cut pipeline
            # compute time (estrank/myrs both rank mlf1 now). Uncomment any
            # line to revive that factor. slope90 stays active (used in the
            # process_universe diagnostics print); RS140_QQQ stays active
            # (referenced by the commented momentum ranking in myrs).
            #'slope120': Slope(window_length=120, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            #'slope220': Slope(window_length=220, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            'slope90': Slope(window_length=90, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            #'slope30': Slope(window_length=30, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            #'stk20w': StochasticOscillatorWeekly(),
            #'above_200dma': Above200DMA(mask=tradable_filter),
            #'walpha': WeightedAlpha(),

            # === Relative Strength ===
            'RS140_QQQ': RelativeStrength(window_length=140, market_sid=symbol('QQQ').sid).shift(),
            #'RS160_QQQ': RelativeStrength(window_length=160, market_sid=symbol('QQQ').sid).shift(),
            #'RS180_QQQ': RelativeStrength(window_length=180, market_sid=symbol('QQQ').sid).shift(),

            # === Return Metrics === (unused downstream; see cleanup note above)
            #'Ret60': Returns(window_length=60, mask=tradable_filter),
            #'Ret120': Returns(window_length=120, mask=tradable_filter),
            #'Ret220': Returns(window_length=220, mask=tradable_filter),

            # === Quality Filters ===
            'publicdays': PublicSince(window_length=121),
            'vol': Volatility(window_length=10, mask=tradable_filter),

            # === Regime Signals ===
            'vixflag': CustomFundamentals4.pred.latest.shift(),
            'vixflag0': CustomFundamentals4.pred.latest,
            'bc1': CustomFundamentals9.bc1.latest,

            # === Machine Learning Factor ===
            # 'MLfactor': MLFactorC(
            #     inputs=[
            #         Returns(window_length=90, mask=tradable_filter),
            #         CustomFundamentals.EnterpriseValueToEBITDA_DailyTimeSeriesRatio_,
            #         CustomFundamentals.LongTermGrowth_Mean,
            #         CustomFundamentals.CombinedAlphaModelSectorRank,
            #         CustomFundamentals.ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_,
            #     ],
            #     window_length=180,
            #     mask=tradable_filter,
            #     shift_target=15
            # ).zscore(mask=tradable_filter).shift(),

            # === Sentiment Factors === (unused downstream; see cleanup note above)
            #'sentcomb': (
            #    SumFactor(CustomFundamentals2.sentvad_neg, window_length=18).zscore() +
            #    SumFactor(CustomFundamentals2.sent2sub, window_length=18).zscore() +
            #    (1 / SumFactor(CustomFundamentals2.sent2pol, window_length=18).zscore())
            #),
            #'sentest': 1 / SumFactor(CustomFundamentals2.sent2pol, window_length=18),

             'mlf1': CustomFundamentals10.predicted_return.latest,
        }
    )
    return pipe


def before_trading_start(context, data):
    """
    Daily preprocessing before market open.

    Called each trading day before market open. Retrieves pipeline output,
    updates market indicators, processes signals, and prepares the universe
    of securities for potential trading.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor

    Processing Steps
    ----------------
    1. Initialize security IDs (first run only)
    2. Get pipeline output
    3. Update market indicators (moving averages, etc.)
    4. Update VIX and Barchart signals (RISKBRAKE-1: prev captured first)
    5. Determine market trend/regime
    6. Process universe and select long/short candidates
    """
    # Initialize SIDs on first call
    initialize_sids(context, data, algo)

    # Get pipeline output
    df = algo.pipeline_output('my_pipeline')

    print(f"Raw stock universe size {df.shape}")
    print(algo.get_datetime(timezone("America/Los_Angeles")))

    # Update moving averages and other market indicators
    update_market_indicators(context, data)

    # Update VIX signal (store previous for change detection)
    # RISKBRAKE-1: capture yesterday's value BEFORE overwriting. The legacy
    # code initialized vixflag_prev once and never updated it, so the
    # intraday VIX-flip handler in initial_allocation could never fire even
    # with live vixdata. The lookup itself is restorable via
    # USE_VIXDATA_REGIME; False preserves the legacy hardcode of 0.
    context.vixflag_prev = context.vixflag
    if USE_VIXDATA_REGIME:
        context.vixflag = df.loc[context.ibm_sid].vixflag.copy()
    else:
        context.vixflag = 0  # vixdata regime disabled (legacy hardcode preserved)
    print('IBM-vixdata', context.vixflag)

    # Update Barchart trend signal
    context.bc1 = df.loc[context.ibm_sid].bc1.copy()



    # Determine market trend based on signals
    compute_trend(context, data)

    # Reset daily flags
    context.daily_flag = 0
    context.daily_print_flag = 0

    # Process universe and select positions
    df = process_universe(context, df, data)
    return


def process_universe(context, df, data):
    """
    Process the stock universe and select long/short candidates.

    Applies filters, calculates alpha scores, and selects the final
    securities for the long and short portfolios.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    df : pandas.DataFrame
        Pipeline output with all factor data
    data : zipline.protocol.BarData
        Market data accessor

    Returns
    -------
    pandas.DataFrame
        Filtered and ranked universe

    Processing Steps
    ----------------
    1. Filter by sector and exclusions
    2. Calculate cash return (z-scored softplus alpha)
    3. MLF1 health guard: warn on NaN-fraction / dispersion collapse /
       constant column; with USE_MLF1_HEALTH_BRAKE also arm the RISKBRAKE-3
       gross scaler for the next rebalance
    4. Calculate momentum ranking (myrs = mlf1 rank)
    5. Filter to top 500 by market cap
    6. Calculate final ranking (estrank = mlf1 rank)
    7. Select long portfolio (value + momentum)
    8. Select short portfolio (lowest cash return)
    9. Calculate portfolio beta ratio (BUGFIX-3: NaN guard + ceiling)
    """
    # Apply sector and stock exclusions from the universe of our stocks
    df = filter_symbols_sectors_universe(df)

    # Calculate dollar volume for liquidity filtering
    #df['doll_vol'] = df['price'] * df['smav']

    # Calculate cash return: (FCF - Interest) / Enterprise Value
    df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
    df['cash_return'] = df['cash_return'].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=['cash_return'], inplace=True)


    # "Z-scored softplus scaling with cross-sectional normalization" applied to cash_return as key alpha and later used in the weight model 
    # A convex, distribution-aware signal-to-weight transform
    cr = df['cash_return']
    z = (cr - cr.mean()) / (cr.std() + 1e-8)                # cross-sectional z-score
    k = 1.0                                                 # do NOT exceed 1.5 with this tail behavior
    alpha = np.log1p(np.exp(k * z))                         # softplus (convex) transform -- 
    alpha = np.minimum(alpha, np.percentile(alpha, 99.5))   # loose tail cap
    alpha /= alpha.mean()                                   # cross-sectional normalization

    df['cash_return_zsoft'] = alpha  # assign computed normalized cash_return back to the column as key alpha

    # Display top sectors by cash return
    sorted_df = df.groupby('sector')['cash_return_zsoft'].agg(['mean']).sort_values(by='mean', ascending=False)
    print(sorted_df.iloc[0].name)

    # MLF1 HEALTH GUARD: estrank, myrs, and the long weight scaling all
    # depend on this one external column. Warn loudly if it looks
    # stale/broken; with USE_MLF1_HEALTH_BRAKE (RISKBRAKE-3) also arm the
    # gross scaler. Never halts the rebalance. See the
    # MLF1_MAX_NAN_FRACTION / MLF1_MIN_CROSS_STD constants for thresholds.
    mlf1_nan_frac = float(df['mlf1'].isna().mean()) if len(df) else 1.0
    mlf1_std = float(df['mlf1'].std(skipna=True)) if len(df) else float('nan')
    mlf1_nunique = int(df['mlf1'].nunique(dropna=True))
    if mlf1_nan_frac > MLF1_MAX_NAN_FRACTION:
        print(f'[MLF1-GUARD] WARNING: {mlf1_nan_frac:.1%} of mlf1 values are NaN '
              f'(threshold {MLF1_MAX_NAN_FRACTION:.0%}) -- the refe-fundamentals-mlf1 '
              f'collection may be stale or broken; estrank/myrs ranks are '
              f'unreliable for this rebalance')
    if not np.isfinite(mlf1_std) or mlf1_std < MLF1_MIN_CROSS_STD:
        print(f'[MLF1-GUARD] WARNING: mlf1 cross-sectional std is {mlf1_std} '
              f'(threshold {MLF1_MIN_CROSS_STD}) -- signal dispersion has '
              f'collapsed; ranks are meaningless. Check the mlf1 data collection.')
    if mlf1_nunique <= 1:
        print(f'[MLF1-GUARD] WARNING: mlf1 has {mlf1_nunique} unique value(s) '
              f'across {len(df)} stocks -- column looks constant/empty. '
              f'Check the mlf1 data collection.')

    # RISKBRAKE-3: arm the acting brake while the column is broken. A broken
    # collection previously traded a random 50-name book at FULL size with
    # only a console warning. The scale is consumed (and reported) by
    # regular_allocation at the next rebalance; it re-arms/di sarms daily as
    # the column's health changes.
    mlf1_broken = (
        (mlf1_nan_frac > MLF1_MAX_NAN_FRACTION) or
        (not np.isfinite(mlf1_std)) or (mlf1_std < MLF1_MIN_CROSS_STD) or
        (mlf1_nunique <= 1)
    )
    if USE_MLF1_HEALTH_BRAKE and mlf1_broken:
        context.mlf1_health_scale = MLF1_BRAKE_SCALE
        print(f'[BRAKE] MLF1 health brake ARMED: gross long will be scaled by '
              f'{MLF1_BRAKE_SCALE} at the next rebalance (broken/stale mlf1 column)')
    else:
        context.mlf1_health_scale = 1.0

    # Calculate momentum ranking
    df['myrs'] =  df.mlf1.rank() #df.slope120.rank() + df.RS140_QQQ.rank() #+ df.mlf1.rank()

    # Display top stocks by market cap with key metrics
    print("cash ret, myrs --->>>>>> ", 
          df.sort_values(by='market_cap', ascending=False)[0:10][['cash_return_zsoft', 'cash_return', 'slope90', 'mlf1']], '\n')
    print("MLF1, --->>>>>> ", 
          df.nlargest(150, 'market_cap') \
            .sort_values('mlf1', ascending=False) \
            .head(10)[['market_cap', 'mlf1']])


    # Filter to top stocks by market cap
    df = df.sort_values(by=['market_cap'], ascending=[False])[0:FILTERED_UNIVERSE_SIZE].copy()
    df['estrank'] = df.mlf1.rank()
    print('spy below spyma80 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
              'spyprice:', context.spyprice, 'spyma80:', context.spyma80)

    # Calculate ranking based on market regime
    # if context.spyprice <= context.spyma80:
    #     # Defensive mode: focus on liquidity and momentum
    #     df['estrank'] = df.mlf1.rank() #+ df['eps_ActualSurprise_prev_Q_percent'].rank()#+#df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1)
    #     print('switch estrank to doll_vol spy below spyma80 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
    #           'spyprice:', context.spyprice, 'spyma80:', context.spyma80)
    # else:
    #     # Normal mode: multi-factor ranking
    #     context.season = 1 if algo.get_datetime().date().month in GROWTH_SEASON_MONTHS else 0

    #     df['estrank'] = (
    #         # df['MLfactor'].rank() +
    #         (df.mlf1.rank()) +
    #         (df['entval'].rank() * 2) +
    #         (df['cash_return_zsoft'].rank()) + 
    #         df['eps_gr_mean'].rank() * (4 if context.season == 1 else 1) +
    #         (df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1) / 3) 
    #     )

    print(f"Filtered stock universe size {df.shape}")

    # Select portfolios
    select_long_portfolio(context, df, data)
    select_short_portfolio(context, df, data)
    context.topmcap = df.sort_values(by=['market_cap'], ascending=[False])[0:7].copy()

    # Combine for universe tracking
    context.universe = np.union1d(context.longs.index.values, context.shorts.index.values)

    # Calculate portfolio beta ratio
    # BUGFIX-3: the legacy max(1.3, compute_beta(...)) had no ceiling and no
    # NaN guard. A short basket with beta near zero makes the ratio explode
    # (inflating every short weight), and max(1.3, nan) = nan propagates
    # straight into the orders. Now: non-finite -> 1.3 fallback, otherwise
    # clipped to [1.3, BETA_RATIO_CAP]. Logs whenever the guard changes the
    # value the legacy code would have used.
    raw_beta_ratio = compute_beta(context, data)
    if not np.isfinite(raw_beta_ratio):
        print(f'[BUGFIX-3] beta_ratio non-finite ({raw_beta_ratio}); '
              f'falling back to 1.3 (legacy would have propagated NaN into orders)')
        context.beta_ratio = 1.3
    else:
        context.beta_ratio = float(np.clip(raw_beta_ratio, 1.3, BETA_RATIO_CAP))
        if raw_beta_ratio > BETA_RATIO_CAP:
            print(f'[BUGFIX-3] beta_ratio {raw_beta_ratio:.4f} capped at '
                  f'{BETA_RATIO_CAP} (legacy would have inflated short weights)')
    print(f'Beta ratio: {context.beta_ratio:.4f}')

    return df


def compute_value_tilt_lambda(context):
    """
    RISKBRAKE-7: drawdown-gated value-tilt factor in [0, 1].

    0.0  -> book at rest: normal momentum-dominant configuration
            (sleeves TOP_MOMENTUM_STOCKS/20, mlf1 exponent 1.8).
    1.0  -> full value posture: momentum sleeve surrenders
            VALUE_TILT_MAX_SHIFT names to the cash-return sleeve and the
            mlf1 weighting exponent sits at VALUE_TILT_MLF1_EXP_MIN.

    Ramps linearly with context.draw_down between VALUE_TILT_DD_START and
    VALUE_TILT_DD_FULL, and unwinds automatically as the account recovers
    toward its high-water mark. Driven by BOOK state (drawdown), not
    market state, because the Jul-2026 unwind demonstrated that factor
    crashes can occur with the index flat and vol quiet -- the account's
    own P&L is the only sensor guaranteed to see them.

    Returns 0.0 whenever USE_VALUE_TILT_SLIDER is False (exact base-file
    behavior).
    """
    if not USE_VALUE_TILT_SLIDER:
        return 0.0
    dd = float(getattr(context, 'draw_down', 0.0) or 0.0)
    span = max(VALUE_TILT_DD_FULL - VALUE_TILT_DD_START, 1e-9)
    lam = (dd - VALUE_TILT_DD_START) / span
    return float(np.clip(lam, 0.0, 1.0))


def select_long_portfolio(context, df, data):
    """
    Select securities for the long portfolio.

    Combines value and momentum selection strategies to build a diversified
    long portfolio. Weights are adjusted by beta and slope factors.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    df : pandas.DataFrame
        Filtered universe with rankings
    data : zipline.protocol.BarData
        Market data accessor

    Selection Process
    -----------------
    1. Value Selection (30 stocks):
       - From top 150 by estrank, select top 30 by cash_return_zsoft

    2. Momentum Selection (20 stocks):
       - If bullish: From top 500 by RS140_QQQ, select top 20 by myrs
       - If bearish: Additional value stocks by cash_return_zsoft

    3. Combine and adjust:
       - Merge value and momentum selections
       - Sort by cash_return_zsoft
       - Adjust by beta (IWM or SPY depending on regime)
       - Apply slope factor boost (slope^2 adjustment)

    Result
    ------
    Sets context.longs with selected securities and adjusted cash_return_zsoft weights
    """
    dfl = df.copy()
    print('drop top two volatility outliers')
    dfl = drop_top_vol_outliers(dfl) # outlier volatility top 2 stocks removal function to help with random MEME stocks


    # RISKBRAKE-7: value-tilt slider. As the account draws down, momentum
    # sleeve slots hand over to the cash-return (value) sleeve -- the book
    # stays fully invested and inside the same signal pipeline; only the
    # internal value/momentum mix moves. lambda=0 reproduces the base
    # sleeve split exactly.
    value_tilt = compute_value_tilt_lambda(context)
    sleeve_shift = int(round(value_tilt * VALUE_TILT_MAX_SHIFT))
    num_momentum_stocks = TOP_MOMENTUM_STOCKS - sleeve_shift
    num_value_stocks = LONG_PORTFOLIO_SIZE - num_momentum_stocks
    if value_tilt > 0:
        tilted_exp = MLF1_WEIGHT_EXPONENT - value_tilt * (MLF1_WEIGHT_EXPONENT - VALUE_TILT_MLF1_EXP_MIN)
        print(f'[BRAKE] value-tilt slider ACTIVE: lambda={value_tilt:.2f} '
              f'(drawdown {float(getattr(context, "draw_down", 0.0)):.1%}) -> '
              f'sleeves value/momentum {num_value_stocks}/{num_momentum_stocks}, '
              f'mlf1 exponent {tilted_exp:.2f}')

    # Select value stocks: high cash return from top-ranked stocks
    context.longs_c = (
        dfl.sort_values(by=['estrank'], ascending=[False])[0:150] # this is the same as mlf1 as assigned 
           .sort_values(by=['cash_return_zsoft'], ascending=[False])[0:num_value_stocks]
           .copy()
    )

    # Select momentum stocks based on market trend
    if context.vix_uptrend_flag:
        context.longs_m = (
            dfl.sort_values(by=['mlf1'], ascending=[False])[0:500]
               .sort_values(by=['myrs'], ascending=[False])[0:num_momentum_stocks] # this is the same as mlf1 as assigned 
               .copy()
        )
    else:
        # In downtrends, select more value-oriented stocks
        context.longs_m = (
            dfl.sort_values(by=['estrank'], ascending=[False])[0:150]
               .sort_values(by=['cash_return_zsoft'], ascending=[False])[num_value_stocks:LONG_PORTFOLIO_SIZE]
               .copy()
        )

    # Combine value and momentum selections
    c_set = set(context.longs_c.index)
    m_set = set(context.longs_m.index)
    context.longs = dfl[dfl.index.isin(c_set.union(m_set))].copy()
    print(f'Long portfolio size: {len(context.longs)}')
    print('portfolio vol:')
    print(context.longs.sort_values(by=['vol'], ascending=[False])['vol'].head(10))

    # Sort by cash return
    context.longs = context.longs.sort_values(by=['cash_return_zsoft'], ascending=[False]).copy()

    # Adjust by beta based on market regime
    if context.spyprice >= context.spyma80:
        # Normal: normalize by IWM beta
        context.longs['cash_return_zsoft'] /= winsorize(context.longs['beta60IWM'], limits=[0.005, 0.4])
    else:
        # Defensive: normalize by SPY beta
        context.longs['cash_return_zsoft'] /= winsorize(context.longs['beta60SPY'], limits=[0.005, 0.1])

    # Ensure positive weights
    context.longs['cash_return_zsoft'] = context.longs['cash_return_zsoft'].clip(lower=0.005)

    ## proportional weighting based on mlf1 strength:
    mlf1 = winsorize(context.longs['mlf1'], limits=[0.1, 0.05])

    # RISKBRAKE-7: the exponent slides from MLF1_WEIGHT_EXPONENT (1.8,
    # momentum dominates the weight ordering) toward VALUE_TILT_MLF1_EXP_MIN
    # as the value tilt engages, compressing the mlf1 multiplier's dispersion
    # so cash_return_zsoft reasserts control of relative sizing in drawdowns.
    # value_tilt=0 (no drawdown, or slider off) reproduces mlf1 ** 1.8 exactly.
    mlf1_exp = MLF1_WEIGHT_EXPONENT - value_tilt * (MLF1_WEIGHT_EXPONENT - VALUE_TILT_MLF1_EXP_MIN)
    mlf1_adjusted = np.where(mlf1 > 0 , mlf1 ** mlf1_exp , mlf1)

    # BUGFIX-1: floor the multiplier at a tiny positive value. Previously a
    # negative mlf1 made the product cash_return_zsoft * mlf1_adjusted
    # negative, and the abs() in get_normalized_weights / the order loop then
    # gave the most ML-bearish stock a LARGE long weight (sign inversion).
    # With the floor, a negative ML prediction now yields a near-zero weight,
    # preserving monotonic "scale by ML strength" intent. Logs when it fires.
    n_neg = int((mlf1_adjusted < MLF1_WEIGHT_FLOOR).sum())
    if n_neg > 0:
        neg_names = context.longs['name'][mlf1_adjusted < MLF1_WEIGHT_FLOOR].tolist()
        print(f'[BUGFIX-1] floored {n_neg} non-positive mlf1 multiplier(s) at '
              f'{MLF1_WEIGHT_FLOOR} (legacy would have sign-inverted these to '
              f'large long weights): {neg_names}')
    mlf1_adjusted = np.clip(mlf1_adjusted, MLF1_WEIGHT_FLOOR, None)

    context.longs['cash_return_zsoft'] = context.longs['cash_return_zsoft'] * mlf1_adjusted

    # Final sorting of long portfolio
    context.longs = context.longs.sort_values(by=['cash_return_zsoft'], ascending=[False])

    print(f'Long portfolio calculated with total cash_return_zsoft {context.longs["cash_return_zsoft"].sum()}')

    return


def select_short_portfolio(context, df, data):
    """
    Select securities for the short portfolio.

    Identifies weak stocks to inform hedging decisions. Avoids shorting
    stocks in the top momentum sector to prevent fighting strong trends.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    df : pandas.DataFrame
        Filtered universe with rankings
    data : zipline.protocol.BarData
        Market data accessor

    Selection Process
    -----------------
    1. Calculate sector momentum using ETF returns
    2. Identify top momentum sector (to exclude from shorts)
    3. Remove top momentum sector from candidates
    4. Select stocks with lowest cash return

    Result
    ------
    Sets context.shorts with selected securities
    Note: Actual shorting is done via IWM, not individual stocks
    """
    dfs = df.copy()

    # Get sector momentum rankings
    mom_list = GenerateMomentumList(context, data, context.sector_etf, 242)
    mom_list = [item[0] for item in mom_list]

    # Identify top sector to avoid shorting
    top_momentum_sector = mom_list[0]
    context.mometf = top_momentum_sector

    # Remove top momentum sector
    dfs = RemoveSectors(context, top_momentum_sector, dfs, "not shorting! %s")
    print('Bottom momentum sector:', mom_list[-1])

    # Select shorts with lowest cash return
    context.shorts = (dfs.sort_values(by=['cash_return_zsoft'], ascending=[True])[0:SHORT_PORTFOLIO_SIZE]
                      .copy())
    return


def update_market_indicators(context, data):
    """
    Update market-level technical indicators.

    Calculates moving averages and trend indicators for SPY and IWM
    used in regime detection and position sizing.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor

    Indicators Updated
    ------------------
    SPY:
        - spyprice: Current price
        - spyma21/50/80/85/150/200: Simple moving averages

    IWM:
        - iwmprice: Current price
        - iwmma50/10: Hull moving averages
        - hulltrend: HMA trend direction

    Price History:
        - price_history_spy100: 200-day SPY prices (for signal detection)
        - price_history_iwm250: 250-day IWM prices (for HMA calculation)
    """
    # SPY indicators
    context.price_history_spy100 = data.history(symbol('SPY'), 'price', 200, '1d')
    context.spyprice = context.price_history_spy100.values[-1]
    context.spyma21 = np.mean(context.price_history_spy100.tail(21).values)
    context.spyma50 = np.mean(context.price_history_spy100.tail(50).values)
    context.spyma80 = np.mean(context.price_history_spy100.tail(80).values)
    context.spyma85 = np.mean(context.price_history_spy100.tail(85).values)
    context.spyma150 = np.mean(context.price_history_spy100.tail(150).values)
    context.spyma200 = np.mean(context.price_history_spy100.tail(200).values)

    # Update SPY MA flag
    context.spy_below80ma = context.spyprice < context.spyma80
    context.spy_below150ma = context.spyprice < context.spyma150

    # IWM indicators with Hull MA
    context.price_history_iwm250 = data.history(symbol('IWM'), 'price', 250, '1d')
    context.iwmprice = context.price_history_iwm250.values[-1]
    context.iwmma50 = hull_moving_average(context.price_history_iwm250.values, 50)[-1]
    context.iwmma10 = hull_moving_average(context.price_history_iwm250.values, 10)[-1]
    context.hulltrend = hull_ma_trend(context.price_history_iwm250.values, 80, lookback=7)


def handle_data(context, data):
    """
    Intraday monitoring and metrics recording.

    Called on each trading bar. Monitors account metrics, tracks
    drawdown, and records performance statistics.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor

    Metrics Tracked
    ---------------
    - Net liquidation value (with cash adjustments)
    - High-water mark for drawdown calculation
    - Current drawdown percentage
    - Account leverage
    """
    time_minute = algo.get_datetime(timezone("America/Los_Angeles")).minute
    time_hour = algo.get_datetime(timezone("America/Los_Angeles")).hour

    context.account_leverage = 2.0

    # End-of-day reporting flag
    printflag = (time_hour == 12 and time_minute == 59)

    if context.daily_flag == 0 or printflag:
        # Calculate net liquidation with adjustments
        my_net_liquidation = context.account.net_liquidation + context.cash_adjustment

        # Update high-water mark
        if my_net_liquidation > context.max_liquid:
            context.max_liquid = my_net_liquidation
            if context.daily_print_flag == 0 or printflag:
                print(f"New equity high! Max liquidation for the trading period: {context.max_liquid:.0f}")

        # Calculate drawdown
        context.draw_down = (context.max_liquid - my_net_liquidation) / context.max_liquid
        if context.draw_down != 0 and (context.daily_print_flag == 0 or printflag):
            print(f"Current account drawdown from high of: {context.max_liquid:.0f}")
            print(f'DD= {context.draw_down:.3%}')

        print(" ")
        print(" ")

        context.daily_print_flag = 1
        if context.cash_adjustment == 0:
            context.daily_flag = 1

    return


def initial_allocation(context, data):
    """
    Handle initial portfolio allocation and intraday signal changes.

    Called daily. Monitors for VIX signal changes and SPY moving average
    crossovers that require immediate action rather than waiting for
    the weekly rebalance.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor

    Signal Handling
    ---------------
    1. VIX Signal Change (vixflag turns negative):
       - If SPY below MA80: Reduce IWM short
       - If SPY above MA80: Full reallocation
       (Reachable again with USE_VIXDATA_REGIME=True -- RISKBRAKE-1 fixed
       the vixflag_prev wiring that kept this branch dead.)

    2. SPY MA80 Crossover (price crosses above):
       - Reduce IWM short position
       - Only acts on non-rebalance days

    3. SPY MA80 Crossover (price crosses BELOW) -- RISKBRAKE-5:
       - Escalate the hedge to the BULL_WEAK sizing the same day instead
       - of waiting for Tuesday. Never reduces an already-larger hedge.

    4. Initial Portfolio:
       - Triggers regular_allocation on first run
    """
    spyprice_1 = context.price_history_spy100.iloc[-1]
    spyprice_2 = context.price_history_spy100.iloc[-2]

    # Setup logging
    # BUGFIX-4: use the shared logger instead of attaching a new
    # FlightlogHandler on every daily call (handler leak / duplicate lines).
    logger = get_algo_logger()

    # Initialize IWM weight
    if context.iwm_w == 0:
        context.iwm_w = -0.4

    # Handle VIX signal change (bearish -> bullish)
    if (context.vixflag_prev > 0 and context.vixflag <= 0 and
        algo.get_datetime().date().weekday() != context.days_offset):

        if spyprice_1 < context.spyma80:
            # SPY below MA80: cautiously reduce short
            if context.shortfact != 0:
                if USE_BETA_TARGETED_HEDGE:
                    # VIX just flipped bullish but SPY is still below MA80:
                    # halve the hedge to capture the regime change, but never
                    # reduce it above the beta-based floor set at the last
                    # rebalance -- the hedge role is preserved intraday.
                    # Both values are negative: min() keeps the shorter one.
                    # RISKBRAKE-6: ordered through the (possibly split)
                    # hedge instrument weights.
                    new_weight = min(context.iwm_w / 2, context.iwm_hedge_floor)
                    place_hedge_orders(algo, context, context.hedge_symbol_weights, new_weight)
                    context.iwm_w = new_weight  # iwm_w == weight actually held
                else:
                    # LEGACY -- verbatim pre-iwmhedge code. Note iwm_w
                    # holds the pre-shortfact total_ws stored at the last
                    # rebalance and is deliberately NOT updated after this
                    # order, exactly as in the baseline.
                    new_weight = min(context.iwm_w / (2 * context.shortfact), -0.4 * context.shortfact * context.bcfactor)
                    algo.order_target_percent(context.iwm_sid, new_weight)

                logger.info(' ')
                logger.info('ALERT ALERT ALERT !!!! ')
                logger.info(f'vix long exit for IWM at {algo.get_datetime(timezone("America/Los_Angeles"))}')
                logger.info(f'switching out to less short IWM -- weight = {new_weight}')

                print("\n\n\n\n\n")
                print(">" * 100)
                print(" ALERT ALERT ALERT !!!! ")
                print("")
                print(algo.get_datetime(timezone("America/Los_Angeles")))
                print("vix long exit for IWM")
                print(f"switching out to less short IWM -- weight = {new_weight}")
                print("\n\n\n\n\n")

        if spyprice_1 >= context.spyma80:
            # SPY above MA80: full reallocation
            logger.info(' ')
            logger.info('ALERT ALERT ALERT !!!! ')
            logger.info(f'vix long exit for IWM at {algo.get_datetime(timezone("America/Los_Angeles"))}')
            logger.info('executing reallocation')

            regular_allocation(context, data)
            exit_positions(context, data)

        logger.info(' ')

    # Initial portfolio setup
    if context.initialized == 0:
        context.initialized = 1
        regular_allocation(context, data)

    # Handle SPY crossing above MA80
    if (spyprice_1 > context.spyma80 and
        spyprice_2 <= context.spyma80 and
        context.vix_uptrend_flag and
        context.spy_below80ma and
        algo.get_datetime().date().weekday() != context.days_offset):

        print("\n\n\n\n\n")
        if context.shortfact != 0:
            if USE_BETA_TARGETED_HEDGE:
                # SPY crossed back above MA80 in a bullish VIX regime: halve
                # the hedge to lean into the recovery, but respect the
                # beta-based floor so the book is never left unprotected
                # before the next weekly rebalance re-sizes the hedge.
                # RISKBRAKE-6: ordered through the split hedge weights.
                new_weight = min(context.iwm_w / 2, context.iwm_hedge_floor)
                place_hedge_orders(algo, context, context.hedge_symbol_weights, new_weight)
                context.iwm_w = new_weight  # iwm_w == weight actually held
            else:
                # LEGACY -- verbatim pre-iwmhedge code (fixed -0.4 *
                # bcfactor reference, iwm_w deliberately not updated).
                new_weight = min(context.iwm_w / 2, -0.4 * context.bcfactor)
                algo.order_target_percent(context.iwm_sid, new_weight)

            print(">" * 100)
            print("ALERT ALERT ALERT !!!! ")
            print("")
            print(algo.get_datetime(timezone("America/Los_Angeles")))
            print(f"spyma cross over, spyprice_1, spyprice_2, spyma80: {spyprice_1}, {spyprice_2}, {context.spyma80}")
            print(f"spyma cross over --- switching out to less short IWM -- weight = {new_weight}")
            context.spy_below80ma = False
            print(f'weekday: {algo.get_datetime().date().weekday()}')
            print("\n\n\n\n\n")

            logger.info(' ')
            logger.info('ALERT ALERT ALERT !!!! ')
            logger.info(f'spyma cross over, spyprice_1: {spyprice_1}, spyprice_2: {spyprice_2}, spyma80: {context.spyma80} at {algo.get_datetime(timezone("America/Los_Angeles"))}')
            logger.info(f"spyma cross over --- switching out to less short IWM -- weight = {new_weight}")
            logger.info(' ')

    # RISKBRAKE-5: handle SPY crossing BELOW MA80 (symmetric trigger).
    # The two intraday events above only ever REDUCE the hedge; a breakdown
    # previously waited up to a week for the Tuesday rebalance to size up
    # protection. This mirrors the above-crossing block: on the first day
    # after SPY closes below its 80-day MA, escalate the hedge to the
    # BULL_WEAK sizing immediately. The beta-neutral anchor is recovered
    # from the floor published by the last rebalance
    # (iwm_hedge_floor = IWM_HEDGE_FLOOR_RATIO * beta_neutral_w), so no new
    # pipeline data is needed intraday. min() on negative weights means the
    # hedge only ever GROWS here -- if the current short already exceeds the
    # escalation target, nothing happens.
    if (USE_INTRADAY_HEDGE_ESCALATION and USE_BETA_TARGETED_HEDGE and
        spyprice_1 < context.spyma80 and
        spyprice_2 >= context.spyma80 and
        context.vix_uptrend_flag and
        algo.get_datetime().date().weekday() != context.days_offset):

        beta_neutral_est = context.iwm_hedge_floor / IWM_HEDGE_FLOOR_RATIO  # negative
        escalated_weight = IWM_HEDGE_RATIO_BULL_WEAK * beta_neutral_est
        new_weight = min(context.iwm_w, escalated_weight)

        if new_weight < context.iwm_w - 1e-9:
            place_hedge_orders(algo, context, context.hedge_symbol_weights, new_weight)
            context.iwm_w = new_weight

            print("\n\n\n\n\n")
            print(">" * 100)
            print("ALERT ALERT ALERT !!!! [BRAKE] RISKBRAKE-5")
            print("")
            print(algo.get_datetime(timezone("America/Los_Angeles")))
            print(f"spyma cross UNDER, spyprice_1, spyprice_2, spyma80: {spyprice_1}, {spyprice_2}, {context.spyma80}")
            print(f"escalating hedge to BULL_WEAK sizing -- weight = {new_weight:.4f} "
                  f"(beta-neutral est {beta_neutral_est:.4f})")
            print("\n\n\n\n\n")

            logger.info(' ')
            logger.info('ALERT ALERT ALERT !!!! [BRAKE] RISKBRAKE-5')
            logger.info(f'spyma cross UNDER at {algo.get_datetime(timezone("America/Los_Angeles"))}: '
                        f'spyprice_1 {spyprice_1}, spyprice_2 {spyprice_2}, spyma80 {context.spyma80}')
            logger.info(f'escalating hedge to BULL_WEAK sizing -- weight = {new_weight:.4f}')
            logger.info(' ')

    return


def regular_allocation(context, data):
    """
    Main portfolio allocation and rebalancing function.

    Called weekly (Tuesday) to rebalance the portfolio. Calculates weights
    for all positions based on alpha signals and market conditions.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor

    Allocation Process
    ------------------
    1. Update trend signals and drawdown factor (RISKBRAKE-4: two-sided)
    2. Normalize long weights by cash_return_zsoft
    3. Apply regime-based multipliers:
       - Long factor (1.5 bullish, variable bearish)
       - Drawdown factor (dip-buy shallow / de-risk deep with RISKBRAKE-4)
       - Market condition factor (1.05 defensive, 1.102 normal)
       - VIX factor (1.6x in uptrend)
       - BC factor (0.65x if SPY < MA150 and bc1 triggered)
       - Risk brakes (RISKBRAKE-2 GARCH vol scaler, RISKBRAKE-3 MLF1 health)
       - Low-vol overlay (gross-preserving reshaping)
    4. Execute long orders (99% individual stocks, 1% SPY)
    5. Calculate legacy short signal (total_ws) with beta adjustment
    6. Size and execute the hedge via compute_iwm_hedge_weight()
       (RISKBRAKE-6: execution optionally split IWM/QQQ)

    Position Sizing
    ---------------
    Long weights: cash_return_zsoft * longfact * dd_factor * 0.637 * adjust * 1.03
                  * vix_mult * bc_mult * risk_brake_scale, then lowvol overlay
    Hedge: blend of (long-book IWM dollar-beta * regime hedge ratio * tilts)
           and the legacy alpha-based short signal, bounded by
           IWM_HEDGE_FLOOR_RATIO / IWM_HEDGE_CAP_RATIO of beta-neutral.
           Because the beta-targeted leg scales with the realized long
           weights, the hedge automatically inherits dd_factor, the risk
           brakes, and all long-side multipliers -- net exposure stays
           balanced when the brakes de-lever the book.
    """
    longs = context.longs.index
    shorts = context.shorts.index

    # Update trend signals
    compute_trend(context, data)
    context.initialized = 1

    # Calculate drawdown adjustment
    # RISKBRAKE-4: two-sided response. The legacy capped dip-buy is
    # preserved bit-for-bit up to DD_DERISK_THRESHOLD; beyond it the
    # response inverts and dd_factor declines linearly to DD_DERISK_FLOOR.
    context.dd_factor = min([context.clip, (1 + context.draw_down * DRAWDOWN_FACTOR_MULTIPLIER)])
    if USE_TWO_SIDED_DD and context.draw_down > DD_DERISK_THRESHOLD:
        legacy_dd_factor = context.dd_factor
        derisked = context.clip - (context.draw_down - DD_DERISK_THRESHOLD) * DD_DERISK_SLOPE
        context.dd_factor = max(DD_DERISK_FLOOR, derisked)
        print(f'[BRAKE] RISKBRAKE-4 two-sided dd_factor: drawdown '
              f'{context.draw_down:.1%} > {DD_DERISK_THRESHOLD:.0%} threshold -> '
              f'dd_factor {context.dd_factor:.3f} (legacy dip-buy would have '
              f'pressed at {legacy_dd_factor:.3f})')

    try:
        print(algo.get_datetime(timezone("America/Los_Angeles")))
        print(f'Beta ratio: {context.beta_ratio:.4f}')
        print(f"Drawdown factor: {context.dd_factor:.4f}")
        print(f"Long factor: {context.longfact:.2f}")
        print(f"Net liquidation: {context.account.net_liquidation:.2f}")
    except:
        print("print ERROR >>>>>")

    # Get normalized weights
    longs_mcw, shorts_mcw = get_normalized_weights(context, data, 'cash_return_zsoft')

    # Calculate regime-based multipliers
    if context.vix_uptrend_flag:
        trend_longfact_multiplier = 0.625
        trend_spy_gt_ma21 = 1
    else:
        trend_longfact_multiplier = 1.625
        trend_spy_gt_ma21 = 1.3 if context.spyprice > context.spyma21 else 1

    # Market condition adjustment
    if context.spyprice < context.spyma80:
        adjust_fact = 1.05  # More conservative below 80-day MA
    else:
        print('SPY above 80-day MA: switch adjust_fact to 1.102 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', 
              'spyprice:', context.spyprice, 'spyma:', context.spyma80)
        adjust_fact = 1.102  # More aggressive above 80-day MA

    print('adjust factor', adjust_fact)

    # Apply long weight multipliers
    longs_mcw['cash_return_zsoft'] = (
        longs_mcw['cash_return_zsoft'] * 
        context.longfact * 
        context.dd_factor * 
        0.637 * 
        #adjust_fact * 
        1.03
    )

    # VIX uptrend boost
    if context.vix_uptrend_flag:
        longs_mcw['cash_return_zsoft'] = longs_mcw['cash_return_zsoft'] * 1.6

    # Barchart trend adjustment
    if context.spyprice < context.spyma150 and context.bc1 == 1 and spy_vol_flag(context, data, window=250) == 1:
        context.bcfactor = 0.73 #0.82
        longs_mcw['cash_return_zsoft'] = longs_mcw['cash_return_zsoft'] * context.bcfactor

    # RISKBRAKE-2 / RISKBRAKE-3: correlated-unwind gross scalers.
    # Applied AFTER the legacy multiplier chain (so all A/B baselines are
    # bit-identical when the brakes are idle) and BEFORE the lowvol overlay
    # (which is gross-preserving and therefore commutes with any scalar).
    # The hedge is sized downstream from these same weights, so the
    # beta-neutral anchor -- and with it the hedge -- shrinks in proportion
    # and the book stays internally balanced while de-levered.
    context.risk_brake_scale = 1.0
    if USE_GARCH_VOL_BRAKE:
        vol_ann = spy_garch_vol(context, data, window=250)
        if vol_ann > GARCH_BRAKE_ON_VOL:
            garch_scale = max(GARCH_BRAKE_MIN_SCALE, GARCH_BRAKE_ON_VOL / vol_ann)
            context.risk_brake_scale *= garch_scale
            print(f'[BRAKE] RISKBRAKE-2 GARCH vol brake ACTIVE: SPY conditional '
                  f'vol {vol_ann:.1%} > {GARCH_BRAKE_ON_VOL:.0%} -> gross scale '
                  f'{garch_scale:.2f}')
        else:
            print(f'[BRAKE] GARCH vol brake idle (SPY conditional vol {vol_ann:.1%} '
                  f'<= {GARCH_BRAKE_ON_VOL:.0%})')
    if USE_MLF1_HEALTH_BRAKE and getattr(context, 'mlf1_health_scale', 1.0) < 1.0:
        context.risk_brake_scale *= context.mlf1_health_scale
        print(f'[BRAKE] RISKBRAKE-3 MLF1 health brake applied: '
              f'x{context.mlf1_health_scale:.2f}')
    if context.risk_brake_scale < 1.0:
        longs_mcw['cash_return_zsoft'] = (
            longs_mcw['cash_return_zsoft'] * context.risk_brake_scale
        )
        print(f'[BRAKE] combined risk-brake scale {context.risk_brake_scale:.2f} '
              f'applied to long book (hedge anchor follows automatically via '
              f'beta-targeting)')

    # LOWVOL OVERLAY: gross-preserving risk reshaping of the final long
    # weights (inverse-vol temper, overbought trim, sector cap, true name
    # cap). Applied after ALL alpha/regime multipliers AND the risk brakes,
    # so the book's gross long weight -- and therefore total_wl, the SPY
    # sleeve, and the hedge's beta-neutral anchor computed downstream -- is
    # unchanged in size; only the cross-sectional mix moves.
    # LOWVOL_MASTER = False makes this a no-op.
    longs_mcw = apply_lowvol_overlay(context, data, longs_mcw)

    # Portfolio split: 99% individual stocks, 1% SPY
    port_weight_factor = 0.99
    spy_weight_factor = 1 - port_weight_factor

    if context.verbose == 1:
        print_positions(longs, longs_mcw, 'cash_return_zsoft', port_weight_factor)

    # Execute long orders
    total_wl = 0
    for sid, w in zip(longs, longs_mcw['cash_return_zsoft'].values):
        w = abs(w)
        algo.order_target_percent(sid, w * port_weight_factor)
        if w > 1.0:
            print(sid, w)
        total_wl = total_wl + w

    # SPY allocation
    algo.order_target_percent(context.spysym, total_wl * spy_weight_factor)

    print(f'Total SPY weight: {total_wl * spy_weight_factor:.4f}')
    print(f'Total port long weight: {total_wl * port_weight_factor:.4f}')
    print(f'Total long weight: {total_wl:.4f}')

    # Calculate short weights
    #shorts_mcw[shorts_mcw['cash_return_zsoft'] > 0.15] = 0.15

    shorts_mcw['cash_return_zsoft'] = (
        -shorts_mcw['cash_return_zsoft'] *
        context.shortfact *
        context.beta_ratio *
        trend_longfact_multiplier *
        trend_spy_gt_ma21 *
        0.637 *
        adjust_fact *
        1.03
    )

    # Apply short multiplier (same for both regimes currently)
    # if context.vix_uptrend_flag and context.spyprice > context.spyma21:
    #     shorts_mcw['cash_return_zsoft'] = shorts_mcw['cash_return_zsoft'] * 1.4
    # else:
    #     shorts_mcw['cash_return_zsoft'] = shorts_mcw['cash_return_zsoft'] * 1.4

    # Calculate total short weight
    total_ws = 0
    for sid, w in zip(shorts, shorts_mcw['cash_return_zsoft'].values):
        w = abs(w)
        w = w * -1
        if w > 1.0:
            print(sid, w)
        total_ws = total_ws + w

    if total_ws > 0:
        print(total_ws)
        print('Error: total short weight is positive')

    context.total_ws = total_ws
    print(f'Total short weight: {total_ws:.4f}')



    if USE_BETA_TARGETED_HEDGE:
        # Size and execute the hedge (beta-targeted, regime-scaled,
        # bounded). The regime branching of the legacy path below (the
        # -0.384 * total_wl floor when SPY is below MA80 in an uptrend) is
        # handled inside the sizing function via IWM_HEDGE_RATIO_BULL_WEAK,
        # scaled to the long book's measured IWM beta instead of a fixed
        # constant. RISKBRAKE-6: execution goes through
        # context.hedge_symbol_weights (IWM-only, or IWM+QQQ split).
        iwm_w = compute_iwm_hedge_weight(context, data, longs_mcw, total_wl, total_ws,
                                         port_weight_factor=port_weight_factor)
        place_hedge_orders(algo, context, context.hedge_symbol_weights, iwm_w)
        print('Short Position Factors: shortfact:', context.shortfact, 'beta_ratio:', context.beta_ratio,
              'trend_multiplier:', trend_longfact_multiplier, 'spy_ma21_factor:', trend_spy_gt_ma21,
              'base_multiplier:', 0.637, 'adjust_factor:', adjust_fact, 'final_multiplier:', 1.03)
        print(f'Hedge weight {iwm_w:.4f} (legacy short signal total_ws {total_ws:.4f})')
        context.iwm_w = iwm_w
    else:
        # LEGACY SIZING -- verbatim pre-iwmhedge production code (git history).
        # Reproduces the legacy return profile exactly, including the
        # shortfact re-application inside place_short_orders and storing the
        # pre-scaling total_ws in context.iwm_w.
        if context.vix_uptrend_flag and context.spy_below80ma:
            iwm_w = min(-1 * 0.384 * total_wl, total_ws)
            place_short_orders(algo, context, context.short_symbol_weights, iwm_w)
            print('Short Position Factors: shortfact:', context.shortfact, 'beta_ratio:', context.beta_ratio,
                  'trend_multiplier:', trend_longfact_multiplier, 'spy_ma21_factor:', trend_spy_gt_ma21,
                  'base_multiplier:', 0.637, 'adjust_factor:', adjust_fact, 'final_multiplier:', 1.03)
            print(f'Min active-> SPY below MA80 - IWM weight {iwm_w:.4f}')
            context.iwm_w = iwm_w
        else:
            place_short_orders(algo, context, context.short_symbol_weights, total_ws)
            print('Short Position Factors: shortfact:', context.shortfact, 'beta_ratio:', context.beta_ratio,
                  'trend_multiplier:', trend_longfact_multiplier, 'spy_ma21_factor:', trend_spy_gt_ma21,
                  'base_multiplier:', 0.637, 'adjust_factor:', adjust_fact, 'final_multiplier:', 1.03)
            print(f'IWM weight {total_ws:.4f}')
            context.iwm_w = total_ws

    # Position count warning
    if len(pd.Series(tuple(context.portfolio.positions.keys()))) > 52:
        print("WARNING: Too many positions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>+++++++++++++++++++++++++++++++++")

    print(" ")
    return


def exit_positions(context, data):
    """
    Exit positions no longer in the target portfolio.

    Called weekly after regular_allocation. Closes any positions that
    are not in the current long portfolio or excluded ETFs.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor

    Protected Positions
    -------------------
    The following are never automatically closed:
    - IWM (hedge position)
    - SPY (core allocation)
    - DIA, QQQ, TLT, SPLV (other ETFs)
    (QQQ's protection also covers the RISKBRAKE-6 hedge sleeve.)
    """
    desired_sids = set(context.longs.index)

    # Find positions to close
    getting_the_boot = [
        sid for sid in context.portfolio.positions.keys()
        if sid not in desired_sids
        and sid != context.iwm_sid
        and sid != context.spysym
        and sid != context.dia_sid
        and sid != context.qqq_sid
        and sid != context.tlt_sid
        and sid != algo.sid(symbol('SPLV').real_sid)
    ]

    if context.verbose == 1 and getting_the_boot:
        print('Exiting positions not in longs and not IWM.')

    # Close each position
    for sid in getting_the_boot:
        if context.verbose == 1:
            print('Exiting', sid)
        try:
            algo.order_target(sid, 0)
        except Exception as e:
            print(f'Failed to exit {sid}:', e)

    return


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# Supporting functions for SID initialization, universe filtering,
# weight normalization, and other algorithm utilities.

def spy_garch_vol(context, data, window=500, refit_every=21):
    """
    Annualized GJR-GARCH(1,1) conditional volatility of SPY (the LEVEL).

    RISKBRAKE-2 refactor: this estimation previously lived inside
    spy_vol_flag, which collapsed it to a binary flag. The GARCH gross
    brake needs the vol level to size its scale factor
    (GARCH_BRAKE_ON_VOL / vol), so the fit and recursion now live here and
    spy_vol_flag is a thin threshold wrapper. Numerical behavior is
    unchanged: same estimator, same rolling window, same refit cadence,
    same cache on context (shared between the brake and the bcfactor gate,
    so enabling the brake adds no extra fitting cost).

    No `arch` dependency -- fits by maximum likelihood with scipy.optimize
    (already in the zipline container). Caches the fit and only
    re-estimates every `refit_every` bars to keep it cheap and stable.

    Returns
    -------
    float
        Annualized conditional volatility (e.g. 0.18 = 18%).
        Returns 0.0 if there is insufficient price history.
    """
    import numpy as np
    from scipy.optimize import minimize

    px = data.history(symbol('SPY'), 'price', window, '1d').values
    rets = np.diff(px) / px[:-1]
    rets = rets[np.isfinite(rets)] * 100.0          # scale x100 for stability
    if rets.size < 100:
        return 0.0

    # ---- negative log-likelihood of GJR-GARCH(1,1) with Normal errors ----
    def neg_loglik(theta, r):
        omega, alpha, gamma, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0:
            return 1e10
        if alpha + gamma/2 + beta >= 1.0:           # stationarity
            return 1e10
        mu = r.mean()
        e = r - mu
        e2 = e ** 2
        n = r.size
        s2 = np.empty(n)
        s2[0] = e2.mean()
        for t in range(1, n):
            s2[t] = omega + (alpha + gamma*(e[t-1] < 0)) * e2[t-1] + beta*s2[t-1]
        s2 = np.maximum(s2, 1e-12)
        ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(s2) + e2/s2)
        return -ll

    # ---- refit only every `refit_every` bars; cache on context ----
    bar = getattr(context, '_garch_bar', 0)
    cached = getattr(context, '_garch_params', None)
    if cached is None or bar % refit_every == 0:
        x0 = np.array([0.02, 0.02, 0.10, 0.90])     # sensible start
        bnds = [(1e-8, None), (0, 1), (0, 1), (0, 1)]
        try:
            res = minimize(neg_loglik, x0, args=(rets,), method='L-BFGS-B',
                           bounds=bnds, options={'maxiter': 200})
            cached = res.x if res.success else x0
        except Exception:
            cached = x0 if cached is None else cached
        context._garch_params = cached
    context._garch_bar = bar + 1

    omega, alpha, gamma, beta = cached

    # ---- run recursion forward to get current conditional vol ----
    mu = rets.mean()
    e = rets - mu
    e2 = e ** 2
    neg = e < 0
    s2 = e2.mean()
    for t in range(e2.size):
        s2 = omega + (alpha + gamma*neg[t]) * e2[t] + beta*s2

    # s2 is in (return*100)^2 space -> /100 back to return units, then annualize
    vol_annual = (np.sqrt(max(s2, 0.0)) / 100.0) * np.sqrt(252)
    return vol_annual


def spy_vol_flag(context, data, window=500, threshold=0.20, refit_every=21):
    """
    Binary SPY vol-regime flag from the GJR-GARCH conditional vol.

        if spy_vol_flag(context, data) == 1:
            # vol elevated -> reduce leverage
        else:
            # normal regime -> maintain normal leverage

    Thin wrapper around spy_garch_vol (RISKBRAKE-2 refactor -- see there);
    the flag's numerical behavior is identical to the pre-refactor version.

    Returns 1 if annualized conditional vol > threshold, else 0.
    """
    return 1 if spy_garch_vol(context, data, window=window,
                              refit_every=refit_every) > threshold else 0


"""
Binary SPY volatility-regime flags using the `arch` package with Student-t errors.

Two model variants, same call signature:

    spy_vol_flag_gjrgarch(context, data, window=250, threshold=0.20)
    spy_vol_flag_egarch(context, data, window=250, threshold=0.20)

Each returns 1 if the 1-step-ahead annualized conditional vol > threshold, else 0.

    if spy_vol_flag_gjrgarch(context, data) == 1:
        # vol elevated -> reduce leverage
    else:
        # normal regime -> maintain normal leverage

NOTE: requires the `arch` package in the zipline container:
    docker exec -u root quantrocket-zipline-1 pip install arch
    docker restart quantrocket-zipline-1
"""




TRADING_DAYS = 252


def _spy_returns(context, data, window):
    """Pull SPY daily prices and return clean simple returns scaled x100."""
    px = data.history(symbol('SPY'), 'price', window, '1d').values
    rets = np.diff(px) / px[:-1]
    rets = rets[np.isfinite(rets)]
    return rets * 100.0                    # arch wants returns ~x100 for stability


def _flag_from_fit(res, threshold):
    """Forecast 1-step vol from a fitted arch result and apply the threshold."""
    fc = res.forecast(horizon=1, reindex=False)
    vol_daily = np.sqrt(fc.variance.values[-1, 0]) / 100.0   # back to return units
    vol_annual = vol_daily * np.sqrt(TRADING_DAYS)
    return 1 if vol_annual > threshold else 0


# ---------------------------------------------------------------------------
# GJR-GARCH(1,1) with Student-t errors
# ---------------------------------------------------------------------------
def spy_vol_flag_gjrgarch(context, data, window=250, threshold=0.20):
    """
    GJR-GARCH(1,1)-t conditional-vol regime flag for SPY.

        sigma2_t = omega + (alpha + gamma*I[eps<0])*eps_{t-1}^2 + beta*sigma2_{t-1}

    Captures the leverage effect (down moves raise vol more than up moves),
    which dominates for equity indices. Returns 1 if vol > threshold, else 0.
    """
    rets = _spy_returns(context, data, window)
    if rets.size < 100:
        return 0
    try:
        am = arch_model(rets, mean='Constant',
                        vol='GARCH', p=1, o=1, q=1, dist='t')   # o=1 -> GJR
        res = am.fit(disp='off', show_warning=False)
        return _flag_from_fit(res, threshold)
    except Exception:
        # realized-vol fallback so a non-convergent fit never crashes the algo
        vol_annual = (np.std(rets) / 100.0) * np.sqrt(TRADING_DAYS)
        return 1 if vol_annual > threshold else 0


# ---------------------------------------------------------------------------
# EGARCH(1,1) with Student-t errors
# ---------------------------------------------------------------------------
def spy_vol_flag_egarch(context, data, window=250, threshold=0.20):
    """
    EGARCH(1,1)-t conditional-vol regime flag for SPY.

        ln(sigma2_t) = omega + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1}
                              + beta*ln(sigma2_{t-1})

    Models log-variance, so no non-negativity constraints on params, and the
    gamma term captures asymmetry directly. Often fits index vol as well as or
    better than GJR. Returns 1 if vol > threshold, else 0.
    """
    rets = _spy_returns(context, data, window)
    if rets.size < 100:
        return 0
    try:
        am = arch_model(rets, mean='Constant',
                        vol='EGARCH', p=1, o=1, q=1, dist='t')
        res = am.fit(disp='off', show_warning=False)
        return _flag_from_fit(res, threshold)
    except Exception:
        vol_annual = (np.std(rets) / 100.0) * np.sqrt(TRADING_DAYS)
        return 1 if vol_annual > threshold else 0



def initialize_sids(context, data, algo):
    """
    Initialize security IDs for ETFs and benchmark instruments.

    Called once on first trading day. Looks up SIDs for all ETFs
    and indexes used by the algorithm.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    algo : module
        Zipline API module

    SIDs Initialized
    ----------------
    - benchmarkSecurity: IWM (for beta calculation)
    - iwm_sid, dia_sid, qqq_sid, tlt_sid: ETF positions
    - ibm_sid: Used for VIX signal lookup
    - spysym: SPY for core allocation
    - sector_etf: List of sector ETFs for momentum ranking
    - sector_etf_dict: Mapping of symbol to SID
    - short_symbol_weights: Legacy short instrument weights (IWM only)
    - hedge_symbol_weights: RISKBRAKE-6 hedge execution split (IWM/QQQ)
    """
    if context.sids_initialized == 0:
        print('Looking up security IDs >>>>>>')

        # Core ETFs
        context.benchmarkSecurity = algo.sid(symbol('IWM').real_sid)
        context.iwm_sid = algo.sid(symbol('IWM').real_sid)
        context.dia_sid = algo.sid(symbol('DIA').real_sid)
        context.ibm_sid = algo.sid(symbol('IBM').real_sid)
        context.spysym = algo.sid(symbol('SPY').real_sid)
        context.qqq_sid = algo.sid(symbol('QQQ').real_sid)
        context.rwm_sid = algo.sid(symbol('RWM').real_sid)
        context.tlt_sid = algo.sid(symbol('TLT').real_sid)
        context.iwb_sid = algo.sid(symbol('IWB').real_sid)

        # Non-tradable symbols (excluded from stock selection)
        context.no_trade_sym = symbols([
            'OEF', 'QQQ', 'IWM', 'SPY', 'TLT',
            'MTUM', 'SPYG', 'QUAL', 'DIA', 'SPHB'
        ])
        print('Non-tradable symbols:', context.no_trade_sym)

        # Sector ETFs for momentum ranking
        context.sector_etf = []
        context.sector_etf_dict = {}
        for sym in ['IYZ', 'XLF', 'XLE', 'XLK', 'XLB', 'XLY', 'XLI', 'XLV', 'XLP', 'XLU']:
            sid_var = algo.sid(symbol(sym).real_sid)
            context.sector_etf.append(sid_var)
            context.sector_etf_dict.update({sym: sid_var})
        context.sector_etf = pd.Series(context.sector_etf)
        print("Sector ETF dictionary:", context.sector_etf_dict)

        # Index ETFs
        context.index_etf = []
        context.index_etf_dict = {}
        for sym in ['IWM', 'QQQ']:
            sid_var = algo.sid(symbol(sym).real_sid)
            context.index_etf.append(sid_var)
            context.index_etf_dict.update({sym: sid_var})
        context.index_etf = pd.Series(context.index_etf)
        print("Index ETF dictionary:", context.index_etf_dict)

        # Short instrument weights (100% IWM) -- legacy path
        context.short_symbol_weights = {
            context.iwm_sid: 1,
        }

        # RISKBRAKE-6: hedge EXECUTION split. The hedge total is still sized
        # against the long book's IWM dollar-beta (compute_iwm_hedge_weight
        # is unchanged); this only splits the order across instruments so
        # part of the short covers the large-cap growth/momentum leg where
        # the mlf1 book concentrates. QQQ_HEDGE_FRACTION = 0 restores the
        # exact IWM-only prior behavior. Used by the beta-targeted weekly
        # rebalance AND all intraday hedge adjustments; the legacy
        # (USE_BETA_TARGETED_HEDGE=False) path keeps short_symbol_weights.
        if QQQ_HEDGE_FRACTION > 0:
            context.hedge_symbol_weights = {
                context.iwm_sid: 1.0 - QQQ_HEDGE_FRACTION,
                context.qqq_sid: QQQ_HEDGE_FRACTION,
            }
            print(f"[BRAKE] RISKBRAKE-6 hedge split active: "
                  f"IWM {1.0 - QQQ_HEDGE_FRACTION:.0%} / QQQ {QQQ_HEDGE_FRACTION:.0%}")
        else:
            context.hedge_symbol_weights = dict(context.short_symbol_weights)

        context.sids_initialized = 1

    return


def filter_symbols_sectors_universe(df):
    """
    Filter universe by sector exclusions and special cases.

    Removes certain stocks and sectors from consideration based on:
    - Merger/acquisition dates (stocks delisted)
    - Sector exclusions (Financials)
    - Sector limits (Energy, Real Estate)
    - Trading history requirements

    Parameters
    ----------
    df : pandas.DataFrame
        Raw pipeline output

    Returns
    -------
    pandas.DataFrame
        Filtered universe

    Exclusions
    ----------
    - GBT: After June 2022 (acquisition)
    - ABMD: After November 2022 (acquisition)
    - XM: After July 2023
    - SPLK: After November 2023 (acquisition)
    - MSTR: After 2010 (crypto exposure)
    - ITCI: After March 2025
    - Financials sector: Excluded entirely
    - Energy: Limited to top 20 by market cap
    - Real Estate: Limited to top 15 by market cap
    - New listings: publicdays must be > 0
    """
    current_date = algo.get_datetime().date()
    current_year = current_date.year
    current_month = current_date.month

    # Build exclusion list based on dates
    to_drop = []
    if (current_year > 2022) or (current_year == 2022 and current_month >= 6):
        to_drop.append('GBT')
    if (current_year > 2022) or (current_year == 2022 and current_month >= 11):
        to_drop.append('ABMD')
    if (current_year > 2023) or (current_year == 2023 and current_month >= 7):
        to_drop.append('XM')
    if (current_year > 2023) or (current_year == 2023 and current_month >= 11):
        to_drop.append('SPLK')
    if current_year > 2010:
        to_drop.append('MSTR')
    if (current_year > 2025) or (current_year == 2025 and current_month >= 3):
        to_drop.append('ITCI')
    if (current_year > 2026) or (current_year == 2026 and current_month >= 2):
        to_drop.append('RNA')
    if (current_year > 2026) or (current_year == 2026 and current_month >= 3):
        to_drop.append('EXAS')
    # if (current_year > 2026) or (current_year == 2026 and current_month >= 4):
    #     to_drop.append('CAR')


    # Remove excluded stocks
    for stock_name in to_drop:
        df = df[df['name'] != stock_name]

    # Remove Financials sector
    df = df[df['sector'] != 'Financials']

    # Limit Energy and Real Estate exposure
    df_energy = df[df['sector'] == 'Energy'].sort_values(by='market_cap', ascending=False)[:20]
    df_real_estate = df[df['sector'] == 'Real Estate'].sort_values(by='market_cap', ascending=False)[:15]

    df = pd.concat([ 
        df[~df['sector'].isin(['Energy', 'Real Estate'])],
        df_energy,
        df_real_estate
    ])

    # Require trading history
    df = df[df['publicdays'] > 0]

    return df


def get_normalized_weights(context, data, target):
    """
    Normalize position weights with size limits.

    Converts raw alpha scores into portfolio weights, applying
    position size constraints to ensure diversification.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    target : str
        Column name containing alpha scores

    Returns
    -------
    tuple
        (longs_mcw, shorts_mcw) - DataFrames with normalized weights

    Normalization Process
    ---------------------
    1. Fill NaN values with small positive number
    2. Normalize to sum to 1
    3. Apply position limits (max 6% long, 2% short)
    4. Clip minimum weights to avoid tiny positions
    5. Re-normalize to sum to 1
    6. Repeat clipping/normalization for stability
    """
    # Fill missing values
    context.longs[target].fillna(value=0.001, inplace=True)
    context.shorts[target].fillna(value=0.001, inplace=True)

    # Display sector statistics
    df1 = context.longs.groupby('sector')[target].agg(['median', 'mean', 'count'])
    print('Mean of target', target)
    print(df1)
    print('Length of target column:', len(context.longs[[target]]), 'Sum:', context.longs[[target]].sum())

    # Normalize long weights
    longs_mcw = abs(context.longs[[target]]) / abs(context.longs[[target]]).sum()

    # Apply position limits (two passes for stability)
    longs_mcw[longs_mcw[target] >= MAX_POSITION_SIZE_LONG] = MAX_POSITION_SIZE_LONG
    longs_mcw[longs_mcw[target] <= 0.002] = 0.02
    longs_mcw = abs(longs_mcw[[target]]) / abs(longs_mcw[[target]]).sum()

    longs_mcw[longs_mcw[target] >= MAX_POSITION_SIZE_LONG] = MAX_POSITION_SIZE_LONG
    longs_mcw[longs_mcw[target] <= 0.002] = 0.02
    longs_mcw = abs(longs_mcw[[target]]) / abs(longs_mcw[[target]]).sum()

    # Normalize short weights
    shorts_mcw = abs(context.shorts[[target]]) / abs(context.shorts[[target]]).sum()
    shorts_mcw[shorts_mcw[target] > MAX_POSITION_SIZE_SHORT] = MAX_POSITION_SIZE_SHORT
    shorts_mcw = abs(shorts_mcw[[target]]) / abs(shorts_mcw[[target]]).sum()

    return longs_mcw, shorts_mcw


def apply_lowvol_overlay(context, data, longs_mcw):
    """
    Gross-preserving low-volatility reshaping of the final long weights.

    Called from regular_allocation AFTER every alpha/regime multiplier
    (longfact, dd_factor, vix boost, bcfactor) and the risk brakes, and
    BEFORE the order loop, so the weights it sees are exactly the weights
    that would be ordered.

    THE INVARIANT: sum(weights) out == sum(weights) in, to float precision.
    The overlay only moves weight BETWEEN the 50 names the base algorithm
    already selected and sized -- it never adds or removes gross exposure.
    Consequently total_wl, the SPY sleeve, and the hedge's beta-neutral
    anchor (computed downstream from these same weights) are unchanged in
    size, and the average market exposure driving the return profile is
    preserved by construction. Only the cross-sectional risk mix moves.

    Levers (each behind its own switch; see the "Low-Volatility Overlay
    Parameters" constants block for tuning guidance):

    1. Inverse-vol temper: w_i *= clip((vol_i/median_vol)^-eta, MIN, MAX)
    2. Overbought trim:    parabolic names (price > (1+ext) * MA50) get
                           w_i *= LOWVOL_OB_TRIM_FACTOR; freed weight goes
                           pro-rata to the UNTRIMMED names (riskbrakes fix:
                           exact redistribution, matching the docstring --
                           the earlier global renorm leaked a sliver of
                           each trim back to its own target)
    3. Sector cap:         no GICS sector > LOWVOL_SECTOR_MAX_FRACTION of
                           the book; excess redistributed pro-rata
    4. True name cap:      water-filled LOWVOL_NAME_CAP_FRACTION ceiling

    Order matters: 1 and 2 are signal-driven tilts, applied first and
    jointly renormalized; 3 and 4 are hard caps, applied last so nothing
    downstream can push a name/sector back over its limit.

    All levers print "[LOWVOL]" lines when they bind, and every call prints
    the naive vol-proxy change (sqrt(sum w^2 sigma^2), correlations
    ignored -- a diagnostic, not a risk model).

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context (context.longs supplies vol/sector/name columns)
    data : zipline.protocol.BarData
        Market data accessor (one price-history fetch powers levers 1 & 2)
    longs_mcw : pandas.DataFrame
        Final long weights in the 'cash_return_zsoft' column

    Returns
    -------
    pandas.DataFrame
        Same frame with the column reshaped (or untouched if LOWVOL_MASTER
        is False / the book is degenerate).
    """
    target = 'cash_return_zsoft'

    if not LOWVOL_MASTER:
        return longs_mcw

    w = longs_mcw[target].abs().copy()
    total_before = float(w.sum())
    if total_before <= 0 or len(w) < 5:
        print(f'[LOWVOL] skipped: degenerate long book '
              f'(n={len(w)}, gross={total_before:.4f})')
        return longs_mcw

    # --- Shared price history: one fetch powers the vol and overbought
    # signals. 50 names x ~61 days weekly is negligible cost.
    hist_window = max(LOWVOL_OB_MA_WINDOW, LOWVOL_VOL_WINDOW) + 1
    px = None
    try:
        px = data.history(list(w.index), 'price', hist_window, '1d')
    except Exception as e:
        print(f'[LOWVOL] WARNING: price history fetch failed ({e}); '
              f'temper falls back to pipeline 10d vol, overbought trim skipped')

    if px is not None:
        vol = px.pct_change().tail(LOWVOL_VOL_WINDOW).std().reindex(w.index)
    else:
        vol = context.longs['vol'].reindex(w.index)
    v = vol.fillna(vol.median())

    # Naive diversification proxy (ignores correlations; logging only).
    proxy_before = float(np.sqrt((w.pow(2) * v.pow(2)).sum()))

    # --- Lever 1: inverse-vol temper (bounded tilt) ---
    if LOWVOL_USE_VOL_TEMPER:
        med = float(v.median())
        if np.isfinite(med) and med > 0:
            f = (v / med).pow(-LOWVOL_VOL_TEMPER_ETA)
            f = f.clip(LOWVOL_TEMPER_MIN, LOWVOL_TEMPER_MAX).fillna(1.0)
            w = w * f
            print(f'[LOWVOL] vol temper (eta={LOWVOL_VOL_TEMPER_ETA}): '
                  f'factors {float(f.min()):.2f}-{float(f.max()):.2f}, '
                  f'{int((f <= LOWVOL_TEMPER_MIN + 1e-12).sum())} at min clip, '
                  f'{int((f >= LOWVOL_TEMPER_MAX - 1e-12).sum())} at max clip')
        else:
            print(f'[LOWVOL] vol temper skipped: bad median vol ({med})')

    # --- Lever 2: overbought (parabolic extension) trim ---
    n_ob = 0
    if LOWVOL_USE_OVERBOUGHT_TRIM and px is not None:
        ma = px.tail(LOWVOL_OB_MA_WINDOW).mean().reindex(w.index)
        last = px.iloc[-1].reindex(w.index)
        ext = last / ma - 1.0
        ob = (ext > LOWVOL_OB_EXTENSION).fillna(False)  # NaN -> not trimmed
        n_ob = int(ob.sum())
        if n_ob > 0:
            names = context.longs['name'].reindex(w.index)
            trimmed = [f'{n} ({e:+.0%})'
                       for n, e in zip(names[ob], ext[ob])]
            print(f'[LOWVOL] overbought trim x{LOWVOL_OB_TRIM_FACTOR} '
                  f'(> {LOWVOL_OB_EXTENSION:.0%} above MA{LOWVOL_OB_MA_WINDOW}): '
                  f'{trimmed}')
            # RISKBRAKES fix: exact redistribution -- the freed weight goes
            # pro-rata to the UNTRIMMED names only, so none of it leaks back
            # to the very positions being trimmed (the earlier global renorm
            # returned ~1-2% of each trim to its own target).
            freed = float((w.loc[ob] * (1.0 - LOWVOL_OB_TRIM_FACTOR)).sum())
            w.loc[ob] = w.loc[ob] * LOWVOL_OB_TRIM_FACTOR
            recv = ~ob
            recv_sum = float(w.loc[recv].sum())
            if recv_sum > 0:
                w.loc[recv] = w.loc[recv] * (1.0 + freed / recv_sum)

    # Re-anchor gross after the signal-driven tilts (levers 1-2) so the
    # hard caps below operate on the restored book size.
    w = w * (total_before / float(w.sum()))

    # --- Levers 3 & 4: hard caps, iterated JOINTLY ---
    # Each cap's pro-rata redistribution can push the OTHER cap back into
    # violation (the name cap's water-fill spills weight into names of an
    # at-cap sector, and vice versa), so both are applied inside an outer
    # loop until they hold simultaneously. Converges in 1-2 passes in
    # practice; if the combination is infeasible (caps too tight for the
    # book) it warns and exits with gross still preserved.
    # .astype(object) first: the pipeline delivers sector as a pandas
    # Categorical, and fillna with a label outside its category set raises
    # "Cannot setitem on a Categorical with a new category".
    sectors = (context.longs['sector'].reindex(w.index)
               .astype(object).fillna('Unknown'))
    cap_w = LOWVOL_NAME_CAP_FRACTION * total_before
    for _joint in range(6):
        # Lever 3: sector cap with pro-rata redistribution
        if LOWVOL_USE_SECTOR_CAP:
            for _pass in range(5):
                sec_frac = w.groupby(sectors).sum() / total_before
                over = sec_frac[sec_frac > LOWVOL_SECTOR_MAX_FRACTION + 1e-9]
                if over.empty:
                    break
                capped = set(over.index)
                excess = 0.0
                for sec in capped:
                    m = (sectors == sec)
                    cur = float(w[m].sum())
                    tgt = LOWVOL_SECTOR_MAX_FRACTION * total_before
                    w.loc[m] = w.loc[m] * (tgt / cur)
                    excess += cur - tgt
                recv = ~sectors.isin(capped)
                recv_sum = float(w[recv].sum())
                if recv_sum <= 0:
                    # Nowhere to redistribute (whole book capped) -- restore
                    # gross and stop; better an over-cap book than lost gross.
                    w = w * (total_before / float(w.sum()))
                    print('[LOWVOL] sector cap: no uncapped sector to receive '
                          'excess; cap released to preserve gross')
                    break
                w.loc[recv] = w.loc[recv] * (1.0 + excess / recv_sum)
                if excess > 5e-4:  # mute the sub-basis-point cleanup passes
                    print(f'[LOWVOL] sector cap (joint pass {_joint + 1}): '
                          f'{sorted(capped)} capped at '
                          f'{LOWVOL_SECTOR_MAX_FRACTION:.0%} of book, '
                          f'{excess:.4f} gross redistributed')

        # Lever 4: true per-name cap (water-filling)
        if LOWVOL_USE_NAME_CAP:
            for _pass in range(10):
                over = w > cap_w + 1e-12
                if not bool(over.any()):
                    break
                if _pass == 0 and _joint == 0:
                    print(f'[LOWVOL] name cap {LOWVOL_NAME_CAP_FRACTION:.0%} '
                          f'of book: {int(over.sum())} name(s) clipped')
                excess = float((w[over] - cap_w).sum())
                w.loc[over] = cap_w
                under = ~over
                under_sum = float(w[under].sum())
                if under_sum <= 0:
                    break
                w.loc[under] = w.loc[under] * (1.0 + excess / under_sum)

        # Exit when both caps hold simultaneously (1e-6-of-book tolerance:
        # sub-basis-point overshoot is noise, not exposure).
        sec_ok = (not LOWVOL_USE_SECTOR_CAP) or bool(
            (w.groupby(sectors).sum() / total_before
             <= LOWVOL_SECTOR_MAX_FRACTION + 1e-6).all())
        name_ok = (not LOWVOL_USE_NAME_CAP) or bool(
            (w <= cap_w + 1e-6 * total_before).all())
        if sec_ok and name_ok:
            break
    else:
        print('[LOWVOL] WARNING: sector/name caps did not converge jointly '
              '(caps may be infeasible for this book); gross is still '
              'preserved but a cap may be slightly exceeded')

    # --- Exact gross restoration: the return-preservation invariant ---
    w = w * (total_before / float(w.sum()))
    drift = abs(float(w.sum()) - total_before)
    if drift > 1e-9:
        print(f'[LOWVOL] WARNING: gross drift {drift:.2e} after final renorm')

    proxy_after = float(np.sqrt((w.pow(2) * v.pow(2)).sum()))
    print(f'[LOWVOL] gross long {total_before:.4f} preserved; naive vol proxy '
          f'{proxy_before:.5f} -> {proxy_after:.5f} '
          f'({proxy_after / proxy_before - 1:+.1%}); overbought trims: {n_ob}')

    longs_mcw[target] = w.reindex(longs_mcw.index).values
    return longs_mcw


def print_positions(port, port_w, target, factor=1):
    """
    Print current position weights for debugging.

    Parameters
    ----------
    port : pandas.Index
        Security IDs
    port_w : pandas.DataFrame
        Weight DataFrame
    target : str
        Column name with weights
    factor : float, optional
        Multiplier for display (default: 1)
    """
    print(algo.get_datetime(timezone("America/Los_Angeles")))
    l = sorted(
        [list(c) for c in zip(port[0:], (port_w[target] * factor).round(6).astype(str).values[0:])],
        key=lambda x: x[1],
        reverse=True
    )
    print(pd.Series(l).values)
    return


def place_short_orders(algo, context, symbol_weights, total_weight):
    """
    Execute short position orders.

    Places orders for short hedging instruments (primarily IWM).
    Weights are distributed proportionally among instruments.

    Parameters
    ----------
    algo : module
        Zipline API module
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    symbol_weights : dict
        Mapping of SID to relative weight
    total_weight : float
        Total short weight to allocate (negative)
    """
    # BUGFIX-4: shared logger instead of attaching a new handler per call.
    logger = get_algo_logger()

    sum_of_weights = sum(symbol_weights.values())

    for symbol, weight in symbol_weights.items():
        target_percent = (weight / sum_of_weights) * total_weight
        print(f"Executing short target {symbol}, {target_percent * context.shortfact:.4f}")
        algo.order_target_percent(symbol, target_percent * context.shortfact)

    return


def place_hedge_orders(algo, context, symbol_weights, total_weight):
    """
    Execute hedge orders at exactly the requested total weight.

    Unlike place_short_orders, this does NOT rescale by context.shortfact at
    order time -- shortfact (and every other regime factor) is already baked
    into the weight produced by compute_iwm_hedge_weight. This removes the
    legacy double-application of shortfact and makes context.iwm_w identical
    to the weight actually held, so the intraday adjustment logic in
    initial_allocation operates on true position size.

    RISKBRAKE-6: symbol_weights is context.hedge_symbol_weights, which may
    split execution across IWM and QQQ. All intraday hedge adjustments order
    through this same helper, so every leg stays proportional and
    context.iwm_w remains the true TOTAL hedge weight held.

    Parameters
    ----------
    algo : module
        Zipline API module
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    symbol_weights : dict
        Mapping of SID to relative weight (IWM-only, or IWM/QQQ split)
    total_weight : float
        Final total hedge weight to allocate (negative)
    """
    sum_of_weights = sum(symbol_weights.values())

    for hedge_sid, weight in symbol_weights.items():
        target_percent = (weight / sum_of_weights) * total_weight
        print(f"Executing hedge target {hedge_sid}, {target_percent:.4f}")
        algo.order_target_percent(hedge_sid, target_percent)

    return


def compute_iwm_hedge_weight(context, data, longs_mcw, total_wl, total_ws,
                             port_weight_factor=0.99):
    """
    Compute the hedge weight by beta-targeting the long book.

    The legacy sizing derived the hedge from the alpha scores of 50 short
    candidates that are never traded, run through a chain of fixed
    multipliers. That number had no link to the exposure the hedge exists
    to offset, did not scale with dd_factor, and double-applied shortfact
    at order time. This function anchors the hedge to the long portfolio's
    measured IWM dollar-beta and keeps the legacy signal only as a
    return-seeking tilt, so the hedge can be made cheaper in strong tapes
    (max return) without ever losing its protective role.

    RISKBRAKE note: because the anchor is built from the realized long
    weights, the hedge automatically inherits the GARCH/MLF1 risk-brake
    scales and the two-sided dd_factor -- when the brakes de-lever the
    book, the hedge shrinks in proportion and net exposure stays balanced.
    RISKBRAKE-6 splits only the EXECUTION across IWM/QQQ (see
    place_hedge_orders); the sizing below is unchanged and remains
    IWM-beta-anchored.

    Sizing Model
    ------------
    1. Beta-neutral weight:
         beta_neutral_w = -(sum_i w_i * beta60IWM_i + spy_sleeve * beta_spy_iwm)
       where w_i are the final long weights (all long multipliers and
       dd_factor included). This is the IWM short that would fully offset
       the long book's IWM beta.

    2. Regime hedge ratio (fraction of beta-neutral actually carried):
         - Bullish, SPY >= MA80 : IWM_HEDGE_RATIO_BULL       (light, max carry)
         - Bullish, SPY <  MA80 : IWM_HEDGE_RATIO_BULL_WEAK  (defensive)
         - Bearish              : IWM_HEDGE_RATIO_BEAR       (protection first)
       In the bullish regime the ratio is scaled by shortfact / 0.9 so the
       existing seasonal short calendar (SHORT_RESTRICTED_MONTHS + bc1)
       carries over unchanged.

    3. Tactical tilt (bounded 0.70x - 1.45x):
         - IWM Hull-MA downtrend and price < HMA50 : 1.25x
           (shorting a falling index adds return AND protection)
         - IWM Hull-MA uptrend and price > HMA50   : 0.80x
           (do not fight small-cap strength; keep only the core hedge)
         - bc1 bear confirmation and SPY < MA150   : 1.15x

    4. Alpha blend:
         iwm_w = IWM_HEDGE_BETA_BLEND * beta_hedge
                 + (1 - IWM_HEDGE_BETA_BLEND) * legacy_signal
       where legacy_signal is what the old path would have ordered
       (total_ws * shortfact), preserving the relative-weakness information
       in the short candidate list.

    5. Bounds (the hedge-role guarantee, applied while the book is invested):
         floor = IWM_HEDGE_FLOOR_RATIO * beta_neutral_w  -> hedge never vanishes
         cap   = IWM_HEDGE_CAP_RATIO   * beta_neutral_w  -> hedge never becomes
                                                            a net-short bet
       When the long book is (near) empty -- bearish regime below MA80 --
       the position is directional, so the legacy signal is used directly,
       clamped to IWM_BEAR_MAX_SHORT.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    longs_mcw : pandas.DataFrame
        Final long weights (cash_return_zsoft column, all multipliers applied)
    total_wl : float
        Total gross long weight
    total_ws : float
        Legacy alpha-based short signal (negative)
    port_weight_factor : float
        Fraction of the long book held in individual stocks (rest is SPY)

    Returns
    -------
    float
        Target hedge weight (negative). Also sets context.iwm_hedge_floor
        for the intraday adjustment logic (and RISKBRAKE-5's anchor).
    """
    # --- 1. Dollar beta of the long book vs IWM ---
    # longs_mcw weights arrive AFTER all long multipliers (longfact, dd_factor,
    # vix boost, bcfactor) have been applied, so they sum to the gross long
    # weight actually being ordered. Multiplying each weight by that stock's
    # 60-day IWM beta gives the book's IWM dollar-beta -- the exposure the
    # hedge exists to offset. Because dd_factor is embedded in these weights,
    # the hedge automatically shrinks/grows with the long book in drawdowns.
    long_w = longs_mcw['cash_return_zsoft'].abs()

    # Per-stock betas from the pipeline. Missing betas default to 1.0 (market-
    # like). Winsorize the cross-section so one mis-estimated 60-day beta
    # (e.g. a post-event name) cannot distort the aggregate, and clip at 0.2
    # so near-zero/negative regression artifacts cannot erase real exposure.
    betas = context.longs['beta60IWM'].reindex(long_w.index).fillna(1.0)
    betas = pd.Series(
        np.asarray(winsorize(betas, limits=[0.02, 0.02])), index=betas.index
    ).clip(lower=0.2)

    # Only port_weight_factor (~99%) of the long book is in individual stocks.
    port_dollar_beta = float((long_w * betas).sum()) * port_weight_factor

    # The remaining sleeve is held in SPY; hedge it at SPY's realized beta to
    # IWM, estimated as cov(SPY, IWM)/var(IWM) over the last ~120 sessions
    # from the price histories already fetched in update_market_indicators.
    # Falls back to 0.85 (long-run typical) if history is too short, and is
    # clipped to [0.4, 1.2] against degenerate estimates in quiet markets.
    spy_rets = context.price_history_spy100.pct_change().dropna().values
    iwm_rets = context.price_history_iwm250.pct_change().dropna().values
    n = min(len(spy_rets), len(iwm_rets), 120)
    if n > 20:
        s, i = spy_rets[-n:], iwm_rets[-n:]
        var_i = np.var(i)
        beta_spy_iwm = float(np.cov(s, i)[0, 1] / var_i) if var_i > 0 else 0.85
    else:
        beta_spy_iwm = 0.85
    beta_spy_iwm = float(np.clip(beta_spy_iwm, 0.4, 1.2))
    port_dollar_beta += total_wl * (1 - port_weight_factor) * beta_spy_iwm

    # The IWM weight that would fully neutralize the book's IWM beta.
    # Everything downstream is expressed as a fraction of this anchor.
    beta_neutral_w = -port_dollar_beta

    # --- 2. Regime hedge ratio ---
    # How much of beta-neutral to actually carry, by regime. The bull-regime
    # scaling by shortfact/0.9 maps the legacy seasonal short calendar
    # (SHORT_RESTRICTED_MONTHS + bc1 sets shortfact to 0.45 vs 0.9 in
    # compute_trend) onto the new scheme: restricted months halve the hedge,
    # exactly as they halved the legacy short.
    if context.vix_uptrend_flag:
        h = IWM_HEDGE_RATIO_BULL_WEAK if context.spy_below80ma else IWM_HEDGE_RATIO_BULL
        h *= context.shortfact / 0.9  # carries the seasonal short calendar over
    else:
        h = IWM_HEDGE_RATIO_BEAR

    # --- 3. Tactical tilt (the return-enhancement layer) ---
    # Overweight the short when IWM itself is weak -- shorting a falling
    # index adds return AND protection, so leaning in never conflicts with
    # the hedge role. Underweight (never remove: the floor below still
    # applies) when small caps are in a confirmed uptrend and the short is
    # pure drag. Signals reuse the Hull-MA state computed daily in
    # update_market_indicators and the Barchart bear confirmation.
    tilt = 1.0
    if context.hulltrend == 'negative' and context.iwmprice < context.iwmma50:
        tilt *= 1.25   # IWM downtrend confirmed by price below Hull MA50
    elif context.hulltrend == 'positive' and context.iwmprice > context.iwmma50:
        tilt *= 0.80   # IWM uptrend confirmed: carry only the core hedge
    if context.bc1 == 1 and context.spyprice < context.spyma150:
        tilt *= 1.15   # Barchart bear signal with SPY under MA150
    # Clamp so stacked tilts stay a tactical adjustment, not a regime change.
    tilt = float(np.clip(tilt, 0.70, 1.45))

    beta_hedge_w = beta_neutral_w * h * tilt

    # --- 4 & 5. Blend with the legacy signal and apply hedge-role bounds ---
    # legacy_w reproduces what the old code path would have ordered: total_ws
    # already contains one application of shortfact, and place_short_orders
    # applied it a second time at order entry. Multiplying here keeps the
    # blend calibrated to historical sizing (the double application becomes
    # explicit and documented instead of accidental).
    legacy_w = total_ws * context.shortfact

    if total_wl > 0.2 and beta_neutral_w < 0:
        # Book is invested: the IWM short is a hedge. Blend the structural
        # beta-targeted size with the legacy alpha signal, then bound it.
        # All quantities are negative weights, so min() enforces "at least
        # this short" and max() enforces "no shorter than".
        iwm_w = IWM_HEDGE_BETA_BLEND * beta_hedge_w + (1 - IWM_HEDGE_BETA_BLEND) * legacy_w
        floor_w = IWM_HEDGE_FLOOR_RATIO * beta_neutral_w
        cap_w = IWM_HEDGE_CAP_RATIO * beta_neutral_w
        iwm_w = min(iwm_w, floor_w)  # hedge-role guarantee: floor short stays on
        iwm_w = max(iwm_w, cap_w)    # never beyond beta-neutral * cap (no net-short bet)
        # Publish the floor so the intraday de-risking events in
        # initial_allocation (VIX flip, SPY/MA80 crossover) can halve the
        # hedge without ever stripping it entirely, and so RISKBRAKE-5 can
        # recover the beta-neutral anchor intraday.
        context.iwm_hedge_floor = floor_w
    else:
        # Long book is (near) empty (bearish below MA80, longfact = 0): the
        # IWM short is the portfolio's directional position, not a hedge, so
        # the beta-based bounds don't apply. Use the legacy signal at full
        # strength -- bear-regime behavior is intentionally unchanged from
        # the prior version -- clamped to IWM_BEAR_MAX_SHORT, and never
        # allowed to flip long.
        iwm_w = max(legacy_w, -IWM_BEAR_MAX_SHORT)
        iwm_w = min(iwm_w, 0.0)
        # Conservative intraday floor while there is no long book to protect.
        context.iwm_hedge_floor = min(-0.10, iwm_w / 2)

    print('IWM hedge sizing -> long_dollar_beta_iwm:', round(port_dollar_beta, 4),
          'beta_neutral_w:', round(beta_neutral_w, 4),
          'beta_spy_iwm:', round(beta_spy_iwm, 3),
          'regime_ratio:', round(h, 3), 'tilt:', round(tilt, 3),
          'legacy_w:', round(legacy_w, 4), 'final iwm_w:', round(iwm_w, 4))

    return iwm_w


def GenerateMomentumList(context, data, etf_list, momlength):
    """
    Generate momentum ranking for sector ETFs.

    Ranks ETFs by price momentum over the specified period.
    Used to identify strong/weak sectors for rotation.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    etf_list : list
        List of ETF SIDs to rank
    momlength : int
        Lookback period in trading days

    Returns
    -------
    list
        List of [SID, momentum] pairs sorted by momentum (descending)
    """
    price_history = data.history(etf_list, 'price', momlength, '1d')
    pct_change = (price_history.iloc[-1] - price_history.iloc[0]) / price_history.iloc[0]
    momentum_df = pct_change.to_frame(name='momentum').reset_index()
    momentum_df = momentum_df.sort_values(by='momentum', ascending=False)
    top_momentum_list = momentum_df.head(context.topMom).values.tolist()
    return top_momentum_list


def RemoveSectors(context, etf, dfs, prt_str):
    """
    Remove stocks from a specific sector based on ETF mapping.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    etf : zipline.assets.Equity
        Sector ETF SID
    dfs : pandas.DataFrame
        Stock universe to filter
    prt_str : str
        Format string for logging

    Returns
    -------
    pandas.DataFrame
        Filtered universe with sector removed

    Sector Mapping
    --------------
    XLB -> Materials
    XLY -> Consumer Discretionary
    XLF -> Financials
    XLP -> Consumer Staples
    XLV -> Health Care
    XLU -> Utilities
    IYZ -> Communication Services
    XLE -> Energy
    XLI -> Industrials
    XLK -> Information Technology
    """
    etf_sector_map = {
        context.sector_etf_dict['XLB']: 'Materials',
        context.sector_etf_dict['XLY']: 'Consumer Discretionary',
        context.sector_etf_dict['XLF']: 'Financials',
        context.sector_etf_dict['XLP']: 'Consumer Staples',
        context.sector_etf_dict['XLV']: 'Health Care',
        context.sector_etf_dict['XLU']: 'Utilities',
        context.sector_etf_dict['IYZ']: 'Communication Services',
        context.sector_etf_dict['XLE']: 'Energy',
        context.sector_etf_dict['XLI']: 'Industrials',
        context.sector_etf_dict['XLK']: 'Information Technology',
    }

    if etf in etf_sector_map:
        sector_to_remove = etf_sector_map[etf]
        dfs = dfs[dfs['sector'] != sector_to_remove]
        print(prt_str % etf)

    return dfs


def compute_trend(context, data):
    """
    Determine market trend and set exposure factors.

    Uses VIX signal to classify market regime and sets appropriate
    exposure multipliers for long and short positions.

    RISKBRAKE-1 note: with USE_VIXDATA_REGIME = False (default) the vixflag
    is always 0, so only the bullish branch below is reachable -- exactly
    the legacy behavior. The bearish branch becomes live again once the
    vixdata collection is verified and the switch enabled.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor

    Regime Classification
    ---------------------
    VIX signal <= 0 (Bullish):
        - vix_uptrend_flag = True
        - longfact = 1.5
        - shortfact = 0.45 or 0.9 (seasonal)
        - clip = 1.2

    VIX signal > 0 (Bearish):
        - vix_uptrend_flag = False
        - longfact = 0.0 (if SPY < MA80) or IWM weight
        - shortfact = 0.5
        - clip = 1.6
    """
    print('VIX flag:', context.vixflag)

    # Store previous value
    context.longfact_last = context.longfact

    if context.vixflag <= 0:
        # Bullish regime
        print('Trend mode: 1.5')
        context.vix_uptrend_flag = True
        context.longfact = 1.5

        # Seasonal short adjustment
        if algo.get_datetime().date().month in SHORT_RESTRICTED_MONTHS and context.bc1 == 0: #not context.spy_below80ma:
            context.shortfact = 0.45
        else:
            context.shortfact = 0.9
        context.clip = 1.2
    else:
        # Bearish regime
        print('Trend mode: 1')
        context.vix_uptrend_flag = False
        context.longfact = 0.0 if context.spy_below80ma else abs(context.iwm_w)
        context.shortfact = 0.5
        context.clip = 1.6

    return


def compute_beta(context, data):
    """
    Compute portfolio beta ratio versus benchmark.

    Calculates the ratio of long portfolio beta to short portfolio beta,
    used for adjusting hedge sizing.

    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor

    Returns
    -------
    float
        Beta ratio (long_beta / short_beta)

    Calculation
    -----------
    Uses exponentially weighted covariance over 120 days:
    beta = cov(portfolio_returns, benchmark_returns) / var(benchmark_returns)

    Notes
    -----
    Minimum return value is 1.3 to ensure adequate hedging.
    """
    benchmark_array = np.array([context.benchmarkSecurity])
    assets_array = np.concatenate((context.universe, benchmark_array))

    prices = data.history(assets_array, 'price', 120, '1d')

    prices_longs = prices[context.longs.index.intersection(prices.columns)]
    prices_shorts = prices[context.shorts.index.intersection(prices.columns)]
    prices_spy = prices[context.benchmarkSecurity]

    # Calculate portfolio returns (sum across positions)
    rets_long_port = prices_longs.pct_change().sum(axis=1)
    rets_short_port = prices_shorts.pct_change().sum(axis=1)
    rets_spy = prices_spy.pct_change()

    beta_span = 120

    # Calculate exponentially weighted covariances and variance
    long_cov = rets_long_port.ewm(span=beta_span, adjust=True).cov(rets_spy)
    short_cov = rets_short_port.ewm(span=beta_span, adjust=True).cov(rets_spy)
    bench_var = rets_spy.ewm(span=beta_span, adjust=True).var()

    # Calculate betas
    long_beta = long_cov.iloc[-1] / bench_var.iloc[-1]
    short_beta = short_cov.iloc[-1] / bench_var.iloc[-1]

    # Calculate ratio
    beta_ratio = long_beta / short_beta

    print("long_beta, short_beta, beta_ratio:", long_beta, short_beta, beta_ratio)

    return beta_ratio

def drop_top_vol_outliers(df, col='vol', n=2, k=1.5, verbose=True):
    """
    Drop the top-n rows of `df` ranked by `col`, but only if they exceed
    the upper fence (p90 + k*IDR), where IDR is the inter-decile range
    (p90 - p10). Rows within the normal range are kept.

    Using the 10th/90th percentiles instead of the 25th/75th widens the
    "normal" band -- useful when the middle 50% is too narrow to reflect
    the full spread of the data (common for fat-tailed series like vol).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Not modified in place.
    col : str, default 'vol'
        Column used to rank rows and compute the outlier threshold.
    n : int, default 2
        Number of top-ranked candidates to consider for removal.
    k : float, default 1.5
        Fence multiplier applied to the inter-decile range.
    verbose : bool, default True
        If True, print the fence, the dropped indexes, and their values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with qualifying outliers removed. The original `df`
        is untouched.
    """
    # Compute the 10th and 90th percentiles of the target column.
    # p10 = value below which 10% of observations fall
    # p90 = value below which 90% of observations fall
    p10, p90 = df[col].quantile([0.10, 0.90])

    # Inter-decile range -- the spread of the middle 80% of the data.
    # Wider than IQR, so the fence sits further out.
    idr = p90 - p10

    # Upper fence: threshold above which values are flagged as outliers.
    upper = p90 + k * idr

    # Index labels of the top-n candidates by `col` value.
    top_idx = df[col].nlargest(n).index

    # Keep only candidates that actually exceed the fence.
    drop_idx = [i for i in top_idx if df.loc[i, col] > upper]

    # Build the cleaned frame (leaves original untouched).
    cleaned = df.drop(drop_idx)

    if verbose:
        print(f"p10={p10:.4f}, p90={p90:.4f}, IDR={idr:.4f}")
        print(f"Fence ({col} > p90 + {k}*IDR) = {upper:.4f}")
        if drop_idx:
            print(f"Dropped {len(drop_idx)} outlier(s):")
            for idx in drop_idx:
                print(f"  index={idx!r}, {col}={df.loc[idx, col]:.4f}")
        else:
            print("Dropped 0 outlier(s) -- top values within normal range.")

    return cleaned
