#!/usr/bin/env python3
"""
Build a comprehensive technical factor set from SHARADAR/SEP OHLCV.

WHY THIS EXISTS
---------------
~78% of the training inputs are quarterly fundamentals that update 4x/year and
are forward-filled in between. On most Tuesdays the model sees features nearly
identical to the previous Tuesday, which is a structural ceiling on a 20-day
forecast. Technical factors update every day. This script generates them so the
hypothesis can actually be measured (--technicals-only) rather than argued.

WHY SEP AND NOT THE LSEG CLOSE
------------------------------
The LSEG feed carries only RefPriceClose. Most of the indicator families --
ATR, Stochastic, ADX, Ichimoku, Donchian, Keltner, Ultimate Oscillator -- read
the high-low range and are simply uncomputable from a close alone. SEP gives
open/high/low/close/volume.

POINT-IN-TIME CORRECTNESS
-------------------------
Every indicator here is a backward-looking function of prices up to and
including date T -- rolling means, EMAs, ranges. None peeks forward. That is
inherent to the library (it is causal by construction), but the `--verify`
flag proves it empirically: scramble every price after a cutoff and confirm
that no indicator value dated before the cutoff changes.

NOTE ON `close` vs `closeadj`: indicators are computed on SPLIT/DIVIDEND-
ADJUSTED prices (closeadj), rescaling open/high/low by the same factor. Using
raw closes would inject artificial gaps at every corporate action.

Usage
-----
    python3 build_technical_features.py \
        --sep sharadar_raw/sharadar_sep_2022-01-01_2026-09-15.parquet \
        --out technical_features_2022_2026.parquet
    # then merge into the training input with --merge-into
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

MIN_BARS = 260          # a symbol needs ~1y of history for 200-day indicators
PREFIX = 't_'           # every generated column is prefixed, so --technicals-only
                        # and --fundamental-only can select on it

# NON-CAUSAL INDICATORS -- MUST BE EXCLUDED.
# Ichimoku's "visual" cloud spans are DELIBERATELY displaced forward in time:
# that forward projection is the whole point of the cloud on a chart, and `ta`
# reproduces it faithfully. In a feature matrix it is straight look-ahead.
# Empirically confirmed by the --verify scramble: these two are the only 2 of 86
# indicators whose PAST values change when FUTURE prices are altered
# (max diff 3.6e+00 and 1.4e+00 respectively; every other indicator: 0).
# The plain trend_ichimoku_a / trend_ichimoku_b are backward-looking and kept.
NON_CAUSAL = {'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b'}


def build_for_symbol(g, add_all):
    """All `ta` indicators for one symbol's OHLCV history."""
    g = g.sort_values('date', kind='stable')
    out = add_all(g[['open', 'high', 'low', 'close', 'volume']].copy(),
                  open='open', high='high', low='low', close='close',
                  volume='volume', fillna=False)
    new = [c for c in out.columns
           if c not in ('open', 'high', 'low', 'close', 'volume')
           and c not in NON_CAUSAL]
    res = out[new].copy()
    res.columns = [PREFIX + c.lower() for c in new]
    res['date'] = g['date'].values
    res['ticker'] = g['ticker'].iloc[0]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sep', required=True, help='SHARADAR/SEP parquet (OHLCV)')
    ap.add_argument('--out', required=True, help='output parquet of technical features')
    ap.add_argument('--start', help='drop output rows before this date (keep lookback)')
    ap.add_argument('--verify', action='store_true',
                    help='prove causality: scramble future prices, confirm past '
                         'indicator values are unchanged')
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"ABORT: {args.out} exists")
    try:
        from ta import add_all_ta_features
    except ImportError:
        sys.exit("`ta` not installed:  pip install ta")

    print("loading SEP...")
    sep = pd.read_parquet(args.sep, engine='pyarrow')
    sep['date'] = pd.to_datetime(sep['date'])
    sep['ticker'] = sep['ticker'].astype(str)

    # Use adjusted prices; rescale OHL by the same close adjustment factor so the
    # bar geometry stays internally consistent across splits.
    if 'closeadj' in sep.columns:
        fac = (sep['closeadj'] / sep['close'].replace(0, np.nan)).fillna(1.0)
        for c in ('open', 'high', 'low'):
            sep[c] = sep[c] * fac
        sep['close'] = sep['closeadj']
        print("  using split/dividend-adjusted prices (closeadj)")

    sep = sep.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    sep = sep[sep['close'] > 0]
    n_all = sep['ticker'].nunique()
    cnt = sep.groupby('ticker')['date'].transform('size')
    sep = sep[cnt >= MIN_BARS]
    print(f"  {len(sep):,} bars | {sep['ticker'].nunique():,} symbols with >= {MIN_BARS} bars "
          f"(dropped {n_all - sep['ticker'].nunique():,} too-short)")

    if args.verify:
        t = sep[sep['ticker'] == sep['ticker'].iloc[0]].copy()
        cut = t['date'].iloc[int(len(t) * 0.7)]
        alt = t.copy()
        fut = alt['date'] > cut
        alt.loc[fut, ['open', 'high', 'low', 'close']] = \
            alt.loc[fut, ['open', 'high', 'low', 'close']].sample(frac=1, random_state=0).values
        a = build_for_symbol(t, add_all_ta_features)
        b = build_for_symbol(alt, add_all_ta_features)
        cols = [c for c in a.columns if c.startswith(PREFIX)]
        m = a[a['date'] < cut][cols].reset_index(drop=True)
        n = b[b['date'] < cut][cols].reset_index(drop=True)
        diff = (m - n).abs().max().max()
        print(f"\n  CAUSALITY: scrambled every price after {cut:%Y-%m-%d}; "
              f"max change in earlier indicator values = {diff:.2e}")
        print(f"  -> {'PASS (all indicators are backward-looking)' if diff < 1e-8 else 'FAIL'}\n")
        if diff >= 1e-8:
            sys.exit("causality check failed - not writing output")

    print("building indicators per symbol...")
    res, done = [], 0
    for tkr, g in sep.groupby('ticker', sort=False):
        try:
            res.append(build_for_symbol(g, add_all_ta_features))
        except Exception:
            pass                        # a pathological series should not kill the run
        done += 1
        if done % 400 == 0:
            print(f"  {done:,} symbols", flush=True)

    tech = pd.concat(res, ignore_index=True)
    feat = [c for c in tech.columns if c.startswith(PREFIX)]
    # drop all-NaN and zero-variance columns: they cost memory and teach nothing
    keep = [c for c in feat if tech[c].notna().any() and tech[c].std(skipna=True) > 0]
    tech = tech[['ticker', 'date'] + keep]
    if args.start:
        tech = tech[tech['date'] >= pd.Timestamp(args.start)]

    # float32: these are derived indicators, not accounting figures. float64
    # doubles the file and the training-time memory for ~7 digits of precision
    # nobody uses. Values outside float32 range are clipped to avoid inf.
    f32max = np.finfo(np.float32).max
    for c in keep:
        tech[c] = tech[c].clip(-f32max, f32max).astype('float32')

    tech.to_parquet(args.out, engine='pyarrow', compression='snappy', index=False)
    print(f"\ndone: {len(tech):,} rows x {len(keep)} indicators -> {args.out} "
          f"({os.path.getsize(args.out)/1e6:.0f} MB)")
    print(f"  dropped {len(feat) - len(keep)} empty/constant columns")
    print(f"  date range {tech['date'].min():%Y-%m-%d} .. {tech['date'].max():%Y-%m-%d}")


if __name__ == '__main__':
    main()
