#!/usr/bin/env python3
"""
Build a 2025+ training input from the PRODUCTION parquet, plus the 77 technical
indicators.

Purpose: measure what the technical set is worth in the configuration that
actually runs live -- production's own data (LSEG intact, NOT substituted),
production's 90d/1d horizon, weekly Tuesday walk -- predicting 2026 only, with
2025 as the lookback year. That makes the output directly comparable to the
live weekly backtest.

TWO COLUMNS MUST BE DROPPED
---------------------------
The source is a production OUTPUT file, so it carries `predicted_return` and
`forward_return` from the previous run. Neither is in the script's exclude list:
  - predicted_return would become a FEATURE, handing the model its own earlier
    (leak-contaminated) forecasts.
  - forward_return is the target and would be recomputed anyway, but a stale
    copy sitting in the frame is an unnecessary hazard.
Both are removed here rather than relied on to be filtered downstream.
"""
import argparse, os, sys
import numpy as np, pandas as pd

DROP = ['predicted_return', 'forward_return']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prod', required=True, help='production predictions parquet')
    ap.add_argument('--tech', required=True, help='technical indicator parquet')
    ap.add_argument('--start', default='2025-01-01')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        sys.exit(f"ABORT: {args.output} exists")

    print(f"loading {args.prod} ...")
    df = pd.read_parquet(args.prod, engine='pyarrow')
    df['Date'] = pd.to_datetime(df['Date'])
    n0 = len(df)
    df = df[df['Date'] >= pd.Timestamp(args.start)]
    print(f"  {n0:,} -> {len(df):,} rows from {args.start}")

    gone = [c for c in DROP if c in df.columns]
    df = df.drop(columns=gone)
    print(f"  dropped prior-run outputs: {gone}")

    t = pd.read_parquet(args.tech, engine='pyarrow')
    t['date'] = pd.to_datetime(t['date'])
    feats = [c for c in t.columns if c.startswith('t_')]
    dup = set(feats) & set(df.columns)
    if dup:
        sys.exit(f"ABORT: indicator names already present: {sorted(dup)[:5]}")
    t = t.drop_duplicates(subset=['ticker', 'date'], keep='first')
    print(f"  tech {len(t):,} rows x {len(feats)} indicators")

    # Case-fold for the Sharadar join only; the LSEG Symbol keeps its case so
    # share classes (BRKa vs BRKA) stay distinct in the training data.
    df['_k'] = df['Symbol'].astype(str).str.upper()
    t = t.rename(columns={'ticker': '_k', 'date': 'Date'})
    t['_k'] = t['_k'].astype(str).str.upper()

    before = len(df)
    df = df.merge(t[['_k', 'Date'] + feats], on=['_k', 'Date'], how='left')
    assert len(df) == before, f"merge changed rows {before} -> {len(df)}"
    cov = df[feats[0]].notna().mean()
    df = df.drop(columns=['_k'])

    df.to_parquet(args.output, engine='pyarrow', compression='snappy', index=False)
    print(f"\ndone: {len(df):,} rows x {df.shape[1]} cols -> {args.output} "
          f"({os.path.getsize(args.output)/1e9:.2f} GB)")
    print(f"  date range          : {df['Date'].min():%Y-%m-%d} .. {df['Date'].max():%Y-%m-%d}")
    print(f"  indicator coverage  : {100*cov:.1f}%")
    print(f"  LSEG intact (PriceTarget_Median non-null): "
          f"{100*df['PriceTarget_Median'].notna().mean():.1f}%"
          if 'PriceTarget_Median' in df.columns else '')


if __name__ == '__main__':
    main()
