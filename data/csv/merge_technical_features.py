#!/usr/bin/env python3
"""Merge the pre-built technical indicator parquet into a training input.

The indicators are keyed (ticker, date) and were computed from SHARADAR/SEP
OHLCV. They are backward-looking by construction and the build step verified
that empirically (scramble the future, past values unchanged), so an exact
same-day join introduces no look-ahead: an indicator dated T uses only prices
up to T, and the model reading it at T is reading history.

Symbol case is preserved (LSEG marks share classes with a lowercase suffix,
e.g. BRKa vs BRKA); the Sharadar side is folded for matching only.
"""
import argparse, os, sys
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True)
    ap.add_argument('--tech', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        sys.exit(f"ABORT: {args.output} exists")

    print("loading...")
    df = pd.read_parquet(args.input, engine='pyarrow')
    dcol = 'date' if 'date' in df.columns else 'Date'
    scol = 'symbol' if 'symbol' in df.columns else 'Symbol'
    df[dcol] = pd.to_datetime(df[dcol])
    print(f"  input {len(df):,} rows x {df.shape[1]} cols")

    t = pd.read_parquet(args.tech, engine='pyarrow')
    t['date'] = pd.to_datetime(t['date'])
    feats = [c for c in t.columns if c.startswith('t_')]
    print(f"  tech  {len(t):,} rows x {len(feats)} indicators")

    dup = set(feats) & set(df.columns)
    if dup:
        sys.exit(f"ABORT: {len(dup)} indicator names already exist in the input: "
                 f"{sorted(dup)[:6]}")

    t = t.drop_duplicates(subset=['ticker', 'date'], keep='first')
    df['_k'] = df[scol].astype(str).str.upper()
    t = t.rename(columns={'ticker': '_k'})
    t['_k'] = t['_k'].astype(str).str.upper()
    t = t.rename(columns={'date': dcol})

    before = len(df)
    df = df.merge(t[['_k', dcol] + feats], on=['_k', dcol], how='left')
    assert len(df) == before, f"merge changed row count {before} -> {len(df)}"
    cov = df[feats[0]].notna().mean()
    df = df.drop(columns=['_k'])

    df.to_parquet(args.output, engine='pyarrow', compression='snappy', index=False)
    print(f"\ndone: {len(df):,} rows x {df.shape[1]} cols -> {args.output} "
          f"({os.path.getsize(args.output)/1e9:.2f} GB)")
    print(f"  indicator coverage: {100*cov:.1f}% of rows")
    if cov < 0.70:
        print("  WARNING: low coverage - check ticker namespaces before training")


if __name__ == '__main__':
    main()
