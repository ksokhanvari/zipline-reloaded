#!/usr/bin/env python3
"""
MAX-RANK blend of the 19-factor GBM with the LIVE PRODUCTION mlf1 -- no retraining.

  score = max(pct-rank 19-factor GBM, pct-rank production mlf1)  per trading day

Production source: MLData/20091231_20260922_forecast_only.csv (what the algo loads
today). Its 2026 predictions from 2026-01-26 are genuinely live; January 2026 and
earlier come from rebuilds (leaky), so only 2026 is written.

Scale: quantile-mapped onto the production file's OWN 2026 distribution, so the
algo's mlf1_crzsoft ** 1.2 sizing sees the units it sees live.
"""
import numpy as np, pandas as pd
START = '2026-01-02'
PROD = 'MLData/20091231_20260922_forecast_only.csv'
OUT = 'experiments/FACTOR19_FORECAST/MAXRANK_F19_PRODLIVE_2026_forecast_only.csv'
def load(f):
    x = pd.read_csv(f); x['Date'] = pd.to_datetime(x['Date']); return x[x['Date'] >= START]
g = load('experiments/FACTOR19_FORECAST/FACTOR19_MONTHLY_forecast_only.csv').rename(columns={'predicted_return': 'gbm'})
p = load(PROD).rename(columns={'predicted_return': 'mlf1'})
B = g.merge(p, on=['Date', 'Symbol'], how='outer')
ra = B.groupby('Date')['gbm'].rank(pct=True); rb = B.groupby('Date')['mlf1'].rank(pct=True)
B['score'] = pd.concat([ra, rb], axis=1).max(axis=1)          # max ignores a missing side
qs = np.linspace(0, 1, 1001); ref = np.quantile(p['mlf1'].dropna().values, qs)
B['predicted_return'] = np.interp(B.groupby('Date')['score'].rank(pct=True), qs, ref)
B = B.dropna(subset=['predicted_return'])
B[['Symbol', 'Date', 'predicted_return']].sort_values(['Date', 'Symbol']).to_csv(OUT, index=False)
both = B['gbm'].notna() & B['mlf1'].notna()
print(f"{OUT}\n  {len(B):,} rows  {B.Symbol.nunique():,} symbols  {B.Date.min():%Y-%m-%d} -> {B.Date.max():%Y-%m-%d}")
print(f"  rows with both models: {100*both.mean():.0f}%   production-only: {100*(B.gbm.isna()).mean():.0f}%   gbm-only: {100*(B.mlf1.isna()).mean():.0f}%")
print(f"  predicted_return mean {B.predicted_return.mean():+.2f} std {B.predicted_return.std():.2f}"
      f"   (production 2026: mean {p.mlf1.mean():+.2f} std {p.mlf1.std():.2f})")
d = B[B.Date == B.Date.max()]
print(f"  latest day: {len(d):,} names; top by blend: {', '.join(d.nlargest(8, 'predicted_return')['Symbol'])}")
