#!/usr/bin/env python3
"""
MAX-RANK (OR) blend of the 19-factor GBM and the honest mlf1, per trading day:
score = max(pct-rank GBM, pct-rank mlf1). A name ranks high if EITHER model
ranks it high, so the top-N is the union of both models' strongest picks.
OOS 2018-26, top-400: top-30 excess +5.21%/90d vs +3.49% for the 50/50 average.
mlf1 = weekly PIT file from 2023-01-03, monthly PIT file before.
Quantile-mapped onto the production mlf1 distribution (sizing units unchanged).
"""
import numpy as np, pandas as pd
def load(f):
    x = pd.read_csv(f); x['Date'] = pd.to_datetime(x['Date']); return x
g = load('experiments/FACTOR19_FORECAST/FACTOR19_MONTHLY_forecast_only.csv').rename(columns={'predicted_return': 'gbm'})
mo = load('experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG_forecast_only.csv')
wk = load('experiments/PIT_BASE_WEEKLY_90d_from2023_LSEG/PIT_BASE_WEEKLY_90d_from2023_LSEG_forecast_only.csv')
ml = pd.concat([mo[mo.Date < '2023-01-03'], wk[wk.Date >= '2023-01-03']]).rename(columns={'predicted_return': 'mlf1'})
B = g.merge(ml, on=['Date', 'Symbol'], how='outer')
B = B[B['Date'] >= g['Date'].min()]
ra = B.groupby('Date')['gbm'].rank(pct=True); rb = B.groupby('Date')['mlf1'].rank(pct=True)
B['score'] = pd.concat([ra, rb], axis=1).max(axis=1)      # max ignores a missing side
qs = np.linspace(0, 1, 1001); ref = np.quantile(wk['predicted_return'].dropna().values, qs)
B['predicted_return'] = np.interp(B.groupby('Date')['score'].rank(pct=True), qs, ref)
out = 'experiments/FACTOR19_FORECAST/MAXRANK_F19_MLF1_forecast_only.csv'
B[['Symbol', 'Date', 'predicted_return']].dropna().to_csv(out, index=False)
print(f'{out}\n  {len(B):,} rows  {B.Symbol.nunique():,} symbols  {B.Date.min():%Y-%m-%d} -> {B.Date.max():%Y-%m-%d}'
      f'\n  predicted_return mean {B.predicted_return.mean():+.2f} std {B.predicted_return.std():.2f}')
