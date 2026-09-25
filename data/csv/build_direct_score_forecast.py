#!/usr/bin/env python3
"""
Forecast file from a DIRECT (non-ML) cross-sectional score of the 19 factors.

Score: equal-weight average of signed, centred cross-sectional ranks
(missing factor -> neutral). Signs fixed from 2010-2017 ICs, so 2018+ is
out-of-sample; 2010-2017 is in-sample for the signs and will look flattered.
No model is trained, so the file can start in 2010.

Same delivery as the GBM file: month-end score carried onto the trading days
after it, then quantile-mapped onto the production mlf1 distribution so the
algo's mlf1_crzsoft ** 1.2 sizing sees familiar units.
"""
import numpy as np, pandas as pd, sys
VAR = sys.argv[1] if len(sys.argv) > 1 else 'S_eq19'
S = pd.read_parquet('experiments/FACTOR_STUDY/direct_scores.parquet', columns=['Date', 'Symbol', VAR]).dropna()
S = S.rename(columns={'Date': 'snap', VAR: 'score'})
dd = pd.read_parquet('experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/input_PIT_BASE_FULLHIST.parquet',
                     columns=['Date'], engine='pyarrow')['Date']
cnt = pd.to_datetime(dd).value_counts()
days = pd.DatetimeIndex(sorted(cnt[cnt >= 1500].index))
sn = sorted(S['snap'].unique()); parts = []
for i, s in enumerate(sn):
    nxt = sn[i + 1] if i + 1 < len(sn) else days.max() + pd.Timedelta(days=1)
    win = days[(days > s) & (days <= nxt)]
    g = S[S['snap'] == s]
    if len(win) == 0 or len(g) == 0: continue
    parts.append(pd.DataFrame({'Date': np.repeat(win.values, len(g)), 'Symbol': np.tile(g['Symbol'].values, len(win)),
                               'score': np.tile(g['score'].values, len(win))}))
D = pd.concat(parts, ignore_index=True)
D = D[D['Date'] >= '2010-01-01']
wk = pd.read_csv('experiments/PIT_BASE_WEEKLY_90d_from2023_LSEG/PIT_BASE_WEEKLY_90d_from2023_LSEG_forecast_only.csv')
qs = np.linspace(0, 1, 1001); ref = np.quantile(wk['predicted_return'].dropna().values, qs)
D['predicted_return'] = np.interp(D.groupby('Date')['score'].rank(pct=True), qs, ref)
out = f'experiments/FACTOR19_FORECAST/DIRECT_{VAR[2:].upper()}_forecast_only.csv'
D[['Symbol', 'Date', 'predicted_return']].to_csv(out, index=False)
print(f'{out}\n  {len(D):,} rows  {D.Symbol.nunique():,} symbols  {D.Date.min():%Y-%m-%d} -> {D.Date.max():%Y-%m-%d}'
      f'\n  predicted_return mean {D.predicted_return.mean():+.2f} std {D.predicted_return.std():.2f}')
