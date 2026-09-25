#!/usr/bin/env python3
"""
Daily forecast files from the 19-factor model, for backtesting in the algo.

Outputs (experiments/FACTOR19_FORECAST/), both Symbol,Date,predicted_return:
  FACTOR19_MONTHLY_forecast_only.csv   the 19-factor GBM alone
  BLEND_F19_MLF1_forecast_only.csv     50/50 rank blend with the honest mlf1
                                       (weekly PIT file from 2023, monthly PIT before)

Model: HistGBR on the 19 factors selected on 2010-2017 (+ sector), cross-
sectional rank features, rank target. Retrained each month-end on a 60-month
window; PIT guard = training snapshots at least 135 calendar days old, so every
90-trading-day target has realised. A forecast made at month-end close is
applied to the trading days AFTER it, through the next month-end.

Scale: the algo sizes with mlf1_crzsoft ** 1.2, which is scale-sensitive, so
per-date ranks are quantile-mapped onto the production mlf1 distribution.
Ranking is untouched; only the units match.
"""
import numpy as np, pandas as pd, warnings, os, sys
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
OUTD = 'experiments/FACTOR19_FORECAST'
START = pd.Timestamp('2015-01-01')

P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet')
P = P[P['Date'] >= '2010-01-01'].copy()
KEEP = pd.read_csv('experiments/FACTOR_STUDY/selected_factors_2010_2017.csv').iloc[:, 0].tolist()
assert len(KEEP) == 19, KEEP
P['sector'] = P['GICSSectorName'].astype('category').cat.codes
P[KEEP] = P.groupby('Date')[KEEP].rank(pct=True)
P['y'] = P.groupby('Date')['fwd90'].rank(pct=True)
FEAT = KEEP + ['sector']

snaps = sorted(P['Date'].unique())
preds = []
for m in [s for s in snaps if s >= START - pd.DateOffset(months=1)]:
    tr = P[(P['Date'] >= m - pd.DateOffset(months=60)) & (P['Date'] <= m - pd.DateOffset(days=135)) & P['fwd90'].notna()]
    te = P[P['Date'] == m]
    if len(tr) < 5000: continue
    mdl = HistGradientBoostingRegressor(max_depth=6, min_samples_leaf=100, l2_regularization=0.2,
                                        learning_rate=0.05, max_iter=300, random_state=0)
    mdl.fit(tr[FEAT], tr['y'])
    preds.append(pd.DataFrame({'snap': m, 'Symbol': te['Symbol'].values, 'score': mdl.predict(te[FEAT])}))
    print(f'  {pd.Timestamp(m):%Y-%m}: trained {len(tr):,} -> scored {len(te):,}', flush=True)
S = pd.concat(preds)

# ---- carry each month-end forecast onto the trading days after it
src = 'experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/input_PIT_BASE_FULLHIST.parquet'
dd = pd.read_parquet(src, columns=['Date'], engine='pyarrow')['Date']
cnt = pd.to_datetime(dd).value_counts()
days = pd.DatetimeIndex(sorted(cnt[cnt >= 1500].index))
sn = sorted(S['snap'].unique())
parts = []
for i, s in enumerate(sn):
    nxt = sn[i + 1] if i + 1 < len(sn) else days.max() + pd.Timedelta(days=1)
    win = days[(days > s) & (days <= nxt)]
    if len(win) == 0: continue
    g = S[S['snap'] == s]
    parts.append(pd.DataFrame({'Date': np.repeat(win.values, len(g)),
                               'Symbol': np.tile(g['Symbol'].values, len(win)),
                               'score': np.tile(g['score'].values, len(win))}))
D = pd.concat(parts, ignore_index=True)
D = D[D['Date'] >= START]

# ---- quantile-map onto the production mlf1 distribution
wk = pd.read_csv('experiments/PIT_BASE_WEEKLY_90d_from2023_LSEG/PIT_BASE_WEEKLY_90d_from2023_LSEG_forecast_only.csv')
qs = np.linspace(0, 1, 1001)
ref = np.quantile(wk['predicted_return'].dropna().values, qs)
def to_mlf1_units(df, col):
    r = df.groupby('Date')[col].rank(pct=True, method='average')
    return np.interp(r, qs, ref)

D['predicted_return'] = to_mlf1_units(D, 'score')
out1 = f'{OUTD}/FACTOR19_MONTHLY_forecast_only.csv'
D[['Symbol', 'Date', 'predicted_return']].to_csv(out1, index=False)

# ---- blend with honest mlf1: weekly PIT from 2023, monthly PIT before
mo = pd.read_csv('experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG_forecast_only.csv')
for x in (mo, wk): x['Date'] = pd.to_datetime(x['Date'])
ml = pd.concat([mo[mo['Date'] < '2023-01-03'], wk[wk['Date'] >= '2023-01-03']])
ml = ml.rename(columns={'predicted_return': 'mlf1'})
B = D[['Date', 'Symbol', 'score']].merge(ml, on=['Date', 'Symbol'], how='outer')
B = B[B['Date'] >= START]
ra = B.groupby('Date')['score'].rank(pct=True); rb = B.groupby('Date')['mlf1'].rank(pct=True)
B['blend'] = pd.concat([ra, rb], axis=1).mean(axis=1)     # uses whichever exists if one is missing
B['predicted_return'] = to_mlf1_units(B, 'blend')
out2 = f'{OUTD}/BLEND_F19_MLF1_forecast_only.csv'
B[['Symbol', 'Date', 'predicted_return']].dropna().to_csv(out2, index=False)

for f in (out1, out2):
    x = pd.read_csv(f); x['Date'] = pd.to_datetime(x['Date'])
    print(f'\n{f}\n  {len(x):,} rows  {x.Symbol.nunique():,} symbols  {x.Date.min():%Y-%m-%d} -> {x.Date.max():%Y-%m-%d}'
          f'\n  predicted_return mean {x.predicted_return.mean():+.2f}  std {x.predicted_return.std():.2f}'
          f'   (production mlf1: mean {wk.predicted_return.mean():+.2f}  std {wk.predicted_return.std():.2f})')
