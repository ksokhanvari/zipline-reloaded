#!/usr/bin/env python3
"""
Daily-data study inputs:
  (1) full-history close-based TECHNICAL factors (split-corrected), sampled at
      month-ends -> merged onto the monthly panel as t_* candidates;
  (2) STALENESS test: each daily-varying factor measured at T and at T-21
      trading days, so we can see how much a month-old value loses.
Everything is backward-looking; target is the panel's fwd90 (T+1 -> T+91).
"""
import numpy as np, pandas as pd
SRC = 'experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/input_PIT_BASE_FULLHIST.parquet'
LS = ['EnterpriseValueToSales_DailyTimeSeriesRatio_', 'EnterpriseValueToEBIT_DailyTimeSeriesRatio_',
      'EnterpriseValueToEBITDA_DailyTimeSeriesRatio_', 'CombinedAlphaModelSectorRank',
      'EarningsQualityRegionRank_Current', 'Recommendation_Median_1_5_']
d = pd.read_parquet(SRC, columns=['Date', 'Symbol', 'RefPriceClose', 'RefVolume', 'CompanyMarketCap',
                                  'FOCFExDividends_Discrete', 'EarningsPerShare_Actual', 'GICSSectorName'] + LS,
                    engine='pyarrow')
d['Date'] = pd.to_datetime(d['Date'])
cnt = d.groupby('Date').size(); real = set(cnt[cnt >= 1500].index)
d = d[d['Date'].isin(real)].sort_values(['Symbol', 'Date'], kind='stable').reset_index(drop=True)
g = d.groupby('Symbol', sort=False)
r = g['RefPriceClose'].pct_change(); mr = g['CompanyMarketCap'].pct_change()
r = r.where(~(((r < -0.45) | (r > 0.8)) & (mr.abs() < 0.15)), mr).clip(-0.9, 5.0)
d['r'] = r.fillna(0.0)
d['idx'] = d.groupby('Symbol', sort=False)['r'].transform(lambda s: (1 + s).cumprod())
g = d.groupby('Symbol', sort=False)
roll = lambda col, w, fn, mp: g[col].transform(lambda s: getattr(s.rolling(w, min_periods=mp), fn)())
I = d['idx']
print('technicals ...', flush=True)
# RSI (Wilder-style via EWM on gains/losses)
up = d['r'].clip(lower=0); dn = (-d['r']).clip(lower=0)
for n in (14, 28):
    au = up.groupby(d['Symbol'], sort=False).transform(lambda s: s.ewm(alpha=1/n, min_periods=n).mean())
    ad = dn.groupby(d['Symbol'], sort=False).transform(lambda s: s.ewm(alpha=1/n, min_periods=n).mean())
    d[f't_rsi{n}'] = 100 - 100 / (1 + au / ad.replace(0, np.nan))
ma = {n: roll('idx', n, 'mean', int(n*0.8)) for n in (20, 50, 200)}
d['t_px_ma50'] = I / ma[50] - 1
d['t_px_ma200'] = I / ma[200] - 1
d['t_ma50_ma200'] = ma[50] / ma[200] - 1
d['t_ma20_ma50'] = ma[20] / ma[50] - 1
vol60 = roll('r', 60, 'std', 40); vol252 = roll('r', 252, 'std', 200)
mom12 = g['idx'].shift(21) / g['idx'].shift(252) - 1
d['t_mom12_voladj'] = mom12 / (vol252 * np.sqrt(252))
d['t_mom6_voladj'] = (g['idx'].shift(21) / g['idx'].shift(126) - 1) / (vol60 * np.sqrt(252))
d['t_dd_252'] = I / roll('idx', 252, 'max', 200) - 1
d['t_dd_63'] = I / roll('idx', 63, 'max', 50) - 1
d['t_skew_60'] = roll('r', 60, 'skew', 40)
d['t_downvol_60'] = d['r'].clip(upper=0).groupby(d['Symbol'], sort=False).transform(lambda s: s.rolling(60, min_periods=40).std())
d['t_vol_ratio'] = roll('r', 20, 'std', 15) / vol60
dv = d['RefPriceClose'] * d['RefVolume']
d['_dv'] = dv
d['t_dvol_trend'] = d.groupby('Symbol', sort=False)['_dv'].transform(lambda s: s.rolling(20, 15).mean() / s.rolling(120, 90).mean()) - 1
d['_amihud'] = (d['r'].abs() / dv.replace(0, np.nan))
d['t_amihud_60'] = np.log(d.groupby('Symbol', sort=False)['_amihud'].transform(lambda s: s.rolling(60, 40).mean()) + 1e-12)
d['t_rev_5d'] = I / g['idx'].shift(5) - 1
ema12 = g['idx'].transform(lambda s: s.ewm(span=12, min_periods=12).mean())
ema26 = g['idx'].transform(lambda s: s.ewm(span=26, min_periods=26).mean())
macd = (ema12 - ema26) / I
d['_macd'] = macd
d['t_macd_hist'] = macd - d.groupby('Symbol', sort=False)['_macd'].transform(lambda s: s.ewm(span=9, min_periods=9).mean())
d['t_up_days_60'] = (d['r'] > 0).astype(float).groupby(d['Symbol'], sort=False).transform(lambda s: s.rolling(60, 40).mean())
TECH = [c for c in d.columns if c.startswith('t_')]

# ---- daily-varying factors for the staleness test (same definitions as the panel)
px = d['RefPriceClose']; mc = d['CompanyMarketCap']
d['s_focf_mc'] = d['FOCFExDividends_Discrete'] / mc.where(mc > 0)
d['s_trail_eps_px'] = d['EarningsPerShare_Actual'] / px.where(px > 0)
d['s_mom_12_1'] = mom12
d['s_high52'] = I / roll('idx', 252, 'max', 200)
for c in LS: d['s_' + c] = d[c]
fin = d['GICSSectorName'] == 'Financials'
for c in ['s_' + x for x in LS if x.startswith('EnterpriseValue')]: d.loc[fin, c] = np.nan
STALE = [c for c in d.columns if c.startswith('s_')]
for c in STALE:
    d[c + '__lag21'] = g[c].shift(21)          # the value a month-held forecast would be using

me = pd.Series(sorted(real)); me = me.groupby(me.dt.to_period('M')).max()
snap = d[d['Date'].isin(set(me.values))][['Date', 'Symbol'] + TECH + STALE + [c + '__lag21' for c in STALE]]
snap.to_parquet('experiments/FACTOR_STUDY/daily_extras.parquet', index=False)
print(f'done: {len(snap):,} rows, {len(TECH)} technicals, {len(STALE)} staleness factors', flush=True)
