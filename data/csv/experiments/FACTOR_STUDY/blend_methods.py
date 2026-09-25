"""Compare ways of combining GBM19, mlf1 and the direct score, top-400, OOS."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet', columns=['Date', 'Symbol', 'fwd90', 'mc_rank'])
P = P[P.fwd90.notna() & (P.Date >= '2015-01-01') & (P.mc_rank <= 400)].copy()
def load(f):
    x = pd.read_csv(f); x['Date'] = pd.to_datetime(x['Date']); return x
g = load('experiments/FACTOR19_FORECAST/FACTOR19_MONTHLY_forecast_only.csv').rename(columns={'predicted_return': 'gbm'})
d = load('experiments/FACTOR19_FORECAST/DIRECT_EQ19_forecast_only.csv').rename(columns={'predicted_return': 'dir'})
mo = load('experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG_forecast_only.csv')
wk = load('experiments/PIT_BASE_WEEKLY_90d_from2023_LSEG/PIT_BASE_WEEKLY_90d_from2023_LSEG_forecast_only.csv')
ml = pd.concat([mo[mo.Date < '2023-01-03'], wk[wk.Date >= '2023-01-03']]).rename(columns={'predicted_return': 'mlf1'})
days = pd.DatetimeIndex(sorted(g.Date.unique()))
P['use'] = days[days.searchsorted(P['Date'].values, side='right').clip(max=len(days) - 1)]
for x, c in [(g, 'gbm'), (d, 'dir'), (ml, 'mlf1')]:
    P = P.merge(x.rename(columns={'Date': 'use'}), on=['use', 'Symbol'], how='left')
P = P.dropna(subset=['gbm', 'dir', 'mlf1'])
for c in ('gbm', 'dir', 'mlf1'):
    P['r_' + c] = P.groupby('Date')[c].rank(pct=True)
G, M, D = P['r_gbm'], P['r_mlf1'], P['r_dir']
B = {'gbm alone': G, 'mlf1 alone': M, 'direct alone': D,
     'avg 50/50 (current)': (G + M) / 2, 'avg 70gbm/30mlf1': .7*G + .3*M, 'avg 30gbm/70mlf1': .3*G + .7*M,
     'three-way avg': (G + M + D) / 3, 'gbm+direct avg': (G + D) / 2,
     'min-rank (AND)': np.minimum(G, M), 'max-rank (OR)': np.maximum(G, M),
     'rank product': np.sqrt(G * M), 'min-rank 3-way': np.minimum(np.minimum(G, M), D)}
# screen-then-pick: top-120 by mlf1, ordered by gbm (others pushed below)
P['_s'] = np.where(P.groupby('Date')['mlf1'].rank(ascending=False) <= 120, 1 + G, G)
B['screen mlf1->gbm'] = P['_s']
P['_s2'] = np.where(P.groupby('Date')['gbm'].rank(ascending=False) <= 120, 1 + M, M)
B['screen gbm->mlf1'] = P['_s2']
# adaptive: weight by trailing 12m realised IC (target realised => lag 5 months)
ic = P.groupby('Date').apply(lambda s: pd.Series({c: s['r_' + c].corr(s['fwd90'], method='spearman') for c in ('gbm', 'mlf1')}))
tw = ic.shift(5).rolling(12, min_periods=6).mean().clip(lower=0)
w = tw.div(tw.sum(axis=1), axis=0).fillna(0.5)
P = P.merge(w.rename(columns={'gbm': 'wg', 'mlf1': 'wm'}), left_on='Date', right_index=True, how='left')
B['adaptive IC-weight'] = P['wg'].fillna(.5) * G + P['wm'].fillna(.5) * M
P['fwd90c'] = P['fwd90'].clip(-0.95, 3.0)
print(f"{'method':<22}{'period':<9}{'IC':>9}{'ICIR':>7}{'%+':>6}{'top30':>9}{'top50':>9}{'top30 %+':>10}")
for name, s in B.items():
    P['_x'] = s.values if hasattr(s, 'values') else s
    for lbl, a, b in [('2018-26', '2018-01-01', '2026-12-31'), ('2023-26', '2023-01-01', '2026-12-31')]:
        Q = P[(P.Date >= a) & (P.Date <= b)]
        i = Q.groupby('Date').apply(lambda t: t['_x'].corr(t['fwd90'], method='spearman'))
        e30 = np.array([t.nlargest(30, '_x')['fwd90c'].mean() - t['fwd90c'].mean() for _, t in Q.groupby('Date')])
        e50 = np.mean([t.nlargest(50, '_x')['fwd90c'].mean() - t['fwd90c'].mean() for _, t in Q.groupby('Date')])
        print(f"{name if lbl=='2018-26' else '':<22}{lbl:<9}{i.mean():>+9.4f}{i.mean()/i.std():>7.2f}{100*(i>0).mean():>5.0f}%"
              f"{100*e30.mean():>+8.2f}%{100*e50:>+8.2f}%{100*(e30>0).mean():>9.0f}%")
