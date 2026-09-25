"""Learned blends (stacking) vs fixed rules. Walk-forward, PIT-guarded, OOS 2018-26, top-400."""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
warnings.filterwarnings('ignore')
P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet', columns=['Date', 'Symbol', 'fwd90', 'mc_rank', 'p_mom_3m'])
P = P[P.Date >= '2015-01-01'].copy()
P['mkt3m'] = P.groupby('Date')['p_mom_3m'].transform('mean')          # regime: universe's trailing 3m return
def load(f):
    x = pd.read_csv(f); x['Date'] = pd.to_datetime(x['Date']); return x
g = load('experiments/FACTOR19_FORECAST/FACTOR19_MONTHLY_forecast_only.csv').rename(columns={'predicted_return': 'gbm'})
d = load('experiments/FACTOR19_FORECAST/DIRECT_EQ19_forecast_only.csv').rename(columns={'predicted_return': 'dir'})
mo = load('experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG_forecast_only.csv')
wk = load('experiments/PIT_BASE_WEEKLY_90d_from2023_LSEG/PIT_BASE_WEEKLY_90d_from2023_LSEG_forecast_only.csv')
ml = pd.concat([mo[mo.Date < '2023-01-03'], wk[wk.Date >= '2023-01-03']]).rename(columns={'predicted_return': 'mlf1'})
days = pd.DatetimeIndex(sorted(g.Date.unique()))
P['use'] = days[days.searchsorted(P['Date'].values, side='right').clip(max=len(days) - 1)]
for x in (g, d, ml):
    P = P.merge(x.rename(columns={'Date': 'use'}), on=['use', 'Symbol'], how='left')
P = P.dropna(subset=['gbm', 'dir', 'mlf1']).copy()
for c in ('gbm', 'dir', 'mlf1'): P['r_' + c] = P.groupby('Date')[c].rank(pct=True)
P['y_rank'] = P.groupby('Date')['fwd90'].rank(pct=True)
P['y_top'] = (P['y_rank'] >= 0.9).astype(int)
BASE = ['r_gbm', 'r_mlf1', 'r_dir']

months = sorted(P.Date.unique())
def walk(feats, kind):
    out = []
    for m in [x for x in months if x >= pd.Timestamp('2018-01-01')]:
        tr = P[(P.Date >= m - pd.DateOffset(months=60)) & (P.Date <= m - pd.DateOffset(days=135)) & P.fwd90.notna()]
        te = P[P.Date == m]
        if tr.Date.nunique() < 24: continue
        if kind == 'rank':
            mdl = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=200, l2_regularization=1.0,
                                                learning_rate=0.05, max_iter=150, random_state=0).fit(tr[feats], tr['y_rank'])
            p = mdl.predict(te[feats])
        else:
            mdl = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=200, l2_regularization=1.0,
                                                 learning_rate=0.05, max_iter=150, random_state=0).fit(tr[feats], tr['y_top'])
            p = mdl.predict_proba(te[feats])[:, 1]
        out.append(pd.Series(p, index=te.index))
    return pd.concat(out)
P['S_stack_rank'] = walk(BASE, 'rank')
P['S_stack_tail'] = walk(BASE, 'tail')
P['S_stack_tail_regime'] = walk(BASE + ['mkt3m'], 'tail')
P['S_max'] = P[['r_gbm', 'r_mlf1']].max(axis=1)
P['S_avg'] = P[['r_gbm', 'r_mlf1']].mean(axis=1)
P['fwd90c'] = P['fwd90'].clip(-0.95, 3.0)
O = P[(P.Date >= '2018-01-01') & P.fwd90.notna() & (P.mc_rank <= 400)]
print(f"{'method':<22}{'period':<9}{'IC':>8}{'top30':>9}{'top30 %+':>10}{'top50':>9}")
for s in ['S_avg', 'S_max', 'S_stack_rank', 'S_stack_tail', 'S_stack_tail_regime']:
    for lbl, a, b in [('2018-26', '2018', '2026-12-31'), ('2018-22', '2018', '2022-12-31'), ('2023-26', '2023', '2026-12-31')]:
        Q = O[(O.Date >= a) & (O.Date <= b) & O[s].notna()]
        ic = Q.groupby('Date').apply(lambda t: t[s].corr(t['fwd90'], method='spearman')).mean()
        e30 = np.array([t.nlargest(30, s)['fwd90c'].mean() - t['fwd90c'].mean() for _, t in Q.groupby('Date')])
        e50 = np.mean([t.nlargest(50, s)['fwd90c'].mean() - t['fwd90c'].mean() for _, t in Q.groupby('Date')])
        print(f"{s[2:] if lbl=='2018-26' else '':<22}{lbl:<9}{ic:>+8.4f}{100*e30.mean():>+8.2f}%{100*(e30>0).mean():>9.0f}%{100*e50:>+8.2f}%")
P[['Date', 'Symbol', 'S_stack_tail', 'S_stack_tail_regime', 'S_stack_rank']].to_parquet('experiments/FACTOR_STUDY/stack_scores.parquet', index=False)
