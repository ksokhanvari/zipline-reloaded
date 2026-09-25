"""Univariate IC scoring of every candidate factor in the PIT panel."""
import numpy as np, pandas as pd
P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet', engine='pyarrow')
P = P[(P['Date'] >= '2010-01-01') & P['fwd90'].notna()].copy()
FAC = [c for c in P.columns if c[:2] in ('p_', 'l_', 'f_')]
print(f"{P['Date'].nunique()} months {P['Date'].min():%Y-%m}..{P['Date'].max():%Y-%m}, {len(FAC)} factors")

def ic_series(D, neutral=False):
    """per-date Spearman IC for all factors at once (rank -> pearson)."""
    R = D.groupby('Date')[FAC + ['fwd90']].rank(pct=True)
    if neutral:
        R = R - R.groupby([D['Date'], D['GICSSectorName']]).transform('mean')
    else:
        R = R - R.groupby(D['Date']).transform('mean')
    y = R['fwd90']
    out = {}
    for c in FAC:
        x = R[c]; m = x.notna() & y.notna()
        num = (x[m] * y[m]).groupby(D.loc[m, 'Date']).sum()
        den = np.sqrt((x[m]**2).groupby(D.loc[m, 'Date']).sum() * (y[m]**2).groupby(D.loc[m, 'Date']).sum())
        cnt = m.groupby(D['Date']).sum()
        s = (num / den)[cnt[num.index] >= 50]
        out[c] = s
    return pd.DataFrame(out)

def nw_t(s, L=4):
    a = s.values - s.mean(); n = len(a); v0 = (a*a).sum()/n
    tot = sum((1-k/(L+1))*((a[k:]*a[:-k]).sum()/n/v0) for k in range(1, L+1))
    return s.mean() / (s.std(ddof=1)*np.sqrt(1+2*tot)/np.sqrt(n))

def summarise(IC, tag):
    rows = []
    for c in FAC:
        s = IC[c].dropna()
        if len(s) < 60: continue
        h1 = s[s.index < '2018-01-01']; h2 = s[s.index >= '2018-01-01']
        yrs = s.groupby(s.index.year).mean()
        sign = np.sign(s.mean())
        rows.append(dict(factor=c, n=len(s), ic=s.mean(), icir=s.mean()/s.std(ddof=1), t=nw_t(s),
                         ic_h1=h1.mean(), ic_h2=h2.mean(),
                         yrs_same=(np.sign(yrs) == sign).mean()))
    return pd.DataFrame(rows).set_index('factor').add_suffix('_'+tag)

t400 = summarise(ic_series(P[P['mc_rank'] <= 400]), '400')
t1k = summarise(ic_series(P), '1k')
tsn = summarise(ic_series(P, neutral=True), 'sn')
cov = P[FAC].notna().mean().rename('coverage')
T = t400.join(t1k).join(tsn[['ic_sn', 't_sn']]).join(cov)
T.to_csv('experiments/FACTOR_STUDY/factor_scores.csv')
# robustness gate
T['stable'] = (np.sign(T['ic_h1_1k']) == np.sign(T['ic_h2_1k'])) & (np.sign(T['ic_h1_400']) == np.sign(T['ic_h2_400']))
T['pass'] = T['stable'] & (T['t_1k'].abs() >= 2) & (T['yrs_same_1k'] >= 0.6) & (T['coverage'] >= 0.6)
pd.set_option('display.width', 250)
show = ['ic_400', 't_400', 'ic_h1_400', 'ic_h2_400', 'ic_1k', 't_1k', 'ic_h1_1k', 'ic_h2_1k', 'yrs_same_1k', 'ic_sn', 't_sn', 'coverage', 'pass']
S = T.reindex(T['t_1k'].abs().sort_values(ascending=False).index)[show]
fmt = {c: '{:+.4f}'.format for c in show if c.startswith('ic')}
fmt.update({c: '{:+.2f}'.format for c in show if c.startswith('t_')})
fmt.update({'yrs_same_1k': '{:.0%}'.format, 'coverage': '{:.0%}'.format})
print(S.to_string(formatters=fmt))
print(f"\nPASS gate (|t_1k|>=2, same sign both halves in 400 AND 1k, >=60% yrs, cov>=60%): {int(T['pass'].sum())} of {len(T)}")
