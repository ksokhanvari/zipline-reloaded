"""Honest gate for the new FMP factors: select 2010-17, test 2018-26; do they add to the 19?"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet')
X = pd.read_parquet('experiments/FACTOR_STUDY/fmp_extra_factors.parquet')
P = P.merge(X, on=['Date', 'Symbol'], how='left'); P = P[P.Date >= '2010-01-01'].copy()
NEW = [c for c in X.columns if c[:2] in ('e_', 'g_', 'i_')]
OLD = pd.read_csv('experiments/FACTOR_STUDY/selected_factors_2010_2017.csv').iloc[:, 0].tolist()
def icd(D, c): return D.groupby('Date').apply(lambda s: s[c].corr(s['fwd90'], method='spearman') if s[c].notna().sum() >= 50 else np.nan).dropna()
def nw_t(s, L=4):
    a = s.values - s.mean(); n = len(a); v0 = (a*a).sum()/n
    tot = sum((1-k/(L+1))*((a[k:]*a[:-k]).sum()/n/v0) for k in range(1, L+1))
    return s.mean()/(s.std(ddof=1)*np.sqrt(1+2*tot)/np.sqrt(n))
V = P[P.fwd90.notna()]; IS = V[V.Date < '2018-01-01']; OS = V[V.Date >= '2018-01-01']
Cm = IS.groupby('Date')[NEW + OLD].corr().groupby(level=1).mean()
print(f"{'factor':<20}{'cov':>5}{'IC 10-17':>10}{'t':>7}{'10-13':>8}{'14-17':>8}{'gate':>6}{'|corr| 19':>10}{'IC 18-26':>10}{'t':>7}")
passed = []
for c in NEW:
    s = icd(IS, c); o = icd(OS, c)
    if len(s) < 40: print(f"{c:<20} too little history in 2010-17 ({len(s)} months)"); continue
    a = s[s.index < '2014-01-01'].mean(); b = s[s.index >= '2014-01-01'].mean(); t = nw_t(s)
    ok = abs(t) >= 2 and np.sign(a) == np.sign(b); mc = Cm.loc[c, OLD].abs().max()
    if ok and mc < 0.6: passed.append(c)
    print(f"{c:<20}{100*IS[c].notna().mean():>4.0f}%{s.mean():>+10.4f}{t:>+7.2f}{a:>+8.4f}{b:>+8.4f}{'PASS' if ok else '':>6}"
          f"{mc:>10.2f}{o.mean():>+10.4f}{nw_t(o):>+7.2f}{'  <- adds' if ok and mc < 0.6 else ''}")
print(f"\npass the 2010-17 gate and not redundant with the 19: {passed if passed else 'none'}")
if passed:
    Q = P.copy(); Q['sector'] = Q['GICSSectorName'].astype('category').cat.codes
    FE = OLD + passed; Q[FE] = Q.groupby('Date')[FE].rank(pct=True); Q['y'] = Q.groupby('Date')['fwd90'].rank(pct=True)
    months = sorted(Q.Date.unique())
    def walk(feats):
        out = []
        for m in [x for x in months if x >= pd.Timestamp('2018-01-01')]:
            tr = Q[(Q.Date >= m - pd.DateOffset(months=60)) & (Q.Date <= m - pd.DateOffset(days=135)) & Q.fwd90.notna()]
            te = Q[Q.Date == m]
            mdl = HistGradientBoostingRegressor(max_depth=6, min_samples_leaf=100, l2_regularization=0.2,
                                                learning_rate=0.05, max_iter=300, random_state=0).fit(tr[feats], tr['y'])
            out.append(pd.Series(mdl.predict(te[feats]), index=te.index))
        return pd.concat(out)
    Q['p19'] = walk(OLD + ['sector']); Q['pnew'] = walk(FE + ['sector'])
    O = Q[(Q.Date >= '2018-01-01') & Q.fwd90.notna() & (Q.mc_rank <= 400)].copy(); O['fc'] = O.fwd90.clip(-0.95, 3)
    print("\nOOS 2018-26, top-400:")
    for k, n in [('p19', '19 factors'), ('pnew', f'19 + {len(passed)} new FMP')]:
        s = icd(O, k); e = [t.nlargest(30, k)['fc'].mean() - t['fc'].mean() for _, t in O.groupby('Date')]
        print(f"   {n:<22} IC {s.mean():+.4f}  ICIR {s.mean()/s.std():.2f}  t {nw_t(s):+.2f}  top30 {100*np.mean(e):+.2f}%")
