"""(A) does a month-old factor value lose signal?  (B) do technicals survive the honest gate?"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet')
X = pd.read_parquet('experiments/FACTOR_STUDY/daily_extras.parquet')
P = P.merge(X, on=['Date', 'Symbol'], how='left')
P = P[(P.Date >= '2010-01-01')].copy()
def ic_by_date(D, col):
    return D.groupby('Date').apply(lambda s: s[col].corr(s['fwd90'], method='spearman') if s[col].notna().sum() >= 50 else np.nan).dropna()
def nw_t(s, L=4):
    a = s.values - s.mean(); n = len(a); v0 = (a*a).sum()/n
    tot = sum((1-k/(L+1))*((a[k:]*a[:-k]).sum()/n/v0) for k in range(1, L+1))
    return s.mean()/(s.std(ddof=1)*np.sqrt(1+2*tot)/np.sqrt(n))

# ---------------- (A) staleness ----------------
V = P[P.fwd90.notna()]
print("(A) STALENESS: same factor measured today vs 21 trading days ago, against the SAME fwd90\n")
print(f"{'factor':<52}{'univ':>5}{'IC today':>10}{'IC -21d':>10}{'retained':>10}")
for c in sorted([c for c in P.columns if c.startswith('s_') and not c.endswith('__lag21')]):
    for band in (400, 1000):
        D = V[V.mc_rank <= band]
        a = ic_by_date(D, c).mean(); b = ic_by_date(D, c + '__lag21').mean()
        print(f"{c[2:52] if band==400 else '':<52}{band:>5}{a:>+10.4f}{b:>+10.4f}{(b/a if abs(a)>0.005 else np.nan):>9.0%}")

# ---------------- (B) technicals through the honest gate ----------------
TECH = [c for c in P.columns if c.startswith('t_')]
OLD = pd.read_csv('experiments/FACTOR_STUDY/selected_factors_2010_2017.csv').iloc[:, 0].tolist()
IS = P[(P.Date < '2018-01-01') & P.fwd90.notna()]
print("\n(B) TECHNICALS, selection period 2010-17 only, top-1000\n")
print(f"{'technical':<20}{'IC 10-17':>10}{'t':>7}{'IC 10-13':>10}{'IC 14-17':>10}{'gate':>6}{'max|corr| vs 19':>17}")
Cm = IS.groupby('Date')[TECH + OLD].corr().groupby(level=1).mean()
passed = []
for c in TECH:
    s = ic_by_date(IS, c)
    if len(s) < 60: continue
    a = s[s.index < '2014-01-01'].mean(); b = s[s.index >= '2014-01-01'].mean(); t = nw_t(s)
    ok = abs(t) >= 2 and np.sign(a) == np.sign(b)
    mc = Cm.loc[c, OLD].abs().max()
    new = ok and mc < 0.6
    if new: passed.append(c)
    print(f"{c:<20}{s.mean():>+10.4f}{t:>+7.2f}{a:>+10.4f}{b:>+10.4f}{'PASS' if ok else '':>6}{mc:>12.2f}{'  <- adds' if new else ''}")
print(f"\ntechnicals that pass AND are not redundant with the 19: {passed if passed else 'none'}")

# ---------------- OOS: 19 vs 19 + surviving technicals ----------------
if passed:
    Q = P.copy(); Q['sector'] = Q['GICSSectorName'].astype('category').cat.codes
    FE = OLD + passed
    Q[FE] = Q.groupby('Date')[FE].rank(pct=True); Q['y'] = Q.groupby('Date')['fwd90'].rank(pct=True)
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
    Q['p19'] = walk(OLD + ['sector']); Q['p19t'] = walk(FE + ['sector'])
    O = Q[(Q.Date >= '2018-01-01') & Q.fwd90.notna() & (Q.mc_rank <= 400)].copy(); O['fc'] = O.fwd90.clip(-0.95, 3)
    print(f"\nOOS 2018-26, top-400:")
    for k, n in [('p19', '19 factors'), ('p19t', f'19 + {len(passed)} technicals')]:
        s = ic_by_date(O, k)
        e = np.mean([t.nlargest(30, k)['fc'].mean() - t['fc'].mean() for _, t in O.groupby('Date')])
        print(f"   {n:<24} IC {s.mean():+.4f}  ICIR {s.mean()/s.std():.2f}  t {nw_t(s):+.2f}  top30 {100*e:+.2f}%")
