"""
Honest factor selection: choose on 2010-2017 ONLY, test on 2018-2026.

Selection (in-sample 2010-01..2017-12, top-1000):
  gate  : |NW t| >= 2 on 2010-17, same sign in 2010-13 and 2014-17, coverage >= 60%
  dedupe: greedy by |t|, drop a factor whose avg cross-sectional |rank corr|
          with any already-kept factor exceeds 0.6
Out-of-sample (2018-01..2026-04), all models trained walk-forward, PIT-guarded:
  A GBM on all factors | B GBM on selected | C signed equal-weight rank composite
  of selected | D cash_return alone (the algo's value factor) | E production mlf1
"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet')
P = P[P['Date'] >= '2010-01-01'].copy()
FAC = [c for c in P.columns if c[:2] in ('p_', 'l_', 'f_')]
P['sector'] = P['GICSSectorName'].astype('category').cat.codes
# cross-sectional rank transform (robust to units/outliers), target = fwd90 rank
R = P.groupby('Date')[FAC].rank(pct=True)
P[FAC] = R
P['y'] = P.groupby('Date')['fwd90'].rank(pct=True)

def ic_by_date(D, col, ycol='fwd90'):
    return D.groupby('Date').apply(lambda s: s[col].corr(s[ycol], method='spearman') if s[col].notna().sum() >= 50 else np.nan).dropna()

def nw_t(s, L=4):
    a = s.values - s.mean(); n = len(a); v0 = (a*a).sum()/n
    tot = sum((1-k/(L+1))*((a[k:]*a[:-k]).sum()/n/v0) for k in range(1, L+1))
    return s.mean()/(s.std(ddof=1)*np.sqrt(1+2*tot)/np.sqrt(n))

# ---------------- SELECTION on 2010-2017 ----------------
IS = P[(P['Date'] < '2018-01-01') & P['fwd90'].notna()]
rows = []
for c in FAC:
    s = ic_by_date(IS, c)
    if len(s) < 60 or IS[c].notna().mean() < 0.6: continue
    a = s[s.index < '2014-01-01'].mean(); b = s[s.index >= '2014-01-01'].mean()
    rows.append((c, s.mean(), nw_t(s), a, b))
S = pd.DataFrame(rows, columns=['f', 'ic', 't', 'ic_a', 'ic_b']).set_index('f')
S['gate'] = (S['t'].abs() >= 2) & (np.sign(S['ic_a']) == np.sign(S['ic_b']))
cand = S[S['gate']].reindex(S[S['gate']]['t'].abs().sort_values(ascending=False).index)
# average cross-sectional rank correlation among candidates (in-sample)
C = IS.groupby('Date')[list(cand.index)].corr().groupby(level=1).mean().loc[cand.index, cand.index]
kept = []
for f in cand.index:
    if all(abs(C.loc[f, k]) < 0.6 for k in kept):
        kept.append(f)
sign = np.sign(cand.loc[kept, 'ic'])
print(f"IN-SAMPLE 2010-17: {int(S['gate'].sum())} pass the gate, {len(kept)} kept after de-duplication (|corr|<0.6)\n")
print(f"{'factor':<58}{'IC 10-17':>9}{'t':>7}{'IC 10-13':>9}{'IC 14-17':>9}")
for f in kept:
    r = cand.loc[f]; print(f"{f:<58}{r.ic:>+9.4f}{r.t:>+7.2f}{r.ic_a:>+9.4f}{r.ic_b:>+9.4f}")
dropped = [f for f in cand.index if f not in kept]
print(f"\nremoved as redundant: {', '.join(dropped) if dropped else 'none'}")

# ---------------- OUT-OF-SAMPLE 2018-2026 ----------------
months = sorted(P['Date'].unique()); oos = [m for m in months if m >= pd.Timestamp('2018-01-01')]
def walk(feats, lookback=60):
    out = []
    for m in oos:
        lo = m - pd.DateOffset(months=lookback)
        # PIT guard: training snapshot's 90-trading-day target must have realised before m
        tr = P[(P['Date'] >= lo) & (P['Date'] <= m - pd.DateOffset(days=135)) & P['fwd90'].notna()]
        te = P[P['Date'] == m]
        if len(tr) < 5000 or len(te) == 0: continue
        mdl = HistGradientBoostingRegressor(max_depth=6, min_samples_leaf=100, l2_regularization=0.2,
                                            learning_rate=0.05, max_iter=300, random_state=0)
        mdl.fit(tr[feats], tr['y'])
        out.append(pd.Series(mdl.predict(te[feats]), index=te.index))
    return pd.concat(out)

print("\ntraining walk-forward models (monthly, 60m window, PIT guard) ...")
P['pred_A'] = walk(FAC + ['sector'])
P['pred_B'] = walk(kept + ['sector'])
P['pred_C'] = sum(sign[f] * (P[f].fillna(0.5) - 0.5) for f in kept)
P['pred_D'] = P['f_cash_return']
# E: production-config honest PIT forecast at the same snapshots
pb = pd.read_parquet('experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG.parquet',
                     columns=['Date', 'Symbol', 'predicted_return'], engine='pyarrow')
pb['Date'] = pd.to_datetime(pb['Date'])
P = P.merge(pb.rename(columns={'predicted_return': 'pred_E'}), on=['Date', 'Symbol'], how='left')

O = P[(P['Date'] >= '2018-01-01') & P['fwd90'].notna()].copy()
O['fwd90c'] = O['fwd90'].clip(-0.95, 3.0)   # one data glitch must not dominate a basket mean
names = {'A': f'GBM all {len(FAC)} factors', 'B': f'GBM selected {len(kept)}', 'C': f'equal-wt composite {len(kept)}',
         'D': 'cash_return alone', 'E': 'production mlf1 (PIT)'}
print(f"\nOUT-OF-SAMPLE 2018-01..{O['Date'].max():%Y-%m}  (selection never saw these years)\n")
print(f"{'model':<28}{'univ':>6}{'IC':>9}{'ICIR':>7}{'t':>7}{'%+':>6}{'top50 exc':>11}{'top30 exc':>11}")
for k in 'ABCDE':
    for band in (400, 1000):
        D = O[(O['mc_rank'] <= band) & O['pred_' + k].notna()]
        s = ic_by_date(D, 'pred_' + k)
        ex = {}
        for n in (50, 30):
            e = [g.nlargest(n, 'pred_' + k)['fwd90c'].mean() - g['fwd90c'].mean() for _, g in D.groupby('Date') if len(g) >= 2*n]
            ex[n] = 100*np.mean(e)
        print(f"{names[k]:<28}{band:>6}{s.mean():>+9.4f}{s.mean()/s.std(ddof=1):>7.3f}{nw_t(s):>+7.2f}"
              f"{100*(s>0).mean():>5.0f}%{ex[50]:>+10.2f}%{ex[30]:>+10.2f}%")
pd.Series(kept).to_csv('experiments/FACTOR_STUDY/selected_factors_2010_2017.csv', index=False)
