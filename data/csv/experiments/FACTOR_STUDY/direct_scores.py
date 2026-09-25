"""Non-ML cross-sectional scores from the 19 factors, scored like the GBM."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet')
P = P[P['Date'] >= '2010-01-01'].copy()
K = pd.read_csv('experiments/FACTOR_STUDY/selected_factors_2010_2017.csv').iloc[:, 0].tolist()
FAM = {'surprise': ['f_sue_pos', 'l_eps_surprise_px', 'l_GrossProfitMargin_ActualSurprise'],
       'quality': ['l_EarningsQualityRegionRank_Current', 'f_fscore7', 'f_gp_assets', 'f_gw_assets'],
       'value': ['l_focf_mc', 'f_fcf_ev', 'f_gp_ev', 'l_trail_eps_px'],
       'issuance': ['f_share_growth'],
       'analyst': ['l_CombinedAlphaModelSectorRank', 'l_LongTermGrowth_Mean'],
       'growth_tilt': ['f_bm', 'f_div_yield', 'f_rnd_rev', 'f_sbc_rev', 'p_dvol_60']}
assert sorted(sum(FAM.values(), [])) == sorted(K)
CORE = [f for fam, fs in FAM.items() if fam != 'growth_tilt' for f in fs]
# centred cross-sectional ranks in [-0.5, 0.5]; missing -> 0 (neutral)
R = P.groupby('Date')[K].rank(pct=True) - 0.5
for c in K: P['r_' + c] = R[c].fillna(0.0)
P['sector'] = P['GICSSectorName']

# fixed signs from the SELECTION period only (2010-17) -- no peeking
IS = P[(P['Date'] < '2018-01-01') & P['fwd90'].notna()]
def ic_by_date(D, col):
    return D.groupby('Date').apply(lambda s: s[col].corr(s['fwd90'], method='spearman') if s[col].notna().sum() >= 50 else np.nan).dropna()
sign = {f: np.sign(ic_by_date(IS, f).mean()) for f in K}

def sc(cols):                       # equal-weight signed average
    return sum(sign[f] * P['r_' + f] for f in cols) / len(cols)
def fam_bal(fams):                  # average within family, then across families
    return sum(sc(FAM[f]) for f in fams) / len(fams)

P['S_eq19'] = sc(K)
P['S_eq14'] = sc(CORE)
P['S_fam19'] = fam_bal(list(FAM))
P['S_fam14'] = fam_bal([f for f in FAM if f != 'growth_tilt'])

# adaptive IC weights, PIT: each month use factor ICs from snapshots whose
# 90d target had realised (<= m-135d) within the prior 60 months
ICm = pd.DataFrame({f: P[P['fwd90'].notna()].groupby('Date').apply(
    lambda s: s['r_' + f].corr(s['fwd90'], method='spearman')) for f in K})
months = sorted(P['Date'].unique())
W = {}
for m in months:
    h = ICm[(ICm.index >= m - pd.DateOffset(months=60)) & (ICm.index <= m - pd.DateOffset(days=135))]
    if len(h) >= 24: W[m] = h.mean()
W = pd.DataFrame(W).T
def icw(cols, keep_sign=True):
    out = pd.Series(np.nan, index=P.index)
    for m, g in P.groupby('Date'):
        if m not in W.index: continue
        w = W.loc[m, cols]
        if keep_sign:               # magnitude adapts, direction fixed from 2010-17
            w = pd.Series({f: sign[f] * max(sign[f] * w[f], 0.0) for f in cols})
        if w.abs().sum() == 0: continue
        out.loc[g.index] = sum(w[f] * g['r_' + f] for f in cols) / w.abs().sum()
    return out
P['S_icw19'] = icw(K)
P['S_icw14'] = icw(CORE)
P['S_icw19_free'] = icw(K, keep_sign=False)
# sector-neutral version of the best-looking family score
P['S_fam14_sn'] = P['S_fam14'] - P.groupby(['Date', 'sector'])['S_fam14'].transform('mean')

def nw_t(s, L=4):
    a = s.values - s.mean(); n = len(a); v0 = (a*a).sum()/n
    tot = sum((1-k/(L+1))*((a[k:]*a[:-k]).sum()/n/v0) for k in range(1, L+1))
    return s.mean()/(s.std(ddof=1)*np.sqrt(1+2*tot)/np.sqrt(n))

S = [c for c in P.columns if c.startswith('S_')]
P['fwd90c'] = P['fwd90'].clip(-0.95, 3.0)
print(f"{'score':<14}{'period':<10}{'IC':>9}{'ICIR':>7}{'t':>7}{'%+':>6}{'top30':>9}{'top50':>9}")
print('-'*71)
for s in S + ['GBM19 (ref)']:
    for lbl, a, b in [('2015-17', '2015-01-01', '2017-12-31'), ('2018-26', '2018-01-01', '2026-12-31')]:
        if s == 'GBM19 (ref)':
            print(f"{s:<14}{lbl:<10}" + ("   -0.0072  -0.05       50%" if lbl == '2015-17' else "   +0.0787   0.58  +3.96   70%    +3.44%   +2.73%"))
            continue
        D = P[(P['Date'] >= a) & (P['Date'] <= b) & P['fwd90'].notna() & (P['mc_rank'] <= 400) & P[s].notna()]
        ic = ic_by_date(D, s)
        ex = {n: 100*np.mean([g.nlargest(n, s)['fwd90c'].mean() - g['fwd90c'].mean() for _, g in D.groupby('Date')]) for n in (30, 50)}
        print(f"{s:<14}{lbl:<10}{ic.mean():>+9.4f}{ic.mean()/ic.std(ddof=1):>7.2f}{nw_t(ic):>+7.2f}{100*(ic>0).mean():>5.0f}%{ex[30]:>+8.2f}%{ex[50]:>+8.2f}%")
P[['Date', 'Symbol', 'mc_rank'] + S].to_parquet('experiments/FACTOR_STUDY/direct_scores.parquet', index=False)
