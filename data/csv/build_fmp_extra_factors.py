#!/usr/bin/env python3
"""
New point-in-time factors from the FMP extras, attached to the monthly panel.
A value is used only if its source event (report / rating / Form-4 filing) is
dated STRICTLY BEFORE the snapshot date.  Output: experiments/FACTOR_STUDY/fmp_extra_factors.parquet
"""
import numpy as np, pandas as pd, os
P = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet',
                    columns=['Date', 'Symbol', 'RefPriceClose', 'CompanyMarketCap'])
P = P.sort_values('Date')
OUT = []

# ---------------- earnings: EPS + revenue surprise ----------------
if os.path.exists('experiments/FMP_EXTRAS/earnings.parquet'):
    e = pd.read_parquet('experiments/FMP_EXTRAS/earnings.parquet')
    e['date'] = pd.to_datetime(e['date'])
    e = e[(e['date'] <= pd.Timestamp.today()) & e['epsActual'].notna()].copy()
    ph = e['revenueEstimated'].notna() & (e['revenueEstimated'] == e['revenueActual'])   # backfilled placeholder
    e.loc[ph, ['revenueEstimated', 'epsEstimated']] = np.nan
    e['rev_surp'] = (e['revenueActual'] - e['revenueEstimated']) / e['revenueEstimated'].abs().where(lambda x: x > 0)
    e['eps_surp'] = e['epsActual'] - e['epsEstimated']
    e['eps_beat'] = np.sign(e['eps_surp'])
    e['rev_beat'] = np.sign(e['rev_surp'])
    e = e.sort_values(['lseg_symbol', 'date'])
    g = e.groupby('lseg_symbol')
    e['eps_beats4'] = g['eps_beat'].transform(lambda s: (s > 0).astype(float).where(s.notna()).rolling(4, min_periods=3).mean())
    e['rev_beats4'] = g['rev_beat'].transform(lambda s: (s > 0).astype(float).where(s.notna()).rolling(4, min_periods=3).mean())
    e['rev_surp_chg'] = e['rev_surp'] - g['rev_surp'].shift(1)
    e['avail'] = e['date'] + pd.Timedelta(days=1)          # strictly after the report date
    cols = ['rev_surp', 'eps_surp', 'eps_beat', 'rev_beat', 'eps_beats4', 'rev_beats4', 'rev_surp_chg']
    m = pd.merge_asof(P, e[['lseg_symbol', 'avail'] + cols].rename(columns={'lseg_symbol': 'Symbol'}).sort_values('avail'),
                      left_on='Date', right_on='avail', by='Symbol', direction='backward')
    stale = (m['Date'] - m['avail']).dt.days > 120                                  # last report too old
    m.loc[stale, cols] = np.nan
    m['eps_surp_px'] = m['eps_surp'] / m['RefPriceClose'].where(m['RefPriceClose'] > 0)
    keep = ['rev_surp', 'eps_surp_px', 'eps_beat', 'rev_beat', 'eps_beats4', 'rev_beats4', 'rev_surp_chg']
    OUT.append(m[['Date', 'Symbol'] + keep].rename(columns={c: 'e_' + c for c in keep}))
    print(f"earnings factors: coverage " + ", ".join(f"{c} {100*m[c].notna().mean():.0f}%" for c in keep))

# ---------------- analyst grades: net upgrades ----------------
if os.path.exists('experiments/FMP_EXTRAS/grades.parquet'):
    gr = pd.read_parquet('experiments/FMP_EXTRAS/grades.parquet')
    gr['date'] = pd.to_datetime(gr['date']); gr = gr[gr['date'] <= pd.Timestamp.today()]
    a = gr['action'].str.lower()
    gr['up'] = (a == 'upgrade').astype(float); gr['dn'] = (a == 'downgrade').astype(float); gr['n'] = 1.0
    daily = gr.groupby(['lseg_symbol', 'date'])[['up', 'dn', 'n']].sum().reset_index()
    rows = []
    for sym, s in daily.groupby('lseg_symbol'):
        s = s.set_index('date').sort_index()
        r = s[['up', 'dn', 'n']].rolling('90D').sum()
        r['lseg_symbol'] = sym; rows.append(r.reset_index())
    R = pd.concat(rows); R['avail'] = R['date'] + pd.Timedelta(days=1)
    m = pd.merge_asof(P, R.rename(columns={'lseg_symbol': 'Symbol'}).sort_values('avail'),
                      left_on='Date', right_on='avail', by='Symbol', direction='backward')
    old = (m['Date'] - m['avail']).dt.days > 90                # no action in the last 90 days -> zero, not missing
    covered = m['Symbol'].isin(set(gr['lseg_symbol'])) & (m['Date'] >= gr['date'].min())
    for c in ('up', 'dn', 'n'):
        m.loc[old | m[c].isna(), c] = 0.0
        m.loc[~covered, c] = np.nan
    m['g_net_upgrades90'] = m['up'] - m['dn']
    m['g_net_up_ratio90'] = (m['up'] - m['dn']) / m['n'].where(m['n'] > 0)
    m['g_attention90'] = m['n']
    OUT.append(m[['Date', 'Symbol', 'g_net_upgrades90', 'g_net_up_ratio90', 'g_attention90']])
    print(f"grades factors: coverage {100*m['g_net_upgrades90'].notna().mean():.0f}%  (grades start {gr['date'].min():%Y-%m})")

# ---------------- insider: open-market buying ----------------
if os.path.exists('experiments/FMP_EXTRAS/insider.parquet'):
    it = pd.read_parquet('experiments/FMP_EXTRAS/insider.parquet')
    it['filingDate'] = pd.to_datetime(it['filingDate'], errors='coerce')
    tt = it['transactionType'].astype(str)
    it['buy$'] = np.where(tt.str.startswith('P'), it['securitiesTransacted'] * it['price'], 0.0)
    it['sell$'] = np.where(tt.str.startswith('S'), it['securitiesTransacted'] * it['price'], 0.0)
    it['buyer'] = np.where(tt.str.startswith('P'), it['reportingCik'], None)
    daily = it.groupby(['lseg_symbol', 'filingDate']).agg(b=('buy$', 'sum'), s=('sell$', 'sum'),
                                                         nb=('buyer', lambda x: x.dropna().nunique())).reset_index()
    rows = []
    for sym, s in daily.groupby('lseg_symbol'):
        s = s.set_index('filingDate').sort_index()
        r = s[['b', 's', 'nb']].rolling('180D').sum(); r['lseg_symbol'] = sym; rows.append(r.reset_index())
    R = pd.concat(rows); R['avail'] = R['filingDate'] + pd.Timedelta(days=1)
    m = pd.merge_asof(P, R.rename(columns={'lseg_symbol': 'Symbol'}).sort_values('avail'),
                      left_on='Date', right_on='avail', by='Symbol', direction='backward')
    old = (m['Date'] - m['avail']).dt.days > 180
    covered = m['Symbol'].isin(set(it['lseg_symbol'])) & (m['Date'] >= it['filingDate'].min())
    for c in ('b', 's', 'nb'):
        m.loc[old | m[c].isna(), c] = 0.0
        m.loc[~covered, c] = np.nan
    mc = m['CompanyMarketCap'].where(m['CompanyMarketCap'] > 0)
    m['i_net_buy_mc180'] = (m['b'] - m['s']) / mc
    m['i_buy_mc180'] = m['b'] / mc
    m['i_buyers180'] = m['nb']
    OUT.append(m[['Date', 'Symbol', 'i_net_buy_mc180', 'i_buy_mc180', 'i_buyers180']])
    print(f"insider factors: coverage {100*m['i_net_buy_mc180'].notna().mean():.0f}%")

F = OUT[0]
for o in OUT[1:]: F = F.merge(o, on=['Date', 'Symbol'], how='outer')
F.to_parquet('experiments/FACTOR_STUDY/fmp_extra_factors.parquet', index=False)
print(f"-> {len(F):,} rows, factors: {[c for c in F.columns if c[:2] in ('e_','g_','i_')]}")
