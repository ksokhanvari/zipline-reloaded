#!/usr/bin/env python3
"""
Build a point-in-time MONTHLY factor panel for factor selection.

One row per (month-end snapshot, symbol) for the top-1000 names by market cap.
Every factor uses only information available at the snapshot's close; the
target is the split-corrected forward return from T+1 to T+91 trading days
(same convention as production: forecast_days=1, target_return_days=90).

Two data defects in the production input are corrected here:
  1. RefPriceClose is NOT split-adjusted (AMZN 2022-01-03 prints -94.9%).
     On days where price jumps >80%/<-45% but market cap moves <15%, the
     market-cap return replaces the price return.
  2. ~10% of FMP filings are stamped ON the fiscal quarter-end with a
     date-only timestamp -- 1-2 months before the numbers were public
     (real filings land a median 38 days after quarter-end). Their
     availability is pushed out 60 days.

FMP fundamentals are rebuilt from the sparse filing rows (Q1..Q4 discrete,
no FY rows): trailing-twelve-month flows are 4-quarter sums over the fiscal
sequence, then attached to snapshots with merge_asof on availability date.
"""
import numpy as np, pandas as pd, sys, os

SRC = 'experiments/PIT_BASE_MONTHLY_90d_FULLHIST_LSEG/input_PIT_BASE_FULLHIST.parquet'
OUT = 'experiments/FACTOR_STUDY/panel_monthly.parquet'
TOPN = 1000
FD, TRD = 1, 90

LSEG = ['EnterpriseValue_DailyTimeSeries_','FOCFExDividends_Discrete','InterestExpense_NetofCapitalizedInterest',
        'Debt_Total','EarningsPerShare_Actual','EarningsPerShare_SmartEstimate_prev_Q','EarningsPerShare_ActualSurprise',
        'EarningsPerShare_SmartEstimate_current_Q','LongTermGrowth_Mean','PriceTarget_Median',
        'CombinedAlphaModelSectorRank','CombinedAlphaModelSectorRankChange','CombinedAlphaModelRegionRank',
        'EarningsQualityRegionRank_Current','EnterpriseValueToEBIT_DailyTimeSeriesRatio_',
        'EnterpriseValueToEBITDA_DailyTimeSeriesRatio_','EnterpriseValueToSales_DailyTimeSeriesRatio_',
        'Dividend_Per_Share_SmartEstimate','CashCashEquivalents_Total','ForwardPEG_DailyTimeSeriesRatio_',
        'PriceEarningsToGrowthRatio_SmartEstimate_','Recommendation_Median_1_5_','ReturnOnEquity_SmartEstimat',
        'ReturnOnAssets_SmartEstimate','ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_',
        'ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_','ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_',
        'GrossProfitMargin_ActualSurprise','Estpricegrowth_percent']

FLOWS = {'rev':'revenue_fmp','gp':'grossprofit_fmp','oi':'operatingincome_fmp','ebitda':'ebitda_fmp',
         'ebit':'ebit_fmp','ni':'netincome_fmp','ocf':'netcashprovidedbyoperatingactivities_fmp',
         'fcf':'freecashflow_fmp','capex':'capitalexpenditure_fmp','sbc':'stockbasedcompensation_fmp',
         'div':'commondividendspaid_fmp','buyback':'commonstockrepurchased_fmp','issue':'commonstockissuance_fmp',
         'intexp':'interestexpense_fmp','rnd':'researchanddevelopmentexpenses_fmp','da':'depreciationandamortization_fmp'}
STOCKS = {'assets':'totalassets_fmp','equity':'totalstockholdersequity_fmp','debt':'totaldebt_fmp',
          'cash':'cashandshortterminvestments_fmp','ca':'totalcurrentassets_fmp','cl':'totalcurrentliabilities_fmp',
          'recv':'netreceivables_fmp','inv':'inventory_fmp','gw':'goodwillandintangibleassets_fmp',
          'shares':'weightedaverageshsoutdil_fmp'}
EPS = {'eps_act':'epsactual_fmp','eps_est':'epsestimated_fmp'}


def sdiv(a, b):
    b = b.where(b.abs() > 0)
    return (a / b).replace([np.inf, -np.inf], np.nan)


def main():
    if os.path.exists(OUT):
        sys.exit(f'ABORT: {OUT} exists')
    base = ['Date', 'Symbol', 'RefPriceClose', 'RefVolume', 'CompanyMarketCap', 'GICSSectorName']
    print('loading daily price / LSEG columns ...')
    d = pd.read_parquet(SRC, columns=base + LSEG, engine='pyarrow')
    d['Date'] = pd.to_datetime(d['Date'])
    d = d.sort_values(['Symbol', 'Date'], kind='stable').reset_index(drop=True)
    g = d.groupby('Symbol', sort=False)

    # ---- split-corrected daily return and price index
    r = g['RefPriceClose'].pct_change()
    mr = g['CompanyMarketCap'].pct_change()
    split = ((r < -0.45) | (r > 0.8)) & (mr.abs() < 0.15)
    print(f'  split artefacts corrected: {int(split.sum())}')
    r = r.where(~split, mr).clip(-0.9, 5.0)
    d['r'] = r.fillna(0.0)
    d['idx'] = d.groupby('Symbol', sort=False)['r'].transform(lambda s: (1 + s).cumprod())
    g = d.groupby('Symbol', sort=False)
    idx = d['idx']

    # ---- target: T+1 -> T+91
    d['fwd90'] = g['idx'].shift(-(FD + TRD)) / g['idx'].shift(-FD) - 1
    d['fwd21'] = g['idx'].shift(-(FD + 21)) / g['idx'].shift(-FD) - 1

    # ---- price factors (backward-looking)
    print('price factors ...')
    d['mom_12_1'] = g['idx'].shift(21) / g['idx'].shift(252) - 1
    d['mom_6_1'] = g['idx'].shift(21) / g['idx'].shift(126) - 1
    d['mom_3m'] = idx / g['idx'].shift(63) - 1
    d['rev_1m'] = idx / g['idx'].shift(21) - 1
    d['high52'] = idx / g['idx'].transform(lambda s: s.rolling(252, min_periods=200).max())
    d['vol_60'] = g['r'].transform(lambda s: s.rolling(60, min_periods=40).std())
    d['vol_252'] = g['r'].transform(lambda s: s.rolling(252, min_periods=200).std())
    d['maxret_21'] = g['r'].transform(lambda s: s.rolling(21, min_periods=15).max())
    dv = d['RefPriceClose'] * d['RefVolume']
    d['dvol_60'] = np.log(dv.groupby(d['Symbol'], sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean()) + 1)
    d['turnover_60'] = sdiv(np.exp(d['dvol_60']) - 1, d['CompanyMarketCap'])
    d['size'] = np.log(d['CompanyMarketCap'].where(d['CompanyMarketCap'] > 0))

    # ---- LSEG change factors (revisions)
    px = d['RefPriceClose']
    for c, nm in [('EarningsPerShare_SmartEstimate_current_Q', 'eps_est_rev63'),
                  ('PriceTarget_Median', 'pt_chg63'),
                  ('Recommendation_Median_1_5_', 'rec_chg63')]:
        prev = g[c].shift(63)
        d[nm] = (d[c] - prev) / px if nm == 'eps_est_rev63' else (sdiv(d[c], prev) - 1 if nm == 'pt_chg63' else d[c] - prev)
    d['pt_upside'] = sdiv(d['PriceTarget_Median'], px) - 1
    mcd = d['CompanyMarketCap']
    d['fwd_ey'] = sdiv(d['EarningsPerShare_SmartEstimate_current_Q'] * 4, px)   # annualised quarter estimate
    d['trail_eps_px'] = sdiv(d['EarningsPerShare_Actual'], px)
    d['dps_yield'] = sdiv(d['Dividend_Per_Share_SmartEstimate'], px)
    d['debt_mc'] = sdiv(d['Debt_Total'], mcd)
    d['int_mc'] = sdiv(d['InterestExpense_NetofCapitalizedInterest'], mcd)
    d['lcash_mc'] = sdiv(d['CashCashEquivalents_Total'], mcd)
    d['ev_mc'] = sdiv(d['EnterpriseValue_DailyTimeSeries_'], mcd)
    d['focf_mc'] = sdiv(d['FOCFExDividends_Discrete'], mcd)
    d['eps_surprise_px'] = sdiv(d['EarningsPerShare_ActualSurprise'], px)

    # ---- month-end snapshots, top-N by mcap
    # Month-end = last date with a REAL cross-section. Stray rows on weekends /
    # odd dates (a handful of junk symbols, market cap 0) otherwise become the
    # month's whole 'snapshot' and rank as the top-N.
    cnt = d.groupby('Date').size()
    real = cnt[cnt >= 1500].index
    me = pd.Series(real, index=real).groupby(real.to_period('M')).max()
    snap = d[d['Date'].isin(set(me.values)) & (d['CompanyMarketCap'] > 0)].copy()
    snap['mc_rank'] = snap.groupby('Date')['CompanyMarketCap'].rank(ascending=False, method='first')
    snap = snap[snap['mc_rank'] <= TOPN].copy()
    print(f'  snapshots: {snap["Date"].nunique()} months, {len(snap):,} rows')
    del d

    # ---- FMP fundamentals, PIT
    print('FMP fundamentals (PIT, TTM) ...')
    cols = ['Symbol', 'Date', 'accepteddate_fmp', 'fiscalyear_fmp', 'period_fmp'] + \
           list(FLOWS.values()) + list(STOCKS.values()) + list(EPS.values())
    f = pd.read_parquet(SRC, columns=cols, engine='pyarrow')
    f = f[f['accepteddate_fmp'].notna()].copy()
    f['Date'] = pd.to_datetime(f['Date'])
    acc = pd.to_datetime(f['accepteddate_fmp'], errors='coerce')
    qend = f['Date'].dt.month.isin([3, 6, 9, 12]) & ((f['Date'] + pd.offsets.BMonthEnd(0)) == f['Date']) \
        | (f['Date'].dt.is_month_end & f['Date'].dt.month.isin([3, 6, 9, 12]))
    dateonly = acc.dt.hour.isin([0, 19, 20])
    flag = qend & dateonly
    f['avail'] = f['Date'] + pd.to_timedelta(np.where(flag, 60, 0), unit='D')
    print(f'  filings: {len(f):,}; quarter-end-stamped pushed +60d: {int(flag.sum()):,}')
    f = f.rename(columns={v: k for k, v in {**FLOWS, **STOCKS, **EPS}.items()})
    f['qn'] = f['fiscalyear_fmp'] * 4 + f['period_fmp'].str[1].astype(float)
    f = f.sort_values(['Symbol', 'qn'], kind='stable').reset_index(drop=True)
    gf = f.groupby('Symbol', sort=False)
    consec = (f['qn'] - gf['qn'].shift(3)) == 3            # 4 consecutive quarters
    for k in FLOWS:
        f[k + '_ttm'] = gf[k].transform(lambda s: s.rolling(4, min_periods=4).sum()).where(consec)
    yoy = (f['qn'] - gf['qn'].shift(4)) == 4
    for k in ['rev', 'gp', 'ni', 'ocf', 'fcf', 'oi']:
        f[k + '_ttm_l4'] = gf[k + '_ttm'].shift(4).where(yoy)
    for k in ['assets', 'shares', 'equity']:
        f[k + '_l4'] = gf[k].shift(4).where(yoy)
    f['eps_ttm'] = gf['eps_act'].transform(lambda s: s.rolling(4, min_periods=4).sum()).where(consec)
    f['eps_ttm_l4'] = gf['eps_ttm'].shift(4).where(yoy)
    f['sue_raw'] = f['eps_act'] - f['eps_est']
    f['gm'] = sdiv(f['gp_ttm'], f['rev_ttm'])
    f['gm_l4'] = sdiv(f['gp_ttm_l4'], f['rev_ttm_l4'])
    f['opm'] = sdiv(f['oi_ttm'], f['rev_ttm'])
    f['opm_l4'] = sdiv(f['oi_ttm_l4'], f['rev_ttm_l4'])
    f['roa'] = sdiv(f['ni_ttm'], f['assets'])
    f['roa_l4'] = sdiv(f['ni_ttm_l4'], f['assets_l4'])

    keep = [c for c in f.columns if c not in ('Date', 'accepteddate_fmp', 'fiscalyear_fmp', 'period_fmp', 'qn')]
    f = f[keep].sort_values('avail')
    snap = snap.sort_values('Date')
    p = pd.merge_asof(snap, f, left_on='Date', right_on='avail', by='Symbol', direction='backward')
    p['filing_age'] = (p['Date'] - p['avail']).dt.days
    stale = p['filing_age'] > 200                          # no filing in >200d: don't trust
    fcols = [c for c in f.columns if c not in ('Symbol', 'avail')]
    p.loc[stale, fcols] = np.nan

    # ---- FMP-derived factors
    mc = p['CompanyMarketCap']
    ev = p['EnterpriseValue_DailyTimeSeries_'].where(p['EnterpriseValue_DailyTimeSeries_'] > 0,
                                                     mc + p['debt'].fillna(0) - p['cash'].fillna(0))
    F = {}
    F['ey'] = sdiv(p['ni_ttm'], mc)
    F['fcf_yield'] = sdiv(p['fcf_ttm'], mc)
    F['fcf_ev'] = sdiv(p['fcf_ttm'], ev)
    F['cash_return'] = sdiv(p['fcf_ttm'] - p['intexp_ttm'].fillna(0), ev)   # the algo's value factor
    F['ocf_yield'] = sdiv(p['ocf_ttm'], mc)
    F['ebit_ev'] = sdiv(p['ebit_ttm'], ev)
    F['ebitda_ev'] = sdiv(p['ebitda_ttm'], ev)
    F['sales_ev'] = sdiv(p['rev_ttm'], ev)
    F['gp_ev'] = sdiv(p['gp_ttm'], ev)
    F['bm'] = sdiv(p['equity'], mc)
    F['cash_mc'] = sdiv(p['cash'], mc)
    F['div_yield'] = sdiv(-p['div_ttm'], mc)
    F['buyback_yield'] = sdiv(-p['buyback_ttm'], mc)
    F['net_payout_yield'] = sdiv(-p['div_ttm'].fillna(0) - p['buyback_ttm'].fillna(0) - p['issue_ttm'].fillna(0), mc)
    F['roe'] = sdiv(p['ni_ttm'], p['equity'])
    F['roa'] = p['roa']
    F['gp_assets'] = sdiv(p['gp_ttm'], p['assets'])
    F['cfo_assets'] = sdiv(p['ocf_ttm'], p['assets'])
    F['gm'] = p['gm']
    F['opm'] = p['opm']
    F['fcf_margin'] = sdiv(p['fcf_ttm'], p['rev_ttm'])
    F['accruals'] = sdiv(p['ni_ttm'] - p['ocf_ttm'], p['assets'])
    F['sbc_rev'] = sdiv(p['sbc_ttm'], p['rev_ttm'])
    F['rnd_rev'] = sdiv(p['rnd_ttm'], p['rev_ttm'])
    F['capex_assets'] = sdiv(-p['capex_ttm'], p['assets'])
    F['lev'] = sdiv(p['debt'], p['assets'])
    F['netdebt_ebitda'] = sdiv(p['debt'] - p['cash'], p['ebitda_ttm'])
    F['current_ratio'] = sdiv(p['ca'], p['cl'])
    F['int_cov'] = sdiv(p['ebit_ttm'], p['intexp_ttm'])
    F['gw_assets'] = sdiv(p['gw'], p['assets'])
    F['asset_growth'] = sdiv(p['assets'], p['assets_l4']) - 1
    F['share_growth'] = sdiv(p['shares'], p['shares_l4']) - 1
    F['rev_growth'] = sdiv(p['rev_ttm'], p['rev_ttm_l4'].where(p['rev_ttm_l4'] > 0)) - 1
    F['gp_growth'] = sdiv(p['gp_ttm'], p['gp_ttm_l4'].where(p['gp_ttm_l4'] > 0)) - 1
    F['d_ni_assets'] = sdiv(p['ni_ttm'] - p['ni_ttm_l4'], p['assets'])
    F['d_ocf_assets'] = sdiv(p['ocf_ttm'] - p['ocf_ttm_l4'], p['assets'])
    F['d_gm'] = p['gm'] - p['gm_l4']
    F['d_opm'] = p['opm'] - p['opm_l4']
    F['d_roa'] = p['roa'] - p['roa_l4']
    F['d_eps_px'] = sdiv(p['eps_ttm'] - p['eps_ttm_l4'], p['RefPriceClose'])
    F['sue_px'] = sdiv(p['sue_raw'], p['RefPriceClose'])
    F['sue_pos'] = np.sign(p['sue_raw'])
    # Piotroski F-score (9 binary signals)
    fs = ((p['ni_ttm'] > 0).astype(float) + (p['ocf_ttm'] > 0).astype(float)
          + (p['roa'] > p['roa_l4']).astype(float) + (p['ocf_ttm'] > p['ni_ttm']).astype(float)
          + (F['share_growth'] <= 0).astype(float) + (p['gm'] > p['gm_l4']).astype(float)
          + (sdiv(p['rev_ttm'], p['assets']) > sdiv(p['rev_ttm_l4'], p['assets_l4'])).astype(float))
    F['fscore7'] = fs.where(p['ni_ttm'].notna() & p['roa_l4'].notna())
    for k, v in F.items():
        p['f_' + k] = v

    # ---- final column set
    price_f = ['mom_12_1', 'mom_6_1', 'mom_3m', 'rev_1m', 'high52', 'vol_60', 'vol_252', 'maxret_21',
               'dvol_60', 'turnover_60', 'size']
    # Dollar LEVELS are size / nominal-price proxies, not factors -> kept only as x_ (not scored).
    LEVELS = ['EnterpriseValue_DailyTimeSeries_', 'FOCFExDividends_Discrete', 'InterestExpense_NetofCapitalizedInterest',
              'Debt_Total', 'EarningsPerShare_Actual', 'EarningsPerShare_SmartEstimate_prev_Q',
              'EarningsPerShare_ActualSurprise', 'EarningsPerShare_SmartEstimate_current_Q', 'PriceTarget_Median',
              'Dividend_Per_Share_SmartEstimate', 'CashCashEquivalents_Total']
    for c in LEVELS: p.rename(columns={c: 'x_' + c}, inplace=True)
    lseg_f = [c for c in LSEG if c not in LEVELS] + ['eps_est_rev63', 'pt_chg63', 'rec_chg63', 'pt_upside',
              'eps_surprise_px', 'fwd_ey', 'trail_eps_px', 'dps_yield', 'debt_mc', 'int_mc', 'lcash_mc', 'ev_mc', 'focf_mc']
    fmp_f = ['f_' + k for k in F]
    for c in price_f: p.rename(columns={c: 'p_' + c}, inplace=True)
    for c in lseg_f: p.rename(columns={c: 'l_' + c}, inplace=True)
    keepc = ['Date', 'Symbol', 'GICSSectorName', 'CompanyMarketCap', 'mc_rank', 'fwd90', 'fwd21', 'filing_age'] + \
            ['p_' + c for c in price_f] + ['l_' + c for c in lseg_f] + fmp_f
    EVDEP = ['l_EnterpriseValueToEBIT_DailyTimeSeriesRatio_', 'l_EnterpriseValueToEBITDA_DailyTimeSeriesRatio_',
             'l_EnterpriseValueToSales_DailyTimeSeriesRatio_',
             'l_ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_', 'l_debt_mc', 'l_int_mc', 'l_ev_mc',
             'f_fcf_ev', 'f_cash_return', 'f_ebit_ev', 'f_ebitda_ev', 'f_sales_ev', 'f_gp_ev', 'f_lev',
             'f_netdebt_ebitda', 'f_int_cov', 'f_current_ratio']
    fin = p['GICSSectorName'] == 'Financials'
    for c in EVDEP:
        if c in p.columns: p.loc[fin, c] = np.nan
    print(f'  EV/debt factors nulled for Financials: {int(fin.sum()):,} rows')
    keepc = [c for c in keepc if c in p.columns] + ['RefPriceClose'] + ['x_' + c for c in LEVELS]
    p = p[keepc]
    p.to_parquet(OUT, engine='pyarrow', compression='snappy', index=False)
    nf = len([c for c in keepc if c[:2] in ('p_', 'l_', 'f_')])
    print(f'\ndone: {len(p):,} rows, {nf} candidate factors -> {OUT}')
    print(f'  {p["Date"].min():%Y-%m} .. {p["Date"].max():%Y-%m}')
    cov = p[keepc[8:]].notna().mean().sort_values()
    print('  lowest-coverage factors:', ', '.join(f'{k} {100*v:.0f}%' for k, v in cov.head(6).items()))


if __name__ == '__main__':
    main()
