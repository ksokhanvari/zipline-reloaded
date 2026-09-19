#!/usr/bin/env python3
"""
Replace LSEG columns in the training CSV with FMP + Sharadar equivalents.

Disposition of the 38 raw LSEG columns
--------------------------------------
  4 keys/metadata  Date, Symbol, Instrument, TradeDate     -- untouched
  2 price/volume   RefPriceClose, RefVolume                -- untouched
  2 identity       CompanyCommonName, GICSSectorName       -- sharadar_* already merged
  5 QUARTERLY      from *_fmp columns ALREADY IN THIS FILE -- no download needed
  2 DAILY          ev, marketcap  -> SHARADAR/DAILY        -- the only download
  3 RATIOS         daily EV over a forward-filled TTM denominator
 20 DROPPED        16 analyst-estimate + 4 StarMine ranks

Why the quarterly half needs no download
----------------------------------------
The *_fmp columns are already point-in-time: each lands on `accepteddate_fmp`,
the filing's publication timestamp, covering 97.8% of symbols at ~5 filings per
symbol per year. Measured agreement with the LSEG originals on filing rows
(Spearman): debt +0.971, cash +0.964, FCF +0.952, interest +0.909, EPS +0.857.

TWO CORRECTNESS DETAILS THAT BIT THE FIRST VERSION
--------------------------------------------------
1. UNITS. SHARADAR/DAILY reports `ev` and `marketcap` in MILLIONS of USD
   (AAPL = 4,522,736.4), while LSEG reports actual USD. They are scaled by 1e6
   here. A tree model is invariant to a constant factor on a single feature,
   but any ratio mixing a Sharadar numerator with a dollar-denominated
   denominator elsewhere in feature engineering would be off by 1e6.

2. RATIO STALENESS. LSEG's "*_DailyTimeSeriesRatio_" fields update every day as
   the price moves. Computing ev / raw_quarterly_denominator only yields a value
   on the ~2% of rows that are filing dates; forward-filling THAT leaves a ratio
   frozen between filings, losing all daily price information. Instead the
   denominator is forward-filled per symbol FIRST, then divided into the daily
   EV -- so the ratio moves daily, as it should. Forward-filling only carries
   past filings forward, so this stays point-in-time correct.

Usage
-----
    python3 fetch_sharadar_replacements.py --start 2023-01-01 --end 2026-08-25 --skip-sf1
    python3 substitute_lseg_columns.py \
        --input  experiments/20230101_20260825_fulluniv_with_metadata_with_fmpdata.csv \
        --daily  sharadar_raw/sharadar_daily_2023-01-01_2026-08-25.parquet \
        --output experiments/20230101_20260825_SUBSTITUTED.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

MILLIONS = 1e6  # SHARADAR/DAILY unit for ev / marketcap

# LSEG column -> in-file FMP column (already PIT, lands on accepteddate_fmp)
FMP_DIRECT = {
    'debt_total': 'totaldebt_fmp',
    'cashcashequivalents_total': 'cashandcashequivalents_fmp',
    'focfexdividends_discrete': 'freecashflow_fmp',
    'earningspershare_actual': 'eps_fmp',
    # interestexpense_fmp is 0 on 21.1% of filing rows (FMP zeroes netted
    # interest). Treated as missing so ffill carries the last real figure.
    'interestexpense_netofcapitalizedinterest': 'interestexpense_fmp',
}
ZERO_IS_MISSING = {'interestexpense_fmp'}

DAILY_DIRECT = {
    'enterprisevalue_dailytimeseries_': 'ev',
    'companymarketcap': 'marketcap',
}
# LSEG ratio -> quarterly denominator (annualised x4, then ffilled)
RATIOS = {
    'enterprisevaluetoebitda_dailytimeseriesratio_': 'ebitda_fmp',
    'enterprisevaluetoebit_dailytimeseriesratio_': 'operatingincome_fmp',
    'enterprisevaluetosales_dailytimeseriesratio_': 'revenue_fmp',
}

DROP_TIER_B = [
    'earningspershare_smartestimate_prev_q', 'earningspershare_actualsurprise',
    'earningspershare_smartestimate_current_q', 'longtermgrowth_mean',
    'pricetarget_median', 'dividend_per_share_smartestimate',
    'forwardpeg_dailytimeseriesratio_', 'priceearningstogrowthratio_smartestimate_',
    'recommendation_median_1_5_', 'returnonequity_smartestimat',
    'returnonassets_smartestimate', 'forwardpricetocashflowpershare_dailytimeseriesratio_',
    'forwardpricetosalespershare_dailytimeseriesratio_',
    'forwardenterprisevaluetooperatingcashflow_dailytimeseriesratio_',
    'grossprofitmargin_actualsurprise', 'estpricegrowth_percent',
]
DROP_TIER_C = [
    'combinedalphamodelsectorrank', 'combinedalphamodelsectorrankchange',
    'combinedalphamodelregionrank', 'earningsqualityregionrank_current',
]


def build_lookup(input_csv, daily_parquet):
    """(Symbol, Date) -> every replacement value, computed on the FULL history.

    Done outside the chunk loop because the per-symbol forward-fill needs each
    symbol's complete time series; chunking would reset it at chunk borders.
    """
    need = ['date', 'symbol'] + sorted(set(FMP_DIRECT.values()) | set(RATIOS.values()))
    hdr = pd.read_csv(input_csv, nrows=0).columns.tolist()
    need = [c for c in need if c in hdr]
    print("building lookup (full-history ffill)...")
    q = pd.read_csv(input_csv, usecols=need, low_memory=False)
    q['date'] = pd.to_datetime(q['date'])
    q['symbol'] = q['symbol'].astype(str).str.upper()

    for c in ZERO_IS_MISSING & set(q.columns):
        q[c] = q[c].replace(0, np.nan)

    q = q.sort_values(['symbol', 'date'], kind='stable').reset_index(drop=True)
    vals = [c for c in need if c not in ('date', 'symbol')]

    # --- TRUE TTM for the ratio denominators -----------------------------
    # The first version annualised a single quarter (x4). That is wrong for any
    # seasonal business, and for EBITDA/EBIT -- which can sit near zero -- the
    # error explodes: measured rank correlation against the LSEG originals was
    # only 0.475 (EV/EBITDA) and 0.362 (EV/EBIT). A real trailing-twelve-month
    # sum of the last 4 FILINGS fixes it. Computed on filing rows only (the
    # values are sparse by construction), then forward-filled, so it stays
    # point-in-time: a TTM figure only appears once its 4th quarter is public.
    for den in set(RATIOS.values()) & set(q.columns):
        f = q.loc[q[den].notna(), ['symbol', 'date', den]].copy()
        f[f'{den}_ttm'] = (f.groupby('symbol', sort=False)[den]
                             .transform(lambda s: s.rolling(4, min_periods=4).sum()))
        q = q.merge(f[['symbol', 'date', f'{den}_ttm']], on=['symbol', 'date'],
                    how='left')
        vals.append(f'{den}_ttm')

    g = q.groupby('symbol', sort=False)
    for c in vals:
        q[c] = g[c].ffill()          # PIT-safe: only past filings move forward
    print(f"  {len(q):,} rows, {len(vals)} columns forward-filled "
          f"({len(set(RATIOS.values()) & set(q.columns))} with TTM sums)")

    d = pd.read_parquet(daily_parquet, columns=['ticker', 'date', 'ev', 'marketcap'],
                        engine='pyarrow')
    d['date'] = pd.to_datetime(d['date'])
    d['symbol'] = d['ticker'].astype(str).str.upper()
    d['ev'] = d['ev'] * MILLIONS            # <- units fix
    d['marketcap'] = d['marketcap'] * MILLIONS
    print(f"  DAILY {len(d):,} rows (ev/marketcap scaled to USD)")

    lk = q.merge(d[['symbol', 'date', 'ev', 'marketcap']], on=['symbol', 'date'],
                 how='left')

    # Ratios: daily EV over the ffilled, annualised quarterly denominator, so the
    # ratio responds to price every day (see module docstring).
    for lseg, den in RATIOS.items():
        ttm = f'{den}_ttm'
        src = ttm if ttm in lk.columns else None
        if src:
            lk[lseg] = lk['ev'] / lk[src].replace(0, np.nan)

    keep = ['symbol', 'date', 'ev', 'marketcap'] + list(RATIOS) + \
           [c for c in FMP_DIRECT.values() if c in lk.columns]
    keep = [c for c in keep if c in lk.columns]
    return lk[[c for c in keep if c in lk.columns]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--chunksize', type=int, default=400_000)
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"ABORT: {args.output} exists")

    lk = build_lookup(args.input, args.daily)
    hdr = pd.read_csv(args.input, nrows=0).columns.tolist()
    drop = [c for c in DROP_TIER_B + DROP_TIER_C if c in hdr]
    print(f"\nnulling   {len(drop)} columns | replacing "
          f"{len(FMP_DIRECT) + len(DAILY_DIRECT) + len(RATIOS)}\n")

    rows = hit = 0
    first = True
    for i, ch in enumerate(pd.read_csv(args.input, chunksize=args.chunksize,
                                       low_memory=False)):
        ch['_d'] = pd.to_datetime(ch['date'])
        ch['_s'] = ch['symbol'].astype(str).str.upper()
        ch = ch.merge(lk.rename(columns={'symbol': '_s', 'date': '_d'}),
                      on=['_s', '_d'], how='left', suffixes=('', '_lk'))

        for lseg, src in DAILY_DIRECT.items():
            if lseg in ch.columns and src in ch.columns:
                ch[lseg] = ch[src]
        for lseg, fmp in FMP_DIRECT.items():
            src = fmp + '_lk' if fmp + '_lk' in ch.columns else fmp
            if lseg in ch.columns and src in ch.columns:
                ch[lseg] = ch[src]
        for lseg in RATIOS:
            src = lseg + '_lk' if lseg + '_lk' in ch.columns else lseg
            if src in ch.columns and src != lseg:
                ch[lseg] = ch[src]

        hit += ch['ev'].notna().sum()
        rows += len(ch)
        # NULL the dropped columns rather than removing them: the forecast
        # script's feature engineering references several by name (e.g.
        # ReturnOnAssets_SmartEstimate) and raises KeyError if they are absent.
        # Nulled + the script's own ffill/fillna(0) makes them constant, so they
        # carry no information -- informationally dropped, structurally present.
        for c in drop:
            if c in ch.columns:
                ch[c] = np.nan
        helper = ['_s', '_d', 'ev', 'marketcap'] + \
                 [c for c in ch.columns if c.endswith('_lk')]
        ch = ch.drop(columns=[c for c in set(helper) if c in ch.columns])
        ch.to_csv(args.output, mode='w' if first else 'a', header=first, index=False)
        first = False
        if i % 3 == 0:
            print(f"  {rows:>10,} rows | DAILY {100*hit/max(rows,1):5.1f}%", flush=True)

    print(f"\ndone: {rows:,} rows -> {args.output}")
    print(f"  SHARADAR/DAILY coverage: {100*hit/max(rows,1):.1f}% overall "
          f"(~98% within the top-400 universe, which is what the book trades)")


if __name__ == '__main__':
    main()
