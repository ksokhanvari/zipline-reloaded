#!/usr/bin/env python3
"""
Download the FMP datasets that carry DATED history (usable point-in-time):

  earnings   /stable/earnings-calendar?from&to     all companies, ~90-day windows
             -> EPS + REVENUE actual vs estimate, per report date (2009+)
  insider    /stable/insider-trading/search?symbol  per symbol, paginated
             -> Form 4 transactions with filingDate
  grades     /stable/grades?symbol                  per symbol
             -> individual analyst rating actions with date (2012+)
  filings    /stable/income-statement?symbol&period=quarter
             -> filingDate / acceptedDate per fiscal quarter (fixes the
                quarter-end-stamped look-ahead in the production input)

Resumable: each symbol/window is cached as its own JSON under
experiments/FMP_EXTRAS/raw/, so a rerun only fetches what is missing.
Key is read from .env (FMP_API_KEY) -- never printed.

usage: python fetch_fmp_extras.py [earnings|insider|grades|filings|all] [--symbols N]
"""
import json, os, re, sys, time, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd

BASE = 'https://financialmodelingprep.com/stable'
OUT = Path('experiments/FMP_EXTRAS'); RAW = OUT / 'raw'
ENV = Path(__file__).resolve().parents[2] / '.env'


def api_key():
    keys = [m.group(1).strip() for line in ENV.read_text().splitlines()
            if (m := re.match(r'\s*FMP_API_KEY\s*=\s*(.+)', line))]
    keys = [k.strip('"\'') for k in keys if k and not k.endswith('#')]
    if not keys:
        sys.exit('ABORT: no FMP_API_KEY in .env')
    return keys[-1]                                   # last definition wins (the .env has duplicates elsewhere)


def get(path, params, key, cache, tries=4):
    cache = RAW / cache
    if cache.exists():
        return json.loads(cache.read_text())
    q = urllib.parse.urlencode({**params, 'apikey': key})
    for i in range(tries):
        try:
            with urllib.request.urlopen(f'{BASE}/{path}?{q}', timeout=60) as r:
                data = json.loads(r.read())
            if isinstance(data, dict) and ('Error Message' in data or 'error' in data):
                raise RuntimeError(str(data)[:200])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
            return data
        except Exception as e:
            if '429' in str(e) or 'Limit' in str(e):
                time.sleep(30 * (i + 1)); continue
            if i == tries - 1:
                print(f'   FAILED {path} {params}: {str(e)[:120]}'); return []
            time.sleep(2 * (i + 1))
    return []


def universe(n):
    """symbols that ever ranked in the top-1000 by market cap in the factor panel"""
    p = pd.read_parquet('experiments/FACTOR_STUDY/panel_monthly.parquet', columns=['Symbol', 'mc_rank'])
    s = p.loc[p['mc_rank'] <= n, 'Symbol'].dropna().unique().tolist()
    return sorted(s)


def fmp_symbol(s):
    # LSEG marks share classes with a lowercase suffix (BRKa); FMP uses BRK-A
    m = re.match(r'^([A-Z0-9]+)([a-z])$', s)
    return f'{m.group(1)}-{m.group(2).upper()}' if m else s


def earnings(key, syms):
    # NOTE: the all-company earnings-calendar endpoint caps each call at 4,000
    # rows (one month already hits it), so it silently drops data. Per-symbol
    # history has no cap and is US-only.
    rows = []
    for i, s in enumerate(syms, 1):
        f = fmp_symbol(s)
        d = get('earnings', {'symbol': f, 'limit': 200}, key, f'earnings_sym/{f}.json')
        rows += [{**r, 'lseg_symbol': s} for r in d]
        if i % 250 == 0: print(f'   earnings: {i}/{len(syms)} symbols', flush=True)
    df = pd.DataFrame(rows).drop_duplicates(['symbol', 'date'])
    df.to_parquet(OUT / 'earnings.parquet', index=False)
    print(f'earnings: {len(df):,} rows, {df.symbol.nunique():,} symbols, {df.date.min()} .. {df.date.max()}')


def per_symbol(key, syms, kind):
    rows = []
    for i, s in enumerate(syms, 1):
        f = fmp_symbol(s)
        if kind == 'insider':
            for page in range(0, 40):
                # Open-market PURCHASES only: the documented insider signal. The unfiltered
                # feed is dominated by award/tax/option filings (thousands of pages for
                # mega-caps) and would take ~8 hours for the universe.
                d = get('insider-trading/search', {'symbol': f, 'page': page, 'limit': 1000,
                                                   'transactionType': 'P-Purchase'}, key, f'insider_buys/{f}_{page}.json')
                rows += [{**r, 'lseg_symbol': s} for r in d]
                if len(d) < 1000: break
        elif kind == 'grades':
            rows += [{**r, 'lseg_symbol': s} for r in get('grades', {'symbol': f}, key, f'grades/{f}.json')]
        elif kind == 'filings':
            d = get('income-statement', {'symbol': f, 'period': 'quarter', 'limit': 120}, key, f'filings/{f}.json')
            rows += [{**{k: r.get(k) for k in ('symbol', 'date', 'fiscalYear', 'period', 'filingDate', 'acceptedDate')}, 'lseg_symbol': s} for r in d]
        if i % 200 == 0: print(f'   {kind}: {i}/{len(syms)} symbols', flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f'{kind}.parquet', index=False)
    print(f'{kind}: {len(df):,} rows, {df["symbol"].nunique() if len(df) else 0:,} symbols')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    n = int(sys.argv[sys.argv.index('--symbols') + 1]) if '--symbols' in sys.argv else 1000
    OUT.mkdir(parents=True, exist_ok=True)
    key = api_key()
    syms = universe(n)
    print(f'universe: {len(syms):,} symbols (ever top-{n} by mcap)')
    if what in ('earnings', 'all'): earnings(key, syms)
    for k in ('filings', 'grades', 'insider'):
        if what in (k, 'all'): per_symbol(key, syms, k)
