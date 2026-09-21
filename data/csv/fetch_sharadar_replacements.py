#!/usr/bin/env python3
"""
Download the Sharadar data needed to replace LSEG columns in the training CSV.

Pulls two tables via the NASDAQ Data Link bulk export API (no nasdaqdatalink
dependency, so it runs outside the container):

  SHARADAR/DAILY  - daily valuation: ev, evebitda, marketcap, pe, pb, ps
  SHARADAR/SF1    - quarterly fundamentals, dimension ARQ (as-reported quarterly)

POINT-IN-TIME CORRECTNESS
-------------------------
SF1 rows carry both `calendardate` (the quarter they describe) and `datekey`
(the date the filing became public). Joining on calendardate would leak: Q1
figures are not knowable until the Q1 filing appears, typically 30-60 days
later. Everything here keys off `datekey`, so a row is only ever visible on
or after the day it was actually published. DAILY needs no such treatment --
it is already a daily observation.

Usage
-----
    python3 fetch_sharadar_replacements.py --start 2009-12-31 --end 2026-09-15
    python3 fetch_sharadar_replacements.py --start 2023-01-01 --end 2026-09-15 \
        --out-dir sharadar_raw
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

API = 'https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json'

# DAILY: everything we need for the daily-valuation replacements
DAILY_COLS = ['ticker', 'date', 'ev', 'evebitda', 'marketcap', 'pe', 'pb', 'ps']

# SEP: full OHLCV. Needed for the technical factor build -- the LSEG feed carries
# only a close, which rules out ATR/Stochastic/ADX/Ichimoku and every other
# indicator that reads the high-low range.
SEP_COLS = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'closeadj']

# SF1 (ARQ): fundamentals for the calculated columns.
#   debt, cashneq, intexp, eps            -> direct replacements
#   fcf, ncfdiv                           -> FOCFExDividends_Discrete
#   ebit, ebitda, revenue                 -> EV/EBIT, EV/EBITDA, EV/Sales
SF1_COLS = ['ticker', 'datekey', 'calendardate', 'dimension',
            'debt', 'cashneq', 'intexp', 'eps',
            'fcf', 'ncfdiv', 'ebit', 'ebitda', 'revenue']


def load_api_key():
    if os.environ.get('NASDAQ_DATA_LINK_API_KEY'):
        return os.environ['NASDAQ_DATA_LINK_API_KEY']
    # NOTE: this .env defines the key more than once (an early placeholder and
    # the real value further down). Take the LAST definition, which is what
    # shell `source` semantics would leave set, and skip obvious placeholders.
    for env in (Path(__file__).resolve().parents[2] / '.env',
                Path.cwd() / '.env'):
        if env.exists():
            found = re.findall(r'^NASDAQ_DATA_LINK_API_KEY\s*=\s*(\S+)',
                               env.read_text(), re.M)
            valid = [k.strip().strip('"\'') for k in found
                     if len(k.strip().strip('"\'')) >= 8 and '#' not in k]
            if valid:
                return valid[-1]
    sys.exit("No usable NASDAQ_DATA_LINK_API_KEY in environment or .env")


def bulk_export(table, api_key, params=None, timeout=60):
    """Request a bulk export, poll until the zip is ready, return a DataFrame."""
    q = {'qopts.export': 'true', 'api_key': api_key}
    q.update(params or {})
    print(f"  requesting {table} export...", flush=True)

    for attempt in range(1, 6):
        r = requests.get(API.format(table=table), params=q, timeout=timeout)
        if r.status_code == 200:
            break
        print(f"    HTTP {r.status_code} (attempt {attempt}/5), retrying in 30s",
              flush=True)
        time.sleep(30)
    else:
        sys.exit(f"{table}: export request failed after 5 attempts")

    info = r.json().get('datatable_bulk_download', {}).get('file', {})
    link, status = info.get('link'), info.get('status')

    # The export is generated asynchronously: the first response normally comes
    # back status='creating' with link=None. Poll until it turns 'fresh'/'regenerating'
    # and a link appears -- do NOT treat the missing link as an error up front.
    waited = 0
    while status == 'creating' and waited < 1800:
        time.sleep(20)
        waited += 20
        r = requests.get(API.format(table=table), params=q, timeout=timeout)
        info = r.json().get('datatable_bulk_download', {}).get('file', {})
        status, link = info.get('status'), info.get('link')
        if waited % 60 == 0:
            print(f"    waiting for export ({waited}s, status={status})", flush=True)

    if not link:
        sys.exit(f"{table}: no download link after {waited}s "
                 f"(status={status}) -- {r.text[:300]}")

    print(f"  downloading {table}...", flush=True)
    blob = requests.get(link, timeout=1800)
    blob.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(blob.content)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_csv(fh, low_memory=False)
    print(f"  {table}: {len(df):,} rows, {len(df.columns)} cols", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', required=True, help='YYYY-MM-DD')
    ap.add_argument('--out-dir', default='sharadar_raw')
    ap.add_argument('--skip-daily', action='store_true')
    ap.add_argument('--skip-sf1', action='store_true')
    ap.add_argument('--sep', action='store_true',
                    help='also download SHARADAR/SEP (OHLCV) for technical features')
    args = ap.parse_args()

    key = load_api_key()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    if not args.skip_daily:
        p = out / f'sharadar_daily_{args.start}_{args.end}.parquet'
        if p.exists():
            print(f"DAILY: {p} exists, skipping")
        else:
            df = bulk_export('DAILY', key, {
                'date.gte': args.start, 'date.lte': args.end,
                'qopts.columns': ','.join(DAILY_COLS),
            })
            df['date'] = pd.to_datetime(df['date'])
            df.to_parquet(p, engine='pyarrow', compression='snappy', index=False)
            print(f"  -> {p} ({p.stat().st_size/1e6:.0f} MB)\n")

    if not args.skip_sf1:
        p = out / f'sharadar_sf1_arq_{args.start}_{args.end}.parquet'
        if p.exists():
            print(f"SF1: {p} exists, skipping")
        else:
            # Filter on datekey, not calendardate: we want everything PUBLISHED
            # in the window. Reach back 1y so early dates have a prior filing.
            back = (pd.Timestamp(args.start) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            df = bulk_export('SF1', key, {
                'dimension': 'ARQ',
                'datekey.gte': back, 'datekey.lte': args.end,
                'qopts.columns': ','.join(SF1_COLS),
            })
            for c in ('datekey', 'calendardate'):
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c])
            df.to_parquet(p, engine='pyarrow', compression='snappy', index=False)
            print(f"  -> {p} ({p.stat().st_size/1e6:.0f} MB)\n")

    if args.sep:
        p = out / f'sharadar_sep_{args.start}_{args.end}.parquet'
        if p.exists():
            print(f'SEP: {p} exists, skipping')
        else:
            df = bulk_export('SEP', key, {
                'date.gte': args.start, 'date.lte': args.end,
                'qopts.columns': ','.join(SEP_COLS),
            })
            df['date'] = pd.to_datetime(df['date'])
            df.to_parquet(p, engine='pyarrow', compression='snappy', index=False)
            print(f'  -> {p} ({p.stat().st_size/1e6:.0f} MB)\n')

    print("done. next: substitute_lseg_columns.py")


if __name__ == '__main__':
    main()
