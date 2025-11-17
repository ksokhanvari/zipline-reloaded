#!/usr/bin/env python
"""
Download Sharadar fundamentals for an existing bundle.

This script downloads SF1 fundamentals and adds them to an existing bundle
without re-downloading pricing data.

Usage:
    python scripts/download_fundamentals_only.py
    python scripts/download_fundamentals_only.py --bundle sharadar-tech
    python scripts/download_fundamentals_only.py --start-date 2020-01-01
"""

import os
import sys
from pathlib import Path
import pandas as pd
import argparse

# Add src to path so we can import zipline modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from zipline.data.bundles.sharadar_bundle import (
    download_sharadar_fundamentals,
    store_fundamentals,
)


def find_latest_bundle(bundle_name):
    """Find the most recent ingestion of a bundle."""
    zipline_root = Path(os.environ.get('ZIPLINE_ROOT', Path.home() / '.zipline'))
    bundle_dir = zipline_root / 'data' / bundle_name

    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"Bundle '{bundle_name}' not found at {bundle_dir}. "
            f"Have you ingested it yet?"
        )

    # Find most recent ingestion
    ingestion_dirs = sorted(bundle_dir.glob('*'))
    if not ingestion_dirs:
        raise FileNotFoundError(
            f"No ingestions found for bundle '{bundle_name}'"
        )

    return ingestion_dirs[-1]


def load_symbol_to_sid(bundle_path):
    """Load symbol → SID mapping from bundle's assets database."""
    import sqlite3

    assets_db = list(bundle_path.glob('assets-*.sqlite'))
    if not assets_db:
        assets_db = list(bundle_path.glob('assets-*.db'))

    if not assets_db:
        raise FileNotFoundError(f"No assets database found in {bundle_path}")

    conn = sqlite3.connect(str(assets_db[0]))
    assets_df = pd.read_sql("""
        SELECT DISTINCT esm.sid, esm.symbol
        FROM equity_symbol_mappings esm
        INNER JOIN (
            SELECT sid, MAX(end_date) as max_end_date
            FROM equity_symbol_mappings
            GROUP BY sid
        ) latest ON esm.sid = latest.sid AND esm.end_date = latest.max_end_date
    """, conn)
    conn.close()

    return dict(zip(assets_df['symbol'], assets_df['sid']))


def main():
    parser = argparse.ArgumentParser(
        description='Download Sharadar fundamentals for an existing bundle'
    )
    parser.add_argument(
        '--bundle',
        default='sharadar',
        help='Bundle name (default: sharadar)'
    )
    parser.add_argument(
        '--start-date',
        default='1998-01-01',
        help='Start date for fundamentals (default: 1998-01-01)'
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='End date for fundamentals (default: today)'
    )
    parser.add_argument(
        '--tickers',
        nargs='*',
        default=None,
        help='Specific tickers to download (default: all tickers in bundle)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get('NASDAQ_DATA_LINK_API_KEY')
    if not api_key:
        print("❌ Error: NASDAQ_DATA_LINK_API_KEY environment variable not set")
        sys.exit(1)

    print("=" * 60)
    print("Sharadar Fundamentals Download (Standalone)")
    print("=" * 60)
    print(f"Bundle: {args.bundle}")
    print(f"Date range: {args.start_date} to {args.end_date or 'today'}")
    print("")

    # Find latest bundle ingestion
    print("Step 1/4: Finding latest bundle ingestion...")
    bundle_path = find_latest_bundle(args.bundle)
    print(f"  ✓ Found: {bundle_path.name}")

    # Load symbol → SID mapping
    print("\nStep 2/4: Loading symbol → SID mapping from bundle...")
    symbol_to_sid = load_symbol_to_sid(bundle_path)
    print(f"  ✓ Loaded {len(symbol_to_sid)} symbols")

    # Download fundamentals
    print("\nStep 3/4: Downloading SF1 fundamentals...")
    sf1_data = download_sharadar_fundamentals(
        api_key=api_key,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if sf1_data.empty:
        print("❌ No fundamentals data downloaded")
        sys.exit(1)

    # Add SIDs
    sf1_data['sid'] = sf1_data['permaticker'].astype(int)

    # Verify SID matching
    pricing_sids = set(symbol_to_sid.values())
    fundamentals_sids = set(sf1_data['sid'].unique())

    matched_sids = fundamentals_sids & pricing_sids
    unmatched_sids = fundamentals_sids - pricing_sids

    print(f"  SID matching: {len(matched_sids)} matched, {len(unmatched_sids)} unmatched")

    # Keep only fundamentals for assets with pricing data
    sf1_data = sf1_data[sf1_data['sid'].isin(pricing_sids)].copy()

    print(f"  ✓ Downloaded {len(sf1_data):,} fundamental records")
    print(f"    Tickers: {sf1_data['ticker'].nunique()}")
    print(f"    Date range: {sf1_data['datekey'].min()} to {sf1_data['datekey'].max()}")

    # Store fundamentals
    print("\nStep 4/4: Storing fundamentals...")
    store_fundamentals(sf1_data, str(bundle_path), symbol_to_sid)

    print("\n" + "=" * 60)
    print("✓ Fundamentals download complete!")
    print("=" * 60)
    print(f"Stored in: {bundle_path / 'fundamentals' / 'sf1.h5'}")
    print("")
    print("You can now use fundamentals in your strategies:")
    print("  from zipline.pipeline.data.sharadar import SharadarFundamentals")
    print("  revenue = SharadarFundamentals.revenue.latest")
    print("")


if __name__ == '__main__':
    main()
