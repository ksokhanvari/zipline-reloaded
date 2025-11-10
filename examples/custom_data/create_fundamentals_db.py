#!/usr/bin/env python
"""
Create and populate the fundamentals custom database.

This script creates a fundamentals database and loads sample data from CSV.
"""

import pandas as pd
from pathlib import Path

from zipline.data.bundles import load as load_bundle
from zipline.data.custom import create_custom_db, insert_data

# Configuration
DB_CODE = "fundamentals"
BAR_SIZE = "1 day"
SAMPLE_DATA_FILE = Path(__file__).parent / "sample_fundamentals.csv"
BUNDLE_NAME = "sharadar"

# Define column types matching the Fundamentals Database class
FUNDAMENTAL_COLUMNS = {
    'Revenue': 'float',
    'NetIncome': 'float',
    'TotalAssets': 'float',
    'TotalEquity': 'float',
    'SharesOutstanding': 'float',
    'EPS': 'float',
    'BookValuePerShare': 'float',
    'ROE': 'float',
    'DebtToEquity': 'float',
    'CurrentRatio': 'float',
    'PERatio': 'float',
    'Sector': 'text',
}


def main():
    print("=" * 80)
    print("CREATING FUNDAMENTALS DATABASE")
    print("=" * 80)
    print()

    # Step 1: Create the database
    print(f"Creating database: {DB_CODE}")
    try:
        db_path = create_custom_db(
            db_code=DB_CODE,
            bar_size=BAR_SIZE,
            columns=FUNDAMENTAL_COLUMNS,
        )
        print(f"✓ Database created: {db_path}")
        print()
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return 1

    # Step 2: Load sample data
    print(f"Loading sample data from: {SAMPLE_DATA_FILE}")
    if not SAMPLE_DATA_FILE.exists():
        print(f"✗ Sample data file not found: {SAMPLE_DATA_FILE}")
        return 1

    df = pd.read_csv(SAMPLE_DATA_FILE)
    print(f"✓ Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    print()

    # Step 3: Load bundle to get asset finder
    print(f"Loading bundle: {BUNDLE_NAME}")
    try:
        bundle_data = load_bundle(BUNDLE_NAME)
        asset_finder = bundle_data.asset_finder
        print(f"✓ Bundle loaded")
        print()
    except Exception as e:
        print(f"✗ Error loading bundle: {e}")
        print("  Make sure you've ingested the bundle first:")
        print(f"    zipline ingest -b {BUNDLE_NAME}")
        return 1

    # Step 4: Convert tickers to SIDs
    print("Converting tickers to SIDs...")
    tickers = df['Ticker'].unique()
    ticker_to_sid = {}

    for ticker in tickers:
        try:
            # Get the most recent asset for this ticker
            assets = asset_finder.lookup_symbols([ticker], as_of_date=None)
            if assets and assets[0] is not None:
                ticker_to_sid[ticker] = assets[0].sid
                print(f"  {ticker} -> SID {assets[0].sid}")
            else:
                print(f"  ⚠ {ticker} not found in bundle")
        except Exception as e:
            print(f"  ⚠ Error looking up {ticker}: {e}")

    if not ticker_to_sid:
        print("✗ No tickers could be converted to SIDs")
        return 1

    print(f"✓ Converted {len(ticker_to_sid)} tickers to SIDs")
    print()

    # Step 5: Prepare data for insertion
    print("Preparing data for insertion...")
    df['Sid'] = df['Ticker'].map(ticker_to_sid)

    # Drop rows where ticker wasn't found
    df = df.dropna(subset=['Sid'])
    df['Sid'] = df['Sid'].astype(int)

    # Drop the Ticker column (we only need Sid)
    df = df.drop(columns=['Ticker'])

    # Ensure Date is in the right format
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    print(f"✓ Prepared {len(df)} rows for insertion")
    print()

    # Step 6: Insert data
    print("Inserting data into database...")
    try:
        insert_data(DB_CODE, df)
        print(f"✓ Inserted {len(df)} rows into {DB_CODE} database")
        print()
    except Exception as e:
        print(f"✗ Error inserting data: {e}")
        return 1

    # Step 7: Verify
    print("=" * 80)
    print("DATABASE CREATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print(f"Database location: {db_path}")
    print(f"Records inserted: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Tickers: {', '.join(sorted(ticker_to_sid.keys()))}")
    print()
    print("You can now run the backtest with fundamentals!")
    print()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
