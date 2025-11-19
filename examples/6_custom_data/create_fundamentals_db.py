#!/usr/bin/env python
"""
Create and populate the fundamentals custom database.

This script creates a fundamentals database and loads sample data from CSV.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.register_bundles import ensure_bundles_registered
from zipline.data.bundles import load as load_bundle
from zipline.data.custom import create_custom_db, load_csv_to_db

# Ensure bundles are registered for standalone execution
ensure_bundles_registered()

# Configuration
DB_CODE = "fundamentals"
BAR_SIZE = "1 day"
SAMPLE_DATA_FILE = Path(__file__).parent / "sample_fundamentals.csv"
BUNDLE_NAME = "sharadar"

# Define column types matching the Fundamentals Database class
FUNDAMENTAL_COLUMNS = {
    'Symbol': 'text',  # Ticker symbol (AAPL, MSFT, etc.)
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

    # Add Symbol column if not present (copy from Ticker column)
    if 'Symbol' not in df.columns and 'Ticker' in df.columns:
        df['Symbol'] = df['Ticker']
        print(f"✓ Added Symbol column from Ticker column")

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

    # Step 4: Build ticker-to-SID mapping
    print("Building ticker-to-SID mapping...")
    tickers = df['Ticker'].unique()
    sid_map = {}

    for ticker in tickers:
        try:
            # Get the most recent asset for this ticker
            assets = asset_finder.lookup_symbols([ticker], as_of_date=None)
            if assets and assets[0] is not None:
                sid_map[ticker] = assets[0].sid
                print(f"  {ticker} -> SID {assets[0].sid}")
            else:
                print(f"  ⚠ {ticker} not found in bundle")
        except Exception as e:
            print(f"  ⚠ Error looking up {ticker}: {e}")

    if not sid_map:
        print("✗ No tickers could be converted to SIDs")
        return 1

    print(f"✓ Converted {len(sid_map)} tickers to SIDs")
    print()

    # Step 5: Save modified DataFrame with Symbol column to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        temp_csv_path = tmp.name

    # Step 6: Load CSV data into database
    print("Loading data into database...")
    try:
        result = load_csv_to_db(
            csv_path=temp_csv_path,  # Use temp file with Symbol column
            db_code=DB_CODE,
            sid_map=sid_map,
            id_col='Ticker',  # CSV uses 'Ticker' not 'Symbol'
            date_col='Date',
            fail_on_unmapped=False,  # Continue even if some tickers aren't found
        )
        print(f"✓ Inserted {result['rows_inserted']} rows into {DB_CODE} database")

        if result['unmapped_ids']:
            print(f"  ⚠ {len(result['unmapped_ids'])} unmapped tickers: {', '.join(result['unmapped_ids'])}")

        if result['errors']:
            print(f"  ⚠ {len(result['errors'])} errors occurred")
            for error in result['errors']:
                print(f"    - {error}")
        print()
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up temp file
        import os
        if os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)

    # Step 7: Verify
    print("=" * 80)
    print("DATABASE CREATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print(f"Database location: {db_path}")
    print(f"Records inserted: {result['rows_inserted']}")
    print(f"Tickers: {', '.join(sorted(sid_map.keys()))}")
    print()
    print("You can now run the backtest with fundamentals!")
    print()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
