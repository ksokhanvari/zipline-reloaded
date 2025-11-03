#!/usr/bin/env python
"""
Test script for the new Sharadar nasdaqdatalink.get_table() implementation.

This script tests the download_sharadar_table function to ensure it works
with the new API method and returns the expected data structure.
"""

import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zipline.data.bundles.sharadar_bundle import download_sharadar_table


def test_sep_download():
    """Test downloading SEP table with limited tickers."""
    print("\n" + "="*70)
    print("Testing SEP Download with nasdaqdatalink.get_table()")
    print("="*70)

    # Get API key
    api_key = os.environ.get('NASDAQ_DATA_LINK_API_KEY')
    if not api_key:
        print("❌ NASDAQ_DATA_LINK_API_KEY not set in environment")
        print("   Set it with: export NASDAQ_DATA_LINK_API_KEY='your_key'")
        return False

    print(f"✓ API key found: {api_key[:10]}...")

    # Test with a small subset of tickers
    test_tickers = ['AAPL', 'MSFT']
    test_start = '2024-01-01'
    test_end = '2024-01-31'

    print(f"\nTest parameters:")
    print(f"  Tickers: {test_tickers}")
    print(f"  Date range: {test_start} to {test_end}")

    try:
        # Download data
        df = download_sharadar_table(
            table='SEP',
            api_key=api_key,
            tickers=test_tickers,
            start_date=test_start,
            end_date=test_end,
        )

        print(f"\n✓ Download successful!")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")

        # Verify expected columns
        expected_cols = ['ticker', 'date', 'open', 'high', 'low', 'close',
                        'volume', 'closeadj', 'closeunadj']

        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            print(f"\n❌ Missing expected columns: {missing_cols}")
            return False

        print(f"\n✓ All expected columns present")

        # Show sample data
        print(f"\nSample data (first 5 rows):")
        print(df.head())

        print(f"\nData types:")
        print(df.dtypes)

        # Verify data types
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            print(f"\n❌ 'date' column is not datetime type")
            return False

        print(f"\n✓ Date column is properly formatted as datetime")

        # Check for data completeness
        print(f"\nData completeness:")
        print(f"  Unique tickers: {df['ticker'].nunique()}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Missing values:")
        print(df.isnull().sum())

        print("\n" + "="*70)
        print("✓ TEST PASSED: SEP download works correctly!")
        print("="*70)
        return True

    except Exception as e:
        print(f"\n❌ Error during download:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_actions_download():
    """Test downloading ACTIONS table."""
    print("\n" + "="*70)
    print("Testing ACTIONS Download")
    print("="*70)

    api_key = os.environ.get('NASDAQ_DATA_LINK_API_KEY')
    if not api_key:
        print("❌ NASDAQ_DATA_LINK_API_KEY not set")
        return False

    test_tickers = ['AAPL']
    test_start = '2024-01-01'
    test_end = '2024-12-31'

    try:
        df = download_sharadar_table(
            table='ACTIONS',
            api_key=api_key,
            tickers=test_tickers,
            start_date=test_start,
            end_date=test_end,
        )

        print(f"✓ Downloaded {len(df):,} corporate actions")
        if not df.empty:
            print(f"  Columns: {list(df.columns)}")
            print(f"\nSample actions:")
            print(df.head())

        print("\n✓ TEST PASSED: ACTIONS download works!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Sharadar nasdaqdatalink.get_table() Implementation Test")
    print("="*70)

    # Run tests
    sep_passed = test_sep_download()
    actions_passed = test_actions_download()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"SEP Download:     {'✓ PASS' if sep_passed else '❌ FAIL'}")
    print(f"ACTIONS Download: {'✓ PASS' if actions_passed else '❌ FAIL'}")
    print("="*70)

    if sep_passed and actions_passed:
        print("\n✓ All tests passed! The implementation is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)
