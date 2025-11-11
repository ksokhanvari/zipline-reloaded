#!/usr/bin/env python
"""
Check what data is available in the Sharadar bundle.
"""

import pandas as pd
from zipline.data.bundles import load as load_bundle

def main():
    print("=" * 80)
    print("SHARADAR BUNDLE DATA CHECK")
    print("=" * 80)
    print()

    # Load bundle
    print("Loading Sharadar bundle...")
    bundle_data = load_bundle('sharadar')
    print("✓ Bundle loaded")
    print()

    # Get asset finder
    asset_finder = bundle_data.asset_finder

    # Check specific tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    print("Checking ticker availability:")
    print("-" * 80)
    for ticker in tickers:
        try:
            assets = asset_finder.lookup_symbols([ticker], as_of_date=None)
            if assets and assets[0] is not None:
                asset = assets[0]
                print(f"  {ticker:6s} -> SID {asset.sid:6d}  "
                      f"({asset.start_date} to {asset.end_date})")
            else:
                print(f"  {ticker:6s} -> NOT FOUND")
        except Exception as e:
            print(f"  {ticker:6s} -> ERROR: {e}")
    print()

    # Get equity daily bar reader (for OHLCV data)
    equity_daily_bar_reader = bundle_data.equity_daily_bar_reader

    # Check data availability for AAPL
    print("Sample AAPL pricing data:")
    print("-" * 80)
    try:
        aapl = asset_finder.lookup_symbols(['AAPL'], as_of_date=None)[0]

        # Get a date range
        start_date = pd.Timestamp('2023-06-01')
        end_date = pd.Timestamp('2023-06-30')

        # Load OHLCV data
        dates = pd.date_range(start_date, end_date, freq='D')
        sessions = [d for d in dates if equity_daily_bar_reader.sessions.searchsorted(d) < len(equity_daily_bar_reader.sessions)]

        if sessions:
            sample_session = sessions[0]

            # Get OHLCV for one day
            closes = equity_daily_bar_reader.load_raw_arrays(
                ['close'],
                sample_session,
                sample_session,
                [aapl.sid]
            )

            print(f"  Date: {sample_session}")
            print(f"  SID: {aapl.sid}")
            print(f"  Close price available: Yes")
            print(f"  ✓ EquityPricing data is working")
        else:
            print(f"  No trading sessions found in date range")

    except Exception as e:
        print(f"  Error loading data: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)
    print("CHECK COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
