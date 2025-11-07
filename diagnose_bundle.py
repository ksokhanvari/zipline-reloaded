#!/usr/bin/env python
"""
Diagnostic script to check bundle data availability.

Usage:
    python diagnose_bundle.py [bundle_name] [symbol]

Example:
    python diagnose_bundle.py quandl SPY
    python diagnose_bundle.py sharadar SPY
"""

import sys
import pandas as pd
from zipline.data.bundles import load, register, bundles
from zipline.utils.calendar_utils import get_calendar


def register_sharadar_bundle():
    """Register the sharadar bundle if not already registered."""
    if 'sharadar' not in bundles:
        try:
            from zipline.data.bundles.sharadar_bundle import sharadar_bundle
            register('sharadar', sharadar_bundle(tickers=None, incremental=True, include_funds=True))
            print("✓ Registered sharadar bundle\n")
        except ImportError:
            print("⚠ Warning: Could not import sharadar_bundle module\n")
        except Exception as e:
            print(f"⚠ Warning: Could not register sharadar bundle: {e}\n")


def diagnose_bundle(bundle_name='quandl', symbol='SPY'):
    """
    Check bundle data availability for a specific symbol.

    Parameters
    ----------
    bundle_name : str
        Name of the bundle to check
    symbol : str
        Symbol to check (e.g., 'SPY')
    """
    print(f"\n{'='*70}")
    print(f"Bundle Data Diagnostics: {bundle_name}")
    print(f"{'='*70}\n")

    # Register sharadar bundle if needed
    if bundle_name.lower() == 'sharadar':
        register_sharadar_bundle()

    # Check if bundle is registered
    if bundle_name not in bundles:
        print(f"✗ Error: Bundle '{bundle_name}' is not registered\n")
        print("Available bundles:")
        for b in sorted(bundles.keys()):
            if not b.startswith('.'):
                print(f"  - {b}")
        print("\nTo register a bundle, see the Zipline documentation.")
        return

    try:
        # Load the bundle
        print(f"Loading bundle '{bundle_name}'...")
        bundle_data = load(bundle_name)
        print("✓ Bundle loaded successfully\n")

        # Get the asset
        print(f"Looking up symbol: {symbol}")
        try:
            asset = bundle_data.asset_finder.lookup_symbol(symbol, as_of_date=None)
            print(f"✓ Found asset: {asset}")
            print(f"  - SID: {asset.sid}")
            print(f"  - Start date: {asset.start_date}")
            print(f"  - End date: {asset.end_date}")
            print(f"  - Exchange: {asset.exchange}\n")
        except Exception as e:
            print(f"✗ Error finding symbol {symbol}: {e}")
            return

        # Check price data availability
        print("Checking price data availability...")
        calendar = get_calendar(asset.exchange)

        # Get all trading days in asset's range
        trading_days = calendar.sessions_in_range(
            asset.start_date,
            asset.end_date
        )

        print(f"  - Asset listed for {len(trading_days)} trading days")
        print(f"  - From: {trading_days[0]}")
        print(f"  - To: {trading_days[-1]}\n")

        # Try to get price data
        print("Fetching daily price data...")
        try:
            # Get the equity daily bar reader
            daily_bar_reader = bundle_data.equity_daily_bar_reader

            # Try to load price data for the asset
            # Make sure dates are timezone-aware for comparison
            end_date = asset.end_date
            if end_date.tz is None:
                end_date = end_date.tz_localize('UTC')

            now = pd.Timestamp.now(tz='UTC')
            sessions = calendar.sessions_in_range(
                asset.start_date,
                min(end_date, now)
            )

            # Get close prices
            closes = daily_bar_reader.load_raw_arrays(
                columns=['close'],
                start_date=sessions[0],
                end_date=sessions[-1],
                assets=[asset.sid]
            )[0][:, 0]

            # Find missing data (NaN or 0)
            valid_data = ~pd.isna(closes) & (closes > 0)
            num_valid = valid_data.sum()
            num_missing = len(closes) - num_valid

            print(f"  - Total sessions checked: {len(sessions)}")
            print(f"  - Sessions with valid data: {num_valid}")
            print(f"  - Sessions with missing data: {num_missing}")
            print(f"  - Data coverage: {num_valid/len(sessions)*100:.1f}%\n")

            if num_missing > 0:
                # Show first and last 10 missing dates
                missing_indices = ~valid_data
                missing_dates = sessions[missing_indices]

                print(f"⚠ WARNING: Found {num_missing} sessions with missing data!\n")

                if len(missing_dates) <= 20:
                    print("Missing dates:")
                    for date in missing_dates:
                        print(f"  - {date.date()}")
                else:
                    print("First 10 missing dates:")
                    for date in missing_dates[:10]:
                        print(f"  - {date.date()}")
                    print(f"  ... ({num_missing - 20} more) ...")
                    print("Last 10 missing dates:")
                    for date in missing_dates[-10:]:
                        print(f"  - {date.date()}")

                print()

                # Check if recent data is missing
                recent_sessions = sessions[-30:]  # Last 30 trading days
                recent_closes = closes[-30:]
                recent_valid = ~pd.isna(recent_closes) & (recent_closes > 0)
                recent_missing = (~recent_valid).sum()

                if recent_missing > 0:
                    print(f"🔴 CRITICAL: Missing {recent_missing} of last 30 trading days!")
                    print("   This will cause backtests to fail.\n")
                    print("   Recent missing dates:")
                    for i, (date, valid) in enumerate(zip(recent_sessions, recent_valid)):
                        if not valid:
                            print(f"     - {date.date()}")
                    print()

            else:
                print("✓ All sessions have valid price data!\n")

            # Show sample of recent data
            print("Recent price data (last 10 sessions):")
            for session, close in zip(sessions[-10:], closes[-10:]):
                status = "✓" if not pd.isna(close) and close > 0 else "✗"
                close_str = f"${close:.2f}" if not pd.isna(close) else "MISSING"
                print(f"  {status} {session.date()}: {close_str}")

        except Exception as e:
            print(f"✗ Error fetching price data: {e}")
            import traceback
            traceback.print_exc()
            return

        print(f"\n{'='*70}")
        print("RECOMMENDATIONS:")
        print(f"{'='*70}\n")

        # Check for critically small dataset (less than 10 days)
        if len(sessions) < 10:
            print("🔴 CRITICAL ISSUE: Bundle has less than 10 days of data!")
            print(f"   Only {len(sessions)} trading day(s) available.\n")
            print("This indicates a failed or incomplete bundle ingestion.\n")
            print("REQUIRED ACTION:")
            print("  1. Delete the incomplete bundle data")
            print("  2. Re-ingest from scratch:\n")
            print(f"     zipline ingest -b {bundle_name}\n")
            print("  This will download the complete historical dataset.")
            print("  Initial ingest may take 15-30 minutes.\n")
            return

        if num_missing > 0:
            print("Your bundle has missing data. To fix this:\n")
            print("1. Re-ingest the bundle to get complete data:")
            print(f"   zipline ingest -b {bundle_name}\n")
            print("2. Or adjust your backtest dates to avoid missing data:")
            if num_valid > 0:
                first_valid_idx = valid_data.argmax()  # First True value
                last_valid_idx = len(valid_data) - valid_data[::-1].argmax() - 1
                print(f"   Start date: {sessions[first_valid_idx].date()}")
                print(f"   End date: {sessions[last_valid_idx].date()}\n")
        else:
            print("✓ Bundle data looks good!")
            print(f"  You can safely backtest from {sessions[0].date()} to {sessions[-1].date()}\n")

    except Exception as e:
        print(f"✗ Error loading bundle: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    bundle_name = sys.argv[1] if len(sys.argv) > 1 else 'quandl'
    symbol = sys.argv[2] if len(sys.argv) > 2 else 'SPY'

    diagnose_bundle(bundle_name, symbol)
