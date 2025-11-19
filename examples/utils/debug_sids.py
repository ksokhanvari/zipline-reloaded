#!/usr/bin/env python
"""
Debug script to check which SIDs are being used for the universe tickers.
"""
import sys
sys.path.insert(0, '/app/examples/shared_modules')

from zipline.data.bundles import register, load as load_bundle
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

# Register bundle
register('sharadar', sharadar_bundle())

# Universe tickers from the notebook
UNIVERSE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V', 'WMT', 'XOM', 'TSLA']

print("=" * 80)
print("CHECKING SIDS FOR UNIVERSE TICKERS")
print("=" * 80)
print()

# Load Sharadar bundle
print("Loading Sharadar bundle...")
bundle_data = load_bundle('sharadar')
asset_finder = bundle_data.asset_finder
print("✓ Bundle loaded")
print()

print("Universe tickers and their SIDs:")
print("-" * 80)
for ticker in UNIVERSE_TICKERS:
    try:
        asset = asset_finder.lookup_symbol(ticker, as_of_date=None)
        if asset is not None:
            print(f"  {ticker:10s} -> SID: {asset.sid}")
        else:
            print(f"  {ticker:10s} -> NOT FOUND")
    except Exception as e:
        print(f"  {ticker:10s} -> ERROR: {e}")

print()
print("=" * 80)
