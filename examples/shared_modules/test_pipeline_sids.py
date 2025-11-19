#!/usr/bin/env python
"""
Test what SIDs the pipeline is actually using.
"""
import sys
sys.path.insert(0, '/app/examples/shared_modules')

import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals
from zipline.pipeline.filters import StaticAssets
from zipline.data.bundles import load as load_bundle, register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle
from zipline.pipeline.engine import SimplePipelineEngine

# Register bundle
register('sharadar', sharadar_bundle())

UNIVERSE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V', 'WMT', 'XOM', 'TSLA']

print("=" * 80)
print("TESTING PIPELINE ASSET SELECTION")
print("=" * 80)
print()

# Load bundle
print("Loading bundle...")
bundle_data = load_bundle('sharadar')
asset_finder = bundle_data.asset_finder

# Get assets for our tickers (same as make_pipeline())
print("Looking up tickers...")
assets = []
for ticker in UNIVERSE_TICKERS:
    try:
        asset = asset_finder.lookup_symbol(ticker, as_of_date=None)
        if asset:
            assets.append(asset)
            print(f"  {ticker:10s} -> SID: {asset.sid:6d} ({asset})")
        else:
            print(f"  {ticker:10s} -> NOT FOUND")
    except Exception as e:
        print(f"  {ticker:10s} -> ERROR: {e}")

print()
print(f"Total assets in pipeline universe: {len(assets)}")
print()

# Create the universe filter
universe = StaticAssets(assets)

# Create a simple pipeline
pipeline = Pipeline(
    columns={
        's_roe': SharadarFundamentals.roe.latest,
    },
    screen=universe,
)

print("Pipeline created with universe of", len(assets), "assets")
print()
print("Asset SIDs in universe:")
for asset in assets:
    print(f"  {asset.symbol:10s} SID: {asset.sid}")

print()
print("=" * 80)
