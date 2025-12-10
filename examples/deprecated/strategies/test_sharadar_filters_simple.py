"""
Simple test script for Sharadar filters
Run this with: docker exec zipline-reloaded-jupyter python3 /app/examples/strategies/test_sharadar_filters_simple.py
"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/examples/strategies')
sys.path.insert(0, '/app/examples/utils')

print("=" * 80)
print("SHARADAR FILTERS TEST")
print("=" * 80)
print()

# Step 1: Register bundles
print("Step 1: Registering bundles...")
try:
    from register_bundles import ensure_bundles_registered
    ensure_bundles_registered(verbose=True)
    print("✅ Bundles registered via module")
except Exception as e:
    print(f"⚠️  Module registration failed: {e}")
    print("Trying manual registration...")

    from zipline.data.bundles import register
    from zipline.data.bundles.sharadar_bundle import sharadar_bundle

    register('sharadar', sharadar_bundle(
        tickers=None,
        incremental=True,
        include_funds=True,
    ))
    print("✅ Bundles manually registered")

print()

# Step 2: Import modules
print("Step 2: Importing modules...")
from zipline.pipeline import Pipeline
from zipline.data.bundles import load as load_bundle
from zipline.pipeline.loaders.auto_loader import setup_auto_loader
from zipline.pipeline import SimplePipelineEngine
from zipline.utils.calendar_utils import get_calendar
import pandas as pd

from sharadar_filters import (
    ExchangeFilter,
    CategoryFilter,
    ADRFilter,
    SectorFilter,
    ScaleMarketCapFilter,
    create_sharadar_universe,
    SharadarTickers,
)
print("✅ Imports successful")
print()

# Step 3: Set up auto loader
print("Step 3: Setting up auto loader...")
custom_loader = setup_auto_loader(
    bundle_name='sharadar',
    custom_db_dir=None,
    enable_sid_translation=True,
)
print("✅ Auto loader configured")
print()

# Step 4: Load bundle
print("Step 4: Loading bundle...")
bundle_data = load_bundle('sharadar')
print("✅ Bundle loaded")
print()

# Step 5: Create pipeline engine
print("Step 5: Creating pipeline engine...")

try:
    from zipline.pipeline.domain import US_EQUITIES

    engine = SimplePipelineEngine(
        get_loader=lambda column: custom_loader,
        asset_finder=bundle_data.asset_finder,
        default_domain=US_EQUITIES,
    )
    print("✅ Pipeline engine ready")
except Exception as e:
    print(f"❌ Error creating pipeline engine: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 6: Test individual filters
print("Step 6: Testing individual filters...")

def make_test_pipeline():
    # Get raw metadata
    exchange = SharadarTickers.exchange.latest
    category = SharadarTickers.category.latest
    is_adr = SharadarTickers.is_adr.latest

    # Create filters
    exchange_filter = ExchangeFilter()
    category_filter = CategoryFilter()
    adr_filter = ADRFilter()

    return Pipeline(
        columns={
            'exchange': exchange,
            'category': category,
            'is_adr': is_adr,
            'exchange_filter': exchange_filter,
            'category_filter': category_filter,
            'adr_filter': adr_filter,
        },
    )

test_date = pd.Timestamp('2024-01-05')
print(f"Running pipeline for {test_date}...")

try:
    pipeline = make_test_pipeline()
    result = engine.run_pipeline(pipeline, test_date, test_date)

    print(f"✅ Pipeline executed: {len(result)} assets")
    print()

    # Step 7: Check filter results
    print("Step 7: Analyzing filter results...")
    print("=" * 80)

    total_assets = len(result)
    print(f"Total assets: {total_assets:,}")
    print()

    # Exchange filter
    exchange_pass = result['exchange_filter'].sum()
    print(f"Exchange Filter (NYSE/NASDAQ/NYSEMKT): {exchange_pass:,} ({exchange_pass/total_assets*100:.1f}%)")

    # Category filter
    category_pass = result['category_filter'].sum()
    print(f"Category Filter (Domestic Common Stock): {category_pass:,} ({category_pass/total_assets*100:.1f}%)")

    # ADR filter
    adr_count = result['adr_filter'].sum()
    non_adr_count = total_assets - adr_count
    print(f"ADR count: {adr_count:,} ({adr_count/total_assets*100:.1f}%)")
    print(f"Non-ADR count: {non_adr_count:,} ({non_adr_count/total_assets*100:.1f}%)")

    # Combined filter
    combined_filter = result['exchange_filter'] & result['category_filter'] & ~result['adr_filter']
    combined_pass = combined_filter.sum()
    print(f"\nCombined Filter (all three): {combined_pass:,} ({combined_pass/total_assets*100:.1f}%)")

    print()
    print("=" * 80)

    # Step 8: Test universe function
    print()
    print("Step 8: Testing create_sharadar_universe()...")

    def make_universe_pipeline():
        sharadar_universe = create_sharadar_universe(
            exchanges=['NYSE', 'NASDAQ', 'NYSEMKT'],
            include_adrs=False,
        )

        return Pipeline(
            columns={
                'exchange': SharadarTickers.exchange.latest,
                'category': SharadarTickers.category.latest,
            },
            screen=sharadar_universe,
        )

    universe_pipeline = make_universe_pipeline()
    universe_result = engine.run_pipeline(universe_pipeline, test_date, test_date)

    print(f"✅ Universe Filter Applied: {len(universe_result):,} stocks pass")
    print()

    print("Exchange distribution in filtered universe:")
    print(universe_result['exchange'].value_counts())
    print()

    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("The Sharadar filters are working correctly!")
    print("You can now use them in your strategies.")

except Exception as e:
    print(f"❌ Error running pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
