#!/usr/bin/env python
"""
Quick test to verify Sharadar fundamentals work in Pipeline.
"""

import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals
from zipline.pipeline.loaders.sharadar_fundamentals import make_sharadar_fundamentals_loader
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.domain import US_EQUITIES
from zipline.data.bundles import load, register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle


def test_fundamentals_pipeline():
    """Test that fundamentals load correctly in Pipeline."""

    print("=" * 60)
    print("Testing Sharadar Fundamentals in Pipeline")
    print("=" * 60)

    # Step 1: Register bundle
    print("\nStep 1/5: Registering bundle...")
    register('sharadar', sharadar_bundle())
    print(f"  ✓ Bundle registered")

    # Step 2: Load bundle
    print("\nStep 2/5: Loading bundle...")
    bundle_data = load('sharadar')
    print(f"  ✓ Bundle loaded")

    # Step 3: Find valid trading session
    print("\nStep 3/5: Finding valid trading session...")
    trading_calendar = bundle_data.equity_daily_bar_reader.trading_calendar
    schedule = trading_calendar.sessions_in_range(
        pd.Timestamp('2024-11-01'),
        pd.Timestamp('2024-11-15')
    )
    test_date = schedule[-1]
    print(f"  ✓ Using test date: {test_date.date()}")

    # Step 4: Create fundamentals loader
    print("\nStep 4/5: Creating fundamentals loader...")
    fundamentals_loader = make_sharadar_fundamentals_loader('sharadar')
    print(f"  ✓ Loader created: {fundamentals_loader.fundamentals_path}")

    # Step 5: Create Pipeline
    print("\nStep 5/5: Creating Pipeline...")
    pipeline = Pipeline(
        columns={
            'revenue': SharadarFundamentals.revenue.latest,
            'netinc': SharadarFundamentals.netinc.latest,
            'assets': SharadarFundamentals.assets.latest,
            'equity': SharadarFundamentals.equity.latest,
            'roe': SharadarFundamentals.roe.latest,
            'marketcap': SharadarFundamentals.marketcap.latest,
        },
    )
    print(f"  ✓ Pipeline created with 6 fundamental columns")

    # Create engine with fundamentals loader
    engine = SimplePipelineEngine(
        get_loader=lambda column: fundamentals_loader,
        asset_finder=bundle_data.asset_finder,
        default_domain=US_EQUITIES,
    )

    print(f"  Running Pipeline for {test_date.date()}...")
    result = engine.run_pipeline(pipeline, test_date, test_date)

    print(f"\n  ✓ Pipeline executed successfully!")
    print(f"    Result shape: {result.shape}")
    print(f"    Assets with data: {result.dropna(how='all').shape[0]:,}")

    # Show sample results
    print(f"\n  Sample results (companies with complete data):")
    sample = result.dropna().head(10)
    if len(sample) > 0:
        print(sample.to_string())
    else:
        print("  Note: No companies with complete data on this date")
        print("  Showing any available data:")
        print(result.dropna(how='all').head(10).to_string())

    # Statistics
    print(f"\n  Data coverage:")
    for col in result.columns:
        non_null = result[col].notna().sum()
        pct = (non_null / len(result) * 100) if len(result) > 0 else 0
        print(f"    {col}: {non_null:,} assets ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("✓ Fundamentals Pipeline test PASSED!")
    print("=" * 60)
    print("\nYou can now use fundamentals in your strategies:")
    print("  from zipline.pipeline.data.sharadar import SharadarFundamentals")
    print("  revenue = SharadarFundamentals.revenue.latest")
    print("  roe = SharadarFundamentals.roe.latest")
    print("")


if __name__ == '__main__':
    test_fundamentals_pipeline()
