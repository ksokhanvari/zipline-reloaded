"""
Test Sharadar fundamentals loader directly.
"""
import pandas as pd
from zipline.pipeline.loaders.sharadar_fundamentals import make_sharadar_fundamentals_loader
from zipline.pipeline.data.sharadar import SharadarFundamentals
from zipline.data.bundles import load as load_bundle, register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

# Register
register('sharadar', sharadar_bundle(tickers=None, incremental=True, include_funds=True))

print("Testing Sharadar fundamentals loader...")

# Create the loader
try:
    loader = make_sharadar_fundamentals_loader('sharadar')
    print(f"✓ Loader created: {loader}")
    print(f"  Bundle path: {loader.bundle_path}")
    print(f"  Fundamentals path: {loader.fundamentals_path}")
    print(f"  File exists: {loader.fundamentals_path.exists()}")

    # Load bundle to get asset finder
    bundle_data = load_bundle('sharadar')
    print(f"\n✓ Bundle loaded")

    # Get some test assets
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    test_date = pd.Timestamp('2023-01-05', tz='UTC')
    test_date_naive = pd.Timestamp('2023-01-05')  # For lookup

    assets = []
    for symbol in test_symbols:
        try:
            asset = bundle_data.asset_finder.lookup_symbol(symbol, as_of_date=test_date_naive)
            assets.append(asset)
            print(f"  {symbol}: SID {asset.sid}")
        except Exception as e:
            print(f"  {symbol}: Not found - {e}")

    if assets:
        sids = pd.Index([asset.sid for asset in assets], dtype='int64')
        dates = pd.DatetimeIndex([test_date])

        # Try to load data
        print(f"\nTrying to load ROE data for {test_date.date()}...")

        columns = [SharadarFundamentals.roe]
        mask = pd.DataFrame(True, index=dates, columns=sids)

        from zipline.pipeline.domain import US_EQUITIES

        result = loader.load_adjusted_array(
            domain=US_EQUITIES,
            columns=columns,
            dates=dates,
            sids=sids,
            mask=mask.values
        )

        print(f"✓ Loaded data")
        for col, arr in result.items():
            print(f"\n{col.name}:")
            print(f"  Shape: {arr.data.shape}")
            print(f"  Values: {arr.data[0]}")  # First (and only) date
            print(f"  Non-NaN: {(~pd.isna(arr.data[0])).sum()} / {len(arr.data[0])}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
