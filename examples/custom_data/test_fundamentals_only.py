#!/usr/bin/env python
"""
Simple test to verify fundamentals data can be loaded without pricing data.
This helps isolate if the string/float error is from custom data or EquityPricing.
"""

import pandas as pd
import numpy as np
from zipline import run_algorithm
from zipline.api import attach_pipeline, pipeline_output
from zipline.pipeline import Pipeline
from zipline.pipeline.filters import StaticAssets
from zipline.data.bundles import load as load_bundle
from zipline.pipeline.data.db import Database, Column
from zipline.data.custom import CustomSQLiteLoader


# Define Fundamentals database
class Fundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 1

    Revenue = Column(float)
    NetIncome = Column(float)
    ROE = Column(float)
    PERatio = Column(float)
    DebtToEquity = Column(float)
    EPS = Column(float)
    CurrentRatio = Column(float)


# Set up custom loader
def setup_custom_loader():
    """Set up custom loader for fundamentals data"""

    class LoaderDict(dict):
        def get(self, key, default=None):
            # First try exact match
            if key in self:
                return self[key]

            # Match by dataset name and column name (ignoring domain)
            if hasattr(key, 'dataset') and hasattr(key, 'name'):
                key_dataset_name = str(key.dataset).split('<')[0]
                key_col_name = key.name

                for registered_col, loader in self.items():
                    if hasattr(registered_col, 'dataset') and hasattr(registered_col, 'name'):
                        reg_dataset_name = str(registered_col.dataset).split('<')[0]
                        reg_col_name = registered_col.name

                        if key_dataset_name == reg_dataset_name and key_col_name == reg_col_name:
                            return loader

            raise KeyError(key)

    custom_loader_dict = LoaderDict()
    loader = CustomSQLiteLoader("fundamentals")

    # Register all Fundamentals columns
    for col_name in ['Revenue', 'NetIncome', 'ROE', 'PERatio', 'DebtToEquity', 'EPS', 'CurrentRatio']:
        col = getattr(Fundamentals, col_name)
        custom_loader_dict[col] = loader
        print(f"  Registered: {col_name}")

    print(f"✓ Custom loader configured with {len(custom_loader_dict)} columns")
    return custom_loader_dict

def make_pipeline():
    """Create a simple pipeline with ONLY fundamentals data (no EquityPricing)"""

    # Load bundle to get asset finder
    bundle_data = load_bundle('sharadar')
    asset_finder = bundle_data.asset_finder

    # Get our universe
    universe_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    universe_assets = []
    for ticker in universe_tickers:
        try:
            assets = asset_finder.lookup_symbols([ticker], as_of_date=None)
            if assets and assets[0] is not None:
                universe_assets.append(assets[0])
                print(f"  {ticker} -> SID {assets[0].sid}")
        except:
            pass

    base_universe = StaticAssets(universe_assets)

    # Get fundamental metrics
    roe = Fundamentals.ROE.latest
    pe = Fundamentals.PERatio.latest

    # Simple filter using ONLY fundamentals
    high_roe = (roe > 5.0)

    print(f"\nCreating pipeline with {len(universe_assets)} assets")

    return Pipeline(
        columns={
            'roe': roe,
            'pe': pe,
        },
        screen=base_universe & high_roe,
    )


def initialize(context):
    """Initialize algorithm"""
    print("\nInitializing algorithm...")
    pipe = make_pipeline()
    attach_pipeline(pipe, 'test')
    print("✓ Pipeline attached")


def handle_data(context, data):
    """No trading, just testing data loading"""
    pass


def before_trading_start(context, data):
    """Get pipeline output"""
    context.output = pipeline_output('test')
    print(f"\nPipeline output ({len(context.output)} rows):")
    if len(context.output) > 0:
        print(context.output)
    else:
        print("  (empty - no stocks passed filter)")


if __name__ == '__main__':
    print("=" * 80)
    print("FUNDAMENTALS-ONLY TEST")
    print("=" * 80)
    print("Testing if fundamentals data works without EquityPricing")
    print()

    print("Setting up custom loader...")
    custom_loader = setup_custom_loader()
    print()

    try:
        result = run_algorithm(
            start=pd.Timestamp('2023-06-01'),
            end=pd.Timestamp('2023-06-30'),
            initialize=initialize,
            handle_data=handle_data,
            before_trading_start=before_trading_start,
            capital_base=100000.0,
            bundle='sharadar',
            custom_loader=custom_loader,
        )

        print()
        print("=" * 80)
        print("✓ TEST PASSED - Fundamentals data works!")
        print("=" * 80)

    except Exception as e:
        print()
        print("=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
