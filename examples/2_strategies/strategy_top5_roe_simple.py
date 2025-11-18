#!/usr/bin/env python3
"""
Simple Top 5 ROE Strategy - Minimal Version

A simplified version of the Top 5 ROE strategy with minimal configuration.
Just define your functions and run the backtest.

For the full-featured version with progress logging, metadata export, and
comprehensive output, see strategy_top5_roe.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    record,
    schedule_function,
    date_rules,
    time_rules,
)
from zipline.pipeline import Pipeline
from zipline.pipeline.data.db import Database, Column
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.data import USEquityPricing
from zipline.data.bundles import load as load_bundle
from zipline.data.custom import CustomSQLiteLoader

# Enable logging
logging.basicConfig(level=logging.INFO)

# Register bundle
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

register('sharadar', sharadar_bundle(tickers=None, incremental=True, include_funds=True))

# ============================================================================
# Define Custom Fundamentals Database
# ============================================================================

class CustomFundamentals(Database):
    """Custom Fundamentals database."""

    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252

    # Fundamental metrics
    ReturnOnEquity_SmartEstimat = Column(float)
    CompanyMarketCap = Column(float)
    GICSSectorName = Column(str)


# ============================================================================
# Build Pipeline Loader
# ============================================================================

def build_pipeline_loaders():
    """Build pipeline loader map with auto-discovery."""

    class LoaderDict(dict):
        """Dict that matches BoundColumns by dataset and column name."""
        def get(self, key, default=None):
            if key in self:
                return super().__getitem__(key)

            if hasattr(key, 'dataset') and hasattr(key, 'name'):
                key_dataset_name = str(key.dataset).split('<')[0]
                key_col_name = key.name

                for registered_col, loader in self.items():
                    if hasattr(registered_col, 'dataset') and hasattr(registered_col, 'name'):
                        reg_dataset_name = str(registered_col.dataset).split('<')[0]
                        reg_col_name = registered_col.name

                        if key_dataset_name == reg_dataset_name and key_col_name == reg_col_name:
                            return loader

            return default

        def __getitem__(self, key):
            result = self.get(key)
            if result is None:
                raise KeyError(key)
            return result

    # Load bundle and create loaders
    bundle_data = load_bundle('sharadar')

    pricing_loader = USEquityPricingLoader.without_fx(
        bundle_data.equity_daily_bar_reader,
        bundle_data.adjustment_reader
    )

    fundamentals_loader = CustomSQLiteLoader(
        db_code=CustomFundamentals.CODE,
        db_dir=Path('/root/.zipline/data/custom')
    )

    # Build loader map
    custom_loader = LoaderDict()

    # Pricing columns
    for attr_name in ['close', 'high', 'low', 'open', 'volume']:
        custom_loader[getattr(USEquityPricing, attr_name)] = pricing_loader

    # Auto-discover fundamental columns
    for attr_name in dir(CustomFundamentals):
        if attr_name.startswith('_') or attr_name in ['CODE', 'LOOKBACK_WINDOW']:
            continue

        attr = getattr(CustomFundamentals, attr_name)
        if hasattr(attr, 'dataset'):
            custom_loader[attr] = fundamentals_loader

    return custom_loader


# ============================================================================
# Pipeline Definition
# ============================================================================

def make_pipeline():
    """Create pipeline for stock selection."""

    roe = CustomFundamentals.ReturnOnEquity_SmartEstimat.latest
    market_cap = CustomFundamentals.CompanyMarketCap.latest
    sector = CustomFundamentals.GICSSectorName.latest

    # Universe: top 100 by market cap
    universe = market_cap.top(100)

    # Selection: top 5 by ROE
    selection = roe.top(5, mask=universe)

    return Pipeline(
        columns={
            'ROE': roe,
            'Market_Cap': market_cap,
            'Sector': sector,
        },
        screen=selection,
    )


# ============================================================================
# Strategy Functions
# ============================================================================

def initialize(context):
    """Initialize strategy."""
    attach_pipeline(make_pipeline(), 'roe_strategy')

    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(hours=1)
    )


def before_trading_start(context, data):
    """Get pipeline output."""
    context.pipeline_data = pipeline_output('roe_strategy')


def rebalance(context, data):
    """Rebalance portfolio."""
    if context.pipeline_data is None or context.pipeline_data.empty:
        return

    # Get tradeable stocks
    all_selected = context.pipeline_data.index
    selected_stocks = [stock for stock in all_selected if data.can_trade(stock)]

    if not selected_stocks:
        return

    # Equal weight
    target_weight = 1.0 / len(selected_stocks)

    # Sell positions no longer selected
    current_positions = set(context.portfolio.positions.keys())
    for stock in current_positions:
        if stock not in selected_stocks and data.can_trade(stock):
            order_target_percent(stock, 0.0)

    # Buy/rebalance selected stocks
    for stock in selected_stocks:
        order_target_percent(stock, target_weight)


def handle_data(context, data):
    """Record metrics."""
    record(
        portfolio_value=context.portfolio.portfolio_value,
        num_positions=len(context.portfolio.positions),
    )


def analyze(context, perf):
    """Analyze results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    total_return = (perf['portfolio_value'].iloc[-1] / perf['portfolio_value'].iloc[0] - 1) * 100
    daily_returns = perf['portfolio_value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    max_dd = ((perf['portfolio_value'] / perf['portfolio_value'].cummax()) - 1).min() * 100

    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Final Value: ${perf['portfolio_value'].iloc[-1]:,.2f}")
    print("=" * 60 + "\n")


# ============================================================================
# Run Backtest
# ============================================================================

if __name__ == '__main__':
    # Build custom loader
    custom_loader = build_pipeline_loaders()

    # Run the backtest
    results = run_algorithm(
        start=pd.Timestamp('2020-01-01'),
        end=pd.Timestamp('2023-12-31'),
        initialize=initialize,
        before_trading_start=before_trading_start,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=100000,
        data_frequency='daily',
        bundle='sharadar',
        custom_loader=custom_loader,
    )
