#!/usr/bin/env python3
"""
Top 5 ROE Strategy using Custom Fundamentals

This strategy:
- Uses custom REFE fundamentals database
- Filters to top 100 stocks by market cap
- Selects top 5 stocks by ROE
- Rebalances weekly with equal weights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

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
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.domain import US_EQUITIES
from zipline.data.bundles import load as load_bundle
from zipline.data.custom import CustomSQLiteLoader
from zipline.utils.calendar_utils import get_calendar

# Enable logging
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Define Custom Fundamentals Database
# ============================================================================

class REFEFundamentals(Database):
    """Custom REFE Fundamentals database."""

    CODE = "refe-fundamentals"
    LOOKBACK_WINDOW = 252

    # Key columns for strategy
    ReturnOnEquity_SmartEstimat = Column(float)
    ReturnOnAssets_SmartEstimate = Column(float)
    CompanyMarketCap = Column(float)
    RefPriceClose = Column(float)
    GICSSectorName = Column(str)
    LongTermGrowth_Mean = Column(float)
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_ = Column(float)


# ============================================================================
# Pipeline Setup
# ============================================================================

# Global cache for loaders
_loader_cache = {}

def get_pipeline_loader(column):
    """
    Pipeline loader factory that routes columns to appropriate loaders.
    """
    dataset = column.dataset
    dataset_name = getattr(dataset, '__name__', '')

    # Route custom fundamentals
    if 'REFEFundamentals' in dataset_name or 'REFEFundamentals' in str(dataset):
        cache_key = REFEFundamentals.CODE
        if cache_key not in _loader_cache:
            db_dir = Path.home() / '.zipline' / 'data' / 'custom'
            _loader_cache[cache_key] = CustomSQLiteLoader(
                db_code=REFEFundamentals.CODE,
                db_dir=db_dir
            )
        return _loader_cache[cache_key]

    # Route pricing data
    if column in USEquityPricing.columns:
        if 'pricing' not in _loader_cache:
            bundle_data = load_bundle('sharadar')
            _loader_cache['pricing'] = USEquityPricingLoader(
                bundle_data.equity_daily_bar_reader,
                bundle_data.adjustment_reader
            )
        return _loader_cache['pricing']

    raise ValueError(f"No loader for {column}")


def make_pipeline():
    """
    Create pipeline: Top 5 ROE stocks from top 100 by market cap.
    """
    # Get fundamentals
    roe = REFEFundamentals.ReturnOnEquity_SmartEstimat.latest
    market_cap = REFEFundamentals.CompanyMarketCap.latest
    sector = REFEFundamentals.GICSSectorName.latest

    # Screen: top 100 by market cap
    top_100_by_mcap = market_cap.top(100)

    # Select top 5 by ROE from top 100
    top_5_roe = roe.top(5, mask=top_100_by_mcap)

    return Pipeline(
        columns={
            'ROE': roe,
            'Market_Cap': market_cap,
            'Sector': sector,
        },
        screen=top_5_roe,
    )


# ============================================================================
# Strategy Logic
# ============================================================================

def initialize(context):
    """Initialize strategy."""
    # Attach pipeline
    attach_pipeline(make_pipeline(), 'roe_strategy')

    # Schedule weekly rebalancing (every Monday)
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(hours=1),
    )

    context.rebalance_count = 0

    logging.info("=" * 60)
    logging.info("TOP 5 ROE STRATEGY")
    logging.info("=" * 60)
    logging.info("Universe: Top 100 stocks by market cap")
    logging.info("Selection: Top 5 by ROE")
    logging.info("Rebalance: Weekly (Mondays)")
    logging.info("=" * 60)


def before_trading_start(context, data):
    """Get pipeline output before market opens."""
    context.pipeline_data = pipeline_output('roe_strategy')


def rebalance(context, data):
    """Execute weekly rebalancing."""
    context.rebalance_count += 1

    # Get pipeline output
    pipeline_data = context.pipeline_data

    if pipeline_data is None or pipeline_data.empty:
        logging.warning("No stocks in pipeline output")
        return

    # Filter for tradeable stocks
    all_selected = pipeline_data.index
    selected_stocks = [stock for stock in all_selected if data.can_trade(stock)]

    if len(selected_stocks) == 0:
        logging.warning("No tradeable stocks")
        return

    # Equal weight
    target_weight = 1.0 / len(selected_stocks)

    # Current positions
    current_positions = set(context.portfolio.positions.keys())

    # Sell positions no longer in target
    for stock in current_positions - set(selected_stocks):
        if data.can_trade(stock):
            order_target_percent(stock, 0.0)

    # Buy/rebalance target positions
    for stock in selected_stocks:
        order_target_percent(stock, target_weight)

    # Log holdings
    holdings = [s.symbol for s in selected_stocks]
    tradeable_data = pipeline_data.loc[selected_stocks]
    avg_roe = tradeable_data['ROE'].mean()

    logging.info(f"Rebalance #{context.rebalance_count}:")
    logging.info(f"  Holdings: {', '.join(holdings)}")
    logging.info(f"  Avg ROE: {avg_roe:.2f}%")


def handle_data(context, data):
    """Record daily metrics."""
    record(
        portfolio_value=context.portfolio.portfolio_value,
        num_positions=len(context.portfolio.positions),
        leverage=context.account.leverage,
    )


# ============================================================================
# Run Backtest
# ============================================================================

if __name__ == '__main__':
    # Load bundle for asset finder
    bundle_data = load_bundle('sharadar')

    # Create pipeline engine with custom loader
    trading_calendar = get_calendar('NYSE')
    engine = SimplePipelineEngine(
        get_loader=get_pipeline_loader,
        asset_finder=bundle_data.asset_finder,
        default_domain=US_EQUITIES,
    )

    # Monkey-patch run_algorithm to use our custom engine
    # This is the key to making custom fundamentals work!
    import zipline.algorithm
    original_make_engine = zipline.algorithm.SimplePipelineEngine

    def patched_engine(*args, **kwargs):
        # Return our pre-configured engine
        return engine

    zipline.algorithm.SimplePipelineEngine = lambda *args, **kwargs: engine

    # Run backtest
    start = pd.Timestamp('2024-05-01', tz='UTC')
    end = pd.Timestamp('2024-11-14', tz='UTC')

    print("\n" + "=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Capital: $100,000")
    print("=" * 60 + "\n")

    results = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        before_trading_start=before_trading_start,
        handle_data=handle_data,
        capital_base=100000,
        data_frequency='daily',
        bundle='sharadar',
    )

    # Restore original
    zipline.algorithm.SimplePipelineEngine = original_make_engine

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Final Portfolio Value: ${results.portfolio_value.iloc[-1]:,.2f}")

    total_return = (results.portfolio_value.iloc[-1] / 100000 - 1) * 100
    print(f"Total Return: {total_return:.2f}%")

    daily_returns = results.returns.dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        print(f"Sharpe Ratio: {sharpe:.2f}")

    max_dd = ((results.portfolio_value / results.portfolio_value.cummax()) - 1).min() * 100
    print(f"Max Drawdown: {max_dd:.2f}%")
    print("=" * 60)

    # Save results
    results.to_pickle('backtest_results.pkl')
    print("\n✓ Results saved to backtest_results.pkl")
    print("  Load with: results = pd.read_pickle('backtest_results.pkl')")
