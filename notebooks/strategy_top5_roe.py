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
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.data import USEquityPricing
from zipline.data.bundles import load as load_bundle
from zipline.data.custom import CustomSQLiteLoader

# Enable logging
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Register Bundle (Required for Jupyter Notebooks)
# ============================================================================

# Register Sharadar bundle (required for Jupyter notebooks)
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

register('sharadar', sharadar_bundle(tickers=None, incremental=True, include_funds=True))
print("✓ Sharadar bundle registered")

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
# Build Pipeline Loader Map (Clean Approach)
# ============================================================================

def build_pipeline_loaders():
    """
    Build a proper PipelineLoader map.

    This is the clean approach - map each column/term to its appropriate loader.
    No monkey-patching required!
    """
    # Custom dict that handles domain-aware columns (e.g., USEquityPricing<US>.close)
    class LoaderDict(dict):
        """
        Dict that matches BoundColumns by dataset and column name.

        This handles the case where columns are registered without a domain
        (USEquityPricing.close) but looked up with a domain
        (USEquityPricing<US_EQUITIES>.close).
        """
        def get(self, key, default=None):
            # First try exact match
            if key in self:
                return super().__getitem__(key)

            # If key is a BoundColumn, try matching by dataset name and column name
            if hasattr(key, 'dataset') and hasattr(key, 'name'):
                # Extract dataset name without domain (remove <DOMAIN> part)
                key_dataset_name = str(key.dataset).split('<')[0]
                key_col_name = key.name

                # Search for a matching column in our registered columns
                for registered_col, loader in self.items():
                    if hasattr(registered_col, 'dataset') and hasattr(registered_col, 'name'):
                        reg_dataset_name = str(registered_col.dataset).split('<')[0]
                        reg_col_name = registered_col.name

                        # Match by dataset name and column name
                        if key_dataset_name == reg_dataset_name and key_col_name == reg_col_name:
                            return loader

            # No match found, return default
            return default

        def __getitem__(self, key):
            result = self.get(key)
            if result is None:
                raise KeyError(key)
            return result

    # Load bundle data
    bundle_data = load_bundle('sharadar')

    # Create loaders
    pricing_loader = USEquityPricingLoader.without_fx(
        bundle_data.equity_daily_bar_reader,
        bundle_data.adjustment_reader
    )

    db_dir = Path.home() / '.zipline' / 'data' / 'custom'
    fundamentals_loader = CustomSQLiteLoader(
        db_code=REFEFundamentals.CODE,
        db_dir=db_dir
    )

    # Build the loader map using LoaderDict
    custom_loader = LoaderDict()

    # Map all pricing columns to pricing loader
    custom_loader[USEquityPricing.close] = pricing_loader
    custom_loader[USEquityPricing.high] = pricing_loader
    custom_loader[USEquityPricing.low] = pricing_loader
    custom_loader[USEquityPricing.open] = pricing_loader
    custom_loader[USEquityPricing.volume] = pricing_loader

    # Map all fundamental columns to fundamentals loader
    fundamental_columns = [
        REFEFundamentals.ReturnOnEquity_SmartEstimat,
        REFEFundamentals.ReturnOnAssets_SmartEstimate,
        REFEFundamentals.CompanyMarketCap,
        REFEFundamentals.RefPriceClose,
        REFEFundamentals.GICSSectorName,
        REFEFundamentals.LongTermGrowth_Mean,
        REFEFundamentals.EnterpriseValueToEBITDA_DailyTimeSeriesRatio_,
    ]

    for column in fundamental_columns:
        custom_loader[column] = fundamentals_loader

    print(f"✓ Pipeline loader map built ({len(custom_loader)} columns mapped)")

    return custom_loader


# ============================================================================
# Pipeline Definition
# ============================================================================

def make_pipeline():
    """
    Create a pipeline that:
    1. Filters to top 100 stocks by market cap
    2. Selects top 5 by ROE
    """
    # Get factors
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
    """
    Initialize strategy and attach pipeline.
    """
    # Attach pipeline
    pipe = make_pipeline()
    attach_pipeline(pipe, 'roe_strategy')

    # Schedule weekly rebalancing
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(hours=1)
    )

    context.rebalance_count = 0

    logging.info("ROE Strategy initialized")
    logging.info("  Rebalancing: Weekly")
    logging.info("  Universe: Top 100 by market cap")
    logging.info("  Selection: Top 5 by ROE")


def before_trading_start(context, data):
    """
    Called daily before market opens.
    Get pipeline output.
    """
    context.pipeline_data = pipeline_output('roe_strategy')


def rebalance(context, data):
    """
    Weekly rebalance based on pipeline output.
    """
    context.rebalance_count += 1

    # Get current pipeline output
    if context.pipeline_data is None or context.pipeline_data.empty:
        logging.warning("No stocks in pipeline output")
        return

    # Get selected stocks - filter out untradeable assets
    all_selected = context.pipeline_data.index
    selected_stocks = [stock for stock in all_selected if data.can_trade(stock)]

    if len(selected_stocks) == 0:
        logging.warning("No tradeable stocks in pipeline output")
        return

    # Log if any stocks were filtered out
    if len(selected_stocks) < len(all_selected):
        filtered_out = [s.symbol for s in all_selected if s not in selected_stocks]
        logging.info(f"  Filtered out untradeable: {', '.join(filtered_out)}")

    # Equal weight
    target_weight = 1.0 / len(selected_stocks)

    # Get current positions
    current_positions = set(context.portfolio.positions.keys())

    # Sell stocks no longer in selection (only if tradeable)
    for stock in current_positions:
        if stock not in selected_stocks and data.can_trade(stock):
            order_target_percent(stock, 0.0)

    # Buy/rebalance selected stocks
    for stock in selected_stocks:
        order_target_percent(stock, target_weight)

    # Log holdings
    holdings = [s.symbol for s in selected_stocks]
    logging.info(f"Rebalance #{context.rebalance_count}: {', '.join(holdings)}")

    # Log factor values
    tradeable_output = context.pipeline_data.loc[selected_stocks]
    if not tradeable_output.empty:
        logging.info(f"  Avg ROE: {tradeable_output['ROE'].mean():.2%}")
        logging.info(f"  Avg Market Cap: ${tradeable_output['Market_Cap'].mean()/1e9:.2f}B")


def handle_data(context, data):
    """
    Record daily metrics.
    """
    record(
        portfolio_value=context.portfolio.portfolio_value,
        num_positions=len(context.portfolio.positions),
        leverage=context.account.leverage,
    )


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Build pipeline loader map (CLEAN APPROACH - NO MONKEY-PATCHING!)
    custom_loader = build_pipeline_loaders()

    # Run backtest
    start = pd.Timestamp('2025-10-01')
    end = pd.Timestamp('2025-11-05')

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
        custom_loader=custom_loader,  # ← THE KEY - No monkey-patching needed!
    )

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    total_return = ((results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1) * 100
    daily_returns = results['portfolio_value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    max_drawdown = ((results['portfolio_value'] / results['portfolio_value'].cummax()) - 1).min() * 100

    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total Rebalances: {results['num_positions'].count()}")
    print("=" * 60)
