"""
Simple Multi-Source Strategy Example

This demonstrates the clean, simple architecture for mixing Sharadar and custom data.

Just import and use - no complex loader setup required!
"""

import pandas as pd
import numpy as np
import logging

from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    schedule_function,
    date_rules,
    time_rules,
)
from zipline.pipeline import Pipeline
from zipline.pipeline.data import sharadar
from zipline.pipeline.data.db import Database, Column

# Import the auto loader - this handles everything!
from zipline.pipeline.loaders.auto_loader import setup_auto_loader

# Enable logging
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Define Custom Fundamentals Database
# ============================================================================

class CustomFundamentals(Database):
    """Custom LSEG fundamentals."""

    CODE = "fundamentals"  # Must match your database name
    LOOKBACK_WINDOW = 252

    # Metrics from your LSEG database
    ReturnOnEquity_SmartEstimat = Column(float)
    ForwardPEG_DailyTimeSeriesRatio_ = Column(float)
    CompanyMarketCap = Column(float)
    Debt_Total = Column(float)


# ============================================================================
# Pipeline Definition - Mix Sharadar and Custom Data
# ============================================================================

def make_pipeline(universe_size=100, selection_size=5):
    """
    Pipeline using BOTH Sharadar and Custom LSEG data.

    This is as simple as using either source alone!
    """
    # Sharadar fundamentals
    s_fundamentals = sharadar.Fundamentals.slice('MRQ', period_offset=0)
    s_roe = s_fundamentals.ROE.latest
    s_fcf = s_fundamentals.FCF.latest
    s_marketcap = s_fundamentals.MARKETCAP.latest

    # Custom LSEG fundamentals
    l_roe = CustomFundamentals.ReturnOnEquity_SmartEstimat.latest
    l_peg = CustomFundamentals.ForwardPEG_DailyTimeSeriesRatio_.latest
    l_marketcap = CustomFundamentals.CompanyMarketCap.latest

    # Universe: top N by market cap (using Sharadar)
    universe = s_marketcap.top(universe_size)

    # Consensus scoring: bonus when both sources agree
    high_roe_sharadar = s_roe > 15.0
    high_roe_lseg = l_roe > 15.0
    both_confirm_quality = high_roe_sharadar & high_roe_lseg

    # Selection: top M by Sharadar ROE, with LSEG confirmation
    selection = s_roe.top(selection_size, mask=universe & both_confirm_quality)

    return Pipeline(
        columns={
            # Sharadar
            's_roe': s_roe,
            's_fcf': s_fcf,
            's_marketcap': s_marketcap,
            # Custom LSEG
            'l_roe': l_roe,
            'l_peg': l_peg,
            'l_marketcap': l_marketcap,
            # Derived
            'both_confirm': both_confirm_quality,
        },
        screen=selection,
    )


# ============================================================================
# Strategy Logic
# ============================================================================

def initialize(context):
    """Initialize strategy."""
    # Attach pipeline
    pipe = make_pipeline()
    attach_pipeline(pipe, 'multi_source')

    # Schedule rebalancing
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(hours=1)
    )

    # Initialize tracking
    context.rebalance_count = 0

    logging.info("="*80)
    logging.info("Multi-Source Fundamentals Strategy Initialized")
    logging.info("="*80)
    logging.info("Data sources:")
    logging.info("  - Sharadar: ROE, FCF, MarketCap")
    logging.info("  - Custom LSEG: ROE, PEG, MarketCap")
    logging.info("Strategy: Top 5 by ROE with multi-source confirmation")
    logging.info("="*80)


def before_trading_start(context, data):
    """Get pipeline data."""
    context.pipeline_data = pipeline_output('multi_source')


def rebalance(context, data):
    """Weekly rebalancing."""
    context.rebalance_count += 1

    if context.pipeline_data is None or context.pipeline_data.empty:
        logging.warning("No stocks in pipeline output")
        return

    # Get tradeable stocks
    all_selected = context.pipeline_data.index
    selected_stocks = [s for s in all_selected if data.can_trade(s)]

    if not selected_stocks:
        logging.warning("No tradeable stocks")
        return

    # Equal weight
    target_weight = 1.0 / len(selected_stocks)

    # Get current positions
    current_positions = set(context.portfolio.positions.keys())
    target_positions = set(selected_stocks)

    # Sell positions no longer selected
    for stock in current_positions - target_positions:
        if data.can_trade(stock):
            order_target_percent(stock, 0.0)

    # Buy/rebalance selected stocks
    for stock in selected_stocks:
        if data.can_trade(stock):
            order_target_percent(stock, target_weight)

    # Log
    confirmed = context.pipeline_data['both_confirm'].sum()
    logging.info(
        f"Rebalance #{context.rebalance_count}: "
        f"{len(selected_stocks)} stocks, "
        f"{confirmed} with LSEG confirmation"
    )


def analyze(context, perf):
    """Analyze results."""
    returns = perf['returns']
    total_return = (perf['portfolio_value'].iloc[-1] / perf['portfolio_value'].iloc[0] - 1) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    logging.info("="*80)
    logging.info("BACKTEST RESULTS")
    logging.info("="*80)
    logging.info(f"Total Return: {total_return:.2f}%")
    logging.info(f"Sharpe Ratio: {sharpe:.2f}")
    logging.info(f"Rebalances: {context.rebalance_count}")
    logging.info("="*80)

    return perf


# ============================================================================
# Run Backtest - Simple!
# ============================================================================

if __name__ == '__main__':
    # Backtest parameters
    START = pd.Timestamp('2023-01-01')
    END = pd.Timestamp('2024-11-01')
    CAPITAL = 100000

    # Run backtest - notice how simple this is!
    results = run_algorithm(
        start=START,
        end=END,
        initialize=initialize,
        before_trading_start=before_trading_start,
        analyze=analyze,
        capital_base=CAPITAL,
        bundle='sharadar',
        custom_loader=setup_auto_loader(),  # That's it! Auto-detects everything
    )

    print("\n✓ Backtest complete!")
    print(f"Final portfolio value: ${results['portfolio_value'].iloc[-1]:,.2f}")
