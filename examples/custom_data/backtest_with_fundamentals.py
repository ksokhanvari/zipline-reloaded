#!/usr/bin/env python
"""
Zipline Backtest with Custom Fundamental Data

This script demonstrates how to combine:
- Sharadar bundle (for pricing data)
- Custom fundamentals database (for fundamental metrics)

Strategy:
- Screen for quality stocks using fundamental criteria (ROE, P/E, Debt)
- Rank by quality score
- Hold top N stocks, rebalance monthly
"""

import pandas as pd
import numpy as np
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
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import EquityPricing
from zipline.pipeline.filters import StaticAssets
from zipline.data.bundles import load as load_bundle
# CustomSQLiteLoader is automatically used based on Database.CODE

# ============================================================================
# CONFIGURATION
# ============================================================================

# Strategy parameters
TOP_N_STOCKS = 10
REBALANCE_FREQUENCY = 'monthly'  # monthly or weekly

# Backtest period
START_DATE = '2023-04-01'
END_DATE = '2024-01-31'
INITIAL_CAPITAL = 100000.0

# ============================================================================
# DEFINE CUSTOM DATABASE CLASS
# ============================================================================

from zipline.pipeline.data.db import Database, Column


class Fundamentals(Database):
    """
    Custom fundamentals database.

    This approach is cleaner than make_custom_dataset_class() and allows
    direct usage like: Fundamentals.ROE.latest
    """

    CODE = "fundamentals"  # Must match your database code
    LOOKBACK_WINDOW = 240  # Days to look back

    # Income Statement
    Revenue = Column(float)
    NetIncome = Column(float)

    # Balance Sheet
    TotalAssets = Column(float)
    TotalEquity = Column(float)
    SharesOutstanding = Column(float)

    # Per-Share Metrics
    EPS = Column(float)
    BookValuePerShare = Column(float)

    # Financial Ratios
    ROE = Column(float)
    DebtToEquity = Column(float)
    CurrentRatio = Column(float)
    PERatio = Column(float)

    # Metadata
    Sector = Column(str)


print("✓ Fundamentals Database class defined")

# ============================================================================
# CUSTOM FACTORS
# ============================================================================

class QualityScore(CustomFactor):
    """
    Composite quality score based on:
    - ROE (higher is better)
    - P/E ratio (lower is better)
    - Debt/Equity (lower is better)

    Returns a normalized score where higher = better quality.
    """
    inputs = [
        Fundamentals.ROE,
        Fundamentals.PERatio,
        Fundamentals.DebtToEquity,
    ]
    window_length = 1

    def compute(self, today, assets, out, roe, pe, debt):
        # Get latest values (window_length=1, so just index 0)
        roe_latest = roe[0]
        pe_latest = pe[0]
        debt_latest = debt[0]

        # Normalize each metric to 0-1 scale using min-max normalization
        # ROE: higher is better
        roe_min, roe_max = np.nanmin(roe_latest), np.nanmax(roe_latest)
        if roe_max > roe_min:
            roe_score = (roe_latest - roe_min) / (roe_max - roe_min)
        else:
            roe_score = np.full_like(roe_latest, 0.5)

        # P/E: lower is better, so invert
        pe_min, pe_max = np.nanmin(pe_latest), np.nanmax(pe_latest)
        if pe_max > pe_min:
            pe_score = 1 - ((pe_latest - pe_min) / (pe_max - pe_min))
        else:
            pe_score = np.full_like(pe_latest, 0.5)

        # Debt: lower is better, so invert
        debt_min, debt_max = np.nanmin(debt_latest), np.nanmax(debt_latest)
        if debt_max > debt_min:
            debt_score = 1 - ((debt_latest - debt_min) / (debt_max - debt_min))
        else:
            debt_score = np.full_like(debt_latest, 0.5)

        # Composite score (equal weights)
        out[:] = (roe_score + pe_score + debt_score) / 3.0


class ProfitMargin(CustomFactor):
    """Calculate profit margin: (Net Income / Revenue) * 100"""
    inputs = [
        Fundamentals.NetIncome,
        Fundamentals.Revenue,
    ]
    window_length = 1

    def compute(self, today, assets, out, net_income, revenue):
        latest_income = net_income[0]
        latest_revenue = revenue[0]

        with np.errstate(divide='ignore', invalid='ignore'):
            profit_margin = (latest_income / latest_revenue) * 100.0
            profit_margin = np.where(latest_revenue == 0, np.nan, profit_margin)

        out[:] = profit_margin


# ============================================================================
# PIPELINE DEFINITION
# ============================================================================

def make_pipeline():
    """
    Create a pipeline that:
    1. Gets fundamental data from custom database
    2. Gets pricing data from sharadar bundle
    3. Screens for quality stocks
    4. Ranks by quality score
    """

    # Get fundamental metrics from custom database
    roe = Fundamentals.ROE.latest
    pe_ratio = Fundamentals.PERatio.latest
    debt_to_equity = Fundamentals.DebtToEquity.latest
    eps = Fundamentals.EPS.latest
    current_ratio = Fundamentals.CurrentRatio.latest
    sector = Fundamentals.Sector.latest

    # Calculate derived metrics
    quality_score = QualityScore()
    profit_margin = ProfitMargin()

    # Get pricing data from bundle
    close_price = EquityPricing.close.latest
    volume = EquityPricing.volume.latest

    # Calculate technical indicators
    avg_volume_20d = EquityPricing.volume.mavg(20)  # 20-day average volume

    # Define screening filters
    # 1. Fundamental quality
    high_roe = (roe > 5.0)  # Lowered threshold for more stocks
    reasonable_pe = (pe_ratio < 50.0)  # Reasonable valuation
    manageable_debt = (debt_to_equity < 5.0)  # Not over-leveraged

    # 2. Liquidity (from pricing data)
    liquid = (avg_volume_20d > 100000)  # Minimum liquidity

    # 3. Valid price
    valid_price = (close_price > 1.0)  # Minimum price

    # Combine all filters
    quality_universe = (
        high_roe &
        reasonable_pe &
        manageable_debt &
        liquid &
        valid_price
    )

    # Rank by quality score within universe
    quality_rank = quality_score.rank(mask=quality_universe, ascending=False)

    return Pipeline(
        columns={
            # Fundamental metrics
            'quality_score': quality_score,
            'quality_rank': quality_rank,
            'roe': roe,
            'pe_ratio': pe_ratio,
            'debt_to_equity': debt_to_equity,
            'eps': eps,
            'current_ratio': current_ratio,
            'profit_margin': profit_margin,
            'sector': sector,

            # Pricing metrics
            'close': close_price,
            'volume': volume,
            'avg_volume_20d': avg_volume_20d,
        },
        screen=quality_universe,
    )


# ============================================================================
# ALGORITHM LOGIC
# ============================================================================

def initialize(context):
    """
    Called once at the start of the backtest.
    """
    # Attach our pipeline
    attach_pipeline(make_pipeline(), 'quality_stocks')

    # Schedule rebalancing
    if REBALANCE_FREQUENCY == 'monthly':
        schedule_function(
            rebalance,
            date_rules.month_start(),
            time_rules.market_open(hours=1),
        )
    else:  # weekly
        schedule_function(
            rebalance,
            date_rules.week_start(),
            time_rules.market_open(hours=1),
        )

    # Initialize tracking
    context.top_n = TOP_N_STOCKS
    context.current_positions = set()

    print(f"\n{'='*70}")
    print(f"BACKTEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"Strategy: Quality Factor (Fundamentals-based)")
    print(f"Top N stocks: {context.top_n}")
    print(f"Rebalance: {REBALANCE_FREQUENCY}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"{'='*70}\n")


def before_trading_start(context, data):
    """
    Called every day before market open.
    Gets fresh pipeline data.
    """
    # Get pipeline output
    context.output = pipeline_output('quality_stocks')

    # Log pipeline stats
    if context.output is not None and len(context.output) > 0:
        avg_quality = context.output['quality_score'].mean()
        avg_roe = context.output['roe'].mean()
        avg_pe = context.output['pe_ratio'].mean()

        print(f"\n{context.get_datetime().date()}")
        print(f"  Universe size: {len(context.output)} stocks")
        print(f"  Avg Quality Score: {avg_quality:.3f}")
        print(f"  Avg ROE: {avg_roe:.1f}%")
        print(f"  Avg P/E: {avg_pe:.1f}")


def rebalance(context, data):
    """
    Rebalance portfolio to hold top N quality stocks.
    """
    if context.output is None or len(context.output) == 0:
        print("  ⚠ No stocks in universe, skipping rebalance")
        return

    # Select top N stocks by quality rank
    top_stocks = context.output.nsmallest(context.top_n, 'quality_rank')

    if len(top_stocks) == 0:
        print("  ⚠ No top stocks found, skipping rebalance")
        return

    # Calculate target weight (equal weight)
    target_weight = 1.0 / len(top_stocks)

    # Get current positions
    current_positions = set(context.portfolio.positions.keys())
    target_positions = set(top_stocks.index)

    # Stocks to sell (in current but not in target)
    to_sell = current_positions - target_positions

    # Stocks to buy (in target but not in current)
    to_buy = target_positions - current_positions

    # Stocks to rebalance (in both)
    to_rebalance = current_positions & target_positions

    print(f"\n  REBALANCING:")
    print(f"    Sell: {len(to_sell)} positions")
    print(f"    Buy: {len(to_buy)} positions")
    print(f"    Rebalance: {len(to_rebalance)} positions")
    print(f"    Target weight: {target_weight:.2%}")

    # Sell positions no longer in top N
    for asset in to_sell:
        if data.can_trade(asset):
            order_target_percent(asset, 0.0)
            print(f"      SELL: {asset.symbol}")

    # Buy/rebalance to equal weight
    for asset in top_stocks.index:
        if data.can_trade(asset):
            order_target_percent(asset, target_weight)
            if asset in to_buy:
                stock_data = top_stocks.loc[asset]
                print(f"      BUY:  {asset.symbol} "
                      f"(Score: {stock_data['quality_score']:.3f}, "
                      f"ROE: {stock_data['roe']:.1f}%, "
                      f"P/E: {stock_data['pe_ratio']:.1f})")

    # Update tracking
    context.current_positions = target_positions


def analyze(context, perf):
    """
    Called at the end of the backtest.
    Analyze performance.
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*70}")

    # Calculate metrics
    total_return = (perf['portfolio_value'].iloc[-1] / INITIAL_CAPITAL - 1) * 100

    # Calculate daily returns
    perf['daily_returns'] = perf['portfolio_value'].pct_change()

    # Sharpe ratio (annualized, assuming 252 trading days)
    sharpe = (perf['daily_returns'].mean() / perf['daily_returns'].std()) * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + perf['daily_returns']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Win rate
    winning_days = (perf['daily_returns'] > 0).sum()
    total_days = len(perf['daily_returns'].dropna())
    win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0

    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Value:     ${perf['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {max_drawdown:.2f}%")
    print(f"Win Rate:        {win_rate:.1f}%")
    print(f"Total Trades:    {len(perf[perf['orders'] != '[]'])}")
    print(f"{'='*70}\n")

    # Save detailed results
    perf.to_csv('backtest_results.csv', index=True)
    print("✓ Detailed results saved to: backtest_results.csv\n")

    return perf


# ============================================================================
# RUN BACKTEST
# ============================================================================

if __name__ == '__main__':
    print("\nStarting backtest with custom fundamental data...")
    print("="*70)

    # Run the algorithm
    results = run_algorithm(
        start=pd.Timestamp(START_DATE, tz='UTC'),
        end=pd.Timestamp(END_DATE, tz='UTC'),
        initialize=initialize,
        before_trading_start=before_trading_start,
        analyze=analyze,
        capital_base=INITIAL_CAPITAL,
        bundle='sharadar',
    )

    print("\n✓ Backtest complete!")
    print(f"\nTo visualize results, run:")
    print(f"  python plot_backtest_results.py")
