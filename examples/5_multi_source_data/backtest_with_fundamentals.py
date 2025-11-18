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
from pathlib import Path
# Add examples directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.register_bundles import ensure_bundles_registered
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
from zipline.pipeline.factors import SimpleMovingAverage
from zipline.data.bundles import load as load_bundle
from zipline.assets._assets import Equity
from zipline.data.custom import CustomSQLiteLoader

# Ensure bundles are registered (needed for make_pipeline which uses load_bundle)
ensure_bundles_registered()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Strategy parameters
TOP_N_STOCKS = 10
REBALANCE_FREQUENCY = 'monthly'  # monthly or weekly

# Stocks with fundamental data (must match tickers in fundamentals database)
UNIVERSE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'WMT', 'XOM', 'V']

# Backtest period (dynamically set to last 3 months)
# These will be overridden with recent dates
START_DATE = None  # Will be set dynamically
END_DATE = None    # Will be set dynamically
INITIAL_CAPITAL = 100000.0

# ============================================================================
# DEFINE CUSTOM DATABASE CLASS
# ============================================================================

from zipline.pipeline.data.db import Database, Column


class CustomFundamentals(Database):
    """
    Custom fundamentals database.

    This approach is cleaner than make_custom_dataset_class() and allows
    direct usage like: CustomFundamentals.ROE.latest
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


print("✓ CustomFundamentals Database class defined")

# ============================================================================
# CUSTOM LOADER SETUP
# ============================================================================

def setup_custom_loader():
    """
    Set up custom loader for fundamentals data with explicit db_dir.
    This ensures the loader looks in the correct directory for the database.
    """
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
    # Explicitly set db_dir to the correct location
    db_dir = Path.home() / '.zipline' / 'data' / 'custom'
    loader = CustomSQLiteLoader("fundamentals", db_dir=db_dir)

    # Register all CustomFundamentals columns
    for attr_name in dir(CustomFundamentals):
        attr = getattr(CustomFundamentals, attr_name)
        # Check if it's a Column (has dataset attribute)
        if hasattr(attr, 'dataset'):
            custom_loader_dict[attr] = loader

    print(f"✓ Custom loader configured with {len(custom_loader_dict)} columns for database at {db_dir}")
    return custom_loader_dict

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
        CustomFundamentals.ROE,
        CustomFundamentals.PERatio,
        CustomFundamentals.DebtToEquity,
    ]
    window_length = 1

    def compute(self, today, assets, out, roe, pe, debt):
        # Get latest values - use [-1] for most recent data
        roe_latest = roe[-1]
        pe_latest = pe[-1]
        debt_latest = debt[-1]

        # Defensive: ensure we have numeric arrays
        if roe_latest.dtype == object or pe_latest.dtype == object or debt_latest.dtype == object:
            import warnings
            warnings.warn(f"QualityScore received object dtype arrays! ROE: {roe_latest.dtype}, PE: {pe_latest.dtype}, Debt: {debt_latest.dtype}")
            # Fill with NaN if we got bad data
            out[:] = np.nan
            return

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
        CustomFundamentals.NetIncome,
        CustomFundamentals.Revenue,
    ]
    window_length = 1

    def compute(self, today, assets, out, net_income, revenue):
        latest_income = net_income[-1]
        latest_revenue = revenue[-1]

        # Defensive: ensure we have numeric arrays
        if latest_income.dtype == object or latest_revenue.dtype == object:
            import warnings
            warnings.warn(f"ProfitMargin received object dtype arrays! NetIncome: {latest_income.dtype}, Revenue: {latest_revenue.dtype}")
            # Fill with NaN if we got bad data
            out[:] = np.nan
            return

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

    # Load bundle to get asset finder
    bundle_data = load_bundle('sharadar')
    asset_finder = bundle_data.asset_finder

    # Create universe of stocks with fundamental data
    universe_assets = []
    for ticker in UNIVERSE_TICKERS:
        try:
            assets = asset_finder.lookup_symbols([ticker], as_of_date=None)
            if assets and assets[0] is not None:
                universe_assets.append(assets[0])
        except:
            pass

    # Create base universe filter
    base_universe = StaticAssets(universe_assets)

    # Get fundamental metrics from custom database
    roe = CustomFundamentals.ROE.latest
    pe_ratio = CustomFundamentals.PERatio.latest
    debt_to_equity = CustomFundamentals.DebtToEquity.latest
    eps = CustomFundamentals.EPS.latest
    current_ratio = CustomFundamentals.CurrentRatio.latest
    sector = CustomFundamentals.Sector.latest

    # Calculate derived metrics
    quality_score = QualityScore()
    profit_margin = ProfitMargin()

    # Get pricing data from bundle
    close_price = EquityPricing.close.latest
    volume = EquityPricing.volume.latest

    # Calculate technical indicators
    avg_volume_20d = SimpleMovingAverage(inputs=[EquityPricing.volume], window_length=20)

    # Define screening filters
    # 0. Base universe (stocks with fundamental data)
    # 1. Fundamental quality
    high_roe = (roe > 5.0)  # Lowered threshold for more stocks
    reasonable_pe = (pe_ratio < 50.0)  # Reasonable valuation
    manageable_debt = (debt_to_equity < 5.0)  # Not over-leveraged

    # 2. Liquidity (from pricing data)
    liquid = (avg_volume_20d > 100000)  # Minimum liquidity

    # 3. Valid price
    valid_price = (close_price > 1.0)  # Minimum price

    # Combine all filters (start with base universe)
    quality_universe = (
        base_universe &
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
            # TEMPORARILY REMOVED: 'sector': sector,  # Object dtype may cause issues

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
    # Note: Fundamentals are quarterly (end of March, June, Sept, Dec)
    # We rebalance monthly but will only see universe changes when new fundamentals arrive
    if REBALANCE_FREQUENCY == 'monthly':
        schedule_function(
            rebalance,
            date_rules.month_end(),  # Changed to month_end to align with fundamentals data
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

    # Use recent dates (last 3 months) to match available data
    end_date = pd.Timestamp.now(tz='UTC').normalize()
    start_date = (end_date - pd.DateOffset(months=3)).normalize()

    print(f"\nUsing date range: {start_date.date()} to {end_date.date()}")

    # Set up custom loader with explicit db_dir
    print("\nSetting up custom loader...")
    custom_loader = setup_custom_loader()

    # Run the algorithm
    results = run_algorithm(
        start=start_date,
        end=end_date,
        initialize=initialize,
        before_trading_start=before_trading_start,
        analyze=analyze,
        capital_base=INITIAL_CAPITAL,
        bundle='sharadar',
        custom_loader=custom_loader,
    )

    print("\n✓ Backtest complete!")
    print(f"\nTo visualize results, run:")
    print(f"  python plot_backtest_results.py")
