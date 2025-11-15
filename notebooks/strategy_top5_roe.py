#!/usr/bin/env python3
"""
Top 5 ROE Strategy using Custom Fundamentals

A quantitative trading strategy that selects high-quality stocks based on
Return on Equity (ROE) from a universe of large-cap companies.

STRATEGY OVERVIEW
=================
This strategy implements a fundamental factor-based approach:
1. Universe Selection: Top 100 stocks by market capitalization
2. Factor Ranking: Select top 5 stocks by Return on Equity (ROE)
3. Rebalancing: Weekly rebalancing at market open + 1 hour
4. Weighting: Equal weight allocation across selected stocks

CUSTOM DATA INTEGRATION
========================
This strategy demonstrates Zipline's clean custom data loader approach:
- Custom fundamentals loaded from SQLite database
- Auto-discovery of fundamental columns (no manual mapping)
- Domain-aware column matching via LoaderDict
- No monkey-patching required

USAGE
=====
Run from command line:
    python strategy_top5_roe.py

Or import and use programmatically:
    from strategy_top5_roe import build_pipeline_loaders, make_pipeline
    custom_loader = build_pipeline_loaders()
    results = run_algorithm(..., custom_loader=custom_loader)

REQUIREMENTS
============
- Zipline Reloaded with custom data support
- Custom fundamentals database at /root/.zipline/data/custom/fundamentals.sqlite
- Sharadar bundle data
- See load_csv_fundamentals.ipynb for database creation

OUTPUT
======
- Console: Backtest statistics and rebalance logs
- Files: Results saved to /notebooks/backtest_results.{csv,pkl}

CUSTOMIZATION
=============
Modify the configuration section below to adjust:
- Backtest date range and starting capital
- Universe size (top N by market cap)
- Selection size (top M by ROE)
- Rebalancing frequency (daily/weekly/monthly)
- Database location and bundle name
- Output options (CSV, pickle, metadata)
- Logging verbosity (progress bar, pipeline stats, rebalance details)

PROGRESS LOGGING
================
The strategy includes Zipline's built-in progress logging system:
- enable_progress_logging(): Real-time backtest progress updates
  - Algorithm name: 'Top5-ROE-Strategy'
  - Update interval: Configurable (default: 10 days)
  - Shows day progress, percentage, and portfolio value
- enable_flightlog(): Optional integration with FlightLog server
  - Real-time monitoring and visualization
  - Requires FlightLog server running
- Pipeline stats: Daily universe statistics (optional)
- Rebalance details: Detailed BUY/SELL logging with metrics (optional)
- Timestamps: Start/end times and execution duration
- Performance summary: Return, Sharpe, drawdown, win rate, trades
- Metadata export: JSON file with configuration and results

Example output during execution:
  [2025-01-15 14:32:10] Top5-ROE-Strategy | Day 100/3450 (2.9%) | Portfolio: $102,345
  [2025-01-15 14:32:20] Top5-ROE-Strategy | Day 200/3450 (5.8%) | Portfolio: $105,678
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

# Backtest Parameters
START_DATE = '2012-03-10'  # Backtest start date (allow lookback window)
END_DATE = '2025-11-11'    # Backtest end date
CAPITAL_BASE = 100000      # Starting capital ($)

# Strategy Parameters
UNIVERSE_SIZE = 100        # Top N stocks by market cap
SELECTION_SIZE = 5         # Top M stocks by ROE to hold
REBALANCE_FREQ = 'weekly'  # weekly, monthly, or daily

# Database Configuration
DB_NAME = "fundamentals"   # Must match load_csv_fundamentals.ipynb
DB_DIR = Path('/root/.zipline/data/custom')

# Bundle Configuration
BUNDLE_NAME = 'sharadar'

# Output Configuration
RESULTS_DIR = Path('/notebooks')
SAVE_CSV = True
SAVE_PICKLE = True
SAVE_METADATA = True

# Logging Configuration
LOG_PIPELINE_STATS = True  # Log daily pipeline stats
LOG_REBALANCE_DETAILS = True  # Log detailed trade logging
PROGRESS_UPDATE_INTERVAL = 10  # Days between progress updates
ENABLE_FLIGHTLOG = False  # Enable FlightLog integration (requires server)
FLIGHTLOG_HOST = 'flightlog'  # FlightLog server host
FLIGHTLOG_PORT = 9020  # FlightLog server port

from datetime import datetime
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
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog

# Enable logging
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Register Bundle (Required for Jupyter Notebooks)
# ============================================================================

# Register Sharadar bundle (required for Jupyter notebooks)
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

register(BUNDLE_NAME, sharadar_bundle(tickers=None, incremental=True, include_funds=True))
print(f"✓ {BUNDLE_NAME} bundle registered")

# ============================================================================
# Define Custom Fundamentals Database
# ============================================================================

class CustomFundamentals(Database):
    """
    Custom Fundamentals Database Schema.

    This class defines the schema for custom fundamental data loaded from
    SQLite. Add columns here to make them available in your pipeline.

    AUTO-DISCOVERY: All Column attributes defined here are automatically
    discovered and mapped to the CustomSQLiteLoader by build_pipeline_loaders().
    No manual mapping required!

    Attributes
    ----------
    CODE : str
        Database identifier (must match database filename without .sqlite)
    LOOKBACK_WINDOW : int
        Number of days to look back for data (252 = ~1 year)

    Columns
    -------
    ReturnOnEquity_SmartEstimat : Column(float)
        Return on equity - key profitability metric
    ReturnOnAssets_SmartEstimate : Column(float)
        Return on assets - efficiency metric
    CompanyMarketCap : Column(float)
        Market capitalization in USD
    RefPriceClose : Column(float)
        Reference closing price
    GICSSectorName : Column(str)
        GICS sector classification
    LongTermGrowth_Mean : Column(float)
        Analyst consensus long-term growth rate
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_ : Column(float)
        EV/EBITDA valuation ratio

    Notes
    -----
    To add new columns:
    1. Add column definition here: `NewColumn = Column(dtype)`
    2. Ensure column exists in SQLite database
    3. Run strategy - column is auto-discovered and mapped
    """

    CODE = DB_NAME           # Database identifier
    LOOKBACK_WINDOW = 252    # Days to look back (~1 year)

    # Profitability Metrics
    ReturnOnEquity_SmartEstimat = Column(float)
    ReturnOnAssets_SmartEstimate = Column(float)

    # Size and Price Metrics
    CompanyMarketCap = Column(float)
    RefPriceClose = Column(float)

    # Classification
    GICSSectorName = Column(str)

    # Growth and Valuation
    LongTermGrowth_Mean = Column(float)
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_ = Column(float)


# ============================================================================
# Build Pipeline Loader Map (Clean Approach)
# ============================================================================

def build_pipeline_loaders(db_dir=DB_DIR, db_code=None, bundle_name=BUNDLE_NAME):
    """
    Build a PipelineLoader map with auto-discovery of fundamental columns.

    This function creates a mapping between pipeline columns and their data loaders.
    It uses introspection to automatically discover all columns defined in
    CustomFundamentals, eliminating manual column listing.

    KEY FEATURES
    ------------
    - Auto-discovers fundamental columns via introspection
    - Domain-aware column matching (handles USEquityPricing<US>.close)
    - No monkey-patching required
    - Single source of truth (columns defined once in Database class)

    Parameters
    ----------
    db_dir : Path, optional
        Directory containing custom SQLite databases (default: DB_DIR)
    db_code : str, optional
        Database identifier/filename without .sqlite extension
        (default: CustomFundamentals.CODE)
    bundle_name : str, optional
        Name of the bundle to load for pricing data (default: BUNDLE_NAME)

    Returns
    -------
    LoaderDict
        Dictionary mapping pipeline columns to their loaders
        - Pricing columns → USEquityPricingLoader
        - Fundamental columns → CustomSQLiteLoader

    Examples
    --------
    >>> custom_loader = build_pipeline_loaders()
    >>> results = run_algorithm(..., custom_loader=custom_loader)

    >>> # Use custom database location
    >>> custom_loader = build_pipeline_loaders(
    ...     db_dir=Path('/custom/path'),
    ...     db_code='my_fundamentals'
    ... )

    Notes
    -----
    The LoaderDict class handles domain-aware column matching, so columns
    registered as USEquityPricing.close will match lookups for
    USEquityPricing<US_EQUITIES>.close.
    """
    if db_code is None:
        db_code = CustomFundamentals.CODE

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
    bundle_data = load_bundle(bundle_name)

    # Create loaders
    pricing_loader = USEquityPricingLoader.without_fx(
        bundle_data.equity_daily_bar_reader,
        bundle_data.adjustment_reader
    )

    fundamentals_loader = CustomSQLiteLoader(
        db_code=db_code,
        db_dir=db_dir
    )

    # Build the loader map using LoaderDict
    custom_loader = LoaderDict()

    # Map all pricing columns to pricing loader
    for attr_name in ['close', 'high', 'low', 'open', 'volume']:
        custom_loader[getattr(USEquityPricing, attr_name)] = pricing_loader

    # Automatically map ALL fundamental columns to fundamentals loader
    # This uses introspection to avoid manual column listing
    fundamental_count = 0
    for attr_name in dir(CustomFundamentals):
        # Skip private/magic attributes and class metadata
        if attr_name.startswith('_') or attr_name in ['CODE', 'LOOKBACK_WINDOW']:
            continue

        attr = getattr(CustomFundamentals, attr_name)

        # Map if it's a Column (has dataset attribute, indicating it's a BoundColumn)
        if hasattr(attr, 'dataset'):
            custom_loader[attr] = fundamentals_loader
            fundamental_count += 1

    print(f"✓ Pipeline loader map built ({len(custom_loader)} columns mapped)")
    print(f"  - Pricing columns: 5")
    print(f"  - Fundamental columns: {fundamental_count} (auto-discovered)")

    return custom_loader


# ============================================================================
# Pipeline Definition
# ============================================================================

def make_pipeline(universe_size=UNIVERSE_SIZE, selection_size=SELECTION_SIZE):
    """
    Create a pipeline for ROE-based stock selection.

    This pipeline implements a two-stage filtering process:
    1. Universe Selection: Filter to top N stocks by market capitalization
    2. Factor Ranking: Select top M stocks by Return on Equity (ROE)

    The pipeline outputs the selected stocks along with their fundamental metrics
    for analysis and rebalancing.

    Parameters
    ----------
    universe_size : int, optional
        Number of stocks to include in universe (top N by market cap)
        Default: UNIVERSE_SIZE (100)
    selection_size : int, optional
        Number of stocks to select from universe (top M by ROE)
        Default: SELECTION_SIZE (5)

    Returns
    -------
    Pipeline
        A Zipline pipeline with the following:
        - Columns: ROE, Market_Cap, Sector
        - Screen: Top M stocks by ROE from top N by market cap

    Examples
    --------
    >>> # Use default parameters
    >>> pipe = make_pipeline()

    >>> # Custom universe and selection
    >>> pipe = make_pipeline(universe_size=200, selection_size=10)

    >>> # Attach to algorithm
    >>> attach_pipeline(pipe, 'my_strategy')

    Notes
    -----
    The .latest attribute accesses the most recent value for each metric.
    The pipeline automatically handles missing data and lookback windows.
    """
    # Get factors (most recent values)
    roe = CustomFundamentals.ReturnOnEquity_SmartEstimat.latest
    market_cap = CustomFundamentals.CompanyMarketCap.latest
    sector = CustomFundamentals.GICSSectorName.latest

    # Stage 1: Universe selection - top N by market cap
    universe = market_cap.top(universe_size)

    # Stage 2: Factor ranking - top M by ROE from universe
    selection = roe.top(selection_size, mask=universe)

    return Pipeline(
        columns={
            'ROE': roe,
            'Market_Cap': market_cap,
            'Sector': sector,
        },
        screen=selection,
    )


# ============================================================================
# Strategy Logic
# ============================================================================

def initialize(context):
    """
    Initialize the trading algorithm.

    Called once at the start of the backtest. Sets up the pipeline,
    schedules rebalancing, and initializes context variables.

    Parameters
    ----------
    context : AlgorithmContext
        Zipline algorithm context object. Used to store state between
        function calls during the backtest.

    Notes
    -----
    This function:
    1. Creates and attaches the pipeline for stock selection
    2. Schedules the rebalance function to run periodically
    3. Initializes context variables for tracking
    4. Logs configuration for verification

    The rebalancing frequency is determined by REBALANCE_FREQ:
    - 'weekly': Runs every Monday at market open + 1 hour
    - 'monthly': Runs first trading day of month
    - 'daily': Runs every day at market open + 1 hour
    """
    # Attach pipeline
    pipe = make_pipeline()
    attach_pipeline(pipe, 'roe_strategy')

    # Schedule rebalancing based on configuration
    if REBALANCE_FREQ == 'weekly':
        schedule_function(
            rebalance,
            date_rules.week_start(),
            time_rules.market_open(hours=1)
        )
    elif REBALANCE_FREQ == 'monthly':
        schedule_function(
            rebalance,
            date_rules.month_start(),
            time_rules.market_open(hours=1)
        )
    elif REBALANCE_FREQ == 'daily':
        schedule_function(
            rebalance,
            date_rules.every_day(),
            time_rules.market_open(hours=1)
        )

    # Initialize context variables
    context.rebalance_count = 0

    # Log configuration
    logging.info("=" * 60)
    logging.info("ROE STRATEGY INITIALIZED")
    logging.info("=" * 60)
    logging.info(f"Rebalancing: {REBALANCE_FREQ.capitalize()}")
    logging.info(f"Universe: Top {UNIVERSE_SIZE} by market cap")
    logging.info(f"Selection: Top {SELECTION_SIZE} by ROE")
    logging.info(f"Weighting: Equal weight")
    logging.info("=" * 60)


def before_trading_start(context, data):
    """
    Called daily before market opens.

    Fetches the latest pipeline output containing stock selections and
    fundamental metrics. This data is used by the rebalance function.

    Parameters
    ----------
    context : AlgorithmContext
        Algorithm context object
    data : BarData
        Object providing methods to get pricing and volume data

    Notes
    -----
    Pipeline output is cached in context.pipeline_data and updated daily.
    The rebalance function accesses this cached data when it runs.
    """
    context.pipeline_data = pipeline_output('roe_strategy')

    # Log daily pipeline statistics
    if LOG_PIPELINE_STATS and context.pipeline_data is not None and not context.pipeline_data.empty:
        current_date = context.get_datetime().date()
        universe_size = len(context.pipeline_data)
        avg_roe = context.pipeline_data['ROE'].mean()
        avg_mcap = context.pipeline_data['Market_Cap'].mean()

        logging.info(f"{current_date} | Universe: {universe_size:2d} stocks | "
                    f"Avg ROE: {avg_roe:.2%} | Avg MCap: ${avg_mcap/1e9:.1f}B")


def rebalance(context, data):
    """
    Rebalance portfolio based on pipeline selections.

    This function:
    1. Gets the latest pipeline output (stock selections)
    2. Filters to tradeable stocks only
    3. Sells positions no longer in selection
    4. Buys/rebalances positions to equal weight
    5. Logs holdings and metrics

    Parameters
    ----------
    context : AlgorithmContext
        Algorithm context object containing portfolio state
    data : BarData
        Object providing methods to check tradeability and get prices

    Notes
    -----
    Equal weighting: Each selected stock gets 1/N of portfolio value.
    Untradeable stocks are filtered out to prevent order errors.
    All trades use order_target_percent for precise allocation.
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
    target_positions = set(selected_stocks)

    # Calculate position changes
    to_sell = current_positions - target_positions
    to_buy = target_positions - current_positions
    to_rebalance = current_positions & target_positions

    # Log rebalance header
    if LOG_REBALANCE_DETAILS:
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"REBALANCE #{context.rebalance_count}")
        logging.info("=" * 70)
        logging.info(f"  Sell: {len(to_sell)} | Buy: {len(to_buy)} | "
                    f"Rebalance: {len(to_rebalance)} | Weight: {target_weight:.2%}")

    # Sell stocks no longer in selection (only if tradeable)
    for stock in to_sell:
        if data.can_trade(stock):
            order_target_percent(stock, 0.0)
            if LOG_REBALANCE_DETAILS:
                logging.info(f"    SELL: {stock.symbol}")

    # Buy/rebalance selected stocks
    tradeable_output = context.pipeline_data.loc[selected_stocks]
    for stock in selected_stocks:
        if data.can_trade(stock):
            order_target_percent(stock, target_weight)
            if LOG_REBALANCE_DETAILS and stock in to_buy:
                stock_data = tradeable_output.loc[stock]
                logging.info(f"    BUY:  {stock.symbol} "
                           f"(ROE: {stock_data['ROE']:.2%}, "
                           f"MCap: ${stock_data['Market_Cap']/1e9:.1f}B)")

    # Log summary statistics
    if LOG_REBALANCE_DETAILS and not tradeable_output.empty:
        logging.info(f"  Portfolio Summary:")
        logging.info(f"    Holdings: {', '.join([s.symbol for s in selected_stocks])}")
        logging.info(f"    Avg ROE: {tradeable_output['ROE'].mean():.2%}")
        logging.info(f"    Avg Market Cap: ${tradeable_output['Market_Cap'].mean()/1e9:.2f}B")
        logging.info("=" * 70)
    elif not LOG_REBALANCE_DETAILS:
        # Minimal logging
        holdings = [s.symbol for s in selected_stocks]
        logging.info(f"Rebalance #{context.rebalance_count}: {', '.join(holdings)}")


def handle_data(context, data):
    """
    Called on every bar (daily in this strategy).

    Records performance metrics for analysis and plotting.

    Parameters
    ----------
    context : AlgorithmContext
        Algorithm context object containing portfolio state
    data : BarData
        Object providing methods to get pricing and volume data

    Notes
    -----
    Recorded metrics are stored in the backtest results DataFrame:
    - portfolio_value: Total portfolio value ($)
    - num_positions: Number of stocks currently held
    - leverage: Portfolio leverage ratio
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
    """
    Main execution block.

    This runs when the script is executed directly (not imported).
    It performs the following steps:
    1. Build pipeline loaders with auto-discovery
    2. Run the backtest with configured parameters
    3. Calculate and display performance metrics
    4. Save results to disk
    """

    # Build pipeline loader map (CLEAN APPROACH - NO MONKEY-PATCHING!)
    print("\n" + "=" * 80)
    print("BUILDING PIPELINE LOADERS")
    print("=" * 80)
    custom_loader = build_pipeline_loaders()
    print()

    # Parse dates
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    # Calculate period duration
    trading_days = pd.bdate_range(start, end)
    years = len(trading_days) / 252

    # Display backtest configuration
    print("=" * 80)
    print("BACKTEST CONFIGURATION")
    print("=" * 80)
    print(f"Strategy: Top {SELECTION_SIZE} ROE from Top {UNIVERSE_SIZE} by Market Cap")
    print(f"Period: {start.date()} to {end.date()} ({years:.1f} years)")
    print(f"Capital: ${CAPITAL_BASE:,}")
    print(f"Rebalancing: {REBALANCE_FREQ.capitalize()}")
    print(f"Bundle: {BUNDLE_NAME}")
    print(f"Database: {DB_DIR / (DB_NAME + '.sqlite')}")
    print(f"Progress updates: Every {PROGRESS_UPDATE_INTERVAL} days")
    print(f"Pipeline logging: {'Enabled' if LOG_PIPELINE_STATS else 'Disabled'}")
    print(f"Rebalance logging: {'Detailed' if LOG_REBALANCE_DETAILS else 'Minimal'}")
    print("=" * 80)
    print()

    # Enable progress logging
    print("=" * 80)
    print("ENABLING PROGRESS LOGGING")
    print("=" * 80)

    # Enable FlightLog if configured
    if ENABLE_FLIGHTLOG:
        enable_flightlog(host=FLIGHTLOG_HOST, port=FLIGHTLOG_PORT)
        print(f"✓ FlightLog enabled: {FLIGHTLOG_HOST}:{FLIGHTLOG_PORT}")

    # Enable progress tracking
    enable_progress_logging(
        algo_name='Top5-ROE-Strategy',
        update_interval=PROGRESS_UPDATE_INTERVAL
    )
    print(f"✓ Progress logging enabled")
    print(f"  Algorithm: Top5-ROE-Strategy")
    print(f"  Update interval: {PROGRESS_UPDATE_INTERVAL} days")
    print()

    # Run backtest
    print("=" * 80)
    print("RUNNING BACKTEST")
    print("=" * 80)
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        before_trading_start=before_trading_start,
        handle_data=handle_data,
        capital_base=CAPITAL_BASE,
        data_frequency='daily',
        bundle=BUNDLE_NAME,
        custom_loader=custom_loader,
    )

    end_time = datetime.now()
    duration = end_time - start_time

    # Display completion
    print()
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()

    # Calculate performance metrics
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    initial_value = results['portfolio_value'].iloc[0]
    final_value = results['portfolio_value'].iloc[-1]
    total_return = ((final_value / initial_value) - 1) * 100

    daily_returns = results['portfolio_value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    max_drawdown = ((results['portfolio_value'] / results['portfolio_value'].cummax()) - 1).min() * 100

    # Win rate
    winning_days = (daily_returns > 0).sum()
    total_days = len(daily_returns)
    win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0

    # Count trades
    num_trades = results['orders'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()

    print(f"Initial Capital:     ${initial_value:,.2f}")
    print(f"Final Value:         ${final_value:,.2f}")
    print(f"Total Return:        {total_return:.2f}%")
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")
    print(f"Max Drawdown:        {max_drawdown:.2f}%")
    print(f"Win Rate:            {win_rate:.1f}%")
    print(f"Total Trades:        {num_trades}")
    print(f"Trading Days:        {len(results):,}")
    print("=" * 80)

    # Save results to disk
    if SAVE_CSV or SAVE_PICKLE or SAVE_METADATA:
        print()
        print("=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Ensure results directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        csv_path = RESULTS_DIR / 'backtest_results.csv'
        pickle_path = RESULTS_DIR / 'backtest_results.pkl'
        metadata_path = RESULTS_DIR / 'backtest_results.json'

        if SAVE_CSV:
            results.to_csv(csv_path)
            print(f"✓ CSV: {csv_path}")
            print(f"  Size: {csv_path.stat().st_size / 1024:.1f} KB")
            print(f"  Rows: {len(results):,}")

        if SAVE_PICKLE:
            results.to_pickle(pickle_path)
            print(f"✓ Pickle: {pickle_path}")
            print(f"  Size: {pickle_path.stat().st_size / 1024:.1f} KB")

        if SAVE_METADATA:
            import json
            metadata = {
                'strategy_name': 'Top 5 ROE Strategy',
                'start_date': start.isoformat(),
                'end_date': end.isoformat(),
                'capital_base': CAPITAL_BASE,
                'universe_size': UNIVERSE_SIZE,
                'selection_size': SELECTION_SIZE,
                'rebalance_frequency': REBALANCE_FREQ,
                'bundle': BUNDLE_NAME,
                'database': str(DB_DIR / (DB_NAME + '.sqlite')),
                'run_timestamp': start_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'performance': {
                    'initial_value': float(initial_value),
                    'final_value': float(final_value),
                    'total_return_pct': float(total_return),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown_pct': float(max_drawdown),
                    'win_rate_pct': float(win_rate),
                    'num_trades': int(num_trades),
                    'trading_days': len(results),
                },
                'files': {
                    'csv': str(csv_path) if SAVE_CSV else None,
                    'pickle': str(pickle_path) if SAVE_PICKLE else None,
                }
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✓ Metadata: {metadata_path}")
            print(f"  Size: {metadata_path.stat().st_size / 1024:.1f} KB")

        print("=" * 80)

    print("\n✓ Backtest complete!")
    print(f"✓ Execution time: {duration}\n")
