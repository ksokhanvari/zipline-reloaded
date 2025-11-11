#!/usr/bin/env python
"""
Momentum Strategy with FlightLog - Production Example

A complete momentum-based trading strategy demonstrating elegant use of
FlightLog for real-time monitoring and debugging.

Strategy:
    - Ranks stocks by 20-day momentum
    - Goes long top 5 performers
    - Rebalances weekly
    - Uses FlightLog for real-time monitoring

Usage:
    Terminal 1: python scripts/flightlog.py --host 0.0.0.0 --level INFO
    Terminal 2: python examples/momentum_strategy_with_flightlog.py

    Watch Terminal 1 for real-time colored logs!
"""

import logging
import pandas as pd
from custom_data.register_bundles import ensure_bundles_registered
from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    schedule_function,
    date_rules,
    time_rules,
    set_commission,
    set_slippage,
    symbol,
)
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Returns
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog

# Ensure bundles are registered
ensure_bundles_registered()


# =============================================================================
# Strategy Configuration
# =============================================================================

LOOKBACK_DAYS = 20      # Momentum lookback period
TOP_N_STOCKS = 5        # Number of stocks to hold
REBALANCE_DAYS = 5      # Rebalance every N days
POSITION_SIZE = 0.95    # Use 95% of capital


# =============================================================================
# Strategy Implementation
# =============================================================================

def initialize(context):
    """
    Initialize the trading strategy.

    Sets up:
        - Universe selection pipeline
        - Rebalancing schedule
        - Execution parameters
        - FlightLog monitoring
    """
    # Setup logging level
    logging.basicConfig(level=logging.INFO)

    log_to_flightlog(
        f'Initializing Momentum Strategy (lookback={LOOKBACK_DAYS}d, top={TOP_N_STOCKS})',
        level='INFO'
    )

    # Define our stock universe
    context.universe = [
        symbol('AAPL'), symbol('MSFT'), symbol('GOOGL'),
        symbol('AMZN'), symbol('META'), symbol('TSLA'),
        symbol('NVDA'), symbol('JPM'), symbol('V'),
        symbol('WMT')
    ]

    # Track rebalancing
    context.days_since_rebalance = 0
    context.rebalance_count = 0

    # Schedule rebalancing function
    schedule_function(
        rebalance,
        date_rules.every_day(),
        time_rules.market_open(hours=1)
    )

    log_to_flightlog(
        f'Strategy initialized with {len(context.universe)} stocks',
        level='INFO'
    )


def rebalance(context, data):
    """
    Rebalance portfolio based on momentum signals.

    Called daily, but only rebalances every REBALANCE_DAYS days.
    """
    context.days_since_rebalance += 1

    # Only rebalance every N days
    if context.days_since_rebalance < REBALANCE_DAYS:
        return

    context.days_since_rebalance = 0
    context.rebalance_count += 1

    try:
        # Calculate momentum for each stock
        momentum_scores = calculate_momentum(context, data)

        if len(momentum_scores) == 0:
            log_to_flightlog(
                'No valid momentum signals - skipping rebalance',
                level='WARNING'
            )
            return

        # Select top N stocks by momentum
        top_stocks = momentum_scores.nlargest(TOP_N_STOCKS)

        # Log rebalancing decision
        log_to_flightlog(
            f'Rebalance #{context.rebalance_count}: Selected {len(top_stocks)} stocks',
            level='INFO'
        )

        # Calculate target position size
        if len(top_stocks) > 0:
            target_weight = POSITION_SIZE / len(top_stocks)
        else:
            target_weight = 0

        # Place orders
        orders_placed = 0
        for stock in context.universe:
            if stock in top_stocks.index:
                # Long position
                order_target_percent(stock, target_weight)
                orders_placed += 1
            else:
                # Close position if held
                if stock in context.portfolio.positions:
                    order_target_percent(stock, 0)

        # Log execution summary
        portfolio_value = context.portfolio.portfolio_value
        cash = context.portfolio.cash
        positions = len(context.portfolio.positions)

        log_to_flightlog(
            f'Rebalance complete: {orders_placed} orders, {positions} positions, '
            f'Portfolio: ${portfolio_value:,.0f}, Cash: ${cash:,.0f}',
            level='INFO'
        )

    except Exception as e:
        log_to_flightlog(
            f'Error during rebalance: {str(e)}',
            level='ERROR'
        )


def calculate_momentum(context, data):
    """
    Calculate momentum scores for all stocks in universe.

    Returns
    -------
    pd.Series
        Momentum scores indexed by stock symbol
    """
    momentum = {}

    for stock in context.universe:
        try:
            # Get historical prices
            prices = data.history(
                stock,
                'price',
                LOOKBACK_DAYS + 1,
                '1d'
            )

            if len(prices) < LOOKBACK_DAYS + 1:
                continue

            # Calculate momentum (percentage change)
            momentum[stock] = (prices[-1] / prices[0] - 1) * 100

        except Exception as e:
            # Log issues with individual stocks
            log_to_flightlog(
                f'Could not calculate momentum for {stock.symbol}: {str(e)}',
                level='DEBUG'
            )
            continue

    return pd.Series(momentum)


def analyze(context, perf):
    """
    Analyze backtest results and log summary.

    Called once at the end of the backtest.
    """
    try:
        # Calculate key metrics
        total_return = perf['algorithm_period_return'].iloc[-1] * 100
        final_value = perf['portfolio_value'].iloc[-1]

        # Get Sharpe ratio if available
        sharpe = perf['sharpe'].iloc[-1] if 'sharpe' in perf.columns else None

        # Get max drawdown if available
        if 'max_drawdown' in perf.columns:
            max_dd = perf['max_drawdown'].iloc[-1] * 100
        else:
            max_dd = None

        # Count winning/losing days
        daily_returns = perf['returns']
        winning_days = (daily_returns > 0).sum()
        losing_days = (daily_returns < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) * 100 if (winning_days + losing_days) > 0 else 0

        # Log comprehensive summary
        log_to_flightlog('=' * 60, level='INFO')
        log_to_flightlog('BACKTEST SUMMARY', level='INFO')
        log_to_flightlog('=' * 60, level='INFO')
        log_to_flightlog(f'Total Return: {total_return:+.2f}%', level='INFO')
        log_to_flightlog(f'Final Portfolio Value: ${final_value:,.0f}', level='INFO')

        if sharpe is not None:
            log_to_flightlog(f'Sharpe Ratio: {sharpe:.2f}', level='INFO')

        if max_dd is not None:
            log_to_flightlog(f'Max Drawdown: {max_dd:.1f}%', level='INFO')

        log_to_flightlog(
            f'Win Rate: {win_rate:.1f}% ({winning_days} up / {losing_days} down)',
            level='INFO'
        )
        log_to_flightlog(f'Total Rebalances: {context.rebalance_count}', level='INFO')
        log_to_flightlog('=' * 60, level='INFO')

    except Exception as e:
        log_to_flightlog(
            f'Error in analysis: {str(e)}',
            level='ERROR'
        )


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Run the momentum strategy with FlightLog monitoring.
    """
    print("=" * 70)
    print("Momentum Strategy with FlightLog")
    print("=" * 70)
    print()
    print("Make sure FlightLog is running in another terminal:")
    print("  Terminal 1: python scripts/flightlog.py --host 0.0.0.0 --level INFO")
    print()
    print("Starting backtest...")
    print()

    # Enable FlightLog (connects to localhost)
    flightlog_connected = enable_flightlog(host='localhost', port=9020)

    if flightlog_connected:
        print("✓ Connected to FlightLog on localhost:9020")
        print("  Watch Terminal 1 for real-time colored logs!")
    else:
        print("⚠ FlightLog not available - logs will appear here only")

    print()

    # Enable progress logging
    enable_progress_logging(
        algo_name='Momentum-Strategy',
        update_interval=5  # Log every 5 days
    )

    # Run the backtest
    result = run_algorithm(
        start=pd.Timestamp('2023-01-01'),
        end=pd.Timestamp('2023-12-31'),
        initialize=initialize,
        analyze=analyze,
        capital_base=100000,
        bundle='sharadar'
    )

    print()
    print("=" * 70)
    print("Backtest Complete!")
    print("=" * 70)
    print()
    print(f"Total Return: {result['algorithm_period_return'].iloc[-1] * 100:+.2f}%")
    print(f"Final Value: ${result['portfolio_value'].iloc[-1]:,.0f}")

    if 'sharpe' in result.columns:
        print(f"Sharpe Ratio: {result['sharpe'].iloc[-1]:.2f}")

    print()
    print("Check FlightLog terminal for detailed execution logs!")

    return result


if __name__ == '__main__':
    result = main()
