#!/usr/bin/env python
"""
Test FlightLog with a real backtest.

Usage:
    Terminal 1: python scripts/flightlog.py --host 0.0.0.0 --level INFO
    Terminal 2: python notebooks/test_flightlog_backtest.py
"""

import logging
from zipline import run_algorithm
from zipline.api import *
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
from zipline.utils.progress import enable_progress_logging
import pandas as pd

# Setup logging FIRST
logging.basicConfig(level=logging.INFO)

# Enable FlightLog
print("Connecting to FlightLog...")
connected = enable_flightlog(host='localhost', port=9020)
if connected:
    print("✓ Connected to FlightLog on localhost:9020")
    print("  Logs will stream to Terminal 1\n")
else:
    print("⚠ Could not connect to FlightLog")
    print("  Make sure it's running: python scripts/flightlog.py\n")

# Enable progress logging
enable_progress_logging(
    algo_name='Test-Strategy',
    update_interval=5  # Log every 5 days
)

def initialize(context):
    """Setup algorithm."""
    log_to_flightlog("Initializing Test Strategy...", level='INFO')

    # Pick a simple stock
    context.stock = symbol('AAPL')

    log_to_flightlog(f"Trading {context.stock.symbol}", level='INFO')

def handle_data(context, data):
    """Execute trades."""
    # Simple buy-and-hold
    if not context.portfolio.positions:
        order_target_percent(context.stock, 0.95)
        log_to_flightlog(
            f"Opened position: {context.stock.symbol}",
            level='INFO'
        )

def analyze(context, perf):
    """Analyze results."""
    total_return = perf['algorithm_period_return'].iloc[-1]
    sharpe = perf['sharpe'].iloc[-1] if 'sharpe' in perf else 0

    log_to_flightlog(
        f"Backtest complete! Return: {total_return*100:.2f}%",
        level='INFO'
    )
    log_to_flightlog(
        f"Sharpe Ratio: {sharpe:.2f}",
        level='INFO'
    )

if __name__ == '__main__':
    print("="*60)
    print("Starting backtest with FlightLog streaming...")
    print("="*60)

    result = run_algorithm(
        start=pd.Timestamp('2020-01-01'),
        end=pd.Timestamp('2020-06-30'),
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=100000,
        bundle='sharadar'
    )

    print("\n" + "="*60)
    print("Backtest Complete!")
    print("="*60)
    print(f"Final return: {result['algorithm_period_return'].iloc[-1]*100:.2f}%")
    print("\nCheck Terminal 1 for detailed logs!")
