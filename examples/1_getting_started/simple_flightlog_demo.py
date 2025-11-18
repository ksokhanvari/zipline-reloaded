#!/usr/bin/env python
"""
Simple FlightLog Demo - Minimal Example

The simplest possible example showing FlightLog in action.

Terminal 1: python scripts/flightlog.py --host 0.0.0.0 --level INFO
Terminal 2: python examples/simple_flightlog_demo.py
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.register_bundles import ensure_bundles_registered
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog

# Ensure bundles are registered
ensure_bundles_registered()


# =============================================================================
# Setup (3 lines)
# =============================================================================
# NOTE: These functions prevent duplicate handlers automatically,
# so it's safe to run this multiple times in Jupyter notebooks

logging.basicConfig(level=logging.INFO, force=True)  # force=True prevents duplicates
enable_flightlog(host='localhost', port=9020)  # Auto-checks for existing handlers
enable_progress_logging(algo_name='Simple-Demo', update_interval=5)  # Clears old handlers


# =============================================================================
# Strategy (Clean and Simple)
# =============================================================================

def initialize(context):
    """Setup strategy."""
    log_to_flightlog('Initializing simple buy-and-hold strategy', level='INFO')
    context.stock = symbol('AAPL')


def handle_data(context, data):
    """Buy and hold AAPL."""
    if not context.portfolio.positions:
        order_target_percent(context.stock, 0.95)
        log_to_flightlog('Purchased AAPL - holding long term', level='INFO')


def analyze(context, perf):
    """Show final results."""
    total_return = perf['algorithm_period_return'].iloc[-1] * 100
    final_value = perf['portfolio_value'].iloc[-1]

    log_to_flightlog(f'Final Return: {total_return:+.2f}%', level='INFO')
    log_to_flightlog(f'Final Value: ${final_value:,.0f}', level='INFO')


# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    print("Simple FlightLog Demo")
    print("=" * 60)
    print("Watch Terminal 1 for real-time colored logs!")
    print()

    result = run_algorithm(
        start=pd.Timestamp('2023-01-01'),
        end=pd.Timestamp('2023-12-31'),
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=100000,
        bundle='sharadar'
    )

    print()
    print("=" * 60)
    print(f"Total Return: {result['algorithm_period_return'].iloc[-1] * 100:+.2f}%")
    print(f"Final Value: ${result['portfolio_value'].iloc[-1]:,.0f}")
    print("=" * 60)
