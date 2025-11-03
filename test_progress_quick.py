#!/usr/bin/env python
"""
Quick test to verify progress logging is working.
"""

import logging
from zipline.utils.progress import BacktestProgressLogger
import pandas as pd
import time

print("="*60)
print("Testing Progress Logging...")
print("="*60)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create progress logger
logger = BacktestProgressLogger(
    algo_name='Quick-Test',
    update_interval=1
)

# Initialize with date range
start = pd.Timestamp('2020-01-01')
end = pd.Timestamp('2020-01-10')
logger.initialize(start, end, calendar=None)

print("\nSimulating backtest progress...\n")

# Simulate backtest with realistic portfolio data
test_data = [
    {'portfolio_value': 100000, 'returns': 0.000},
    {'portfolio_value': 100500, 'returns': 0.005},
    {'portfolio_value': 102000, 'returns': 0.015},
    {'portfolio_value': 99500, 'returns': -0.025},
    {'portfolio_value': 103500, 'returns': 0.040},
    {'portfolio_value': 104800, 'returns': 0.013},
    {'portfolio_value': 105200, 'returns': 0.004},
    {'portfolio_value': 107000, 'returns': 0.017},
    {'portfolio_value': 108500, 'returns': 0.014},
    {'portfolio_value': 110000, 'returns': 0.014},
]

for i, data in enumerate(test_data):
    date = start + pd.Timedelta(days=i)
    logger.update(date, data)
    time.sleep(0.3)

# Create a simple performance dataframe for finalize
perf_data = {
    'algorithm_period_return': [0.10],
    'sharpe': [1.45],
    'max_drawdown': [-0.05],
    'portfolio_value': [110000]
}
perf = pd.DataFrame(perf_data)

print("\n")
logger.finalize(perf)

print("\n" + "="*60)
print("✓ Progress logging test complete!")
print("="*60)
print("\nExpected output above:")
print("1. Header row with column names")
print("2. Progress bars (█) increasing from 10% to 100%")
print("3. Cumulative Returns showing percentages")
print("4. Sharpe Ratio with decimal values")
print("5. Max Drawdown as percentage")
print("6. Cumulative PNL in dollars")
print("\n✓ If you see aligned columns, everything is working!")
print("="*60)
