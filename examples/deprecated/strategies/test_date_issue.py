"""
Test script to debug date issue
"""
import sys
sys.path.insert(0, '/app/examples/utils')

from backtest_helpers import backtest
import traceback

try:
    results = backtest(
        algo_filename='/app/examples/strategies/LS-ZR-ported.py',
        name='date-test',
        start_date='2021-01-01',
        end_date='2025-12-04',
        capital_base=1000000,
        bundle='sharadar',
        algo_name='LS-ZR-ported',
        output_file='date-test.pkl'
    )
except Exception as e:
    print("\n" + "=" * 80)
    print("FULL ERROR TRACEBACK:")
    print("=" * 80)
    traceback.print_exc()
    print("=" * 80)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    raise
