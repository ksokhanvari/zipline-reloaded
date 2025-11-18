#!/usr/bin/env python3
"""
Helper to run strategy files from notebooks or Python console

Usage:
    from run_strategy import run_strategy

    results = run_strategy(
        'strategy_top5_roe_simple.py',
        start='2018-01-01',
        end='2023-12-31',
        capital_base=100000
    )
"""

import importlib.util
import sys
from pathlib import Path
import pandas as pd
from zipline import run_algorithm


def load_module_from_file(filepath):
    """Load a Python module from a file path."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Strategy file not found: {filepath}")

    # Create module spec
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {filepath}")

    # Load the module
    module = importlib.util.module_from_spec(spec)
    sys.modules[filepath.stem] = module
    spec.loader.exec_module(module)

    return module


def run_strategy(
    strategy_file,
    start='2020-01-01',
    end='2023-12-31',
    capital_base=100000,
    data_frequency='daily',
    bundle='sharadar',
    **kwargs
):
    """
    Run a strategy from a Python file.

    Parameters
    ----------
    strategy_file : str
        Path to strategy Python file (e.g., 'strategy_top5_roe_simple.py')
    start : str or pd.Timestamp
        Backtest start date
    end : str or pd.Timestamp
        Backtest end date
    capital_base : float
        Starting capital
    data_frequency : str
        'daily' or 'minute'
    bundle : str
        Data bundle name
    **kwargs
        Additional arguments passed to run_algorithm()

    Returns
    -------
    pd.DataFrame
        Backtest results

    Examples
    --------
    >>> # Basic usage
    >>> results = run_strategy('strategy_top5_roe_simple.py')

    >>> # Custom dates and capital
    >>> results = run_strategy(
    ...     'strategy_top5_roe_simple.py',
    ...     start='2018-01-01',
    ...     end='2023-12-31',
    ...     capital_base=500000
    ... )

    >>> # From a notebook
    >>> from run_strategy import run_strategy
    >>> results = run_strategy(
    ...     'strategy_top5_roe_simple.py',
    ...     start='2020-01-01',
    ...     end='2021-12-31'
    ... )
    """
    print(f"Loading strategy from: {strategy_file}")

    # Load the strategy module
    module = load_module_from_file(strategy_file)

    # Extract required functions
    initialize = getattr(module, 'initialize', None)
    handle_data = getattr(module, 'handle_data', None)
    before_trading_start = getattr(module, 'before_trading_start', None)
    analyze = getattr(module, 'analyze', None)

    if initialize is None:
        raise ValueError(f"Strategy file must define 'initialize' function")

    print(f"✓ Strategy loaded successfully")
    if handle_data:
        print(f"  - handle_data: found")
    if before_trading_start:
        print(f"  - before_trading_start: found")
    if analyze:
        print(f"  - analyze: found")

    # Get custom loader if available
    custom_loader = None
    if hasattr(module, 'build_pipeline_loaders'):
        print(f"\nBuilding custom loaders...")
        custom_loader = module.build_pipeline_loaders()
        print(f"✓ Custom loaders built")
        if 'custom_loader' not in kwargs:
            kwargs['custom_loader'] = custom_loader

    # Convert dates to Timestamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    print(f"\nRunning backtest:")
    print(f"  Start: {start.date()}")
    print(f"  End: {end.date()}")
    print(f"  Capital: ${capital_base:,.2f}")
    print(f"  Bundle: {bundle}")
    print()

    # Run the backtest
    results = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        handle_data=handle_data,
        before_trading_start=before_trading_start,
        analyze=analyze,
        capital_base=capital_base,
        data_frequency=data_frequency,
        bundle=bundle,
        **kwargs
    )

    return results


if __name__ == '__main__':
    # Example usage
    print("Example usage:")
    print()
    print("from run_strategy import run_strategy")
    print()
    print("results = run_strategy(")
    print("    'strategy_top5_roe_simple.py',")
    print("    start='2020-01-01',")
    print("    end='2023-12-31',")
    print("    capital_base=100000")
    print(")")
