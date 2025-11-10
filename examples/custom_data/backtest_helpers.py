"""
Backtest Helper Functions for Zipline

Provides easy-to-use functions for running backtests and analyzing results with pyfolio.

Usage:
    from backtest_helpers import backtest, analyze_results

    # Run backtest
    backtest(
        "my_strategy.py",
        "my-strategy-v1",
        bundle="sharadar",
        start_date="2021-01-01",
        end_date="2025-01-01",
        capital_base=1000000,
        filepath_or_buffer="results.csv"
    )

    # Analyze results
    analyze_results("results.csv", benchmark_symbol="SPY")
"""

import os
import sys
import importlib.util
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from zipline import run_algorithm
from zipline.data.bundles import load as load_bundle
from zipline.utils.calendar_utils import get_calendar
from zipline.data.custom import CustomSQLiteLoader


def _setup_custom_loaders(algo_module):
    """
    Detect Database classes in the algorithm module and create custom loaders.

    Parameters
    ----------
    algo_module : module
        Loaded algorithm module

    Returns
    -------
    dict or None
        Dictionary mapping BoundColumns to CustomSQLiteLoader instances,
        or None if no Database classes found
    """
    try:
        from zipline.pipeline.data.db import Database
        from zipline.pipeline.data.dataset import BoundColumn
    except ImportError:
        # Database class not available, no custom loaders needed
        return None

    # Use a custom dict that handles .get() properly
    class LoaderDict(dict):
        """Dict that raises KeyError on .get() for missing keys"""
        def get(self, key, default=None):
            # Check if key exists, if not raise KeyError
            if key not in self:
                raise KeyError(key)
            return self[key]

    custom_loader_dict = LoaderDict()

    # Scan module for Database subclasses
    for attr_name in dir(algo_module):
        attr = getattr(algo_module, attr_name)

        # Check if it's a Database subclass (but not Database itself)
        if (isinstance(attr, type) and
            issubclass(attr, Database) and
            attr is not Database):

            # Get the database CODE
            code = getattr(attr, 'CODE', None)
            if not code:
                continue

            print(f"  Found custom database: {attr_name} (code: {code})")

            # Create a CustomSQLiteLoader for this database
            loader = CustomSQLiteLoader(code)

            # Get the underlying dataset class that was generated
            dataset_class = getattr(attr, '_dataset_class', None)
            if dataset_class:
                # Use columns from the dataset class (these are the actual BoundColumns used in pipeline)
                column_count = 0
                for col_name in dir(dataset_class):
                    # Skip private and special attributes
                    if col_name.startswith('_'):
                        continue
                    try:
                        col = getattr(dataset_class, col_name)
                        # Check if it's a BoundColumn instance
                        if isinstance(col, BoundColumn):
                            custom_loader_dict[col] = loader
                            column_count += 1
                            print(f"    - Registered column: {col_name} -> {col}")
                    except AttributeError:
                        continue
                print(f"    Total: {column_count} columns registered")
            else:
                # Fallback to using Database class attributes
                column_count = 0
                for col_name in dir(attr):
                    if col_name.startswith('_'):
                        continue
                    try:
                        col = getattr(attr, col_name)
                        if isinstance(col, BoundColumn):
                            custom_loader_dict[col] = loader
                            column_count += 1
                            print(f"    - Registered column: {col_name} -> {col}")
                    except AttributeError:
                        continue
                print(f"    Total: {column_count} columns registered")

    if custom_loader_dict:
        print(f"  Total custom columns registered: {len(custom_loader_dict)}")
        return custom_loader_dict
    else:
        return None


def backtest(
    algo_filename,
    name,
    bundle="sharadar",
    data_frequency='daily',
    segment=None,
    progress='D',
    start_date="2020-01-01",
    end_date="2023-12-31",
    capital_base=100000,
    filepath_or_buffer=None,
    output_dir="./backtest_results",
    save_pickle=True,
    **kwargs
):
    """
    Run a Zipline backtest and save results to disk.

    Parameters
    ----------
    algo_filename : str
        Path to Python file containing algorithm (must have initialize, handle_data,
        or before_trading_start functions)
    name : str
        Name/identifier for this backtest run
    bundle : str, default 'sharadar'
        Name of data bundle to use
    data_frequency : str, default 'daily'
        Data frequency: 'daily' or 'minute'
    segment : str, optional
        Segment identifier (for tracking)
    progress : str, default 'D'
        Progress bar frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), or None
    start_date : str or pd.Timestamp
        Backtest start date (YYYY-MM-DD)
    end_date : str or pd.Timestamp
        Backtest end date (YYYY-MM-DD)
    capital_base : float, default 100000
        Starting capital in dollars
    filepath_or_buffer : str, optional
        Output CSV filename. If None, uses "{name}.csv"
    output_dir : str, default './backtest_results'
        Directory to save results
    save_pickle : bool, default True
        Also save results as pickle for full state preservation
    **kwargs : dict
        Additional arguments passed to run_algorithm()

    Returns
    -------
    pd.DataFrame
        Backtest performance results

    Examples
    --------
    >>> backtest(
    ...     "my_strategy.py",
    ...     "momentum-v1",
    ...     bundle="sharadar",
    ...     start_date="2021-01-01",
    ...     end_date="2023-12-31",
    ...     capital_base=1000000,
    ...     filepath_or_buffer="momentum_results.csv"
    ... )
    """
    print("=" * 80)
    print("BACKTEST CONFIGURATION")
    print("=" * 80)
    print(f"Algorithm: {algo_filename}")
    print(f"Name: {name}")
    print(f"Bundle: {bundle}")
    print(f"Data frequency: {data_frequency}")
    if segment:
        print(f"Segment: {segment}")
    print(f"Progress: {progress}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Capital base: ${capital_base:,.2f}")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if filepath_or_buffer is None:
        filepath_or_buffer = f"{name}.csv"

    # Make sure it's in the output directory
    if not Path(filepath_or_buffer).is_absolute():
        csv_path = output_dir / filepath_or_buffer
    else:
        csv_path = Path(filepath_or_buffer)

    # Load algorithm from file
    print(f"Loading algorithm from: {algo_filename}")
    algo_module = _load_algorithm_from_file(algo_filename)

    # Extract required functions
    initialize_func = getattr(algo_module, 'initialize', None)
    handle_data_func = getattr(algo_module, 'handle_data', None)
    before_trading_start_func = getattr(algo_module, 'before_trading_start', None)
    analyze_func = getattr(algo_module, 'analyze', None)

    if initialize_func is None:
        raise ValueError(f"Algorithm file {algo_filename} must define an 'initialize' function")

    print(f"✓ Algorithm loaded successfully")
    print(f"  Functions found: initialize", end="")
    if handle_data_func:
        print(", handle_data", end="")
    if before_trading_start_func:
        print(", before_trading_start", end="")
    if analyze_func:
        print(", analyze", end="")
    print()
    print()

    # Set up custom loaders for Database classes
    print("Detecting custom databases...")
    custom_loader = _setup_custom_loaders(algo_module)
    if custom_loader:
        print("✓ Custom loaders configured")
        # Add to kwargs if not already specified
        if 'custom_loader' not in kwargs:
            kwargs['custom_loader'] = custom_loader
    else:
        print("  No custom databases found")
    print()

    # Load bundle
    print(f"Loading bundle: {bundle}")
    try:
        bundle_data = load_bundle(bundle)
        print(f"✓ Bundle loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Error loading bundle: {e}")
        raise

    # Parse dates (must be timezone-naive for exchange_calendars)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Run backtest
    print("=" * 80)
    print("RUNNING BACKTEST")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        perf = run_algorithm(
            start=start_date,
            end=end_date,
            initialize=initialize_func,
            handle_data=handle_data_func,
            before_trading_start=before_trading_start_func,
            analyze=analyze_func,
            capital_base=capital_base,
            data_frequency=data_frequency,
            bundle=bundle,
            **kwargs
        )

        print()
        print("=" * 80)
        print("BACKTEST COMPLETE")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("BACKTEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        raise

    # Add metadata columns
    perf['backtest_name'] = name
    perf['algo_file'] = algo_filename
    if segment:
        perf['segment'] = segment

    # Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save CSV
    perf.to_csv(csv_path)
    print(f"✓ Saved CSV: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024:.2f} KB")
    print(f"  Rows: {len(perf):,}")

    # Save pickle for full state preservation
    if save_pickle:
        pickle_path = csv_path.with_suffix('.pkl')
        perf.to_pickle(pickle_path)
        print(f"✓ Saved pickle: {pickle_path}")
        print(f"  Size: {pickle_path.stat().st_size / 1024:.2f} KB")

    # Save metadata
    metadata = {
        'backtest_name': name,
        'algo_file': algo_filename,
        'bundle': bundle,
        'data_frequency': data_frequency,
        'segment': segment,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'capital_base': capital_base,
        'run_timestamp': datetime.now().isoformat(),
        'num_rows': len(perf),
        'csv_path': str(csv_path),
        'pickle_path': str(pickle_path) if save_pickle else None,
    }

    metadata_path = csv_path.with_suffix('.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    print()

    # Print summary statistics
    print("=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    total_return = (perf['portfolio_value'].iloc[-1] / capital_base - 1) * 100
    max_drawdown = ((perf['portfolio_value'] / perf['portfolio_value'].cummax()) - 1).min() * 100

    print(f"Initial capital: ${capital_base:,.2f}")
    print(f"Final value: ${perf['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Max drawdown: {max_drawdown:.2f}%")
    print(f"Total trades: {perf['orders'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()}")

    # Check for sharpe ratio
    if 'sharpe' in perf.columns:
        print(f"Sharpe ratio: {perf['sharpe'].iloc[-1]:.3f}")
    elif 'algorithm_period_return' in perf.columns:
        returns = perf['algorithm_period_return'].pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            print(f"Sharpe ratio (approx): {sharpe:.3f}")

    print("=" * 80)
    print()

    return perf


def analyze_results(
    filepath_or_buffer,
    benchmark_symbol='SPY',
    output_file=None,
    live_start_date=None,
    show_plots=True,
    save_plots=True,
    figsize=(14, 10),
    return_dict=False,
):
    """
    Load backtest results and generate pyfolio analysis.

    Parameters
    ----------
    filepath_or_buffer : str
        Path to CSV or pickle file containing backtest results
    benchmark_symbol : str, default 'SPY'
        Symbol to use as benchmark for comparison
    output_file : str, optional
        File path to save tearsheet plots (e.g., 'tearsheet.html' or 'tearsheet.png')
    live_start_date : str or pd.Timestamp, optional
        Date when live trading began (separates in-sample vs out-of-sample)
    show_plots : bool, default True
        Display plots in notebook/interactive environment
    save_plots : bool, default True
        Save plots to file
    figsize : tuple, default (14, 10)
        Figure size for plots
    return_dict : bool, default False
        Return dictionary with detailed metrics

    Returns
    -------
    results : dict (if return_dict=True)
        Dictionary containing:
        - 'returns': Daily returns series
        - 'positions': Positions DataFrame
        - 'transactions': Transactions DataFrame
        - 'metrics': Performance metrics
        - 'benchmark_returns': Benchmark returns (if available)

    Examples
    --------
    >>> # Basic usage
    >>> analyze_results("momentum_results.csv", benchmark_symbol="SPY")

    >>> # Get detailed metrics
    >>> results = analyze_results(
    ...     "momentum_results.csv",
    ...     benchmark_symbol="SPY",
    ...     live_start_date="2024-01-01",
    ...     return_dict=True
    ... )
    >>> print(results['metrics'])
    """
    print("=" * 80)
    print("LOADING BACKTEST RESULTS")
    print("=" * 80)

    # Load results
    filepath = Path(filepath_or_buffer)
    print(f"Loading from: {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Try pickle first (preserves all data types), then CSV
    if filepath.suffix == '.pkl':
        perf = pd.read_pickle(filepath)
        print(f"✓ Loaded pickle file")
    elif filepath.suffix == '.csv':
        perf = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✓ Loaded CSV file")
    else:
        # Try both
        try:
            perf = pd.read_pickle(filepath)
            print(f"✓ Loaded as pickle")
        except:
            perf = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"✓ Loaded as CSV")

    print(f"  Rows: {len(perf):,}")
    print(f"  Date range: {perf.index[0]} to {perf.index[-1]}")
    print(f"  Columns: {len(perf.columns)}")
    print()

    # Extract returns
    print("=" * 80)
    print("EXTRACTING RETURNS")
    print("=" * 80)

    if 'returns' in perf.columns:
        returns = perf['returns']
        print("✓ Using 'returns' column")
    elif 'algorithm_period_return' in perf.columns:
        returns = perf['algorithm_period_return'].pct_change().fillna(0)
        print("✓ Calculated from 'algorithm_period_return'")
    else:
        # Calculate from portfolio value
        returns = perf['portfolio_value'].pct_change().fillna(0)
        print("✓ Calculated from 'portfolio_value'")

    print(f"  Total return: {(returns + 1).cumprod()[-1] - 1:.2%}")
    print(f"  Average daily return: {returns.mean():.4%}")
    print(f"  Daily volatility: {returns.std():.4%}")
    print()

    # Extract positions if available
    positions = None
    transactions = None

    if 'positions' in perf.columns:
        print("✓ Positions data available")
        # positions = perf['positions']  # This is complex to extract

    if 'transactions' in perf.columns:
        print("✓ Transactions data available")
        # transactions = perf['transactions']

    # Try to import pyfolio
    try:
        import pyfolio as pf
        print()
        print("=" * 80)
        print("GENERATING PYFOLIO TEARSHEET")
        print("=" * 80)

        # Parse live start date if provided
        if live_start_date is not None:
            live_start_date = pd.Timestamp(live_start_date, tz='UTC')
            print(f"Live start date: {live_start_date}")

        # Create tearsheet
        if show_plots:
            try:
                pf.create_simple_tear_sheet(
                    returns,
                    benchmark_rets=None,  # Could load benchmark here
                    live_start_date=live_start_date,
                )
                print("✓ Tearsheet generated")
            except Exception as e:
                print(f"⚠ Warning: Could not create full tearsheet: {e}")
                print("  Generating basic plots instead...")
                _create_basic_plots(perf, returns, figsize)

        # Calculate metrics
        print()
        print("=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)

        metrics = _calculate_metrics(returns, perf)

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        print()

        if return_dict:
            return {
                'returns': returns,
                'positions': positions,
                'transactions': transactions,
                'metrics': metrics,
                'perf': perf,
            }

    except ImportError:
        print()
        print("=" * 80)
        print("PYFOLIO NOT AVAILABLE")
        print("=" * 80)
        print("Install with: pip install pyfolio-reloaded")
        print()
        print("Generating basic analysis instead...")
        print()

        # Create basic plots
        if show_plots:
            _create_basic_plots(perf, returns, figsize)

        # Calculate basic metrics
        metrics = _calculate_metrics(returns, perf)

        print("=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        print()

        if return_dict:
            return {
                'returns': returns,
                'metrics': metrics,
                'perf': perf,
            }


def _load_algorithm_from_file(filepath):
    """Load algorithm module from Python file."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Algorithm file not found: {filepath}")

    # Load module from file
    spec = importlib.util.spec_from_file_location("algo_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _calculate_metrics(returns, perf):
    """Calculate performance metrics."""
    metrics = {}

    # Total return
    total_return = (returns + 1).cumprod()[-1] - 1
    metrics['Total Return'] = f"{total_return:.2%}"

    # Annual return
    years = (perf.index[-1] - perf.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    metrics['Annual Return'] = f"{annual_return:.2%}"

    # Volatility
    annual_vol = returns.std() * np.sqrt(252)
    metrics['Annual Volatility'] = f"{annual_vol:.2%}"

    # Sharpe ratio
    sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0
    metrics['Sharpe Ratio'] = f"{sharpe:.3f}"

    # Max drawdown
    cumulative = (returns + 1).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    metrics['Max Drawdown'] = f"{max_dd:.2%}"

    # Calmar ratio
    calmar = (annual_return / abs(max_dd)) if max_dd != 0 else 0
    metrics['Calmar Ratio'] = f"{calmar:.3f}"

    # Win rate (if transactions available)
    if 'returns' in perf.columns:
        win_rate = (returns > 0).sum() / len(returns)
        metrics['Win Rate'] = f"{win_rate:.2%}"

    # Final portfolio value
    if 'portfolio_value' in perf.columns:
        final_value = perf['portfolio_value'].iloc[-1]
        metrics['Final Portfolio Value'] = f"${final_value:,.2f}"

    return metrics


def _create_basic_plots(perf, returns, figsize=(14, 10)):
    """Create basic performance plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Portfolio value
    axes[0].plot(perf.index, perf['portfolio_value'])
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value ($)')
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Cumulative returns
    cumulative = (returns + 1).cumprod() - 1
    axes[1].plot(perf.index, cumulative * 100)
    axes[1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Return (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Drawdown
    cumulative_val = (returns + 1).cumprod()
    running_max = cumulative_val.cummax()
    drawdown = (cumulative_val - running_max) / running_max * 100
    axes[2].fill_between(perf.index, drawdown, 0, alpha=0.3, color='red')
    axes[2].plot(perf.index, drawdown, color='red')
    axes[2].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Drawdown (%)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Monthly returns heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    monthly_returns = returns.resample('M').apply(lambda x: (x + 1).prod() - 1)
    monthly_returns_pct = monthly_returns * 100

    # Create pivot table for heatmap
    monthly_pivot = pd.DataFrame({
        'Year': monthly_returns_pct.index.year,
        'Month': monthly_returns_pct.index.month,
        'Return': monthly_returns_pct.values
    }).pivot(index='Month', columns='Year', values='Return')

    # Plot heatmap
    import seaborn as sns
    sns.heatmap(
        monthly_pivot,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Return (%)'},
        ax=ax
    )

    ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Month')
    ax.set_xlabel('Year')

    # Set month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticklabels(month_names, rotation=0)

    plt.tight_layout()
    plt.show()


# Convenience function for quick analysis
def quick_backtest(
    algo_filename,
    name=None,
    start_date="2021-01-01",
    end_date="2023-12-31",
    capital_base=100000,
    analyze=True,
    **kwargs
):
    """
    Run backtest and immediately analyze results.

    Parameters
    ----------
    algo_filename : str
        Path to algorithm file
    name : str, optional
        Backtest name (defaults to algo filename without extension)
    start_date : str
        Start date
    end_date : str
        End date
    capital_base : float
        Starting capital
    analyze : bool, default True
        Run analysis after backtest
    **kwargs : dict
        Additional arguments for backtest()

    Returns
    -------
    perf : pd.DataFrame
        Backtest results
    """
    if name is None:
        name = Path(algo_filename).stem

    # Run backtest
    perf = backtest(
        algo_filename=algo_filename,
        name=name,
        start_date=start_date,
        end_date=end_date,
        capital_base=capital_base,
        filepath_or_buffer=f"{name}.csv",
        **kwargs
    )

    # Analyze
    if analyze:
        analyze_results(f"./backtest_results/{name}.csv")

    return perf


if __name__ == "__main__":
    # Example usage
    print(__doc__)
