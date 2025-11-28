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

# FlightLog integration for print redirection
# Port 9020: Log info (structured logging)
# Port 9021: Print statements (stdout redirection)
_flightlog_log_enabled = False
_flightlog_print_enabled = False
_flightlog_log_handler = None
_flightlog_print_handler = None

def _connect_flightlog():
    """Connect to flightlog servers. Called at backtest start time."""
    global _flightlog_log_enabled, _flightlog_print_enabled
    global _flightlog_log_handler, _flightlog_print_handler

    # Check if flightlog script exists
    flightlog_paths = [
        '/app/scripts/flightlog.py',
        'scripts/flightlog.py',
        '../scripts/flightlog.py',
    ]

    flightlog_exists = any(os.path.exists(p) for p in flightlog_paths)

    if not flightlog_exists:
        return False

    import logging
    import logging.handlers

    # Hosts to try for port 9020 (log info)
    # localhost first for JupyterLab terminals in same container
    log_hosts = [
        'localhost',              # JupyterLab terminal in same container
        'flightlog',              # docker-compose service name
        'zipline-flightlog',      # container name
        'host.docker.internal',
    ]

    # Hosts to try for port 9021 (print statements)
    # localhost first for JupyterLab terminals in same container
    print_hosts = [
        'localhost',              # JupyterLab terminal in same container
        'flightlog-print',        # docker-compose service name
        'zipline-flightlog-print', # container name
        'host.docker.internal',
    ]

    # Connect to port 9020 for log info
    for host in log_hosts:
        try:
            _flightlog_log_handler = logging.handlers.SocketHandler(host, 9020)
            _flightlog_log_handler.setLevel(logging.INFO)

            # Test connection
            test_record = logging.LogRecord(
                name='backtest.log',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg='FlightLog LOG channel connected',
                args=(),
                exc_info=None
            )
            _flightlog_log_handler.emit(test_record)
            _flightlog_log_enabled = True
            print(f"FlightLog LOG connected to {host}:9020")
            break
        except:
            if _flightlog_log_handler:
                _flightlog_log_handler.close()
            _flightlog_log_handler = None
            continue

    # Connect to port 9021 for print statements
    for host in print_hosts:
        try:
            _flightlog_print_handler = logging.handlers.SocketHandler(host, 9021)
            _flightlog_print_handler.setLevel(logging.INFO)

            # Test connection
            test_record = logging.LogRecord(
                name='backtest.print',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg='FlightLog PRINT channel connected',
                args=(),
                exc_info=None
            )
            _flightlog_print_handler.emit(test_record)
            _flightlog_print_enabled = True
            print(f"FlightLog PRINT connected to {host}:9021")
            break
        except:
            if _flightlog_print_handler:
                _flightlog_print_handler.close()
            _flightlog_print_handler = None
            continue

    if not _flightlog_log_enabled and not _flightlog_print_enabled:
        print("FlightLog connection failed: Could not connect to any host")
        return False

    return _flightlog_log_enabled or _flightlog_print_enabled

def _disconnect_flightlog():
    """Disconnect from flightlog servers."""
    global _flightlog_log_enabled, _flightlog_print_enabled
    global _flightlog_log_handler, _flightlog_print_handler

    if _flightlog_log_handler is not None:
        try:
            _flightlog_log_handler.close()
        except:
            pass
        _flightlog_log_handler = None
    _flightlog_log_enabled = False

    if _flightlog_print_handler is not None:
        try:
            _flightlog_print_handler.close()
        except:
            pass
        _flightlog_print_handler = None
    _flightlog_print_enabled = False

def log_to_flightlog(message, level='INFO'):
    """Send structured log message to port 9020."""
    global _flightlog_log_enabled, _flightlog_log_handler

    if not _flightlog_log_enabled or _flightlog_log_handler is None:
        return

    import logging
    try:
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level = level_map.get(level.upper(), logging.INFO)

        record = logging.LogRecord(
            name='backtest.log',
            level=log_level,
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        _flightlog_log_handler.emit(record)
    except:
        pass

def print_to_flightlog(message):
    """Send print statement to port 9021."""
    global _flightlog_print_enabled, _flightlog_print_handler

    if not _flightlog_print_enabled or _flightlog_print_handler is None:
        return

    import logging
    try:
        record = logging.LogRecord(
            name='backtest.print',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        _flightlog_print_handler.emit(record)
    except:
        pass

class FlightLogPrintRedirector:
    """Redirects print statements to flightlog port 9021 only (suppresses notebook output)."""

    def __init__(self, original_stdout, suppress_stdout=True):
        self.original_stdout = original_stdout
        self.suppress_stdout = suppress_stdout

    def write(self, text):
        # Send to flightlog print channel (port 9021) if enabled
        if _flightlog_print_enabled and text.strip():
            print_to_flightlog(text.strip())
            # Only write to notebook if we're NOT suppressing stdout
            if not self.suppress_stdout:
                self.original_stdout.write(text)
        else:
            # FlightLog not connected - always write to stdout
            self.original_stdout.write(text)

    def flush(self):
        self.original_stdout.flush()

# Bundle registration (must be before other zipline imports)
# Handle import from different locations
try:
    from register_bundles import ensure_bundles_registered
except ImportError:
    try:
        from utils.register_bundles import ensure_bundles_registered
    except ImportError:
        # If register_bundles not available, define a no-op
        def ensure_bundles_registered():
            pass

ensure_bundles_registered()

from zipline import run_algorithm
from zipline.data.bundles import load as load_bundle
from zipline.utils.calendar_utils import get_calendar
from zipline.data.custom import CustomSQLiteLoader
from zipline.pipeline.loaders.auto_loader import setup_auto_loader
from zipline.utils.progress import enable_progress_logging


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
    algo_name=None,
    output_file=None,
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
    algo_name : str, optional
        Algorithm name for logging/progress display. If None, uses algo filename stem
    output_file : str, optional
        Custom output filename (overrides filepath_or_buffer). If None, uses filepath_or_buffer
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
    ...     algo_name="Momentum-Strategy",
    ...     output_file="momentum_2021_2023.pkl"
    ... )
    """
    # Connect to FlightLog servers FIRST (before any prints)
    _connect_flightlog()

    # Enable FlightLog print redirection if print channel connected
    # ALL prints go to port 9021, none to notebook
    original_stdout = None
    if _flightlog_print_enabled:
        original_stdout = sys.stdout
        sys.stdout = FlightLogPrintRedirector(original_stdout)

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

    # Determine output filename (output_file overrides filepath_or_buffer)
    if output_file is not None:
        filepath_or_buffer = output_file
    elif filepath_or_buffer is None:
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

    # Set up auto loader for both custom databases and Sharadar fundamentals
    print("Setting up auto loader...")
    if 'custom_loader' not in kwargs:
        import os
        custom_db_dir = os.environ.get('ZIPLINE_CUSTOM_DATA_DIR', None)
        kwargs['custom_loader'] = setup_auto_loader(
            bundle_name=bundle,
            custom_db_dir=custom_db_dir,
            enable_sid_translation=True,
        )
        print("✓ Auto loader configured (Sharadar + custom databases)")
    else:
        print("  Using provided custom_loader")
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

    # Log backtest start to log channel (port 9020)
    log_to_flightlog("Backtest started")

    # Enable progress logging (update daily)
    # Use algo_name if provided, otherwise extract strategy name from algo_filename
    if algo_name is None:
        algo_display_name = Path(algo_filename).stem  # e.g., "LS-ZR-ported" from "/app/examples/strategies/LS-ZR-ported.py"
    else:
        algo_display_name = algo_name

    # Configure progress logger to use LOG channel (port 9020) instead of stdout
    # This keeps progress separate from print clutter on port 9021
    import logging
    progress_logger = logging.getLogger('zipline.progress')
    progress_logger.handlers.clear()
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False

    # If LOG channel is connected, send progress there
    if _flightlog_log_enabled and _flightlog_log_handler is not None:
        progress_logger.addHandler(_flightlog_log_handler)
    else:
        # Fallback to original stdout (before redirection) if no LOG channel
        handler = logging.StreamHandler(original_stdout if original_stdout else sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        progress_logger.addHandler(handler)

    # Import and call enable_progress_logging - it will use the logger we just configured
    from zipline.utils.progress import BacktestProgressLogger, _global_progress_logger
    import zipline.utils.progress as progress_module

    # Create progress logger instance that uses our pre-configured logger
    progress_module._global_progress_logger = BacktestProgressLogger(
        algo_name=algo_display_name,
        update_interval=1,  # Show progress every trading day
        show_metrics=True,
        logger=progress_logger  # Use our configured logger
    )

    print(f"Progress logging enabled for '{algo_display_name}'")

    # Filter out our custom parameters from kwargs before passing to run_algorithm
    # run_algorithm doesn't know about algo_name or output_file
    run_algo_kwargs = {k: v for k, v in kwargs.items() if k not in ['algo_name', 'output_file']}

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
            **run_algo_kwargs
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
    finally:
        # Restore original stdout and disconnect flightlog
        if original_stdout is not None:
            sys.stdout = original_stdout
        log_to_flightlog("Backtest completed")
        _disconnect_flightlog()

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
        sharpe_value = perf['sharpe'].iloc[-1]
        if sharpe_value is not None and not np.isnan(sharpe_value):
            print(f"Sharpe ratio: {sharpe_value:.3f}")
        else:
            print(f"Sharpe ratio: N/A (insufficient data)")
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
    """Load algorithm module from Python file.

    Always loads a fresh copy of the module, clearing any cached version.
    This ensures code changes are picked up without requiring kernel restart.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Algorithm file not found: {filepath}")

    # Clear any cached version of the module to ensure fresh load
    # This is important when the file has been modified
    module_name = f"algo_module_{filepath.stem}"
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Load module from file
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)

    # Add to sys.modules before exec to handle circular imports
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Clean up on failure
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise

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
