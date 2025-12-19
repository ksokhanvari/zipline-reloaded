"""
Sharadar Fundamentals Query Helper

Provides easy functions to query Sharadar SF1 fundamentals data for specific symbols.

Usage:
    # First, add the utils directory to your path:
    import sys
    sys.path.insert(0, '/app/examples/utils')

    from sharadar_query import query_fundamentals, list_available_columns, help

    # Print help
    help()

    # Get last 10 values of FCF and ROE for AAPL
    df = query_fundamentals('AAPL', ['fcf', 'roe', 'marketcap'], n=10)

    # List all available columns
    cols = list_available_columns()
"""

import pandas as pd


def help():
    """Print usage instructions for the sharadar_query module."""
    help_text = """
================================================================================
SHARADAR FUNDAMENTALS QUERY HELPER
================================================================================

SETUP (run once per notebook/script):
-------------------------------------
    import sys
    sys.path.insert(0, '/app/examples/utils')

BASIC USAGE:
------------
    from sharadar_query import query_fundamentals, list_available_columns, help

    # Print this help
    help()

    # Get last 10 values of FCF, ROE, and PE for AAPL
    df = query_fundamentals('AAPL', ['fcf', 'roe', 'pe'], n=10)
    print(df)

    # List all available columns
    cols = list_available_columns()
    print(cols)

AVAILABLE FUNCTIONS:
--------------------

1. query_fundamentals(symbol, columns, n=10)
   Get the last n values of specified columns for a symbol.

   Example:
       df = query_fundamentals('AAPL', ['fcf', 'revenue', 'netinc'], n=5)

2. list_available_columns()
   List all available Sharadar SF1 columns.

   Example:
       cols = list_available_columns()
       print(cols)

3. query_multiple_symbols(symbols, columns, n=5, latest_only=True)
   Query data for multiple symbols at once.

   Example:
       df = query_multiple_symbols(['AAPL', 'MSFT', 'GOOGL'], ['fcf', 'roe'])

4. compare_symbols(symbols, columns)
   Compare latest values across multiple symbols (pivot table format).

   Example:
       df = compare_symbols(['AAPL', 'MSFT', 'GOOGL'], ['fcf', 'roe', 'pe'])

5. get_symbol_history(symbol, column, start_date=None, end_date=None)
   Get full time series history of a single column.

   Example:
       fcf_series = get_symbol_history('AAPL', 'fcf', start_date='2020-01-01')
       fcf_series.plot()

6. print_columns()
   Print all available column names in a formatted display.

   Example:
       print_columns()

QUICK ACCESS FUNCTIONS:
-----------------------
    from sharadar_query import fcf, roe, pe, marketcap

    df = fcf('AAPL', n=5)        # Free cash flow
    df = roe('MSFT', n=10)       # Return on equity
    df = pe('GOOGL', n=5)        # Price to earnings
    df = marketcap('AMZN', n=5)  # Market capitalization

COMMON COLUMNS:
---------------
Financial Metrics:
    fcf, fcfps       - Free cash flow
    roe, roa, roic   - Return ratios
    revenue          - Revenue
    netinc           - Net income
    ebitda           - EBITDA
    eps              - Earnings per share

Valuation:
    marketcap        - Market capitalization
    pe, pe1          - Price to earnings
    pb               - Price to book
    ps               - Price to sales

Balance Sheet:
    assets           - Total assets
    debt, debtusd    - Total debt
    equity           - Total equity
    cashnequsd       - Cash and equivalents

EXAMPLE SESSION:
----------------
    >>> from sharadar_query import *

    >>> # Check what columns are available
    >>> cols = list_available_columns()
    >>> print(f"Available: {len(cols)} columns")

    >>> # Get AAPL fundamentals
    >>> df = query_fundamentals('AAPL', ['fcf', 'roe', 'pe', 'marketcap'], n=5)
    >>> print(df)

    >>> # Compare tech giants
    >>> comparison = compare_symbols(
    ...     ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    ...     ['fcf', 'roe', 'pe', 'marketcap']
    ... )
    >>> print(comparison)

    >>> # Plot FCF history
    >>> import matplotlib.pyplot as plt
    >>> fcf_history = get_symbol_history('AAPL', 'fcf', start_date='2018-01-01')
    >>> fcf_history.plot(title='AAPL Free Cash Flow')
    >>> plt.show()

================================================================================
"""
    print(help_text)


import numpy as np
from pathlib import Path
import os


def get_sharadar_data_path(bundle_name='sharadar'):
    """
    Get the path to Sharadar SF1 fundamentals data.

    Parameters
    ----------
    bundle_name : str, default 'sharadar'
        Bundle name

    Returns
    -------
    Path
        Path to sf1.h5 file
    """
    zipline_root = Path(os.environ.get('ZIPLINE_ROOT', Path.home() / '.zipline'))
    bundle_dir = zipline_root / 'data' / bundle_name

    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle '{bundle_name}' not found at {bundle_dir}")

    # Find most recent ingestion
    ingestion_dirs = sorted(bundle_dir.glob('*'))
    if not ingestion_dirs:
        raise FileNotFoundError(f"No ingestions found for bundle '{bundle_name}'")

    latest_ingestion = ingestion_dirs[-1]
    fundamentals_path = latest_ingestion / 'fundamentals' / 'sf1.h5'

    if not fundamentals_path.exists():
        raise FileNotFoundError(
            f"Fundamentals data not found at {fundamentals_path}. "
            f"Re-ingest the bundle with fundamentals enabled."
        )

    return fundamentals_path


def load_sharadar_data(bundle_name='sharadar'):
    """
    Load the full Sharadar SF1 fundamentals DataFrame.

    Parameters
    ----------
    bundle_name : str, default 'sharadar'
        Bundle name

    Returns
    -------
    pd.DataFrame
        Full SF1 fundamentals data
    """
    path = get_sharadar_data_path(bundle_name)
    return pd.read_hdf(path, key='sf1')


def list_available_columns(bundle_name='sharadar'):
    """
    List all available columns in Sharadar SF1 data.

    Parameters
    ----------
    bundle_name : str, default 'sharadar'
        Bundle name

    Returns
    -------
    list
        List of column names
    """
    df = load_sharadar_data(bundle_name)
    # Exclude metadata columns
    exclude = ['sid', 'ticker', 'dimension', 'calendardate', 'datekey', 'reportperiod', 'lastupdated']
    cols = [c for c in df.columns if c not in exclude]
    return sorted(cols)


def print_columns(bundle_name='sharadar', columns_per_row=5):
    """
    Print all available columns in a formatted display.

    Parameters
    ----------
    bundle_name : str, default 'sharadar'
        Bundle name
    columns_per_row : int, default 5
        Number of columns to display per row

    Examples
    --------
    >>> print_columns()
    """
    cols = list_available_columns(bundle_name)

    print("=" * 80)
    print(f"SHARADAR SF1 AVAILABLE COLUMNS ({len(cols)} total)")
    print("=" * 80)

    # Print in rows
    for i in range(0, len(cols), columns_per_row):
        row = cols[i:i + columns_per_row]
        print("  " + ", ".join(f"{c:<15}" for c in row))

    print("=" * 80)
    print("\nCommon columns by category:")
    print("-" * 40)
    print("Profitability:  roe, roa, roic, ros, grossmargin")
    print("Cash Flow:      fcf, fcfps, ncfo, capex")
    print("Valuation:      pe, pb, ps, ev, evebitda")
    print("Income:         revenue, netinc, ebitda, eps")
    print("Balance Sheet:  assets, debt, equity, cashnequsd")
    print("Per Share:      bvps, dps, revenueps, ncfops")
    print("-" * 40)


def query_fundamentals(
    symbol,
    columns,
    n=10,
    bundle_name='sharadar',
    include_dates=True,
    sort_descending=True,
):
    """
    Query Sharadar fundamentals for a specific symbol.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    columns : list of str
        List of column names to retrieve (e.g., ['fcf', 'roe', 'marketcap'])
    n : int, default 10
        Number of most recent values to return
    bundle_name : str, default 'sharadar'
        Bundle name
    include_dates : bool, default True
        Include datekey and calendardate columns
    sort_descending : bool, default True
        Sort by date descending (most recent first)

    Returns
    -------
    pd.DataFrame
        DataFrame with requested columns for the symbol

    Examples
    --------
    >>> df = query_fundamentals('AAPL', ['fcf', 'roe', 'pe'], n=5)
    >>> print(df)

    >>> df = query_fundamentals('MSFT', ['revenue', 'netinc', 'eps'], n=20)
    """
    # Load data
    df = load_sharadar_data(bundle_name)

    # Filter by ticker
    symbol = symbol.upper()
    mask = df['ticker'] == symbol

    if not mask.any():
        available_tickers = df['ticker'].unique()[:20]
        raise ValueError(
            f"Symbol '{symbol}' not found in Sharadar data. "
            f"Example available symbols: {list(available_tickers)}"
        )

    symbol_data = df[mask].copy()

    # Validate columns
    available_cols = set(df.columns)
    invalid_cols = [c for c in columns if c not in available_cols]
    if invalid_cols:
        raise ValueError(
            f"Invalid columns: {invalid_cols}. "
            f"Use list_available_columns() to see available columns."
        )

    # Select columns
    select_cols = []
    if include_dates:
        if 'datekey' in df.columns:
            select_cols.append('datekey')
        if 'calendardate' in df.columns:
            select_cols.append('calendardate')

    select_cols.extend(columns)

    result = symbol_data[select_cols].copy()

    # Sort by date
    sort_col = 'datekey' if 'datekey' in result.columns else 'calendardate'
    if sort_col in result.columns:
        result = result.sort_values(sort_col, ascending=not sort_descending)

    # Return last n rows
    return result.head(n).reset_index(drop=True)


def query_multiple_symbols(
    symbols,
    columns,
    n=5,
    bundle_name='sharadar',
    latest_only=True,
):
    """
    Query Sharadar fundamentals for multiple symbols.

    Parameters
    ----------
    symbols : list of str
        List of stock symbols
    columns : list of str
        List of column names to retrieve
    n : int, default 5
        Number of most recent values per symbol (if latest_only=False)
    bundle_name : str, default 'sharadar'
        Bundle name
    latest_only : bool, default True
        Only return the most recent value for each symbol

    Returns
    -------
    pd.DataFrame
        DataFrame with requested data

    Examples
    --------
    >>> df = query_multiple_symbols(['AAPL', 'MSFT', 'GOOGL'], ['fcf', 'roe'])
    """
    # Load data
    df = load_sharadar_data(bundle_name)

    # Normalize symbols
    symbols = [s.upper() for s in symbols]

    # Filter by tickers
    mask = df['ticker'].isin(symbols)
    filtered = df[mask].copy()

    if filtered.empty:
        raise ValueError(f"No data found for symbols: {symbols}")

    # Validate columns
    available_cols = set(df.columns)
    invalid_cols = [c for c in columns if c not in available_cols]
    if invalid_cols:
        raise ValueError(f"Invalid columns: {invalid_cols}")

    # Select columns
    select_cols = ['ticker', 'datekey'] + columns
    select_cols = [c for c in select_cols if c in filtered.columns]

    result = filtered[select_cols].copy()

    # Sort by ticker and date
    result = result.sort_values(['ticker', 'datekey'], ascending=[True, False])

    if latest_only:
        # Get most recent for each symbol
        result = result.groupby('ticker').first().reset_index()
    else:
        # Get n most recent for each symbol
        result = result.groupby('ticker').head(n).reset_index(drop=True)

    return result


def compare_symbols(
    symbols,
    columns,
    bundle_name='sharadar',
):
    """
    Compare latest fundamentals across multiple symbols.

    Parameters
    ----------
    symbols : list of str
        List of stock symbols to compare
    columns : list of str
        List of column names to compare
    bundle_name : str, default 'sharadar'
        Bundle name

    Returns
    -------
    pd.DataFrame
        Pivot table with symbols as rows and columns as values

    Examples
    --------
    >>> df = compare_symbols(['AAPL', 'MSFT', 'GOOGL'], ['fcf', 'roe', 'pe'])
    """
    df = query_multiple_symbols(symbols, columns, bundle_name=bundle_name, latest_only=True)

    # Pivot to get symbols as rows
    result = df.set_index('ticker')[columns]

    return result


def get_symbol_history(
    symbol,
    column,
    start_date=None,
    end_date=None,
    bundle_name='sharadar',
):
    """
    Get full history of a single column for a symbol.

    Parameters
    ----------
    symbol : str
        Stock symbol
    column : str
        Column name to retrieve
    start_date : str or pd.Timestamp, optional
        Start date filter
    end_date : str or pd.Timestamp, optional
        End date filter
    bundle_name : str, default 'sharadar'
        Bundle name

    Returns
    -------
    pd.Series
        Time series of the column values, indexed by datekey

    Examples
    --------
    >>> fcf = get_symbol_history('AAPL', 'fcf', start_date='2020-01-01')
    >>> fcf.plot()
    """
    # Load data
    df = load_sharadar_data(bundle_name)

    # Filter by ticker
    symbol = symbol.upper()
    mask = df['ticker'] == symbol
    symbol_data = df[mask].copy()

    if symbol_data.empty:
        raise ValueError(f"Symbol '{symbol}' not found")

    # Filter by date
    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        symbol_data = symbol_data[symbol_data['datekey'] >= start_date]

    if end_date is not None:
        end_date = pd.Timestamp(end_date)
        symbol_data = symbol_data[symbol_data['datekey'] <= end_date]

    # Create series
    result = symbol_data.set_index('datekey')[column].sort_index()
    result.name = f"{symbol}_{column}"

    return result


# Quick access functions
def fcf(symbol, n=10, bundle_name='sharadar'):
    """Get free cash flow history for a symbol."""
    return query_fundamentals(symbol, ['fcf'], n=n, bundle_name=bundle_name)


def roe(symbol, n=10, bundle_name='sharadar'):
    """Get return on equity history for a symbol."""
    return query_fundamentals(symbol, ['roe'], n=n, bundle_name=bundle_name)


def pe(symbol, n=10, bundle_name='sharadar'):
    """Get price to earnings history for a symbol."""
    return query_fundamentals(symbol, ['pe'], n=n, bundle_name=bundle_name)


def marketcap(symbol, n=10, bundle_name='sharadar'):
    """Get market cap history for a symbol."""
    return query_fundamentals(symbol, ['marketcap'], n=n, bundle_name=bundle_name)


if __name__ == '__main__':
    # Example usage
    print("Sharadar Fundamentals Query Helper")
    print("=" * 50)

    # List available columns
    try:
        cols = list_available_columns()
        print(f"\nAvailable columns ({len(cols)} total):")
        print(cols[:20])  # Show first 20
        print("...")

        # Example query
        print("\nExample: AAPL last 5 FCF values")
        df = query_fundamentals('AAPL', ['fcf', 'revenue', 'netinc'], n=5)
        print(df)

        # Compare symbols
        print("\nExample: Compare AAPL, MSFT, GOOGL")
        comparison = compare_symbols(['AAPL', 'MSFT', 'GOOGL'], ['fcf', 'roe', 'pe', 'marketcap'])
        print(comparison)

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have ingested the sharadar bundle:")
        print("  zipline ingest -b sharadar")
