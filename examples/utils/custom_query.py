"""
Custom Fundamentals Query Helper

Provides easy functions to query custom fundamentals data from SQLite databases.

Usage:
    # First, add the utils directory to your path:
    import sys
    sys.path.insert(0, '/app/examples/utils')

    from custom_query import query_fundamentals, list_available_columns, help

    # Print help
    help()

    # Get last 10 values of columns for AAPL
    df = query_fundamentals('AAPL', ['pred', 'bc1', 'ROE'], n=10)

    # List all available columns
    cols = list_available_columns()
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sqlite3


def help():
    """Print usage instructions for the custom_query module."""
    help_text = """
================================================================================
CUSTOM FUNDAMENTALS QUERY HELPER
================================================================================

SETUP (run once per notebook/script):
-------------------------------------
    import sys
    sys.path.insert(0, '/app/examples/utils')

BASIC USAGE:
------------
    from custom_query import query_fundamentals, list_available_columns, help

    # Print this help
    help()

    # Get last 10 values of columns for AAPL
    df = query_fundamentals('AAPL', ['pred', 'bc1', 'ROE'], n=10)
    print(df)

    # List all available columns
    cols = list_available_columns()
    print(cols)

AVAILABLE FUNCTIONS:
--------------------

1. query_fundamentals(symbol, columns, n=10)
   Get the last n values of specified columns for a symbol.

   Example:
       df = query_fundamentals('AAPL', ['pred', 'bc1', 'ROE'], n=5)

2. list_available_columns()
   List all available columns in the custom database.

   Example:
       cols = list_available_columns()
       print(cols)

3. query_multiple_symbols(symbols, columns, n=5, latest_only=True)
   Query data for multiple symbols at once.

   Example:
       df = query_multiple_symbols(['AAPL', 'MSFT', 'GOOGL'], ['pred', 'ROE'])

4. compare_symbols(symbols, columns)
   Compare latest values across multiple symbols (pivot table format).

   Example:
       df = compare_symbols(['AAPL', 'MSFT', 'GOOGL'], ['pred', 'ROE'])

5. get_symbol_history(symbol, column, start_date=None, end_date=None)
   Get full time series history of a single column.

   Example:
       pred_series = get_symbol_history('AAPL', 'pred', start_date='2020-01-01')
       pred_series.plot()

6. print_columns()
   Print all available column names in a formatted display.

   Example:
       print_columns()

7. list_symbols()
   List all available symbols in the database.

   Example:
       symbols = list_symbols()
       print(symbols)

EXAMPLE SESSION:
----------------
    >>> from custom_query import *

    >>> # Check what columns are available
    >>> print_columns()

    >>> # Get AAPL fundamentals
    >>> df = query_fundamentals('AAPL', ['pred', 'bc1', 'ROE'], n=5)
    >>> print(df)

    >>> # Compare multiple symbols
    >>> comparison = compare_symbols(
    ...     ['AAPL', 'MSFT', 'GOOGL'],
    ...     ['pred', 'ROE', 'CompanyMarketCap']
    ... )
    >>> print(comparison)

================================================================================
"""
    print(help_text)


def get_custom_db_path(db_name='fundamentals'):
    """
    Get the path to custom fundamentals SQLite database.

    Parameters
    ----------
    db_name : str, default 'fundamentals'
        Database name (without .sqlite extension)

    Returns
    -------
    Path
        Path to the SQLite database file
    """
    # Check ZIPLINE_CUSTOM_DATA_DIR first (from docker-compose)
    custom_dir = os.environ.get('ZIPLINE_CUSTOM_DATA_DIR')
    if custom_dir:
        db_path = Path(custom_dir) / f'{db_name}.sqlite'
        if db_path.exists():
            return db_path

    # Fall back to default location
    zipline_root = Path(os.environ.get('ZIPLINE_ROOT', Path.home() / '.zipline'))
    db_path = zipline_root / 'data' / 'custom' / f'{db_name}.sqlite'

    if not db_path.exists():
        raise FileNotFoundError(
            f"Custom database not found at {db_path}. "
            f"Check ZIPLINE_CUSTOM_DATA_DIR environment variable."
        )

    return db_path


def get_table_name(db_path):
    """
    Get the main table name from the database.

    Parameters
    ----------
    db_path : Path
        Path to SQLite database

    Returns
    -------
    str
        Table name
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Prefer known table names in order
        preferred_tables = ['fundamentals', 'Price', 'data']
        for preferred in preferred_tables:
            if preferred in tables:
                return preferred
        # Otherwise return first non-system table
        for table in tables:
            if not table.startswith('sqlite_'):
                return table

        raise ValueError(f"No tables found in database. Available: {tables}")
    finally:
        conn.close()


def load_custom_data(db_name='fundamentals', symbol=None, columns=None, limit=None):
    """
    Load custom fundamentals DataFrame with optional filtering.

    Parameters
    ----------
    db_name : str, default 'fundamentals'
        Database name
    symbol : str, optional
        Filter by symbol
    columns : list of str, optional
        Specific columns to load
    limit : int, optional
        Limit number of rows

    Returns
    -------
    pd.DataFrame
        Fundamentals data
    """
    db_path = get_custom_db_path(db_name)
    table_name = get_table_name(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        # Get actual column names from database
        db_columns = get_all_columns(db_path, table_name)

        # Build query
        if columns:
            # Always include metadata columns for filtering (find actual names case-insensitively)
            meta_cols = []
            for meta in ['symbol', 'date', 'permaticker', 'tradedate']:
                actual = find_column(db_columns, meta)
                if actual:
                    meta_cols.append(actual)

            all_cols = list(set(meta_cols + columns))
            col_str = ', '.join([c for c in all_cols if c in db_columns])
        else:
            col_str = '*'

        query = f"SELECT {col_str} FROM {table_name}"

        # Add symbol filter (handle case-insensitive column names)
        if symbol:
            all_cols_in_table = get_all_columns(db_path, table_name)
            symbol_col = None
            for col in all_cols_in_table:
                if col.lower() == 'symbol':
                    symbol_col = col
                    break
                elif col.lower() == 'permaticker':
                    symbol_col = col
                    break
            if symbol_col:
                query += f" WHERE {symbol_col} = '{symbol.upper()}'"

        # Add limit
        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn)

        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df
    finally:
        conn.close()


def get_all_columns(db_path, table_name):
    """Get all column names from a table."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]
    finally:
        conn.close()


def find_column(columns, name):
    """Find column name case-insensitively."""
    name_lower = name.lower()
    for col in columns:
        if col.lower() == name_lower:
            return col
    return None


def list_available_columns(db_name='fundamentals'):
    """
    List all available columns in the custom database.

    Parameters
    ----------
    db_name : str, default 'fundamentals'
        Database name

    Returns
    -------
    list
        List of column names
    """
    db_path = get_custom_db_path(db_name)
    table_name = get_table_name(db_path)
    all_cols = get_all_columns(db_path, table_name)
    # Exclude metadata columns (case-insensitive)
    exclude_lower = ['sid', 'symbol', 'date', 'permaticker', 'tradedate']
    cols = [c for c in all_cols if c.lower() not in exclude_lower]
    return sorted(cols)


def print_columns(db_name='fundamentals', columns_per_row=2):
    """
    Print all available columns in a formatted display.

    Parameters
    ----------
    db_name : str, default 'fundamentals'
        Database name
    columns_per_row : int, default 2
        Number of columns to display per row

    Examples
    --------
    >>> print_columns()
    """
    cols = list_available_columns(db_name)

    print("=" * 70)
    print(f"CUSTOM FUNDAMENTALS AVAILABLE COLUMNS ({len(cols)} total)")
    print("=" * 70)

    # Print single column list for clarity
    for i, col in enumerate(cols, 1):
        print(f"  {i:2}. {col}")

    print("=" * 70)
    print("\nCommon aliases:")
    print("-" * 50)
    print("  ROE  = ReturnOnEquity_SmartEstimat")
    print("  ROA  = ReturnOnAssets_SmartEstimate")
    print("  ROIC = ReturnOnInvestedCapital_BrokerEstimate")
    print("  FCF  = FOCFExDividends_Discrete")
    print("  EPS  = EarningsPerShare_Actual")
    print("  PEG  = ForwardPEG_DailyTimeSeriesRatio_")
    print("-" * 50)


def list_symbols(db_name='fundamentals'):
    """
    List all available symbols in the database.

    Parameters
    ----------
    db_name : str, default 'fundamentals'
        Database name

    Returns
    -------
    list
        List of unique symbols
    """
    db_path = get_custom_db_path(db_name)
    table_name = get_table_name(db_path)
    all_cols = get_all_columns(db_path, table_name)

    conn = sqlite3.connect(str(db_path))
    try:
        # Check for symbol column (case-insensitive)
        symbol_col = None
        for col in all_cols:
            if col.lower() == 'symbol':
                symbol_col = col
                break
            elif col.lower() == 'permaticker':
                symbol_col = col
                break

        if symbol_col:
            cursor = conn.execute(f"SELECT DISTINCT {symbol_col} FROM {table_name}")
            return sorted([row[0] for row in cursor.fetchall() if row[0]])
        else:
            raise ValueError(f"No symbol or permaticker column found in database. Available columns: {all_cols[:10]}")
    finally:
        conn.close()


def query_fundamentals(
    symbol,
    columns,
    n=10,
    db_name='fundamentals',
    include_dates=True,
    sort_descending=True,
):
    """
    Query custom fundamentals for a specific symbol.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    columns : list of str
        List of column names to retrieve (e.g., ['pred', 'bc1', 'ROE'])
    n : int, default 10
        Number of most recent values to return
    db_name : str, default 'fundamentals'
        Database name
    include_dates : bool, default True
        Include date column
    sort_descending : bool, default True
        Sort by date descending (most recent first)

    Returns
    -------
    pd.DataFrame
        DataFrame with requested columns for the symbol

    Examples
    --------
    >>> df = query_fundamentals('AAPL', ['pred', 'bc1', 'ROE'], n=5)
    >>> print(df)
    """
    # Validate columns first
    available_cols = list_available_columns(db_name) + ['symbol', 'date', 'permaticker']
    invalid_cols = [c for c in columns if c not in available_cols]
    if invalid_cols:
        raise ValueError(
            f"Invalid columns: {invalid_cols}. "
            f"Use list_available_columns() to see available columns."
        )

    # Load only the data we need for this symbol
    symbol_data = load_custom_data(db_name, symbol=symbol, columns=columns)

    if symbol_data.empty:
        available_symbols = list_symbols(db_name)[:20]
        raise ValueError(
            f"Symbol '{symbol}' not found in custom data. "
            f"Example available symbols: {available_symbols}"
        )

    # Select columns
    select_cols = []
    if include_dates and 'date' in symbol_data.columns:
        select_cols.append('date')

    select_cols.extend(columns)

    # Filter to only existing columns
    select_cols = [c for c in select_cols if c in symbol_data.columns]
    result = symbol_data[select_cols].copy()

    # Sort by date
    if 'date' in result.columns:
        result = result.sort_values('date', ascending=not sort_descending)

    # Return last n rows
    return result.head(n).reset_index(drop=True)


def query_multiple_symbols(
    symbols,
    columns,
    n=5,
    db_name='fundamentals',
    latest_only=True,
):
    """
    Query custom fundamentals for multiple symbols.

    Parameters
    ----------
    symbols : list of str
        List of stock symbols
    columns : list of str
        List of column names to retrieve
    n : int, default 5
        Number of most recent values per symbol (if latest_only=False)
    db_name : str, default 'fundamentals'
        Database name
    latest_only : bool, default True
        Only return the most recent value for each symbol

    Returns
    -------
    pd.DataFrame
        DataFrame with requested data

    Examples
    --------
    >>> df = query_multiple_symbols(['AAPL', 'MSFT', 'GOOGL'], ['pred', 'ROE'])
    """
    # Validate columns first
    available_cols = list_available_columns(db_name) + ['symbol', 'date', 'permaticker']
    invalid_cols = [c for c in columns if c not in available_cols]
    if invalid_cols:
        raise ValueError(f"Invalid columns: {invalid_cols}")

    # Normalize symbols
    symbols = [s.upper() for s in symbols]

    # Load data for each symbol and concatenate
    dfs = []
    for symbol in symbols:
        try:
            symbol_df = load_custom_data(db_name, symbol=symbol, columns=columns)
            if not symbol_df.empty:
                dfs.append(symbol_df)
        except:
            pass

    if not dfs:
        raise ValueError(f"No data found for symbols: {symbols}")

    filtered = pd.concat(dfs, ignore_index=True)

    # Determine symbol column (case-insensitive)
    symbol_col = find_column(list(filtered.columns), 'symbol') or find_column(list(filtered.columns), 'permaticker')

    if not symbol_col:
        raise ValueError(f"No symbol column found. Available columns: {list(filtered.columns)}")

    # Determine date column (case-insensitive)
    date_col = find_column(list(filtered.columns), 'date') or find_column(list(filtered.columns), 'tradedate')

    # Select columns
    select_cols = [symbol_col] if symbol_col else []
    if date_col:
        select_cols.append(date_col)
    select_cols.extend(columns)
    select_cols = [c for c in select_cols if c in filtered.columns]

    result = filtered[select_cols].copy()

    # Sort by symbol and date
    sort_cols = []
    ascending = []
    if symbol_col and symbol_col in result.columns:
        sort_cols.append(symbol_col)
        ascending.append(True)
    if date_col and date_col in result.columns:
        sort_cols.append(date_col)
        ascending.append(False)

    if sort_cols:
        result = result.sort_values(sort_cols, ascending=ascending)

    if latest_only:
        # Get most recent for each symbol
        result = result.groupby(symbol_col).first().reset_index()
    else:
        # Get n most recent for each symbol
        result = result.groupby(symbol_col).head(n).reset_index(drop=True)

    return result


def compare_symbols(
    symbols,
    columns,
    db_name='fundamentals',
):
    """
    Compare latest fundamentals across multiple symbols.

    Parameters
    ----------
    symbols : list of str
        List of stock symbols to compare
    columns : list of str
        List of column names to compare
    db_name : str, default 'fundamentals'
        Database name

    Returns
    -------
    pd.DataFrame
        Pivot table with symbols as rows and columns as values

    Examples
    --------
    >>> df = compare_symbols(['AAPL', 'MSFT', 'GOOGL'], ['pred', 'ROE'])
    """
    df = query_multiple_symbols(symbols, columns, db_name=db_name, latest_only=True)

    # Determine symbol column
    symbol_col = 'symbol' if 'symbol' in df.columns else 'permaticker'

    # Pivot to get symbols as rows
    result = df.set_index(symbol_col)[columns]

    return result


def get_symbol_history(
    symbol,
    column,
    start_date=None,
    end_date=None,
    db_name='fundamentals',
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
    db_name : str, default 'fundamentals'
        Database name

    Returns
    -------
    pd.Series
        Time series of the column values, indexed by date

    Examples
    --------
    >>> pred = get_symbol_history('AAPL', 'pred', start_date='2020-01-01')
    >>> pred.plot()
    """
    # Load only the data we need for this symbol
    symbol_data = load_custom_data(db_name, symbol=symbol, columns=[column])

    if symbol_data.empty:
        raise ValueError(f"Symbol '{symbol}' not found")

    # Filter by date
    if 'date' in symbol_data.columns:
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            symbol_data = symbol_data[symbol_data['date'] >= start_date]

        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            symbol_data = symbol_data[symbol_data['date'] <= end_date]

        # Create series
        result = symbol_data.set_index('date')[column].sort_index()
    else:
        result = symbol_data[column]

    result.name = f"{symbol}_{column}"

    return result


if __name__ == '__main__':
    # Example usage
    print("Custom Fundamentals Query Helper")
    print("=" * 50)

    try:
        # List available columns
        cols = list_available_columns()
        print(f"\nAvailable columns ({len(cols)} total):")
        print(cols[:20])  # Show first 20
        if len(cols) > 20:
            print("...")

        # List symbols
        symbols = list_symbols()
        print(f"\nAvailable symbols ({len(symbols)} total):")
        print(symbols[:20])
        if len(symbols) > 20:
            print("...")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have created the custom fundamentals database.")
