"""
Insert utilities for custom data databases.

Provides different strategies for inserting data into SQLite databases:
- replace: Update existing records, insert new ones
- ignore: Skip duplicate records
- fail: Raise error on duplicates
"""

import logging
import sqlite3
from typing import List

import pandas as pd

log = logging.getLogger(__name__)


def insert_or_replace(
    df: pd.DataFrame,
    table: str,
    conn: sqlite3.Connection
) -> int:
    """
    Insert data with REPLACE strategy (update on conflict).

    If a row with the same primary key exists, it will be replaced.

    Parameters
    ----------
    df : DataFrame
        Data to insert. Must have columns matching table schema.
    table : str
        Table name (typically 'Price')
    conn : sqlite3.Connection
        Database connection

    Returns
    -------
    int
        Number of rows inserted/updated

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Sid': ['101', '102'],
    ...     'Date': ['2020-01-01', '2020-01-01'],
    ...     'Revenue': [1000, 2000]
    ... })
    >>> insert_or_replace(df, 'Price', conn)
    2
    """
    if df.empty:
        return 0

    # Use DataFrame.to_sql with if_exists='replace' would replace whole table
    # Instead, use INSERT OR REPLACE for row-level replacement
    columns = df.columns.tolist()
    placeholders = ', '.join(['?' for _ in columns])
    column_names = ', '.join(columns)

    sql = f"INSERT OR REPLACE INTO {table} ({column_names}) VALUES ({placeholders})"

    cursor = conn.cursor()
    rows = [tuple(row) for row in df.values]

    try:
        cursor.executemany(sql, rows)
        conn.commit()
        rows_affected = cursor.rowcount
        log.debug(f"Inserted/replaced {rows_affected} rows into {table}")
        return rows_affected

    except Exception as e:
        conn.rollback()
        log.error(f"Error inserting data: {e}")
        raise


def insert_or_ignore(
    df: pd.DataFrame,
    table: str,
    conn: sqlite3.Connection
) -> int:
    """
    Insert data with IGNORE strategy (skip duplicates).

    If a row with the same primary key exists, the new row is ignored.

    Parameters
    ----------
    df : DataFrame
        Data to insert
    table : str
        Table name
    conn : sqlite3.Connection
        Database connection

    Returns
    -------
    int
        Number of new rows inserted (duplicates not counted)
    """
    if df.empty:
        return 0

    columns = df.columns.tolist()
    placeholders = ', '.join(['?' for _ in columns])
    column_names = ', '.join(columns)

    sql = f"INSERT OR IGNORE INTO {table} ({column_names}) VALUES ({placeholders})"

    cursor = conn.cursor()
    rows = [tuple(row) for row in df.values]

    try:
        cursor.executemany(sql, rows)
        conn.commit()
        rows_affected = cursor.rowcount
        log.debug(f"Inserted {rows_affected} new rows into {table} (ignored duplicates)")
        return rows_affected

    except Exception as e:
        conn.rollback()
        log.error(f"Error inserting data: {e}")
        raise


def insert_or_fail(
    df: pd.DataFrame,
    table: str,
    conn: sqlite3.Connection
) -> int:
    """
    Insert data with FAIL strategy (raise on duplicates).

    If a row with the same primary key exists, raises IntegrityError.

    Parameters
    ----------
    df : pd.DataFrame
        Data to insert
    table : str
        Table name
    conn : sqlite3.Connection
        Database connection

    Returns
    -------
    int
        Number of rows inserted

    Raises
    ------
    sqlite3.IntegrityError
        If any duplicate keys are found
    """
    if df.empty:
        return 0

    columns = df.columns.tolist()
    placeholders = ', '.join(['?' for _ in columns])
    column_names = ', '.join(columns)

    sql = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"

    cursor = conn.cursor()
    rows = [tuple(row) for row in df.values]

    try:
        cursor.executemany(sql, rows)
        conn.commit()
        rows_affected = cursor.rowcount
        log.debug(f"Inserted {rows_affected} rows into {table}")
        return rows_affected

    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise ValueError(
            f"Duplicate key violation: {e}\n"
            "Use on_duplicate='replace' or 'ignore' to handle duplicates"
        )
    except Exception as e:
        conn.rollback()
        log.error(f"Error inserting data: {e}")
        raise


def get_insert_function(on_duplicate: str):
    """
    Get the appropriate insert function for the given strategy.

    Parameters
    ----------
    on_duplicate : str
        One of 'replace', 'ignore', or 'fail'

    Returns
    -------
    callable
        Insert function

    Raises
    ------
    ValueError
        If strategy is not recognized
    """
    strategies = {
        'replace': insert_or_replace,
        'ignore': insert_or_ignore,
        'fail': insert_or_fail,
    }

    if on_duplicate not in strategies:
        raise ValueError(
            f"Invalid on_duplicate strategy: {on_duplicate}. "
            f"Must be one of: {', '.join(strategies.keys())}"
        )

    return strategies[on_duplicate]


def batch_insert(
    df: pd.DataFrame,
    table: str,
    conn: sqlite3.Connection,
    on_duplicate: str = 'replace',
    batch_size: int = 10000,
) -> int:
    """
    Insert data in batches for better performance.

    Parameters
    ----------
    df : pd.DataFrame
        Data to insert
    table : str
        Table name
    conn : sqlite3.Connection
        Database connection
    on_duplicate : str, optional
        Strategy for handling duplicates ('replace', 'ignore', 'fail')
    batch_size : int, optional
        Number of rows per batch

    Returns
    -------
    int
        Total number of rows inserted/updated
    """
    if df.empty:
        return 0

    insert_func = get_insert_function(on_duplicate)
    total_rows = 0

    # Process in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        rows_affected = insert_func(batch, table, conn)
        total_rows += rows_affected

        if (i + batch_size) % (batch_size * 10) == 0:
            log.info(f"Processed {i + batch_size:,} rows...")

    return total_rows
