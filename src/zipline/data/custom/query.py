"""
Query API for custom data databases.

Provides functions to retrieve custom data from SQLite databases.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from .config import DEFAULT_DB_DIR
from .db_manager import connect_db, describe_custom_db

log = logging.getLogger(__name__)


def get_prices(
    db_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sids: Optional[Sequence[Union[int, str]]] = None,
    fields: Optional[Sequence[str]] = None,
    db_dir: Union[str, Path] = None,
) -> pd.DataFrame:
    """
    Retrieve custom data from database.

    Parameters
    ----------
    db_code : str
        Database code/identifier
    start_date : str, optional
        Start date (inclusive). Format: 'YYYY-MM-DD' or ISO 8601.
    end_date : str, optional
        End date (inclusive). Format: 'YYYY-MM-DD' or ISO 8601.
    sids : sequence of int or str, optional
        List of Sids to retrieve. If None, retrieves all Sids.
    fields : sequence of str, optional
        List of field/column names to retrieve. If None, retrieves all fields.
    db_dir : str or Path, optional
        Directory containing database

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Sid', 'Date'] plus requested fields.
        Date column is parsed to datetime.

    Examples
    --------
    >>> # Get all data
    >>> df = get_prices('fundamentals')
    >>>
    >>> # Get specific date range
    >>> df = get_prices('fundamentals', start_date='2020-01-01', end_date='2020-12-31')
    >>>
    >>> # Get specific fields for specific sids
    >>> df = get_prices(
    ...     'fundamentals',
    ...     sids=[101, 102],
    ...     fields=['Revenue', 'EPS']
    ... )
    """
    # Get database schema
    db_info = describe_custom_db(db_code, db_dir)
    available_fields = set(db_info['columns'].keys())

    # Determine which fields to select
    if fields is None:
        select_fields = ['*']
    else:
        # Validate fields
        invalid_fields = set(fields) - available_fields
        if invalid_fields:
            raise ValueError(
                f"Unknown fields: {', '.join(invalid_fields)}. "
                f"Available fields: {', '.join(available_fields)}"
            )
        select_fields = ['Sid', 'Date'] + list(fields)

    # Build SQL query
    sql_parts = [f"SELECT {', '.join(select_fields)} FROM Price WHERE 1=1"]
    params = []

    # Add date filters
    if start_date:
        sql_parts.append("AND Date >= ?")
        params.append(start_date)

    if end_date:
        sql_parts.append("AND Date <= ?")
        params.append(end_date)

    # Add Sid filter
    if sids:
        # Convert all sids to strings for comparison
        sids_str = [str(sid) for sid in sids]
        placeholders = ','.join(['?' for _ in sids_str])
        sql_parts.append(f"AND Sid IN ({placeholders})")
        params.extend(sids_str)

    # Add ordering
    sql_parts.append("ORDER BY Date, Sid")

    sql = ' '.join(sql_parts)

    # Execute query
    conn = connect_db(db_code, db_dir)
    try:
        df = pd.read_sql_query(sql, conn, params=params)

        # Parse dates
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        log.debug(f"Retrieved {len(df)} rows from {db_code}")
        return df

    finally:
        conn.close()


def get_prices_reindexed_like(
    db_code: str,
    template_df: pd.DataFrame,
    fields: Optional[Sequence[str]] = None,
    db_dir: Union[str, Path] = None,
) -> pd.DataFrame:
    """
    Retrieve custom data aligned to a template DataFrame.

    This is useful for ensuring custom data has the same shape as other
    price data (e.g., from a bundle).

    Parameters
    ----------
    db_code : str
        Database code/identifier
    template_df : pd.DataFrame
        Template DataFrame with index (dates) and columns (sids)
    fields : sequence of str, optional
        Fields to retrieve. If None, retrieves all fields.
    db_dir : str or Path, optional
        Directory containing database

    Returns
    -------
    dict
        Dictionary mapping field names to DataFrames, each with the same
        shape as template_df (dates as index, sids as columns)

    Examples
    --------
    >>> # Get price data from bundle
    >>> bundle_prices = data_portal.get_history_window(...)
    >>>
    >>> # Get custom data aligned to same shape
    >>> custom_data = get_prices_reindexed_like(
    ...     'fundamentals',
    ...     bundle_prices,
    ...     fields=['Revenue', 'EPS']
    ... )
    >>>
    >>> # Now bundle_prices and custom_data['Revenue'] have matching shape
    >>> assert bundle_prices.shape == custom_data['Revenue'].shape
    """
    # Extract dates and sids from template
    dates = template_df.index
    sids = template_df.columns

    # Get date range
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = dates.max().strftime('%Y-%m-%d')

    # Query custom data
    df = get_prices(
        db_code,
        start_date=start_date,
        end_date=end_date,
        sids=sids,
        fields=fields,
        db_dir=db_dir,
    )

    if df.empty:
        log.warning(f"No data found for {db_code} in date range")
        # Return empty DataFrames with correct shape
        result = {}
        if fields:
            for field in fields:
                result[field] = pd.DataFrame(index=dates, columns=sids)
        return result

    # Pivot each field into separate DataFrame
    result = {}

    # Get list of data fields (excluding Sid and Date)
    data_fields = [col for col in df.columns if col not in ('Sid', 'Date')]

    for field in data_fields:
        # Pivot: index=Date, columns=Sid, values=field
        pivoted = df.pivot(index='Date', columns='Sid', values=field)

        # Reindex to match template exactly
        reindexed = pivoted.reindex(index=dates, columns=sids)

        result[field] = reindexed

    return result


def get_latest_values(
    db_code: str,
    sids: Optional[Sequence[Union[int, str]]] = None,
    fields: Optional[Sequence[str]] = None,
    as_of_date: Optional[str] = None,
    db_dir: Union[str, Path] = None,
) -> pd.DataFrame:
    """
    Get the most recent values for each sid.

    Parameters
    ----------
    db_code : str
        Database code/identifier
    sids : sequence of int or str, optional
        List of Sids. If None, gets latest for all Sids.
    fields : sequence of str, optional
        List of fields. If None, gets all fields.
    as_of_date : str, optional
        Get latest values as of this date. If None, uses most recent date in database.
    db_dir : str or Path, optional
        Directory containing database

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sid, containing latest values

    Examples
    --------
    >>> # Get latest fundamentals for all companies
    >>> latest = get_latest_values('fundamentals')
    >>>
    >>> # Get latest as of specific date
    >>> latest = get_latest_values(
    ...     'fundamentals',
    ...     as_of_date='2020-12-31',
    ...     fields=['Revenue', 'EPS']
    ... )
    """
    # Get database schema
    db_info = describe_custom_db(db_code, db_dir)
    available_fields = set(db_info['columns'].keys())

    # Determine fields to select
    if fields is None:
        select_fields = list(available_fields)
    else:
        invalid_fields = set(fields) - available_fields
        if invalid_fields:
            raise ValueError(
                f"Unknown fields: {', '.join(invalid_fields)}. "
                f"Available fields: {', '.join(available_fields)}"
            )
        select_fields = list(fields)

    # Build query - get latest date for each sid
    field_list = ', '.join(select_fields)

    sql_parts = [
        f"""
        SELECT Sid, Date, {field_list}
        FROM Price p1
        WHERE Date = (
            SELECT MAX(Date)
            FROM Price p2
            WHERE p2.Sid = p1.Sid
        """
    ]
    params = []

    if as_of_date:
        sql_parts.append("AND p2.Date <= ?")
        params.append(as_of_date)

    sql_parts.append(")")

    if sids:
        sids_str = [str(sid) for sid in sids]
        placeholders = ','.join(['?' for _ in sids_str])
        sql_parts.append(f"AND Sid IN ({placeholders})")
        params.extend(sids_str)

    sql_parts.append("ORDER BY Sid")

    sql = ' '.join(sql_parts)

    # Execute query
    conn = connect_db(db_code, db_dir)
    try:
        df = pd.read_sql_query(sql, conn, params=params)

        # Parse dates
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        log.debug(f"Retrieved latest values for {len(df)} sids from {db_code}")
        return df

    finally:
        conn.close()
