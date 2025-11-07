"""
CSV data loader for custom databases.

Handles loading CSV files into custom data databases with identifier mapping,
date normalization, and duplicate handling.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_DB_DIR
from .db_manager import connect_db, describe_custom_db
from .insert_utils import batch_insert

log = logging.getLogger(__name__)


def normalize_dates(
    df: pd.DataFrame,
    date_col: str,
    date_format: Optional[str] = None,
    tz: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalize date column to ISO 8601 format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column
    date_col : str
        Name of date column
    date_format : str, optional
        strptime format string. If None, pandas will infer.
    tz : str, optional
        Timezone to localize to. If None, dates are naive.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized dates
    """
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    else:
        df[date_col] = pd.to_datetime(df[date_col])

    # Handle timezone
    if tz:
        if df[date_col].dt.tz is None:
            df[date_col] = df[date_col].dt.tz_localize(tz)
        else:
            df[date_col] = df[date_col].dt.tz_convert(tz)

    # Convert to ISO 8601 string format for storage
    df[date_col] = df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df


def map_identifiers_to_sids(
    df: pd.DataFrame,
    id_col: str,
    sid_map: Union[Dict[str, int], pd.DataFrame],
) -> tuple:
    """
    Map identifiers (e.g., symbols) to Sids.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with identifier column
    id_col : str
        Name of identifier column (e.g., 'Symbol')
    sid_map : dict or DataFrame
        Either:
        - Dict mapping identifiers to Sids: {'AAPL': 101, ...}
        - DataFrame with columns [id_col, 'Sid']

    Returns
    -------
    tuple
        (mapped_df, unmapped_ids)
        - mapped_df: DataFrame with 'Sid' column added
        - unmapped_ids: List of identifiers that couldn't be mapped
    """
    # Convert DataFrame sid_map to dict if needed
    if isinstance(sid_map, pd.DataFrame):
        if id_col not in sid_map.columns or 'Sid' not in sid_map.columns:
            raise ValueError(
                f"sid_map DataFrame must have columns '{id_col}' and 'Sid'"
            )
        sid_map = dict(zip(sid_map[id_col], sid_map['Sid']))

    # Create Sid column by mapping
    df['Sid'] = df[id_col].map(sid_map)

    # Identify unmapped rows
    unmapped_mask = df['Sid'].isna()
    unmapped_ids = df.loc[unmapped_mask, id_col].unique().tolist()

    # Keep only mapped rows
    mapped_df = df[~unmapped_mask].copy()

    # Convert Sid to string (for consistency with database schema)
    mapped_df['Sid'] = mapped_df['Sid'].astype(int).astype(str)

    return mapped_df, unmapped_ids


def deduplicate_data(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'last',
) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    subset : list of str, optional
        Columns to use for identifying duplicates. Default is ['Sid', 'Date'].
    keep : str, optional
        Which duplicates to keep: 'first', 'last', or False (drop all)

    Returns
    -------
    pd.DataFrame
        De-duplicated DataFrame
    """
    if subset is None:
        subset = ['Sid', 'Date']

    original_len = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    dropped = original_len - len(df)

    if dropped > 0:
        log.warning(f"Dropped {dropped} duplicate rows (kept='{keep}')")

    return df


def load_csv_to_db(
    csv_path: Union[str, Path],
    db_code: str,
    sid_map: Optional[Union[Dict[str, int], pd.DataFrame]] = None,
    id_col: str = "Symbol",
    date_col: str = "Date",
    date_format: Optional[str] = None,
    tz: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    on_duplicate: str = "replace",
    fail_on_unmapped: bool = True,
    db_dir: Union[str, Path] = None,
) -> Dict[str, Any]:
    """
    Load CSV data into custom database.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file
    db_code : str
        Database code/identifier
    sid_map : dict or DataFrame, optional
        Mapping from identifiers to Sids. If None, will attempt to use
        identifiers as-is (assumes they're already Sids).
    id_col : str, optional
        Name of identifier column in CSV (default: 'Symbol')
    date_col : str, optional
        Name of date column in CSV (default: 'Date')
    date_format : str, optional
        strptime format string for dates. If None, pandas infers format.
    tz : str, optional
        Timezone for dates. If None, dates are naive.
    chunk_size : int, optional
        Number of rows to process at once
    on_duplicate : str, optional
        How to handle duplicate (Sid, Date) pairs: 'replace', 'ignore', or 'fail'
    fail_on_unmapped : bool, optional
        If True, raise error when identifiers can't be mapped to Sids
    db_dir : str or Path, optional
        Directory containing database

    Returns
    -------
    dict
        Diagnostic information:
        - rows_inserted: Number of rows successfully inserted
        - rows_skipped: Number of rows skipped (unmapped or duplicates)
        - unmapped_ids: List of identifiers that couldn't be mapped
        - errors: List of error messages

    Raises
    ------
    FileNotFoundError
        If CSV file or database doesn't exist
    ValueError
        If required columns are missing or fail_on_unmapped=True with unmapped IDs

    Examples
    --------
    >>> sid_map = {'AAPL': 101, 'MSFT': 102}
    >>> result = load_csv_to_db(
    ...     'fundamentals.csv',
    ...     'fundamentals',
    ...     sid_map=sid_map,
    ...     id_col='Symbol',
    ...     date_col='Date'
    ... )
    >>> print(f"Inserted {result['rows_inserted']} rows")
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Get database info
    db_info = describe_custom_db(db_code, db_dir)
    expected_columns = set(db_info['columns'].keys())

    log.info(f"Loading data from {csv_path} into database '{db_code}'")

    # Initialize counters
    total_rows_inserted = 0
    total_rows_skipped = 0
    all_unmapped_ids = []
    errors = []

    # Process CSV in chunks
    try:
        chunks = pd.read_csv(csv_path, chunksize=chunk_size)
        chunk_num = 0

        for chunk in chunks:
            chunk_num += 1
            log.debug(f"Processing chunk {chunk_num} ({len(chunk)} rows)...")

            try:
                # Validate required columns
                if id_col not in chunk.columns:
                    raise ValueError(f"Column '{id_col}' not found in CSV")
                if date_col not in chunk.columns:
                    raise ValueError(f"Column '{date_col}' not found in CSV")

                # Normalize dates
                chunk = normalize_dates(chunk, date_col, date_format, tz)

                # Map identifiers to Sids
                if sid_map is not None:
                    chunk, unmapped_ids = map_identifiers_to_sids(chunk, id_col, sid_map)

                    if unmapped_ids:
                        all_unmapped_ids.extend(unmapped_ids)
                        log.warning(
                            f"Chunk {chunk_num}: {len(unmapped_ids)} unmapped identifiers"
                        )

                    if chunk.empty:
                        log.warning(f"Chunk {chunk_num}: All rows unmapped, skipping")
                        total_rows_skipped += len(chunk)
                        continue
                else:
                    # Assume id_col contains Sids already
                    chunk['Sid'] = chunk[id_col].astype(str)

                # Rename date column to 'Date' if different
                if date_col != 'Date':
                    chunk = chunk.rename(columns={date_col: 'Date'})

                # Select only columns that exist in database schema
                # (Sid, Date, plus data columns)
                available_cols = set(chunk.columns)
                required_cols = {'Sid', 'Date'}
                data_cols = expected_columns & available_cols

                if not data_cols:
                    log.warning(
                        f"Chunk {chunk_num}: No matching data columns found. "
                        f"Expected: {expected_columns}, Got: {available_cols}"
                    )
                    continue

                select_cols = list(required_cols | data_cols)
                chunk = chunk[select_cols]

                # Deduplicate within chunk
                chunk = deduplicate_data(chunk, subset=['Sid', 'Date'])

                # Insert into database
                conn = connect_db(db_code, db_dir)
                try:
                    rows_inserted = batch_insert(
                        chunk,
                        table='Price',
                        conn=conn,
                        on_duplicate=on_duplicate,
                        batch_size=10000,
                    )
                    total_rows_inserted += rows_inserted
                    log.info(f"Chunk {chunk_num}: Inserted {rows_inserted} rows")

                finally:
                    conn.close()

            except Exception as e:
                error_msg = f"Error processing chunk {chunk_num}: {e}"
                log.error(error_msg)
                errors.append(error_msg)
                # Continue with next chunk
                continue

    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # Check for unmapped identifiers
    if all_unmapped_ids:
        unique_unmapped = list(set(all_unmapped_ids))
        log.warning(f"Total unmapped identifiers: {len(unique_unmapped)}")

        if fail_on_unmapped:
            raise ValueError(
                f"Found {len(unique_unmapped)} unmapped identifiers: "
                f"{', '.join(map(str, unique_unmapped[:10]))}"
                + ("..." if len(unique_unmapped) > 10 else "")
            )

    # Return diagnostics
    result = {
        'rows_inserted': total_rows_inserted,
        'rows_skipped': total_rows_skipped,
        'unmapped_ids': list(set(all_unmapped_ids)),
        'errors': errors,
    }

    log.info(
        f"Load complete: {total_rows_inserted} rows inserted, "
        f"{len(result['unmapped_ids'])} unmapped identifiers"
    )

    return result
