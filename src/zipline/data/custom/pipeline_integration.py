"""
Zipline Pipeline integration for custom data.

Provides DataSet classes and loaders to use custom data in Pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Type, Union

import numpy as np
import pandas as pd
from zipline.lib.adjusted_array import AdjustedArray
from zipline.pipeline.data import Column, DataSet
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.utils.numpy_utils import float64_dtype, int64_dtype

from .config import DEFAULT_DB_DIR
from .db_manager import connect_db, describe_custom_db

log = logging.getLogger(__name__)

# Map our custom types to numpy dtypes
TYPE_TO_DTYPE = {
    'int': int64_dtype,
    'float': float64_dtype,
    'text': object,  # Object dtype for strings
    'date': object,
    'datetime': object,
}


def make_custom_dataset_class(
    db_code: str,
    columns: Dict[str, str],
    base_name: Optional[str] = None,
) -> Type[DataSet]:
    """
    Create a DataSet class for custom data.

    Parameters
    ----------
    db_code : str
        Database code/identifier
    columns : dict
        Dictionary mapping column names to types ('int', 'float', 'text', etc.)
    base_name : str, optional
        Base name for the DataSet class. If None, uses db_code.

    Returns
    -------
    type
        A DataSet subclass with appropriate Column definitions

    Examples
    --------
    >>> columns = {'Revenue': 'int', 'EPS': 'float', 'Currency': 'text'}
    >>> FundamentalsDataSet = make_custom_dataset_class(
    ...     'fundamentals',
    ...     columns
    ... )
    >>>
    >>> # Use in pipeline
    >>> pipeline = Pipeline(columns={
    ...     'eps': FundamentalsDataSet.EPS.latest,
    ...     'revenue': FundamentalsDataSet.Revenue.latest,
    ... })
    """
    if base_name is None:
        base_name = db_code.replace('_', ' ').title().replace(' ', '')

    class_name = f"{base_name}DataSet"

    # Build column definitions
    column_defs = {}

    for col_name, col_type in columns.items():
        # Get numpy dtype
        dtype = TYPE_TO_DTYPE.get(col_type, float64_dtype)

        # Determine missing value based on dtype
        if dtype == int64_dtype:
            missing_value = -1
        elif dtype == float64_dtype:
            missing_value = np.nan
        else:
            missing_value = None

        # Create Column
        column_defs[col_name] = Column(
            dtype=dtype,
            missing_value=missing_value,
            doc=f"{col_name} from {db_code} custom data"
        )

    # Create DataSet class dynamically
    dataset_class = type(
        class_name,
        (DataSet,),
        column_defs
    )

    log.debug(
        f"Created DataSet class '{class_name}' with columns: "
        f"{', '.join(columns.keys())}"
    )

    return dataset_class


class CustomSQLiteLoader(PipelineLoader):
    """
    PipelineLoader for custom data stored in SQLite.

    This loader reads data from a custom SQLite database and provides it
    to Zipline Pipeline in the expected format.

    Parameters
    ----------
    db_code : str
        Database code/identifier
    db_dir : str or Path, optional
        Directory containing database

    Examples
    --------
    >>> loader = CustomSQLiteLoader('fundamentals')
    >>> engine = SimplePipelineEngine(
    ...     get_loader=lambda column: loader,
    ...     asset_finder=asset_finder,
    ...     default_domain=US_EQUITIES,
    ... )
    """

    def __init__(
        self,
        db_code: str,
        db_dir: Union[str, Path] = None,
    ):
        self.db_code = db_code
        self.db_dir = db_dir
        self._cache = {}

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        """
        Load data as AdjustedArrays for Pipeline.

        Parameters
        ----------
        domain : zipline.pipeline.domain.Domain
            Pipeline domain
        columns : list[BoundColumn]
            Columns to load
        dates : pd.DatetimeIndex
            Dates for which to load data
        sids : pd.Int64Index
            Asset IDs for which to load data
        mask : np.ndarray[bool]
            Boolean mask indicating active assets

        Returns
        -------
        dict
            Mapping from BoundColumn to AdjustedArray
        """
        # Get all unique column names
        column_names = list(set(col.name for col in columns))

        log.debug(
            f"Loading {len(column_names)} columns for "
            f"{len(dates)} dates and {len(sids)} sids"
        )

        # Query database for all needed columns at once
        # Pass columns with their dtypes so we know which to convert to numeric
        data = self._query_database(dates, sids, columns)

        # Build AdjustedArrays for each column
        arrays = {}
        for column in columns:
            col_data = data.get(column.name)

            if col_data is None:
                # Column not found, create empty array with missing values
                log.warning(
                    f"Column '{column.name}' not found in database, "
                    "using missing values"
                )
                col_data = np.full(
                    (len(dates), len(sids)),
                    column.missing_value,
                    dtype=column.dtype
                )
            else:
                # Ensure correct dtype - data already converted in _query_database
                try:
                    col_data = col_data.astype(column.dtype)
                except Exception as e:
                    log.error(
                        f"Error converting column '{column.name}' to {column.dtype}: {e}. "
                        f"Sample values: {col_data.flatten()[:5]}"
                    )
                    # Fall back to creating an array of missing values
                    col_data = np.full(
                        (len(dates), len(sids)),
                        column.missing_value,
                        dtype=column.dtype
                    )

            # Create AdjustedArray
            # AdjustedArrays allow for point-in-time adjustments (splits, etc.)
            # For custom data, we typically don't have adjustments
            arrays[column] = AdjustedArray(
                data=col_data,
                adjustments={},  # No adjustments for custom data
                missing_value=column.missing_value,
            )

        return arrays

    def _query_database(
        self,
        dates: pd.DatetimeIndex,
        sids: pd.Index,
        columns: list,
    ) -> Dict[str, np.ndarray]:
        """
        Query database and return data as dict of arrays.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Dates to query
        sids : pd.Index
            Sids to query
        columns : list of BoundColumn
            Columns to retrieve (with dtype information)

        Returns
        -------
        dict
            Mapping from column name to 2D array of shape (len(dates), len(sids))
        """
        if not columns:
            return {}

        # Extract column names and build dtype mapping
        column_names = [col.name for col in columns]
        column_dtypes = {col.name: col.dtype for col in columns}

        # Convert dates to strings for SQL query
        start_date = dates.min().strftime('%Y-%m-%d')
        end_date = dates.max().strftime('%Y-%m-%d')

        # Convert sids to strings
        sids_str = [str(sid) for sid in sids]

        # Build SQL query
        field_list = ', '.join(column_names)
        placeholders = ','.join(['?' for _ in sids_str])

        sql = f"""
            SELECT Sid, Date, {field_list}
            FROM Price
            WHERE Date >= ? AND Date <= ?
            AND Sid IN ({placeholders})
            ORDER BY Date, Sid
        """

        params = [start_date, end_date] + sids_str

        # Execute query
        conn = connect_db(self.db_code, self.db_dir)
        try:
            df = pd.read_sql_query(sql, conn, params=params)

            log.debug(
                f"Loaded {len(df)} rows for {len(column_names)} columns "
                f"from {self.db_code} database"
            )

            if len(df) == 0:
                log.warning(
                    f"No data found in {self.db_code} database for "
                    f"dates {start_date} to {end_date}, sids {sids_str[:5]}..."
                )

            # Parse dates
            df['Date'] = pd.to_datetime(df['Date'])

            # Convert Sid to int for indexing
            df['Sid'] = df['Sid'].astype(int)

        except Exception as e:
            log.error(f"Error querying {self.db_code} database: {e}")
            conn.close()
            raise
        finally:
            conn.close()

        # Build result dictionary
        result = {}

        # If dataframe is empty, create empty arrays with proper dtypes
        if len(df) == 0:
            for col_name in column_names:
                col_dtype = column_dtypes.get(col_name, np.float64)
                # Create empty array with proper dtype
                arr = np.full((len(dates), len(sids)), np.nan, dtype=col_dtype)
                result[col_name] = arr
                log.warning(f"DEBUG: Column '{col_name}' created empty array with dtype: {arr.dtype}")
            return result

        for col_name in column_names:
            # Pivot data to 2D array: (dates, sids)
            if col_name not in df.columns:
                log.warning(f"Column '{col_name}' not in query results")
                continue

            # Convert to numeric if the column's dtype is numeric
            # This handles cases where SQLite returns numeric values as strings
            col_dtype = column_dtypes.get(col_name)
            if col_dtype in (np.float64, np.float32, np.int64, np.int32):
                # This column should be numeric, force conversion
                try:
                    original_dtype = df[col_name].dtype
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    log.warning(
                        f"DEBUG: Converted column '{col_name}' from {original_dtype} to {df[col_name].dtype}. "
                        f"Sample values: {df[col_name].head(3).tolist()}"
                    )
                except Exception as e:
                    log.error(f"Could not convert column '{col_name}' to numeric: {e}")
            # else: keep as-is for text/object columns

            # Create a pivot table
            pivot = df.pivot(index='Date', columns='Sid', values=col_name)

            # Reindex to match exactly the requested dates and sids
            # This ensures we have the right shape and fills missing values with NaN
            reindexed = pivot.reindex(index=dates, columns=sids)

            # Convert to numpy array
            arr = reindexed.values

            # Final safety check: if we got an object array but need numeric, convert it
            # This handles edge cases where pivot/reindex creates object dtype despite conversion
            if col_dtype in (np.float64, np.float32, np.int64, np.int32) and arr.dtype == object:
                log.warning(f"Column '{col_name}' has object dtype after pivot, forcing conversion to {col_dtype}")
                try:
                    # Flatten, convert to numeric, reshape, and cast to target dtype
                    flat = arr.flatten()
                    numeric_flat = pd.to_numeric(flat, errors='coerce')
                    arr = numeric_flat.reshape(arr.shape).astype(col_dtype)
                except Exception as e:
                    log.error(f"Failed to convert object array to {col_dtype}: {e}")
                    # Create array of NaN with correct dtype as fallback
                    arr = np.full(arr.shape, np.nan, dtype=col_dtype)

            # Log array dtype for debugging
            log.warning(f"DEBUG: Column '{col_name}' final array dtype: {arr.dtype}, shape: {arr.shape}, sample: {arr.flatten()[:3]}")

            result[col_name] = arr

        return result


def register_custom_loader(engine, dataset_class: Type[DataSet], loader: CustomSQLiteLoader):
    """
    Register a custom data loader with a Pipeline engine.

    Parameters
    ----------
    engine : SimplePipelineEngine
        Pipeline engine
    dataset_class : type
        DataSet class created with make_custom_dataset_class
    loader : CustomSQLiteLoader
        Loader instance

    Examples
    --------
    >>> from zipline.pipeline import Pipeline, SimplePipelineEngine
    >>> from zipline.data.custom import (
    ...     make_custom_dataset_class,
    ...     CustomSQLiteLoader,
    ...     register_custom_loader,
    ... )
    >>>
    >>> # Create DataSet and loader
    >>> FundamentalsDataSet = make_custom_dataset_class('fundamentals', {...})
    >>> loader = CustomSQLiteLoader('fundamentals')
    >>>
    >>> # Register with engine
    >>> engine = SimplePipelineEngine(...)
    >>> register_custom_loader(engine, FundamentalsDataSet, loader)
    >>>
    >>> # Now use in pipeline
    >>> pipeline = Pipeline(columns={'eps': FundamentalsDataSet.EPS.latest})
    >>> results = engine.run_pipeline(pipeline, start_date, end_date)
    """
    # Get all columns from the dataset
    columns = []
    for attr_name in dir(dataset_class):
        attr = getattr(dataset_class, attr_name)
        if hasattr(attr, 'dtype'):  # It's a BoundColumn
            columns.append(attr)

    # Register the loader for each column
    # This depends on the engine implementation
    # For SimplePipelineEngine, we typically pass a get_loader callable
    # For testing, we might directly manipulate loaders dict

    if hasattr(engine, '_loaders'):
        # Direct access to loaders dict (for testing/simple engines)
        for col in columns:
            engine._loaders[col] = loader
        log.info(
            f"Registered {len(columns)} columns from "
            f"{dataset_class.__name__} with engine"
        )
    elif hasattr(engine, 'get_loader'):
        # Engine uses get_loader function
        # We need to wrap our loader
        original_get_loader = engine.get_loader

        def new_get_loader(column):
            if column in columns:
                return loader
            return original_get_loader(column)

        engine.get_loader = new_get_loader
        log.info(
            f"Wrapped get_loader for {dataset_class.__name__} "
            f"({len(columns)} columns)"
        )
    else:
        log.warning(
            "Engine does not have _loaders dict or get_loader method. "
            "Manual registration may be required."
        )
