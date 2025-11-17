"""
Pipeline loader for Sharadar SF1 fundamentals data.

This loader reads quarterly fundamental data from the Sharadar bundle and makes it
available to the Pipeline API with proper point-in-time handling.

Point-in-Time Logic
-------------------
Fundamental data is made available based on the 'datekey' field from Sharadar's SF1
table, which represents the date when the data became publicly available (filing date).
This prevents look-ahead bias by ensuring strategies only access data that was actually
known on each historical date.

For example:
- Q4 2022 earnings (calendardate=2022-12-31) filed on 2023-02-15 (datekey=2023-02-15)
- This data becomes available on 2023-02-15, not 2022-12-31
- The loader will return NaN for requests before 2023-02-15
- The loader will return the Q4 2022 values for requests on or after 2023-02-15

Forward-Filling
---------------
Quarterly fundamentals are forward-filled across trading days until the next quarter's
data becomes available. This means that once Q4 2022 data is filed on 2023-02-15,
it remains the "latest" data until Q1 2023 data is filed (typically ~45 days later).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from zipline.lib.adjusted_array import AdjustedArray
from zipline.utils.numpy_utils import float64_dtype
from .base import PipelineLoader


class SharadarFundamentalsLoader(PipelineLoader):
    """
    Pipeline loader for Sharadar SF1 fundamentals.

    This loader reads fundamentals data stored during bundle ingestion and provides
    it to the Pipeline API with point-in-time correctness.

    Parameters
    ----------
    bundle_path : str
        Path to the bundle directory containing the fundamentals/ subdirectory.
        Typically: ~/.zipline/data/{bundle_name}/{timestamp}/
    """

    def __init__(self, bundle_path):
        """
        Initialize the Sharadar fundamentals loader.

        Parameters
        ----------
        bundle_path : str or Path
            Path to bundle directory containing fundamentals/sf1.h5
        """
        self.bundle_path = Path(bundle_path)
        self.fundamentals_path = self.bundle_path / 'fundamentals' / 'sf1.h5'

        # Cache for loaded data (lazy loading)
        self._data_cache = None
        self._pivot_cache = {}  # Cache pivoted dataframes by column name

    def _load_fundamentals_data(self):
        """
        Load fundamentals data from HDF5 file.

        Returns
        -------
        pd.DataFrame
            Fundamentals data with sid and datekey
        """
        if self._data_cache is not None:
            return self._data_cache

        if not self.fundamentals_path.exists():
            raise FileNotFoundError(
                f"Fundamentals data not found at {self.fundamentals_path}. "
                f"Did you ingest the bundle with include_fundamentals=True?"
            )

        # Load from HDF5
        self._data_cache = pd.read_hdf(self.fundamentals_path, key='sf1')

        # Ensure datekey is datetime
        if 'datekey' in self._data_cache.columns:
            self._data_cache['datekey'] = pd.to_datetime(self._data_cache['datekey'])
        else:
            raise ValueError("Fundamentals data missing required 'datekey' column")

        # Ensure sid column exists
        if 'sid' not in self._data_cache.columns:
            raise ValueError("Fundamentals data missing required 'sid' column")

        return self._data_cache

    def _get_pivoted_data(self, column_name, dates, sids):
        """
        Get pivoted fundamentals data for a specific column with point-in-time handling.

        This method:
        1. Filters SF1 data to the requested column
        2. Pivots data to (datekey, sid) format
        3. Reindexes to all requested (dates, sids)
        4. Forward-fills values (quarterly data persists until next quarter)
        5. Handles missing data with NaN

        Parameters
        ----------
        column_name : str
            Name of the fundamental column (e.g., 'revenue', 'netinc', 'roe')
        dates : pd.DatetimeIndex
            Trading dates for which data is requested
        sids : pd.Int64Index
            SIDs for which data is requested

        Returns
        -------
        np.ndarray
            2D array of shape (len(dates), len(sids)) with point-in-time data
        """
        cache_key = (column_name, tuple(dates), tuple(sids))
        if cache_key in self._pivot_cache:
            return self._pivot_cache[cache_key]

        # Load raw fundamentals data
        df = self._load_fundamentals_data()

        # Check if column exists
        if column_name not in df.columns:
            # Column doesn't exist - return array of NaNs
            result = np.full((len(dates), len(sids)), np.nan, dtype=float64_dtype)
            self._pivot_cache[cache_key] = result
            return result

        # Filter to relevant columns
        subset = df[['sid', 'datekey', column_name]].copy()

        # Remove rows where the column value is NaN (no data available)
        subset = subset[subset[column_name].notna()]

        # If no data at all, return NaNs
        if len(subset) == 0:
            result = np.full((len(dates), len(sids)), np.nan, dtype=float64_dtype)
            self._pivot_cache[cache_key] = result
            return result

        # Handle duplicates: keep the most recent value for each (sid, datekey)
        # This can happen if there are restated earnings
        subset = subset.sort_values('datekey')
        subset = subset.drop_duplicates(subset=['sid', 'datekey'], keep='last')

        # Pivot: rows = datekey, columns = sid, values = metric
        try:
            pivoted = subset.pivot(index='datekey', columns='sid', values=column_name)
        except ValueError as e:
            # If pivot fails due to duplicate index, handle it
            if "duplicate" in str(e).lower():
                # Group by and take last value
                subset = subset.groupby(['datekey', 'sid'])[column_name].last().reset_index()
                pivoted = subset.pivot(index='datekey', columns='sid', values=column_name)
            else:
                raise

        # Ensure index is sorted
        pivoted = pivoted.sort_index()

        # Ensure timezone consistency
        # Pipeline dates are always UTC, but pivoted index might be naive
        if pivoted.index.tz is None:
            pivoted.index = pivoted.index.tz_localize('UTC')
        elif str(pivoted.index.tz) != 'UTC':
            pivoted.index = pivoted.index.tz_convert('UTC')

        # Create complete index combining:
        # 1. All datekeys from the data
        # 2. All requested dates
        # This ensures we have coverage for both filing dates and query dates
        all_dates = pivoted.index.union(dates).sort_values()

        # Reindex to complete date range and all requested SIDs
        pivoted = pivoted.reindex(index=all_dates, columns=sids)

        # Forward-fill within each column (SID)
        # Quarterly data persists until next quarter is filed
        pivoted = pivoted.fillna(method='ffill')

        # Now filter to only the requested dates
        # Use reindex with method='ffill' to handle dates between filings
        result_df = pivoted.reindex(dates, method='ffill')

        # Convert to numpy array
        result = result_df.values.astype(float64_dtype)

        # Cache the result
        self._pivot_cache[cache_key] = result

        return result

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        """
        Load fundamentals data as AdjustedArrays.

        This is the main entry point called by the Pipeline engine.

        Parameters
        ----------
        domain : zipline.pipeline.domain.Domain
            Pipeline domain (e.g., US_EQUITIES)
        columns : list[BoundColumn]
            Columns to load (e.g., [SharadarFundamentals.revenue])
        dates : pd.DatetimeIndex
            Trading dates for which data is requested
        sids : pd.Int64Index
            Asset SIDs for which data is requested
        mask : np.ndarray[bool]
            Boolean mask indicating valid (date, sid) pairs

        Returns
        -------
        dict[BoundColumn -> AdjustedArray]
            Map from each requested column to its data as an AdjustedArray
        """
        # Convert dates to UTC if not already
        if dates.tz is None:
            dates = dates.tz_localize('UTC')
        elif str(dates.tz) != 'UTC':
            dates = dates.tz_convert('UTC')

        # Load data for each requested column
        out = {}
        for column in columns:
            # Get the column name (e.g., 'revenue' from SharadarFundamentals.revenue)
            column_name = column.name

            # Get pivoted data with point-in-time handling
            data = self._get_pivoted_data(column_name, dates, sids)

            # Apply mask: set invalid entries to missing_value
            data = data.copy()
            data[~mask] = column.missing_value

            # Create AdjustedArray (no adjustments needed for fundamentals)
            out[column] = AdjustedArray(
                data=data,
                adjustments={},  # No adjustments for fundamentals
                missing_value=column.missing_value,
            )

        return out


def make_sharadar_fundamentals_loader(bundle_name='sharadar'):
    """
    Factory function to create a SharadarFundamentalsLoader for a specific bundle.

    This function finds the most recent ingestion of the specified bundle and
    creates a loader that reads from it.

    Parameters
    ----------
    bundle_name : str, default 'sharadar'
        Name of the bundle to load fundamentals from

    Returns
    -------
    SharadarFundamentalsLoader
        Configured loader ready to provide fundamentals data

    Raises
    ------
    FileNotFoundError
        If the bundle or fundamentals data cannot be found

    Examples
    --------
    >>> loader = make_sharadar_fundamentals_loader('sharadar')
    >>> # Use in Pipeline:
    >>> from zipline.pipeline import Pipeline
    >>> from zipline.pipeline.data.sharadar import SharadarFundamentals
    >>> pipe = Pipeline(
    ...     columns={'revenue': SharadarFundamentals.revenue.latest},
    ...     loader_overrides={SharadarFundamentals: loader}
    ... )
    """
    from pathlib import Path
    import os

    # Find bundle directory
    zipline_root = Path(os.environ.get('ZIPLINE_ROOT', Path.home() / '.zipline'))
    bundle_dir = zipline_root / 'data' / bundle_name

    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"Bundle '{bundle_name}' not found at {bundle_dir}. "
            f"Have you ingested the bundle?"
        )

    # Find most recent ingestion
    ingestion_dirs = sorted(bundle_dir.glob('*'))
    if not ingestion_dirs:
        raise FileNotFoundError(
            f"No ingestions found for bundle '{bundle_name}' in {bundle_dir}"
        )

    latest_ingestion = ingestion_dirs[-1]

    # Verify fundamentals data exists
    fundamentals_path = latest_ingestion / 'fundamentals' / 'sf1.h5'
    if not fundamentals_path.exists():
        raise FileNotFoundError(
            f"Fundamentals data not found in {latest_ingestion}. "
            f"Re-ingest the bundle with include_fundamentals=True:\n"
            f"  zipline ingest -b {bundle_name}"
        )

    return SharadarFundamentalsLoader(latest_ingestion)
