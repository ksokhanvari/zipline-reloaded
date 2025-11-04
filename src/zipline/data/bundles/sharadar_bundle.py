"""
Sharadar Equity and Fund Prices bundle for Zipline.

This bundle uses NASDAQ Data Link's Sharadar dataset, which provides
high-quality US equity, ETF, and fund pricing data.

Sharadar is a PREMIUM dataset that requires a subscription.
Sign up at: https://data.nasdaq.com/databases/SFA

The bundle supports three main tables:
- SEP (Sharadar Equity Prices): Daily OHLCV pricing data for stocks
- SFP (Sharadar Fund Prices): Daily OHLCV pricing data for ETFs and funds
- ACTIONS: Corporate actions (splits, dividends, etc.)

Example usage:
    from zipline.data.bundles import register
    from zipline.data.bundles.sharadar_bundle import sharadar_bundle

    # Include both equities and funds (default)
    register('sharadar', sharadar_bundle(include_funds=True))

    # Equities only
    register('sharadar-equities', sharadar_bundle(include_funds=False))

    # Then ingest:
    # zipline ingest -b sharadar
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import time
import requests
from io import BytesIO
from zipfile import ZipFile

try:
    import nasdaqdatalink
except ImportError:
    # Fallback to old name
    import quandl as nasdaqdatalink

from zipline.data.bundles import core as bundles


def sharadar_bundle(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    incremental: bool = True,
    use_chunks: bool = False,
    include_funds: bool = True,
):
    """
    Create a zipline data bundle from Sharadar Equity and Fund Prices.

    Parameters
    ----------
    tickers : list of str, optional
        List of ticker symbols to include. If None, downloads all available tickers.
        Note: Downloading all tickers requires significant storage (~10GB+) and time.
        It's recommended to start with a subset.
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, defaults to 1998-01-01
        (beginning of Sharadar historical data).
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, defaults to today.
    api_key : str, optional
        NASDAQ Data Link API key. If None, will use NASDAQ_DATA_LINK_API_KEY
        environment variable.
    incremental : bool, default True
        Enable incremental updates. When True:
        - First ingestion: Downloads all data from start_date to end_date
        - Subsequent ingestions: Only downloads new data since last ingestion
        - Significantly faster for daily updates
        When False: Always downloads full date range (slower but ensures consistency)
    use_chunks : bool, default False
        Download data in 5-year chunks to avoid timeouts on very large date ranges.
        Recommended for initial downloads spanning 15+ years. Slower but more reliable.
    include_funds : bool, default True
        Include fund pricing data (SFP table) in addition to equity prices (SEP table).
        Funds include ETFs, mutual funds, and other investment vehicles.

    Returns
    -------
    callable
        Bundle ingest function for zipline.

    Examples
    --------
    # Daily incremental updates (recommended)
    register('sharadar', sharadar_bundle(incremental=True))

    # All tickers (large download)
    register('sharadar', sharadar_bundle())

    # Specific tickers with incremental updates
    register('sharadar-tech', sharadar_bundle(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        incremental=True,
    ))

    # Force full download (no incremental)
    register('sharadar-full', sharadar_bundle(incremental=False))

    Notes
    -----
    Sharadar is a PREMIUM dataset. You need a paid subscription from:
    https://data.nasdaq.com/databases/SFA

    Incremental updates are highly recommended for daily data refreshes:
    - First run: Downloads all historical data from 1998 (~10-20 minutes)
    - Daily updates: Downloads only yesterday's data (~10-30 seconds)
    """
    # Get API key
    if api_key is None:
        api_key = os.environ.get('NASDAQ_DATA_LINK_API_KEY')
        if api_key is None:
            raise ValueError(
                "NASDAQ Data Link API key required. "
                "Set NASDAQ_DATA_LINK_API_KEY environment variable or pass api_key parameter."
            )

    # Default date range
    if start_date is None:
        # Sharadar data goes back to January 1998
        # Default to getting all available historical data
        start_date = '1998-01-01'

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    def ingest(
        environ,
        asset_db_writer,
        minute_bar_writer,
        daily_bar_writer,
        adjustment_writer,
        calendar,
        start_session,
        end_session,
        cache,
        show_progress,
        output_dir,
    ):
        """
        Ingest Sharadar data into zipline format.
        """
        # Determine effective start date for incremental updates
        effective_start_date = start_date
        is_incremental_update = False

        if incremental:
            # Check for existing bundle data
            from pathlib import Path
            import sqlite3

            # DEBUG: Print paths being searched
            print(f"\n🔍 DEBUG: Checking for previous ingestions...")
            print(f"   Current output_dir: {output_dir}")
            print(f"   Searching in: {Path(output_dir).parent}")
            print(f"   Pattern: */assets-*.sqlite (and */assets-*.db for backwards compatibility)")

            # Look for existing asset database (try both .sqlite and .db extensions)
            existing_ingestions = sorted(Path(output_dir).parent.glob('*/assets-*.sqlite'))
            if not existing_ingestions:
                existing_ingestions = sorted(Path(output_dir).parent.glob('*/assets-*.db'))

            print(f"   Found {len(existing_ingestions)} previous ingestion(s)")
            if existing_ingestions:
                for ing in existing_ingestions:
                    print(f"     - {ing}")
            else:
                # Try alternative detection using parent directory listing
                parent_dir = Path(output_dir).parent
                if parent_dir.exists():
                    print(f"   Parent directory exists, listing contents:")
                    for item in sorted(parent_dir.iterdir()):
                        print(f"     - {item.name}/")
                        if item.is_dir():
                            # Check for both .sqlite and .db extensions
                            assets_dbs = list(item.glob('assets-*.sqlite'))
                            if not assets_dbs:
                                assets_dbs = list(item.glob('assets-*.db'))
                            if assets_dbs:
                                existing_ingestions.extend(assets_dbs)
                                print(f"       Found: {assets_dbs[0].name}")
                    print(f"   Re-checked: Found {len(existing_ingestions)} ingestion(s) total")
                else:
                    print(f"   Parent directory does not exist yet")

            if existing_ingestions:
                latest_db = existing_ingestions[-1]
                print(f"   Using latest database: {latest_db}")
                try:
                    conn = sqlite3.connect(str(latest_db))
                    cursor = conn.cursor()

                    # Get the latest end_date from existing equities
                    print(f"   Querying database for last end_date...")
                    cursor.execute("SELECT MAX(end_date) FROM equities")
                    result = cursor.fetchone()
                    print(f"   Query result: {result}")
                    conn.close()

                    if result and result[0]:
                        # Convert Unix timestamp to date
                        last_date = pd.to_datetime(result[0], unit='s')
                        # Start from the day after last date
                        effective_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                        is_incremental_update = True

                        print(f"\n{'='*60}")
                        print("🔄 INCREMENTAL UPDATE DETECTED")
                        print(f"{'='*60}")
                        print(f"Last ingestion ended: {last_date.date()}")
                        print(f"Downloading new data from: {effective_start_date} to {end_date}")
                        print(f"This will be much faster than a full download!")
                        print(f"{'='*60}\n")
                    else:
                        print(f"   ⚠️  Database query returned no results or NULL")
                        print(f"   Performing full download from {start_date}")
                except Exception as e:
                    print(f"  ℹ️  Could not detect previous ingestion: {e}")
                    print(f"     Exception type: {type(e).__name__}")
                    import traceback
                    print(f"     {traceback.format_exc()}")
                    print(f"  Performing full download from {start_date}")
            else:
                print(f"   No previous ingestions found, will perform full download")

        if not is_incremental_update:
            print(f"\n{'='*60}")
            print("Sharadar Bundle Ingestion")
            print(f"{'='*60}")
            print(f"Date range: {effective_start_date} to {end_date}")
            if tickers:
                print(f"Tickers: {len(tickers)} symbols")
            else:
                print("Tickers: ALL (this may take a while and use significant storage)")
            print(f"{'='*60}\n")

        # Download SEP (equity pricing) data
        total_steps = 4 if include_funds else 3
        if is_incremental_update:
            print(f"Step 1/{total_steps}: Downloading NEW Sharadar Equity Prices (incremental)...")
        else:
            print(f"Step 1/{total_steps}: Downloading Sharadar Equity Prices (SEP table)...")

        sep_data = download_sharadar_table(
            table='SEP',
            api_key=api_key,
            tickers=tickers,
            start_date=effective_start_date,
            end_date=end_date,
            use_chunks=use_chunks,
        )

        if sep_data.empty:
            raise ValueError("No equity pricing data downloaded. Check your subscription and filters.")

        print(f"Downloaded {len(sep_data):,} equity price records for {sep_data['ticker'].nunique()} tickers")

        # Download SFP (fund pricing) data if requested
        sfp_data = pd.DataFrame()
        if include_funds:
            if is_incremental_update:
                print(f"\nStep 2/{total_steps}: Downloading NEW Sharadar Fund Prices (incremental)...")
            else:
                print(f"\nStep 2/{total_steps}: Downloading Sharadar Fund Prices (SFP table)...")

            sfp_data = download_sharadar_table(
                table='SFP',
                api_key=api_key,
                tickers=tickers,
                start_date=effective_start_date,
                end_date=end_date,
                use_chunks=use_chunks,
            )

            if not sfp_data.empty:
                print(f"Downloaded {len(sfp_data):,} fund price records for {sfp_data['ticker'].nunique()} tickers")
            else:
                print("No fund pricing data available (this is normal if you only have equity tickers)")

        # Combine equity and fund data
        if include_funds and not sfp_data.empty:
            # Mark data source for each record
            sep_data['asset_type'] = 'equity'
            sfp_data['asset_type'] = 'fund'

            # Combine
            all_pricing_data = pd.concat([sep_data, sfp_data], ignore_index=True)
            print(f"\nCombined {len(all_pricing_data):,} total price records ({len(sep_data):,} equities + {len(sfp_data):,} funds)")
        else:
            sep_data['asset_type'] = 'equity'
            all_pricing_data = sep_data

        # Validate date range to match available data
        actual_start = all_pricing_data['date'].min()
        actual_end = all_pricing_data['date'].max()
        print(f"Downloaded data range: {actual_start.date()} to {actual_end.date()}")

        # Download ACTIONS (corporate actions) data
        step_num = 3 if include_funds else 2
        if is_incremental_update:
            print(f"\nStep {step_num}/{total_steps}: Downloading NEW corporate actions (incremental)...")
        else:
            print(f"\nStep {step_num}/{total_steps}: Downloading corporate actions (ACTIONS table)...")

        actions_data = download_sharadar_table(
            table='ACTIONS',
            api_key=api_key,
            tickers=tickers,
            start_date=effective_start_date,
            end_date=end_date,
            use_chunks=use_chunks,
        )

        if not actions_data.empty:
            print(f"Downloaded {len(actions_data):,} corporate action records")
            # Filter actions to match the actual data range
            actions_data = actions_data[actions_data['date'] <= actual_end]
            print(f"Filtered to {len(actions_data):,} actions within data range")
        else:
            print("No corporate actions data available")

        # Process data
        step_num = 4 if include_funds else 3
        print(f"\nStep {step_num}/{total_steps}: Processing data for zipline...")
        print(f"Using date range: {actual_start.date()} to {actual_end.date()}")

        # Prepare metadata first (to get sid assignments)
        print("Preparing asset metadata...")
        # Use actual dates from the data to ensure consistency
        metadata = prepare_asset_metadata(
            all_pricing_data,
            actual_start.strftime('%Y-%m-%d'),
            actual_end.strftime('%Y-%m-%d')
        )

        # Create symbol to sid mapping
        symbol_to_sid = {row['symbol']: idx for idx, row in metadata.iterrows()}

        # Add sid to pricing data
        all_pricing_data['sid'] = all_pricing_data['ticker'].map(symbol_to_sid)

        # Write metadata
        print("Writing asset metadata...")
        asset_db_writer.write(equities=metadata)

        # Prepare and write pricing data
        print("Writing daily bars...")

        # Get trading calendar for the full date range to handle missing sessions
        calendar_sessions = calendar.sessions_in_range(
            pd.Timestamp(actual_start),
            pd.Timestamp(actual_end)
        )

        def data_generator():
            """Generator that yields (sid, dataframe) for each symbol"""
            for sid in sorted(all_pricing_data['sid'].unique()):
                if pd.isna(sid):
                    continue  # Skip any unmapped tickers

                symbol_data = all_pricing_data[all_pricing_data['sid'] == sid].copy()

                # Set date as index
                symbol_data = symbol_data.set_index('date')

                # Ensure timezone-naive
                if symbol_data.index.tz is not None:
                    symbol_data.index = symbol_data.index.tz_localize(None)

                # Use closeunadj for close (adjustments handled separately)
                symbol_data['close'] = symbol_data['closeunadj']

                # Ensure all required columns and proper types
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                symbol_data = symbol_data[required_cols].copy()

                # Convert to float
                for col in required_cols:
                    symbol_data[col] = symbol_data[col].astype(float)

                # Remove any NaN rows
                symbol_data = symbol_data.dropna()

                # Sort by date
                symbol_data = symbol_data.sort_index()

                # Get this symbol's actual date range
                symbol_start = symbol_data.index.min()
                symbol_end = symbol_data.index.max()

                # Get trading days for this symbol's range
                symbol_sessions = calendar.sessions_in_range(symbol_start, symbol_end)

                # Reindex to include all trading days, forward-fill gaps
                symbol_data = symbol_data.reindex(symbol_sessions, method='ffill')

                # Drop any remaining NaN rows (at the start before first real data)
                symbol_data = symbol_data.dropna()

                yield int(sid), symbol_data

        daily_bar_writer.write(data_generator(), show_progress=show_progress)

        # Prepare and write adjustments
        print("Writing adjustments...")
        adjustments = prepare_adjustments(actions_data, symbol_to_sid)
        adjustment_writer.write(**adjustments)

        print(f"\n{'='*60}")
        print("✓ Sharadar bundle ingestion complete!")
        print(f"{'='*60}\n")

    return ingest


def download_sharadar_table(
    table: str,
    api_key: str,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_chunks: bool = False,
) -> pd.DataFrame:
    """
    Download a Sharadar table using hybrid approach.

    Uses nasdaqdatalink.get_table() for filtered queries (faster with optimized columns),
    and falls back to bulk export API for large/unfiltered datasets.

    Parameters
    ----------
    table : str
        Table name (e.g., 'SEP', 'ACTIONS', 'SF1', 'TICKERS')
    api_key : str
        NASDAQ Data Link API key
    tickers : list of str, optional
        Filter by tickers
    start_date : str, optional
        Start date filter
    end_date : str, optional
        End date filter
    use_chunks : bool, optional
        If True, download in 5-year chunks to avoid timeouts (slower but more reliable)

    Returns
    -------
    pd.DataFrame
        Downloaded data
    """
    # Set API key
    nasdaqdatalink.ApiConfig.api_key = api_key

    # Handle chunked downloads for very large date ranges
    if use_chunks and start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Calculate number of years
        years_span = (end_dt - start_dt).days / 365.25

        if years_span > 10:  # Only chunk if more than 10 years
            print(f"  Using chunked download (5-year intervals) for {years_span:.1f} years of data...")
            return download_sharadar_table_chunked(
                table=table,
                api_key=api_key,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )

    # Define columns based on table type
    if table == 'SEP':
        # SEP (Sharadar Equity Prices) - optimized column selection
        columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'closeadj', 'closeunadj']
        qopts = {"columns": columns}
    elif table == 'SFP':
        # SFP (Sharadar Fund Prices) - same structure as SEP
        columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'closeadj', 'closeunadj']
        qopts = {"columns": columns}
    elif table == 'ACTIONS':
        # ACTIONS table - corporate actions (splits, dividends)
        columns = ['ticker', 'date', 'action', 'value']
        qopts = {"columns": columns}
    else:
        # For other tables, don't specify columns (get all)
        qopts = None

    # Try get_table() first (works well for filtered queries)
    try:
        print(f"  Downloading {table} table using nasdaqdatalink.get_table()...")

        # Build filter conditions
        filters = {}
        if tickers:
            filters['ticker'] = tickers
        if start_date:
            filters['date.gte'] = start_date
        if end_date:
            filters['date.lte'] = end_date

        # Download data using nasdaqdatalink
        if qopts:
            df = nasdaqdatalink.get_table(
                f'SHARADAR/{table}',
                qopts=qopts,
                **filters,
                paginate=True  # Automatically handle pagination
            )
        else:
            df = nasdaqdatalink.get_table(
                f'SHARADAR/{table}',
                **filters,
                paginate=True
            )

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        print(f"  ✓ Downloaded {len(df):,} records from {table}")

        # Add closeunadj if not present (for backwards compatibility)
        if table in ('SEP', 'SFP') and 'closeunadj' not in df.columns and 'close' in df.columns:
            df['closeunadj'] = df['close']

        return df

    except Exception as e:
        # Check if it's a data limit error
        error_msg = str(e).lower()
        if 'limit' in error_msg or 'exceeds' in error_msg:
            print(f"  ⚠️  Dataset too large for get_table(), using bulk export API...")
            return download_sharadar_bulk_export(table, api_key, tickers, start_date, end_date, qopts)
        else:
            print(f"  ❌ Error downloading {table}: {str(e)}")
            print(f"     This may indicate an API key issue or subscription problem.")
            print(f"     Verify your subscription at: https://data.nasdaq.com/databases/SFA")
            raise


def download_sharadar_table_chunked(
    table: str,
    api_key: str,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chunk_years: int = 5,
) -> pd.DataFrame:
    """
    Download Sharadar table in date-range chunks to avoid timeouts.

    This function splits large date ranges into smaller chunks (default 5 years)
    and downloads each chunk separately, then concatenates the results.

    Parameters
    ----------
    table : str
        Table name
    api_key : str
        NASDAQ Data Link API key
    tickers : list of str, optional
        Filter by tickers
    start_date : str, optional
        Start date
    end_date : str, optional
        End date
    chunk_years : int, default 5
        Number of years per chunk

    Returns
    -------
    pd.DataFrame
        Combined data from all chunks
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Generate date ranges for chunks
    chunks = []
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + pd.DateOffset(years=chunk_years), end_dt)
        chunks.append((
            current_start.strftime('%Y-%m-%d'),
            current_end.strftime('%Y-%m-%d')
        ))
        current_start = current_end + pd.DateOffset(days=1)

    print(f"  Downloading {len(chunks)} chunks ({chunk_years} years each)...")

    # Download each chunk
    all_data = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"\n  Chunk {i}/{len(chunks)}: {chunk_start} to {chunk_end}")

        chunk_data = download_sharadar_table(
            table=table,
            api_key=api_key,
            tickers=tickers,
            start_date=chunk_start,
            end_date=chunk_end,
            use_chunks=False  # Prevent recursion
        )

        if not chunk_data.empty:
            all_data.append(chunk_data)
            print(f"  ✓ Chunk {i}: {len(chunk_data):,} records")
        else:
            print(f"  ⚠️  Chunk {i}: No data")

    # Combine all chunks
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Remove duplicates (if any at chunk boundaries)
        if 'ticker' in combined_df.columns and 'date' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['ticker', 'date'], keep='first')
        print(f"\n  ✓ Combined {len(combined_df):,} total records from all chunks")
        return combined_df
    else:
        return pd.DataFrame()


def download_sharadar_bulk_export(
    table: str,
    api_key: str,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    qopts: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Download Sharadar table using bulk export API.

    This method is used for large datasets that exceed get_table() limits.
    Uses the export=true parameter to download complete datasets as ZIP files.

    Parameters
    ----------
    table : str
        Table name
    api_key : str
        NASDAQ Data Link API key
    tickers : list of str, optional
        Filter by tickers
    start_date : str, optional
        Start date filter
    end_date : str, optional
        End date filter
    qopts : dict, optional
        Query options including column selection

    Returns
    -------
    pd.DataFrame
        Downloaded data
    """
    print(f"  Using bulk export API for {table} table...")

    # Build URL for bulk download
    base_url = f'https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json'
    params = {
        'qopts.export': 'true',
        'api_key': api_key,
    }

    # Add column selection if provided
    if qopts and 'columns' in qopts:
        params['qopts.columns'] = ','.join(qopts['columns'])

    # Add filters if provided
    if tickers:
        params['ticker'] = ','.join(tickers)
    if start_date:
        params['date.gte'] = start_date
    if end_date:
        params['date.lte'] = end_date

    # Request bulk download
    print(f"  Requesting bulk export...")
    response = requests.get(base_url, params=params)
    response.raise_for_status()

    result = response.json()

    # Check if bulk download is available
    if 'datatable_bulk_download' not in result:
        raise ValueError(f"Bulk export not available for {table} table")

    bulk_info = result['datatable_bulk_download']

    # Get file status and link
    file_status = bulk_info['file']['status']
    file_link = bulk_info['file']['link']

    # Wait for file if needed
    valid_statuses = ['fresh', 'regenerating']  # File is ready
    wait_statuses = ['generating', 'creating']  # Need to wait
    max_wait_seconds = 7200  # 2 hours (large datasets from 1998 take 60-90 min)
    waited_seconds = 0
    check_interval = 60

    if file_status in wait_statuses:
        print(f"  ⏳ File is being generated by NASDAQ servers...")
        print(f"     This can take 60-90 minutes for full historical data (1998-present)")
        print(f"     Maximum wait time: {max_wait_seconds // 60} minutes")
        print(f"     Status will be checked every {check_interval} seconds\n")

    while file_status in wait_statuses and waited_seconds < max_wait_seconds:
        time.sleep(check_interval)
        waited_seconds += check_interval

        # Check status again
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        result = response.json()

        bulk_info = result['datatable_bulk_download']
        file_status = bulk_info['file']['status']
        file_link = bulk_info['file']['link']

        minutes_elapsed = waited_seconds // 60
        hours_elapsed = waited_seconds // 3600
        if hours_elapsed > 0:
            print(f"  ⏳ Status: {file_status} ({hours_elapsed}h {minutes_elapsed % 60}m elapsed)")
        else:
            print(f"  ⏳ Status: {file_status} ({minutes_elapsed} min elapsed)")

    # Check if file is ready
    if file_status not in valid_statuses:
        hours_waited = waited_seconds / 3600
        raise RuntimeError(
            f"Bulk download file not ready after {hours_waited:.1f} hours.\n"
            f"Final status: {file_status}\n\n"
            f"The NASDAQ servers are taking longer than expected to prepare your dataset.\n"
            f"Options:\n"
            f"1. Try again in a few minutes - the file might be ready now\n"
            f"2. Download a smaller date range first (e.g., last 5 years)\n"
            f"3. Contact NASDAQ Data Link support if the issue persists\n"
            f"4. Manually download from: {file_link}"
        )

    # Download the file
    print(f"  ✓ File ready, downloading...")
    download_response = requests.get(file_link)
    download_response.raise_for_status()

    # Extract from ZIP
    print(f"  Extracting data from ZIP...")
    with ZipFile(BytesIO(download_response.content)) as zf:
        csv_filename = zf.namelist()[0]
        with zf.open(csv_filename) as csv_file:
            df = pd.read_csv(csv_file, parse_dates=['date'])

    print(f"  ✓ Downloaded {len(df):,} records from {table}")

    # Add closeunadj if not present (for backwards compatibility)
    if table in ('SEP', 'SFP') and 'closeunadj' not in df.columns and 'close' in df.columns:
        df['closeunadj'] = df['close']

    return df


def prepare_asset_metadata(
    pricing_data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Prepare asset metadata for zipline.

    Parameters
    ----------
    pricing_data : pd.DataFrame
        Raw pricing data (SEP and/or SFP)
    start_date : str
        Bundle start date
    end_date : str
        Bundle end date

    Returns
    -------
    pd.DataFrame
        Asset metadata
    """
    # Determine aggregation columns
    agg_dict = {'date': ['min', 'max']}

    # Include asset_type if present (to distinguish equities from funds)
    if 'asset_type' in pricing_data.columns:
        agg_dict['asset_type'] = 'first'

    # Get first and last date for each ticker
    metadata = pricing_data.groupby('ticker').agg(agg_dict).reset_index()

    # Flatten column names
    if 'asset_type' in pricing_data.columns:
        metadata.columns = ['symbol', 'start_date', 'end_date', 'asset_type']
    else:
        metadata.columns = ['symbol', 'start_date', 'end_date']
        metadata['asset_type'] = 'equity'  # Default to equity if not specified

    # Add required columns
    metadata['exchange'] = 'NASDAQ'  # Sharadar is primarily NASDAQ/NYSE
    metadata['asset_name'] = metadata['symbol']

    # Convert dates to pandas Timestamp
    metadata['start_date'] = pd.to_datetime(metadata['start_date'])
    metadata['end_date'] = pd.to_datetime(metadata['end_date'])

    # Ensure dates are within bundle range
    bundle_start = pd.Timestamp(start_date)
    bundle_end = pd.Timestamp(end_date)

    metadata['start_date'] = metadata['start_date'].clip(lower=bundle_start)
    metadata['end_date'] = metadata['end_date'].clip(upper=bundle_end)

    return metadata


def prepare_adjustments(
    actions_data: pd.DataFrame,
    ticker_to_sid: dict,
) -> dict:
    """
    Prepare splits and dividends from ACTIONS table.

    Parameters
    ----------
    actions_data : pd.DataFrame
        Corporate actions data
    ticker_to_sid : dict
        Mapping of ticker symbols to sid integers

    Returns
    -------
    dict
        Dictionary with 'splits' and 'dividends' DataFrames
    """
    if actions_data.empty:
        # Return empty adjustments
        return {
            'splits': pd.DataFrame(columns=['sid', 'ratio', 'effective_date']),
            'dividends': pd.DataFrame(columns=['sid', 'amount', 'ex_date', 'record_date', 'declared_date', 'pay_date']),
        }

    # Process splits
    splits = actions_data[actions_data['action'] == 'Split'].copy()
    if not splits.empty:
        splits['sid'] = splits['ticker'].map(ticker_to_sid)
        splits['ratio'] = splits['value'].astype(float)
        splits['effective_date'] = pd.to_datetime(splits['date'])
        splits = splits[['sid', 'ratio', 'effective_date']].dropna()
    else:
        splits = pd.DataFrame(columns=['sid', 'ratio', 'effective_date'])

    # Process dividends
    dividends = actions_data[actions_data['action'] == 'Dividend'].copy()
    if not dividends.empty:
        dividends['sid'] = dividends['ticker'].map(ticker_to_sid)
        dividends['amount'] = dividends['value'].astype(float)
        dividends['ex_date'] = pd.to_datetime(dividends['date'])
        # Sharadar doesn't provide record/declared/pay dates, use ex_date for all
        dividends['record_date'] = dividends['ex_date']
        dividends['declared_date'] = dividends['ex_date']
        dividends['pay_date'] = dividends['ex_date']
        dividends = dividends[['sid', 'amount', 'ex_date', 'record_date', 'declared_date', 'pay_date']].dropna()
    else:
        dividends = pd.DataFrame(columns=['sid', 'amount', 'ex_date', 'record_date', 'declared_date', 'pay_date'])

    return {
        'splits': splits,
        'dividends': dividends,
    }


# Pre-configured bundle variants
def sharadar_tech_bundle():
    """Sharadar bundle with major tech stocks."""
    return sharadar_bundle(
        tickers=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            'INTC', 'AMD', 'ORCL', 'CSCO', 'AVGO',
        ],
    )


def sharadar_sp500_sample_bundle():
    """Sharadar bundle with S&P 500 sample (top 30 by market cap)."""
    return sharadar_bundle(
        tickers=[
            # Top tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK.B', 'V', 'UNH',
            # Financials
            'JPM', 'JNJ', 'WMT', 'MA', 'PG',
            'XOM', 'CVX', 'HD', 'MRK', 'ABBV',
            # Others
            'PFE', 'KO', 'PEP', 'COST', 'AVGO',
            'TMO', 'DIS', 'CSCO', 'ABT', 'ACN',
        ],
    )


def sharadar_all_bundle():
    """
    Sharadar bundle with ALL tickers.

    WARNING: This downloads ALL US equities (~8,000+ tickers).
    - Download time: 10-30 minutes
    - Storage: 10-20 GB
    - Recommended for production use only.
    """
    return sharadar_bundle(tickers=None)  # None = all tickers


# Register default bundles
def register_sharadar_bundles():
    """Register common Sharadar bundle configurations."""
    from zipline.data.bundles import register

    register('sharadar-tech', sharadar_tech_bundle())
    register('sharadar-sp500', sharadar_sp500_sample_bundle())
    register('sharadar-all', sharadar_all_bundle())
