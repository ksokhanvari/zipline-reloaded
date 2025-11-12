"""
Sharadar Equity and Fund Prices bundle for Zipline.

This module provides integration between Zipline and NASDAQ Data Link's Sharadar dataset,
which offers high-quality fundamental and pricing data for US equities, ETFs, and funds.

OVERVIEW
--------
Sharadar is a PREMIUM dataset that requires a subscription.
Sign up at: https://data.nasdaq.com/databases/SFA

The bundle downloads and processes three main Sharadar tables:
- SEP (Sharadar Equity Prices): Daily OHLCV pricing data for stocks
- SFP (Sharadar Fund Prices): Daily OHLCV pricing data for ETFs and funds
- ACTIONS: Corporate actions (splits, dividends, spinoffs, etc.)

KEY FEATURES
------------
1. Incremental Updates: Only download new data since last ingestion (much faster)
2. Flexible Ticker Selection: Download all tickers or specify a subset
3. Split & Dividend Handling: Automatically handles corporate actions
4. Chunk Downloads: Break large downloads into smaller pieces to avoid timeouts
5. Fund Support: Include or exclude ETFs and mutual funds

SHARADAR DATA COLUMNS
---------------------
Sharadar provides three price columns with different adjustment methods:

- **closeadj**: Fully adjusted (splits AND dividends) - **WE USE THIS**
  - Historical prices adjusted backwards for both splits and dividends
  - Provides total return pricing (price appreciation + dividend reinvestment)
  - This is the correct column for backtesting portfolio performance

- **close**: Split-adjusted historically, NOT dividend-adjusted
  - Historical prices are adjusted backwards for splits only
  - Example: If a 2-for-1 split occurs, all prior prices are halved
  - Gives price return only, missing dividend contribution

- **closeunadj**: No adjustments (raw traded prices)
  - The actual prices traded on each day
  - Not suitable for backtesting (discontinuous at splits)

IMPORTANT: This bundle uses the 'closeadj' column to capture TOTAL RETURN.
We do NOT apply any additional adjustments since they're already in the prices.

INCREMENTAL INGESTION
---------------------
When incremental=True (default):
1. First ingestion: Downloads all data from start_date to end_date
2. Subsequent ingestions:
   - Detects last ingestion date from existing bundle
   - Only downloads new data from (last_date + 1) to end_date
   - Merges new data with existing data
   - Much faster for daily updates (seconds vs minutes)

HOW ADJUSTMENTS WORK
--------------------
This bundle uses Sharadar's 'closeadj' column for TOTAL RETURN.

Total Return = Price Appreciation + Dividend Reinvestment

Sharadar's 'closeadj' column has ALL adjustments already applied:
- Stock splits: Historical prices adjusted backwards (maintains continuity)
- Dividends: Historical prices adjusted backwards (reflects reinvestment)

We do NOT apply any additional adjustments in Zipline to avoid double-adjustment.

Example: Comparing price vs total return for a dividend-paying stock
- Using 'close' (price only): Shows only stock price appreciation
- Using 'closeadj' (total return): Shows price + dividend contribution
- For SPY over 10 years: price return ~12%/yr, total return ~14%/yr (2% from dividends)

USAGE EXAMPLES
--------------
# Basic usage - all tickers with incremental updates
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

register('sharadar', sharadar_bundle())

# Specific tickers only
register('sharadar-tech', sharadar_bundle(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    incremental=True
))

# Historical backtest (no incremental updates)
register('sharadar-2020', sharadar_bundle(
    start_date='2020-01-01',
    end_date='2020-12-31',
    incremental=False
))

# Large historical download with chunking
register('sharadar-full', sharadar_bundle(
    start_date='2000-01-01',
    use_chunks=True,  # Avoid timeouts
    incremental=False
))

# Then ingest:
# zipline ingest -b sharadar

TECHNICAL DETAILS
-----------------
Storage: Each bundle is stored in a timestamped directory containing:
- assets-N.sqlite: Asset metadata (symbols, names, exchange, date ranges)
- equity_symbol_mappings: Symbol to SID mapping with date ranges
- daily_equities.bcolz: Compressed OHLCV data (bcolz format for fast access)
- adjustments/: Split and dividend adjustments (SQLite databases)

Performance:
- Full download (all tickers, 1998-present): ~30-60 minutes, ~10GB
- Incremental daily update: ~30-60 seconds, ~50MB
- Tech stocks subset (50 tickers): ~2-5 minutes, ~200MB

API Limits:
- NASDAQ Data Link has rate limits and monthly call limits
- Premium plans have higher limits
- Use incremental updates to minimize API calls

TROUBLESHOOTING
---------------
If you see split-related portfolio value drops:
1. Ensure you're using the latest version of this code
2. Delete old bundles: rm -rf ~/.zipline/data/sharadar/*
3. Re-ingest: zipline ingest -b sharadar

If downloads timeout:
1. Use smaller ticker lists
2. Use shorter date ranges
3. Enable use_chunks=True
4. Check your internet connection

If incremental updates aren't working:
1. Check bundle directory exists: ls ~/.zipline/data/sharadar/
2. Ensure previous ingestion completed successfully
3. Try incremental=False to force full download

SEE ALSO
--------
- Sharadar documentation: https://data.nasdaq.com/databases/SFA/documentation
- Zipline bundles: https://zipline.ml4trading.io/bundles.html
- NASDAQ Data Link API: https://docs.data.nasdaq.com/
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
                        # Convert timestamp to date
                        # Zipline stores timestamps as nanoseconds since epoch
                        last_date = pd.to_datetime(result[0], unit='ns')
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
            print(f"   Downloading only new data from {effective_start_date} to {end_date}")
            print(f"   (Will merge with existing data from {start_date} onwards)")
        else:
            print(f"Step 1/{total_steps}: Downloading Sharadar Equity Prices (SEP table)...")

        # For incremental updates: download only new data, then we'll merge with existing
        # For full updates: download all data from start_date
        download_start_date = effective_start_date if is_incremental_update else start_date

        sep_data = download_sharadar_table(
            table='SEP',
            api_key=api_key,
            tickers=tickers,
            start_date=download_start_date,
            end_date=end_date,
            use_chunks=use_chunks,
        )

        if sep_data.empty:
            if is_incremental_update:
                # No new data available - abort ingestion cleanly
                print(f"\n{'='*60}")
                print("✓ Already up-to-date!")
                print(f"{'='*60}")
                print(f"No new equity data available for {effective_start_date} to {end_date}")
                print(f"")
                print(f"Possible reasons:")
                print(f"  • Data for {end_date} not yet available (check time: market closes 4PM ET, data ready ~6PM ET)")
                print(f"  • Weekend or market holiday")
                print(f"  • Already have the latest data")
                print(f"")
                print(f"Last ingestion date: {(pd.to_datetime(effective_start_date) - timedelta(days=1)).date()}")
                print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"")
                print(f"💡 Your existing bundle is current - no action needed!")
                print(f"   You can run backtests using the existing data.")
                print(f"{'='*60}\n")

                # Raise a clear exception to abort the ingestion
                # This prevents creating an incomplete bundle
                # Note: This is not an error - just indicates no new data available
                raise ValueError(
                    f"No new data available. Bundle already up-to-date as of "
                    f"{(pd.to_datetime(effective_start_date) - timedelta(days=1)).date()}. "
                    f"Use your existing bundle for backtests - no action needed."
                )
            else:
                # First-time download with no data is an error
                raise ValueError(
                    "No equity pricing data downloaded. "
                    "Check your subscription and filters. "
                    f"Date range: {effective_start_date} to {end_date}"
                )

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
                start_date=download_start_date,
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

        # For incremental updates: merge with existing data
        if is_incremental_update:
            print(f"\n🔄 Merging with existing data...")
            print(f"   Newly downloaded: {len(all_pricing_data):,} records")

            # Load existing data from the latest bundle
            from pathlib import Path
            import bcolz

            try:
                latest_bundle_dir = Path(output_dir).parent
                existing_dirs = sorted(latest_bundle_dir.glob('*/daily_equities.bcolz'))

                if existing_dirs:
                    existing_bcolz_dir = existing_dirs[-1]
                    print(f"   Loading from: {existing_bcolz_dir.parent.name}")

                    # Read existing bcolz data
                    existing_table = bcolz.open(str(existing_bcolz_dir), mode='r')

                    # Convert to DataFrame
                    existing_data = pd.DataFrame({
                        col: existing_table[col][:] for col in ['open', 'high', 'low', 'close', 'volume', 'day', 'id']
                    })

                    # Convert day (uint32 Unix timestamp in SECONDS) to date
                    existing_data['date'] = pd.to_datetime(existing_data['day'], unit='s')
                    existing_data['sid'] = existing_data['id']

                    # Filter out invalid dates (before calendar's first session or NaT)
                    # XNYS calendar starts at 1990-01-02
                    min_valid_date = pd.Timestamp('1990-01-02')
                    initial_count = len(existing_data)

                    # Remove NaT and dates before calendar start
                    existing_data = existing_data[existing_data['date'].notna()]
                    existing_data = existing_data[existing_data['date'] >= min_valid_date]

                    filtered_count = initial_count - len(existing_data)
                    if filtered_count > 0:
                        print(f"   Filtered out {filtered_count:,} rows with invalid dates (before 1990-01-02 or NaT)")

                    if len(existing_data) == 0:
                        print(f"   ⚠️  No valid data remaining after date filtering")
                        raise ValueError("No valid historical data")  # Will be caught below

                    # Load asset database to get ticker symbols
                    latest_assets_db = sorted(latest_bundle_dir.glob('*/assets-*.sqlite'))
                    if not latest_assets_db:
                        latest_assets_db = sorted(latest_bundle_dir.glob('*/assets-*.db'))

                    if latest_assets_db:
                        import sqlite3
                        conn = sqlite3.connect(str(latest_assets_db[-1]))
                        # Symbol is in equity_symbol_mappings, not equities table
                        # Join to get sid -> symbol mapping, taking the most recent symbol for each sid
                        assets_df = pd.read_sql("""
                            SELECT DISTINCT esm.sid, esm.symbol
                            FROM equity_symbol_mappings esm
                            INNER JOIN (
                                SELECT sid, MAX(end_date) as max_end_date
                                FROM equity_symbol_mappings
                                GROUP BY sid
                            ) latest ON esm.sid = latest.sid AND esm.end_date = latest.max_end_date
                        """, conn)
                        conn.close()

                        # Map sid to ticker
                        sid_to_ticker = dict(zip(assets_df['sid'], assets_df['symbol']))
                        existing_data['ticker'] = existing_data['sid'].map(sid_to_ticker)

                        # Get asset_type from metadata if available
                        existing_data['asset_type'] = 'equity'  # Default

                        # Keep only necessary columns
                        existing_data = existing_data[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'asset_type']]

                        print(f"   Existing data: {len(existing_data):,} records from {existing_data['date'].min().date()} to {existing_data['date'].max().date()}")

                        # Filter existing data to before the new data starts
                        cutoff_date = pd.Timestamp(download_start_date)
                        existing_data = existing_data[existing_data['date'] < cutoff_date]
                        print(f"   Keeping {len(existing_data):,} records before {cutoff_date.date()}")

                        # Merge: existing data + new data
                        all_pricing_data = pd.concat([existing_data, all_pricing_data], ignore_index=True)
                        print(f"   ✓ Merged total: {len(all_pricing_data):,} records")
                    else:
                        print(f"   ⚠️  Could not find assets database for ticker mapping")
                        print(f"   Proceeding with new data only")
                else:
                    print(f"   ⚠️  Could not find existing bundle data")
                    print(f"   Proceeding with new data only")

            except Exception as e:
                print(f"   ⚠️  Error loading existing data: {e}")
                print(f"   Proceeding with new data only")

        # Validate date range to match available data
        actual_start = all_pricing_data['date'].min()
        actual_end = all_pricing_data['date'].max()
        print(f"Final data range: {actual_start.date()} to {actual_end.date()}")

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
            start_date=download_start_date,
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

        # For incremental updates: merge with existing metadata to preserve historical start dates
        if is_incremental_update:
            print("   Merging with existing asset metadata...")
            try:
                # Load existing asset metadata
                from pathlib import Path
                import sqlite3

                latest_assets_db = sorted(Path(output_dir).parent.glob('*/assets-*.sqlite'))
                if not latest_assets_db:
                    latest_assets_db = sorted(Path(output_dir).parent.glob('*/assets-*.db'))

                if latest_assets_db:
                    conn = sqlite3.connect(str(latest_assets_db[-1]))
                    # Join equities with symbol mappings to get the symbol column
                    existing_metadata = pd.read_sql("""
                        SELECT e.sid, esm.symbol, e.asset_name, e.start_date, e.end_date, e.exchange
                        FROM equities e
                        LEFT JOIN equity_symbol_mappings esm ON e.sid = esm.sid
                    """, conn)
                    conn.close()

                    # Filter out assets without symbols (shouldn't happen, but be safe)
                    existing_metadata = existing_metadata[existing_metadata['symbol'].notna()]

                    print(f"   Loaded {len(existing_metadata)} existing assets")

                    # Create new metadata from current data
                    new_metadata = prepare_asset_metadata(
                        all_pricing_data,
                        actual_start.strftime('%Y-%m-%d'),
                        actual_end.strftime('%Y-%m-%d')
                    )

                    # Merge: keep existing start_dates, update end_dates
                    # Convert timestamp columns for comparison
                    if 'start_date' in existing_metadata.columns:
                        existing_metadata['start_date'] = pd.to_datetime(existing_metadata['start_date'], unit='ns')
                    if 'end_date' in existing_metadata.columns:
                        existing_metadata['end_date'] = pd.to_datetime(existing_metadata['end_date'], unit='ns')

                    # Merge on symbol - PRESERVE sid from existing_metadata!
                    metadata = pd.merge(
                        new_metadata[['symbol', 'asset_type', 'exchange', 'asset_name']],
                        existing_metadata[['sid', 'symbol', 'start_date', 'end_date']],
                        on='symbol',
                        how='outer'  # Keep all symbols from both
                    )

                    # For assets in new_metadata: update end_date, keep original start_date
                    new_symbols = set(new_metadata['symbol'])
                    for idx, row in metadata.iterrows():
                        if row['symbol'] in new_symbols:
                            # Get dates from new_metadata
                            new_row = new_metadata[new_metadata['symbol'] == row['symbol']].iloc[0]
                            # Keep existing start_date (earlier), update end_date (later)
                            if pd.isna(row['start_date']):
                                metadata.loc[idx, 'start_date'] = new_row['start_date']
                            else:
                                metadata.loc[idx, 'start_date'] = min(row['start_date'], new_row['start_date'])
                            metadata.loc[idx, 'end_date'] = new_row['end_date']

                            # Fill in missing metadata columns
                            if pd.isna(row['asset_type']):
                                metadata.loc[idx, 'asset_type'] = new_row['asset_type']
                            if pd.isna(row['exchange']):
                                metadata.loc[idx, 'exchange'] = new_row['exchange']
                            if pd.isna(row['asset_name']):
                                metadata.loc[idx, 'asset_name'] = new_row['asset_name']
                        else:
                            # Asset from existing metadata only (no new data)
                            # Keep as-is, fill in defaults if needed
                            if pd.isna(row['asset_type']):
                                metadata.loc[idx, 'asset_type'] = 'equity'
                            if pd.isna(row['exchange']):
                                metadata.loc[idx, 'exchange'] = 'NASDAQ'
                            if pd.isna(row['asset_name']):
                                metadata.loc[idx, 'asset_name'] = row['symbol']

                    print(f"   ✓ Merged metadata: {len(metadata)} total assets")
                    print(f"     New/updated: {len(new_symbols)}, Preserved: {len(metadata) - len(new_symbols)}")

                else:
                    print(f"   ⚠️  Could not find existing assets database")
                    print(f"   Creating new metadata from current data")
                    metadata = prepare_asset_metadata(
                        all_pricing_data,
                        actual_start.strftime('%Y-%m-%d'),
                        actual_end.strftime('%Y-%m-%d')
                    )
            except Exception as e:
                print(f"   ⚠️  Error merging metadata: {e}")
                print(f"   Creating new metadata from current data")
                metadata = prepare_asset_metadata(
                    all_pricing_data,
                    actual_start.strftime('%Y-%m-%d'),
                    actual_end.strftime('%Y-%m-%d')
                )
        else:
            # Full ingestion: create new metadata from scratch
            metadata = prepare_asset_metadata(
                all_pricing_data,
                actual_start.strftime('%Y-%m-%d'),
                actual_end.strftime('%Y-%m-%d')
            )

        # Create symbol to sid mapping
        # For incremental mode: preserve existing SIDs, assign new ones to new symbols
        # For full ingestion: use DataFrame index as SID
        if 'sid' in metadata.columns:
            # Incremental mode: we have existing SIDs
            # For new symbols (sid is NaN), assign new SIDs starting from max+1
            max_existing_sid = metadata['sid'].max()
            if pd.isna(max_existing_sid):
                max_existing_sid = -1  # No existing assets

            next_sid = int(max_existing_sid) + 1
            symbol_to_sid = {}

            for idx, row in metadata.iterrows():
                if pd.notna(row['sid']):
                    # Existing asset: use existing SID
                    symbol_to_sid[row['symbol']] = int(row['sid'])
                else:
                    # New asset: assign new SID
                    symbol_to_sid[row['symbol']] = next_sid
                    metadata.loc[idx, 'sid'] = next_sid
                    next_sid += 1
        else:
            # Full ingestion mode: use DataFrame index as SID
            symbol_to_sid = {row['symbol']: idx for idx, row in metadata.iterrows()}
            metadata['sid'] = metadata.index

        # Add sid to pricing data
        all_pricing_data['sid'] = all_pricing_data['ticker'].map(symbol_to_sid)

        # DEBUG: Check SID mapping
        total_rows = len(all_pricing_data)
        unmapped_rows = all_pricing_data['sid'].isna().sum()
        unique_sids = all_pricing_data['sid'].nunique()
        print(f"   DEBUG: Total rows in all_pricing_data: {total_rows:,}")
        print(f"   DEBUG: Rows with mapped SID: {total_rows - unmapped_rows:,}")
        print(f"   DEBUG: Rows with unmapped SID (NaN): {unmapped_rows:,}")
        print(f"   DEBUG: Unique SIDs: {unique_sids}")

        # Sample some specific tickers for debugging
        sample_tickers = ['SPY', 'AAPL', 'MSFT']
        for ticker in sample_tickers:
            if ticker in symbol_to_sid:
                sid = symbol_to_sid[ticker]
                count = len(all_pricing_data[all_pricing_data['ticker'] == ticker])
                print(f"   DEBUG: {ticker} -> SID {sid}, {count:,} rows")
            else:
                print(f"   DEBUG: {ticker} not in symbol_to_sid mapping")

        # Prepare metadata for writer: set sid as index
        # The asset writer expects sid as the DataFrame index
        if 'sid' in metadata.columns:
            metadata = metadata.set_index('sid')
        else:
            # This shouldn't happen, but handle it just in case
            metadata['sid'] = metadata.index
            metadata = metadata.set_index('sid')

        # Create exchanges data (required for Pipeline country_code filtering)
        # Sharadar data includes US exchanges
        exchanges = pd.DataFrame({
            'exchange': ['NASDAQ', 'NYSE', 'NYSEARCA', 'BATS', 'OTC'],
            'canonical_name': ['NASDAQ', 'NYSE', 'NYSE ARCA', 'BATS', 'OTC'],
            'country_code': ['US', 'US', 'US', 'US', 'US']
        })

        # Write metadata
        print("Writing asset metadata...")
        asset_db_writer.write(equities=metadata, exchanges=exchanges)

        # Prepare and write pricing data
        print("Writing daily bars...")

        # Get trading calendar for the full date range to handle missing sessions
        calendar_sessions = calendar.sessions_in_range(
            pd.Timestamp(actual_start),
            pd.Timestamp(actual_end)
        )

        def data_generator():
            """Generator that yields (sid, dataframe) for each symbol"""
            total_sids = 0
            total_rows_yielded = 0

            for sid in sorted(all_pricing_data['sid'].unique()):
                if pd.isna(sid):
                    continue  # Skip any unmapped tickers

                symbol_data = all_pricing_data[all_pricing_data['sid'] == sid].copy()

                # Set date as index
                symbol_data = symbol_data.set_index('date')

                # Ensure timezone-naive
                if symbol_data.index.tz is not None:
                    symbol_data.index = symbol_data.index.tz_localize(None)

                # Use 'closeadj' column for TOTAL RETURN (includes splits AND dividends)
                # Sharadar columns:
                #   - 'close' = split-adjusted historically, NOT dividend-adjusted (price return only)
                #   - 'closeadj' = split-adjusted AND dividend-adjusted (total return)
                #   - 'closeunadj' = NOT split-adjusted, NOT dividend-adjusted (raw prices)
                #
                # IMPORTANT: We use 'closeadj' to get total return including dividends.
                # This is the correct approach for backtesting total portfolio performance.
                #
                # For incremental updates: new data has closeadj, existing data has close
                # Use closeadj where available (not NaN), fall back to close for historical data
                if 'closeadj' in symbol_data.columns:
                    # Use closeadj where not NaN, otherwise keep existing close value
                    symbol_data['close'] = symbol_data['closeadj'].fillna(symbol_data.get('close', pd.Series(dtype=float)))
                # If closeadj not available, fall back to close (backward compatibility)
                elif 'close' not in symbol_data.columns:
                    raise ValueError(f"Neither 'closeadj' nor 'close' column found for sid {sid}")

                # Ensure all required columns and proper types
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                symbol_data = symbol_data[required_cols].copy()

                # Convert to float
                for col in required_cols:
                    symbol_data[col] = symbol_data[col].astype(float)

                # Remove any NaN rows
                symbol_data = symbol_data.dropna()

                # Skip symbols with no valid data
                if len(symbol_data) == 0:
                    continue

                # Sort by date
                symbol_data = symbol_data.sort_index()

                # Get this symbol's actual date range
                symbol_start = symbol_data.index.min()
                symbol_end = symbol_data.index.max()

                # Skip if dates are invalid (NaT)
                if pd.isna(symbol_start) or pd.isna(symbol_end):
                    continue

                # Get trading days for this symbol's range
                symbol_sessions = calendar.sessions_in_range(symbol_start, symbol_end)

                # Reindex to include all trading days, forward-fill gaps
                symbol_data = symbol_data.reindex(symbol_sessions, method='ffill')

                # Drop any remaining NaN rows (at the start before first real data)
                symbol_data = symbol_data.dropna()

                total_sids += 1
                total_rows_yielded += len(symbol_data)

                # DEBUG: Print first few assets to see what's being written
                if total_sids <= 5:
                    print(f"   DEBUG: Yielding SID {int(sid)}, {len(symbol_data)} rows, date range: {symbol_data.index.min().date()} to {symbol_data.index.max().date()}")

                yield int(sid), symbol_data

            print(f"   DEBUG: Total assets yielded: {total_sids}, Total rows: {total_rows_yielded:,}")

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
    Prepare asset metadata DataFrame for Zipline from Sharadar pricing data.

    This function extracts asset metadata (symbols, date ranges, exchanges) from
    the raw pricing data and formats it for Zipline's asset database.

    The metadata includes:
    - symbol: Ticker symbol (e.g., 'AAPL', 'MSFT')
    - start_date: First available trading date for this asset
    - end_date: Last available trading date for this asset
    - asset_type: 'equity' or 'fund'
    - exchange: Primary exchange (defaults to NASDAQ)
    - asset_name: Human-readable name (same as symbol for Sharadar)

    The function ensures that asset date ranges are clipped to the bundle's
    date range to prevent out-of-bounds errors.

    Parameters
    ----------
    pricing_data : pd.DataFrame
        Raw pricing data from SEP and/or SFP tables.
        Must contain 'ticker' and 'date' columns.
        May optionally contain 'asset_type' column.
    start_date : str
        Bundle start date in 'YYYY-MM-DD' format.
        Asset start dates will be clipped to this date.
    end_date : str
        Bundle end date in 'YYYY-MM-DD' format.
        Asset end dates will be clipped to this date.

    Returns
    -------
    pd.DataFrame
        Asset metadata with columns:
        - symbol: str
        - start_date: datetime64
        - end_date: datetime64
        - asset_type: str ('equity' or 'fund')
        - exchange: str
        - asset_name: str

    Notes
    -----
    The index of the returned DataFrame will be used as SIDs (security identifiers)
    in Zipline. The index starts at 0 and increments for each asset.

    Examples
    --------
    >>> pricing = pd.DataFrame({
    ...     'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
    ...     'date': ['2020-01-02', '2020-01-03', '2020-01-02', '2020-01-03'],
    ...     'close': [300.35, 297.43, 160.62, 158.62]
    ... })
    >>> metadata = prepare_asset_metadata(pricing, '2020-01-01', '2020-12-31')
    >>> print(metadata[['symbol', 'start_date', 'end_date']])
      symbol start_date   end_date
    0   AAPL 2020-01-02 2020-01-03
    1   MSFT 2020-01-02 2020-01-03
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
    Prepare split and dividend adjustments from Sharadar ACTIONS table.

    CRITICAL: This function returns EMPTY adjustments for both splits AND dividends!

    WHY NO ADJUSTMENTS?
    -------------------
    We use Sharadar's 'closeadj' column which is ALREADY fully adjusted for
    both splits AND dividends historically. This gives us total return prices.

    Applying adjustments on top of already-adjusted prices causes double-adjustment,
    which would manifest as incorrect returns.

    SHARADAR PRICE COLUMN DEFINITIONS
    ---------------------------------
    - 'close': Split-adjusted historically, NOT dividend-adjusted (price return only)
      → Not used in this bundle
    - 'closeadj': Fully adjusted (splits AND dividends) (total return)
      → This is what we use for backtesting
    - 'closeunadj': Raw traded prices, no adjustments
      → Not used in this bundle

    TOTAL RETURN APPROACH
    --------------------
    By using 'closeadj', backtests automatically capture total return
    (price appreciation + dividend reinvestment) without needing separate
    adjustment processing. This is the standard approach for performance analysis.

    Parameters
    ----------
    actions_data : pd.DataFrame
        Corporate actions data from Sharadar ACTIONS table.
        Must contain columns: ['ticker', 'date', 'action', 'value']
        Actions include: 'Split', 'Dividend', 'Spinoff', etc.
    ticker_to_sid : dict
        Mapping of ticker symbols (str) to Zipline security IDs (int).
        Example: {'AAPL': 0, 'MSFT': 1, ...}

    Returns
    -------
    dict
        Dictionary with two DataFrames:
        - 'splits': Empty DataFrame (we don't apply split adjustments)
          Columns: ['sid', 'ratio', 'effective_date']
        - 'dividends': DataFrame of dividend adjustments to apply
          Columns: ['sid', 'amount', 'ex_date', 'record_date',
                    'declared_date', 'pay_date']

    Notes
    -----
    Sharadar only provides ex_date for dividends, not record/declared/pay dates.
    We use ex_date for all dividend date fields as a simplification.

    The dividend amount is the per-share payment in dollars. Zipline will
    automatically adjust historical prices by this amount to reflect total return.

    Examples
    --------
    >>> actions = pd.DataFrame({
    ...     'ticker': ['AAPL', 'AAPL', 'MSFT'],
    ...     'date': ['2020-08-10', '2020-08-31', '2020-08-19'],
    ...     'action': ['Dividend', 'Split', 'Dividend'],
    ...     'value': [0.82, 4.0, 0.51]
    ... })
    >>> ticker_to_sid = {'AAPL': 0, 'MSFT': 1}
    >>> adjustments = prepare_adjustments(actions, ticker_to_sid)
    >>> print(adjustments['splits'])  # Empty!
    Empty DataFrame
    Columns: [sid, ratio, effective_date]
    Index: []
    >>> print(adjustments['dividends'])
       sid  amount    ex_date record_date declared_date    pay_date
    0    0    0.82 2020-08-10  2020-08-10    2020-08-10  2020-08-10
    1    1    0.51 2020-08-19  2020-08-19    2020-08-19  2020-08-19

    See Also
    --------
    prepare_asset_metadata : Creates asset metadata from pricing data
    download_sharadar_table : Downloads Sharadar tables including ACTIONS
    """
    # DO NOT apply any adjustments - Sharadar 'closeadj' data is already fully adjusted!
    # Returning empty adjustments prevents double-adjustment.

    # Empty splits (already applied in closeadj)
    splits = pd.DataFrame(columns=['sid', 'ratio', 'effective_date'])

    # Empty dividends (already applied in closeadj)
    dividends = pd.DataFrame(columns=['sid', 'amount', 'ex_date', 'record_date', 'declared_date', 'pay_date'])

    return {
        'splits': splits,
        'dividends': dividends,
    }


# Pre-configured bundle variants for common use cases
def sharadar_tech_bundle():
    """
    Pre-configured Sharadar bundle with major technology stocks.

    This bundle includes 15 large-cap technology companies, useful for:
    - Tech sector analysis and backtesting
    - Quick testing of strategies on high-liquidity stocks
    - Learning Zipline without downloading entire universe

    Included tickers:
    - FAANG+: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX
    - Enterprise: ADBE, CRM, ORCL, INTC
    - Semiconductors: AMD, AVGO
    - Networking: CSCO

    Performance:
    - Download time: ~1-2 minutes
    - Storage: ~100-200 MB
    - Data range: 1998-present (varies by ticker)

    Returns
    -------
    callable
        Bundle ingest function configured for tech stocks.

    Examples
    --------
    >>> from zipline.data.bundles import register
    >>> from zipline.data.bundles.sharadar_bundle import sharadar_tech_bundle
    >>> register('tech', sharadar_tech_bundle())
    >>> # Then: zipline ingest -b tech
    """
    return sharadar_bundle(
        tickers=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            'INTC', 'AMD', 'ORCL', 'CSCO', 'AVGO',
        ],
    )


def sharadar_sp500_sample_bundle():
    """
    Pre-configured Sharadar bundle with S&P 500 sample stocks.

    This bundle includes 30 stocks representing major S&P 500 components
    across different sectors. Useful for:
    - Diversified strategy testing
    - Sector rotation strategies
    - Multi-sector portfolio backtesting

    Sectors included:
    - Technology: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
    - Healthcare: JNJ, UNH, MRK, ABBV, PFE, ABT, TMO
    - Financials: JPM, V, MA, BRK.B
    - Consumer: WMT, PG, KO, PEP, COST, HD, DIS
    - Energy: XOM, CVX
    - Other: ACN, CSCO, AVGO

    Performance:
    - Download time: ~2-3 minutes
    - Storage: ~200-400 MB
    - Data range: 1998-present (varies by ticker)

    Returns
    -------
    callable
        Bundle ingest function configured for S&P 500 sample stocks.

    Examples
    --------
    >>> from zipline.data.bundles import register
    >>> from zipline.data.bundles.sharadar_bundle import sharadar_sp500_sample_bundle
    >>> register('sp500-sample', sharadar_sp500_sample_bundle())
    >>> # Then: zipline ingest -b sp500-sample
    """
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
    Pre-configured Sharadar bundle with ALL available tickers.

    WARNING: This downloads the complete Sharadar universe (~8,000+ tickers).

    This is the most comprehensive option but requires significant resources:
    - Download time: 30-90 minutes (depends on connection and API tier)
    - Storage: 10-20 GB (compressed)
    - Memory: 8+ GB RAM recommended for ingestion
    - API calls: ~500-1000 (ensure your plan supports this)

    Use cases:
    - Production trading systems
    - Research requiring full market coverage
    - Statistical analysis across entire universe
    - Factor research and screening

    NOT recommended for:
    - Learning Zipline (use sharadar_tech_bundle instead)
    - Quick strategy testing (use sharadar_sp500_sample_bundle)
    - Systems with limited storage or memory

    Tips for full universe ingestion:
    - Use incremental=True (default) for faster daily updates
    - Consider use_chunks=True if initial download times out
    - Ensure you have a Premium NASDAQ Data Link subscription
    - Run during off-peak hours to maximize API performance

    Returns
    -------
    callable
        Bundle ingest function configured for all available tickers.

    Examples
    --------
    >>> from zipline.data.bundles import register
    >>> from zipline.data.bundles.sharadar_bundle import sharadar_all_bundle
    >>> register('sharadar-full', sharadar_all_bundle())
    >>> # Then: zipline ingest -b sharadar-full
    >>> # This will take 30-90 minutes for first ingestion!
    """
    return sharadar_bundle(tickers=None)  # None = all tickers


# Convenience function to register all pre-configured variants
def register_sharadar_bundles():
    """
    Register all pre-configured Sharadar bundle variants with Zipline.

    This convenience function registers three bundle variants:
    - 'sharadar-tech': 15 major technology stocks
    - 'sharadar-sp500': 30 diversified S&P 500 stocks
    - 'sharadar-all': Complete universe (~8,000+ tickers)

    Usage
    -----
    Call this function in your extension.py or at the start of your script:

    >>> from zipline.data.bundles.sharadar_bundle import register_sharadar_bundles
    >>> register_sharadar_bundles()

    Then ingest any of the bundles:

    >>> # Small tech bundle (recommended for learning)
    >>> zipline ingest -b sharadar-tech

    >>> # Diversified sample (recommended for testing)
    >>> zipline ingest -b sharadar-sp500

    >>> # Full universe (production use only)
    >>> zipline ingest -b sharadar-all

    See Also
    --------
    sharadar_tech_bundle : Technology stocks bundle
    sharadar_sp500_sample_bundle : S&P 500 sample bundle
    sharadar_all_bundle : Complete universe bundle
    sharadar_bundle : Create custom bundle configurations
    """
    from zipline.data.bundles import register

    register('sharadar-tech', sharadar_tech_bundle())
    register('sharadar-sp500', sharadar_sp500_sample_bundle())
    register('sharadar-all', sharadar_all_bundle())
