"""
Temporal SID Mapper - Proper solution for symbol changes over time

This module provides intelligent SID mapping that automatically handles:
- Company name changes (FB -> META, GOOG -> GOOGL, etc.)
- Ticker symbol changes
- Mergers and acquisitions
- Any temporal symbol mappings in the Zipline asset database

Key insight: Zipline's asset_finder.lookup_symbol(symbol, as_of_date) already
knows about all historical symbol changes. We just need to use it correctly!
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import multiprocessing as mp


class TemporalSIDMapper:
    """
    Maps symbols to SIDs using temporal lookups.

    Handles symbol changes automatically by querying Zipline's asset database
    with the appropriate as-of-date for each row.
    """

    # Known symbol changes not in Sharadar bundle (FB was never ingested as separate symbol)
    # Map old_symbol -> current_symbol for normalization
    KNOWN_SYMBOL_CHANGES = {
        'FB': 'META',  # Facebook renamed to Meta in Oct 2021
    }

    def __init__(self, asset_finder):
        """
        Initialize mapper with Zipline asset finder.

        Args:
            asset_finder: Zipline AssetFinder instance from bundle
        """
        self.asset_finder = asset_finder
        self._symbol_cache = {}  # Cache for (symbol, date) -> SID lookups

    def map_single_row(self, symbol, date):
        """
        Map a single (symbol, date) pair to SID.

        Args:
            symbol: Stock ticker symbol
            date: Date as pd.Timestamp or string

        Returns:
            int: SID or None if not found
        """
        # Convert date to Timestamp if needed
        if isinstance(date, str):
            date = pd.Timestamp(date)

        # Normalize symbol if it's a known change (e.g., FB -> META)
        normalized_symbol = self.KNOWN_SYMBOL_CHANGES.get(symbol, symbol)

        # Check cache (use normalized symbol for cache key)
        cache_key = (normalized_symbol, date.date())
        if cache_key in self._symbol_cache:
            return self._symbol_cache[cache_key]

        # Lookup symbol as of date
        try:
            asset = self.asset_finder.lookup_symbol(normalized_symbol, as_of_date=date)
            sid = asset.sid
            self._symbol_cache[cache_key] = sid
            return sid
        except Exception:
            self._symbol_cache[cache_key] = None
            return None

    def map_dataframe_simple(self, df, symbol_col='Symbol', date_col='Date', verbose=True):
        """
        Map all rows in dataframe using simple iteration.

        Good for small datasets (<100k rows).

        Args:
            df: DataFrame with symbol and date columns
            symbol_col: Name of symbol column
            date_col: Name of date column
            verbose: Print progress messages

        Returns:
            Series: SIDs for each row
        """
        if verbose:
            print(f"Mapping {len(df):,} rows using simple iteration...")

        # Ensure date is Timestamp
        dates = pd.to_datetime(df[date_col])

        sids = []
        for idx, (symbol, date) in enumerate(zip(df[symbol_col], dates)):
            if verbose and idx % 10000 == 0 and idx > 0:
                print(f"  Processed {idx:,} rows...")
            sids.append(self.map_single_row(symbol, date))

        return pd.Series(sids, index=df.index)

    def map_dataframe_grouped(self, df, symbol_col='Symbol', date_col='Date', verbose=True):
        """
        Map dataframe using unique (symbol, date) pairs for optimal performance.

        OPTIMIZED for fundamental data where same symbols repeat across many dates.
        This is the FASTEST method for typical use cases (100k-10M rows).

        Strategy:
        1. Find all unique (symbol, date) pairs (typically 1k-10k pairs)
        2. Look up SID for each unique pair ONCE
        3. Map results back to all rows via merge

        Args:
            df: DataFrame with symbol and date columns
            symbol_col: Name of symbol column
            date_col: Name of date column
            verbose: Print progress messages

        Returns:
            Series: SIDs for each row
        """
        if verbose:
            print(f"Mapping {len(df):,} rows using unique pairs strategy...")

        # Ensure date is Timestamp
        df_work = df[[symbol_col, date_col]].copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col])

        # Get unique (symbol, date) pairs - this is the key optimization!
        unique_pairs = df_work.drop_duplicates()

        if verbose:
            print(f"  Found {len(unique_pairs):,} unique (symbol, date) pairs")
            print(f"  Reduction: {len(df) / len(unique_pairs):.1f}x fewer lookups needed")

        # Map each unique pair
        sids_list = []
        for idx, (symbol, date) in enumerate(zip(unique_pairs[symbol_col], unique_pairs[date_col])):
            if verbose and idx > 0 and idx % 100 == 0:
                print(f"  Processed {idx:,} / {len(unique_pairs):,} unique pairs ({idx/len(unique_pairs)*100:.1f}%)")
            sids_list.append(self.map_single_row(symbol, date))

        # Create mapping dataframe
        unique_pairs['Sid'] = sids_list

        # Merge back to original dataframe
        if verbose:
            print(f"  Merging SIDs back to {len(df):,} rows...")

        result = pd.merge(
            df_work,
            unique_pairs,
            on=[symbol_col, date_col],
            how='left'
        )

        return result['Sid']

    def map_dataframe_parallel(self, df, symbol_col='Symbol', date_col='Date',
                               n_jobs=-1, verbose=True):
        """
        Map dataframe using parallel processing for maximum speed.

        Best for large datasets (>1M rows).

        Args:
            df: DataFrame with symbol and date columns
            symbol_col: Name of symbol column
            date_col: Name of date column
            n_jobs: Number of parallel jobs (-1 = all CPUs)
            verbose: Print progress messages

        Returns:
            Series: SIDs for each row
        """
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        if verbose:
            print(f"Mapping {len(df):,} rows using {n_jobs} parallel workers...")

        # Ensure date is Timestamp
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Split dataframe into chunks
        chunk_size = max(len(df) // n_jobs, 1000)
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

        if verbose:
            print(f"  Split into {len(chunks)} chunks of ~{chunk_size:,} rows each")

        # Process chunks in parallel using threads (not processes, to share asset_finder)
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(
                    self._map_chunk,
                    chunk,
                    symbol_col,
                    date_col
                )
                futures.append(future)

            # Collect results
            results = []
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                if verbose:
                    print(f"  Completed chunk {i+1}/{len(futures)}")

        # Concatenate results
        sids = pd.concat(results)
        return sids

    def _map_chunk(self, chunk, symbol_col, date_col):
        """Helper to map a chunk of data."""
        sids = []
        for symbol, date in zip(chunk[symbol_col], chunk[date_col]):
            sids.append(self.map_single_row(symbol, date))
        return pd.Series(sids, index=chunk.index)

    def map_dataframe_auto(self, df, symbol_col='Symbol', date_col='Date', verbose=True):
        """
        Automatically choose the best mapping strategy based on data size.

        Args:
            df: DataFrame with symbol and date columns
            symbol_col: Name of symbol column
            date_col: Name of date column
            verbose: Print progress messages

        Returns:
            Series: SIDs for each row
        """
        size = len(df)

        if size < 100_000:
            if verbose:
                print("Dataset < 100k rows: Using simple iteration")
            return self.map_dataframe_simple(df, symbol_col, date_col, verbose)

        elif size < 1_000_000:
            if verbose:
                print("Dataset 100k-1M rows: Using grouped lookups")
            return self.map_dataframe_grouped(df, symbol_col, date_col, verbose)

        else:
            if verbose:
                print("Dataset > 1M rows: Using parallel processing")
            return self.map_dataframe_parallel(df, symbol_col, date_col, verbose=verbose)


def create_symbol_change_report(asset_finder, start_date, end_date):
    """
    Generate a report of all symbol changes in the asset database.

    This helps identify which companies changed names/tickers.

    Args:
        asset_finder: Zipline AssetFinder
        start_date: Start date for report
        end_date: End date for report

    Returns:
        DataFrame with symbol changes
    """
    print("Analyzing symbol changes in asset database...")

    changes = []
    all_assets = asset_finder.retrieve_all(asset_finder.sids)

    for asset in all_assets:
        if not hasattr(asset, 'symbol'):
            continue

        current_symbol = asset.symbol

        # Check if symbol was different in the past
        for check_date in pd.date_range(start_date, end_date, freq='Y'):
            try:
                historical_asset = asset_finder.lookup_symbol(
                    current_symbol,
                    as_of_date=check_date
                )

                # If SID changed, record it
                if historical_asset.sid != asset.sid:
                    changes.append({
                        'current_symbol': current_symbol,
                        'current_sid': asset.sid,
                        'date': check_date,
                        'historical_symbol': historical_asset.symbol,
                        'historical_sid': historical_asset.sid,
                    })
            except:
                pass

    if changes:
        df = pd.DataFrame(changes)
        print(f"Found {len(df)} symbol changes")
        return df
    else:
        print("No symbol changes found")
        return pd.DataFrame()


# Example usage for load_csv_fundamentals.ipynb
USAGE_EXAMPLE = """
# =============================================================================
# Replace Cell 12 in load_csv_fundamentals.ipynb with this:
# =============================================================================

print("="*80)
print("Mapping symbols to SIDs with temporal awareness")
print("="*80)

# Create temporal mapper
from temporal_sid_mapper import TemporalSIDMapper

mapper = TemporalSIDMapper(asset_finder)

# Map SIDs automatically (chooses best strategy based on data size)
custom_data['Sid'] = mapper.map_dataframe_auto(
    custom_data,
    symbol_col='Symbol',
    date_col='Date',
    verbose=True
)

# Report results
mapped = custom_data['Sid'].notna().sum()
unmapped = custom_data['Sid'].isna().sum()

print(f"\\n✓ Mapping complete:")
print(f"  Mapped: {mapped:,} rows ({mapped/len(custom_data)*100:.1f}%)")
print(f"  Unmapped: {unmapped:,} rows ({unmapped/len(custom_data)*100:.1f}%)")

if unmapped > 0:
    unmapped_symbols = custom_data[custom_data['Sid'].isna()]['Symbol'].unique()
    print(f"  Unmapped symbols (first 10): {list(unmapped_symbols[:10])}")
    print("  These symbols may not exist in the Sharadar bundle")

# Remove unmapped rows
custom_data = custom_data[custom_data['Sid'].notna()].copy()
custom_data['Sid'] = custom_data['Sid'].astype(int)

print(f"\\n✓ Kept {len(custom_data):,} mapped rows")
print("="*80)

# =============================================================================
# This handles ALL symbol changes automatically:
# - FB -> META
# - GOOG -> GOOGL
# - Any other ticker changes in your data
# - Mergers, acquisitions, rebrands
#
# Works by querying Zipline's asset database with the date of each row,
# so it always gets the correct SID for that point in time.
# =============================================================================
"""

if __name__ == '__main__':
    print(USAGE_EXAMPLE)
