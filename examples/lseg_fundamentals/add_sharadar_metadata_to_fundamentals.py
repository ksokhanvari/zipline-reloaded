#!/usr/bin/env python3
"""
Add Sharadar Metadata to LSEG Fundamentals CSV

This script enriches your LSEG fundamentals CSV with Sharadar ticker metadata
(exchange, category, ADR status, sector, industry, market cap scale, etc.)
so that all information is available in a single database table for Pipeline filtering.

Usage:
    # Auto-detect newest CSV file in /data/csv/ directory
    python add_sharadar_metadata_to_fundamentals.py

    # Specify input and output files explicitly
    python add_sharadar_metadata_to_fundamentals.py \
        --input /data/csv/20091231_20251118.csv \
        --output /data/csv/20091231_20251118_with_metadata.csv

    # Auto-detect from custom directory
    python add_sharadar_metadata_to_fundamentals.py \
        --csv-dir /path/to/csv/files/

    # Specify input only (output auto-generated with "_with_metadata" suffix)
    python add_sharadar_metadata_to_fundamentals.py \
        --input /data/csv/20091231_20251118.csv

Auto-Detection:
    - If --input is not provided, searches for CSV files in /data/csv/ directory
    - Selects file with newest end date (format: YYYYMMDD_YYYYMMDD.csv)
    - If --output is not provided, adds "_with_metadata" suffix to input filename

The output CSV will have these additional columns:
    - sharadar_exchange: Exchange (NYSE, NASDAQ, NYSEMKT, etc.)
    - sharadar_category: Stock category (Domestic Common Stock, ADR, ETF, etc.)
    - sharadar_is_adr: Boolean ADR flag (True/False)
    - sharadar_location: Company location (USA, etc.)
    - sharadar_sector: Sharadar sector
    - sharadar_industry: Sharadar industry
    - sharadar_sicsector: SIC sector
    - sharadar_sicindustry: SIC industry
    - sharadar_scalemarketcap: Market cap scale (1-6: Nano to Mega)

Author: Kamran Sokhanvari / Hidden Point Capital
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re


def find_newest_csv(directory='/data/csv/', pattern='*_*.csv'):
    """
    Find the CSV file with the newest date in the filename.

    Assumes filename format: YYYYMMDD_YYYYMMDD.csv
    Returns the file with the most recent end date (second date).

    Parameters
    ----------
    directory : str
        Directory to search for CSV files
    pattern : str
        Glob pattern for CSV files (default: *_*.csv)

    Returns
    -------
    str or None
        Path to newest CSV file, or None if no files found

    Examples
    --------
    >>> find_newest_csv('/data/csv/')
    '/data/csv/20091231_20251118.csv'
    """
    csv_dir = Path(directory)

    if not csv_dir.exists():
        print(f"Warning: Directory {csv_dir} does not exist")
        return None

    # Find all CSV files matching pattern
    csv_files = list(csv_dir.glob(pattern))

    if not csv_files:
        print(f"Warning: No CSV files found in {csv_dir} matching pattern {pattern}")
        return None

    print(f"Found {len(csv_files)} CSV files in {csv_dir}")

    # Extract dates from filenames
    # Pattern: YYYYMMDD_YYYYMMDD.csv
    date_pattern = r'(\d{8})_(\d{8})\.csv'

    files_with_dates = []
    for csv_file in csv_files:
        match = re.search(date_pattern, csv_file.name)
        if match:
            start_date = match.group(1)
            end_date = match.group(2)
            files_with_dates.append((csv_file, start_date, end_date))
            print(f"  {csv_file.name}: {start_date} to {end_date}")

    if not files_with_dates:
        print(f"Warning: No files with date pattern YYYYMMDD_YYYYMMDD.csv found")
        # Return the first file as fallback
        return str(csv_files[0])

    # Sort by end date (most recent last)
    files_with_dates.sort(key=lambda x: x[2])

    newest_file = files_with_dates[-1][0]
    newest_start = files_with_dates[-1][1]
    newest_end = files_with_dates[-1][2]

    print(f"\nNewest file: {newest_file.name} ({newest_start} to {newest_end})")

    return str(newest_file)


def generate_output_filename(input_csv, suffix='_with_metadata'):
    """
    Generate output filename by adding suffix before .csv extension.

    Parameters
    ----------
    input_csv : str
        Input CSV filename
    suffix : str
        Suffix to add before .csv (default: '_with_metadata')

    Returns
    -------
    str
        Output filename

    Examples
    --------
    >>> generate_output_filename('/data/csv/20091231_20251118.csv')
    '/data/csv/20091231_20251118_with_metadata.csv'
    """
    input_path = Path(input_csv)
    output_name = input_path.stem + suffix + input_path.suffix
    output_path = input_path.parent / output_name
    return str(output_path)


def load_sharadar_tickers(bundle_name='sharadar'):
    """
    Load Sharadar ticker metadata from the bundle.

    Parameters
    ----------
    bundle_name : str
        Name of the Sharadar bundle (default: 'sharadar')

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, exchange, category, location,
        sector, industry, sicsector, sicindustry, scalemarketcap, is_adr
    """
    print(f"Loading Sharadar tickers from bundle: {bundle_name}")

    # Find the most recent bundle ingestion
    bundle_dir = Path.home() / '.zipline' / 'data' / bundle_name

    if not bundle_dir.exists():
        # Try Docker path
        bundle_dir = Path('/root/.zipline/data') / bundle_name

    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"Sharadar bundle '{bundle_name}' not found. "
            f"Please ingest the bundle first with: zipline ingest -b {bundle_name}"
        )

    # Get most recent ingestion
    ingestions = sorted([d for d in bundle_dir.iterdir() if d.is_dir()],
                       reverse=True)

    if not ingestions:
        raise FileNotFoundError(f"No ingestions found in {bundle_dir}")

    latest_ingestion = ingestions[0]
    tickers_file = latest_ingestion / 'fundamentals' / 'tickers.h5'

    if not tickers_file.exists():
        raise FileNotFoundError(f"Tickers file not found: {tickers_file}")

    print(f"Loading from: {tickers_file}")

    # Load tickers
    tickers = pd.read_hdf(tickers_file, key='tickers')

    print(f"Loaded {len(tickers)} tickers")

    # IMPORTANT: Deduplicate tickers - keep only one entry per ticker
    # Sharadar has multiple entries per ticker:
    # 1. Active + historical delisted entries (different isdelisted values)
    # 2. Multiple active entries with same ticker (same isdelisted='N')
    if 'isdelisted' in tickers.columns:
        # Sort: non-delisted first (N before Y), then by index
        tickers = tickers.sort_values(['isdelisted', tickers.index.name or 'index'])
        tickers = tickers.drop_duplicates(subset='ticker', keep='first')
        print(f"After deduplication: {len(tickers)} unique tickers")
    else:
        # Fallback: just deduplicate by ticker
        tickers = tickers.drop_duplicates(subset='ticker', keep='first')
        print(f"After deduplication: {len(tickers)} unique tickers")

    # Select relevant columns
    metadata_cols = [
        'ticker', 'exchange', 'category', 'location',
        'sector', 'industry', 'sicsector', 'sicindustry',
        'scalemarketcap'
    ]

    # Keep only columns that exist
    available_cols = [col for col in metadata_cols if col in tickers.columns]
    tickers_subset = tickers[available_cols].copy()

    # Add is_adr flag
    if 'category' in tickers_subset.columns:
        tickers_subset['is_adr'] = tickers_subset['category'].str.contains(
            'ADR', na=False, case=False
        ).astype(int)
    else:
        tickers_subset['is_adr'] = 0

    # Rename ticker to Symbol for merging
    tickers_subset = tickers_subset.rename(columns={'ticker': 'Symbol'})

    return tickers_subset


def add_metadata_to_fundamentals(fundamentals_df, metadata_df):
    """
    Add Sharadar metadata columns to fundamentals DataFrame.

    Parameters
    ----------
    fundamentals_df : pd.DataFrame
        LSEG fundamentals data with at least a 'Symbol' column
    metadata_df : pd.DataFrame
        Sharadar metadata with Symbol column

    Returns
    -------
    pd.DataFrame
        Fundamentals with added metadata columns (prefixed with 'sharadar_')
    """
    print(f"\nMerging metadata...")
    print(f"Fundamentals shape: {fundamentals_df.shape}")
    print(f"Metadata shape: {metadata_df.shape}")

    # Check if Symbol column exists
    if 'Symbol' not in fundamentals_df.columns:
        raise ValueError("Fundamentals DataFrame must have a 'Symbol' column")

    # Prefix metadata columns (except Symbol)
    metadata_cols = [col for col in metadata_df.columns if col != 'Symbol']
    rename_dict = {col: f'sharadar_{col.lower()}' for col in metadata_cols}
    metadata_df_renamed = metadata_df.rename(columns=rename_dict)

    # Merge on Symbol (left join to keep all fundamental rows)
    merged = fundamentals_df.merge(
        metadata_df_renamed,
        on='Symbol',
        how='left'
    )

    print(f"Merged shape: {merged.shape}")

    # Report matching statistics
    matched_symbols = merged['sharadar_exchange'].notna().sum() if 'sharadar_exchange' in merged.columns else 0
    total_rows = len(merged)
    unique_symbols = fundamentals_df['Symbol'].nunique()

    print(f"\nMatching statistics:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Unique symbols: {unique_symbols:,}")
    print(f"  Rows with metadata: {matched_symbols:,} ({matched_symbols/total_rows*100:.1f}%)")

    # Fill missing metadata with defaults
    metadata_columns = [col for col in merged.columns if col.startswith('sharadar_')]
    for col in metadata_columns:
        if col == 'sharadar_is_adr':
            merged[col] = merged[col].fillna(0).astype(int)
        else:
            merged[col] = merged[col].fillna('')

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Add Sharadar metadata to LSEG fundamentals CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input', '-i',
        default=None,
        help='Input LSEG fundamentals CSV file (if not provided, auto-detects newest file in /data/csv/)'
    )

    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output CSV file with metadata added (if not provided, adds "_with_metadata" suffix to input filename)'
    )

    parser.add_argument(
        '--csv-dir', '-d',
        default='/data/csv/',
        help='Directory to search for CSV files when auto-detecting (default: /data/csv/)'
    )

    parser.add_argument(
        '--sharadar-bundle', '-b',
        default='sharadar',
        help='Sharadar bundle name (default: sharadar)'
    )

    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Show preview of first 10 rows with metadata'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ADD SHARADAR METADATA TO LSEG FUNDAMENTALS")
    print("=" * 80)

    # Auto-detect input file if not provided
    if args.input is None:
        print(f"\nNo input file specified - auto-detecting newest CSV in {args.csv_dir}...")
        args.input = find_newest_csv(directory=args.csv_dir)
        if args.input is None:
            print("ERROR: Could not auto-detect input file. Please specify with --input")
            sys.exit(1)
    else:
        print(f"\nUsing specified input file: {args.input}")

    # Auto-generate output filename if not provided
    if args.output is None:
        print(f"No output file specified - auto-generating from input filename...")
        args.output = generate_output_filename(args.input)
        print(f"Generated output filename: {args.output}")
    else:
        print(f"Using specified output file: {args.output}")

    # Load LSEG fundamentals
    print(f"\nLoading LSEG fundamentals from: {args.input}")
    try:
        fundamentals = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    print(f"Loaded {len(fundamentals)} rows, {len(fundamentals.columns)} columns")
    print(f"Date range: {fundamentals['Date'].min()} to {fundamentals['Date'].max()}" if 'Date' in fundamentals.columns else "No Date column")

    # Load Sharadar metadata
    try:
        metadata = load_sharadar_tickers(args.sharadar_bundle)
    except Exception as e:
        print(f"ERROR loading Sharadar metadata: {e}")
        sys.exit(1)

    # Merge
    enriched = add_metadata_to_fundamentals(fundamentals, metadata)

    # Preview
    if args.preview:
        print("\n" + "=" * 80)
        print("PREVIEW (first 10 rows with metadata):")
        print("=" * 80)

        metadata_cols = [col for col in enriched.columns if col.startswith('sharadar_')]
        preview_cols = ['Date', 'Symbol'] + metadata_cols
        preview_cols = [col for col in preview_cols if col in enriched.columns]

        print(enriched[preview_cols].head(10).to_string())

        print("\n" + "=" * 80)
        print("METADATA COLUMN SUMMARY:")
        print("=" * 80)

        for col in metadata_cols:
            unique_count = enriched[col].nunique()
            null_count = enriched[col].isna().sum()
            print(f"{col:40s}: {unique_count:6,} unique values, {null_count:8,} nulls")

    # Save
    print(f"\nSaving enriched data to: {args.output}")
    enriched.to_csv(args.output, index=False)

    print(f"✓ Saved {len(enriched)} rows to {args.output}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original columns: {len(fundamentals.columns)}")
    print(f"New columns: {len(enriched.columns)}")
    print(f"Metadata columns added: {len([c for c in enriched.columns if c.startswith('sharadar_')])}")
    print("\nNew metadata columns:")
    for col in sorted([c for c in enriched.columns if c.startswith('sharadar_')]):
        print(f"  - {col}")

    print("\n✓ Done! You can now load this CSV into fundamentals.sqlite")
    print(f"  and use the metadata columns for Pipeline filtering.")


if __name__ == '__main__':
    main()
