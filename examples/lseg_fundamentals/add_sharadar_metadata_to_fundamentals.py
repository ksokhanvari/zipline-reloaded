#!/usr/bin/env python3
"""
Add Sharadar Metadata to LSEG Fundamentals CSV

This script enriches your LSEG fundamentals CSV with Sharadar ticker metadata
(exchange, category, ADR status, sector, industry, market cap scale, etc.)
so that all information is available in a single database table for Pipeline filtering.

Usage:
    python add_sharadar_metadata_to_fundamentals.py \
        --input lseg_fundamentals.csv \
        --output lseg_fundamentals_with_metadata.csv \
        --sharadar-bundle sharadar

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
        required=True,
        help='Input LSEG fundamentals CSV file'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file with metadata added'
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
