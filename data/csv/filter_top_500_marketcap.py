#!/usr/bin/env python3
"""
Filter CSV to keep top 500 companies by market cap for each date.

Usage:
    python filter_top_500_marketcap.py input.csv
    python filter_top_500_marketcap.py input.csv --output custom_output.csv
    python filter_top_500_marketcap.py input.csv --top 1000
"""

import argparse
import pandas as pd
from pathlib import Path


def filter_top_n_by_marketcap(input_path, output_path=None, top_n=500):
    """
    Read CSV, keep top N companies by CompanyMarketCap for each date.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (optional)
        top_n: Number of top companies to keep per date (default: 500)

    Returns:
        Path to output file
    """
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)

    # Validate required columns
    if 'Date' not in df.columns:
        raise ValueError("CSV must have 'Date' column")
    if 'Symbol' not in df.columns:
        raise ValueError("CSV must have 'Symbol' column")
    if 'CompanyMarketCap' not in df.columns:
        raise ValueError("CSV must have 'CompanyMarketCap' column")

    print(f"Input: {len(df):,} rows, {df['Symbol'].nunique()} unique symbols, {df['Date'].nunique()} unique dates")

    # Group by Date and keep top N by CompanyMarketCap
    print(f"Filtering to top {top_n} companies by market cap for each date...")

    # Sort by CompanyMarketCap descending and group by Date
    df_sorted = df.sort_values('CompanyMarketCap', ascending=False)
    df_filtered = df_sorted.groupby('Date', group_keys=False).head(top_n)

    # Sort by Date and Symbol for clean output
    df_filtered = df_filtered.sort_values(['Date', 'Symbol'])

    print(f"Output: {len(df_filtered):,} rows, {df_filtered['Symbol'].nunique()} unique symbols, {df_filtered['Date'].nunique()} unique dates")

    # Generate output filename if not provided
    if output_path is None:
        input_path_obj = Path(input_path)
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = input_path_obj.parent / f"{stem}_{top_n}mc{suffix}"

    print(f"Saving to {output_path}...")
    df_filtered.to_csv(output_path, index=False)
    print(f"Done! Saved {len(df_filtered):,} rows to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Filter CSV to keep top N companies by market cap for each date',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep top 500 companies (default)
  python filter_top_500_marketcap.py input.csv

  # Keep top 1000 companies
  python filter_top_500_marketcap.py input.csv --top 1000

  # Specify custom output path
  python filter_top_500_marketcap.py input.csv --output filtered.csv
        """
    )

    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (optional)')
    parser.add_argument('--top', '-n', type=int, default=500,
                        help='Number of top companies to keep per date (default: 500)')

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    try:
        filter_top_n_by_marketcap(args.input, args.output, args.top)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
