#!/usr/bin/env python3
"""
Extract data for specific symbols from prediction output files.

This utility script filters prediction output files to extract data for one or
all symbols, with optional forecast-only mode for lean output files.

Usage:
    # Extract all AAPL data
    python extract_symbol.py predictions.csv --symbol AAPL

    # Extract only forecast columns for AAPL
    python extract_symbol.py predictions.csv --symbol AAPL --forecast-only

    # Extract ALL symbols (entire dataset)
    python extract_symbol.py predictions.csv --symbol ALLSYMBOLS

    # Extract ALL symbols, forecast-only (lean output for trading)
    python extract_symbol.py predictions.csv --symbol ALLSYMBOLS --forecast-only

    # Custom output name
    python extract_symbol.py predictions.csv --symbol MSFT --output msft_data.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def extract_symbol_data(input_path, symbol, output_path=None, forecast_only=False):
    """
    Extract all data for a specific symbol from a CSV file.

    Args:
        input_path (Path): Path to input CSV file
        symbol (str): Stock symbol to extract (e.g., 'AAPL', 'MSFT', 'ALLSYMBOLS')
        output_path (Path, optional): Path to output file. If None, auto-generate.
        forecast_only (bool): If True, only keep Date, Symbol, predicted_return columns

    Returns:
        Path: Path to the output file
        int: Number of rows extracted
    """
    print(f"📂 Reading {input_path}...")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None, 0

    # Validate required column
    if 'Symbol' not in df.columns:
        print(f"❌ Error: Input file must have a 'Symbol' column")
        print(f"   Available columns: {', '.join(df.columns)}")
        return None, 0

    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  ✓ Total unique symbols: {df['Symbol'].nunique()}")
    print(f"  ✓ Columns: {len(df.columns)}")

    # Convert symbol to uppercase for case-insensitive matching
    symbol_upper = symbol.upper()

    # Check if user wants all symbols
    if symbol_upper == 'ALLSYMBOLS':
        print(f"\n🎯 Extracting ALL symbols")
        symbol_data = df.copy()
    else:
        # Filter for the specific symbol
        symbol_data = df[df['Symbol'] == symbol_upper].copy()

    if len(symbol_data) == 0 and symbol_upper != 'ALLSYMBOLS':
        print(f"\n⚠️  Warning: No data found for symbol '{symbol}'")
        print(f"\n💡 Available symbols in file (first 20):")
        available_symbols = sorted(df['Symbol'].unique())
        for sym in available_symbols[:20]:
            print(f"   - {sym}")
        if len(available_symbols) > 20:
            print(f"   ... and {len(available_symbols) - 20} more")
        return None, 0

    # Sort by date for easier inspection
    if 'Date' in symbol_data.columns:
        symbol_data = symbol_data.sort_values(['Date', 'Symbol'])

    # Filter to forecast-only columns if requested
    if forecast_only:
        print(f"\n📊 Filtering to forecast-only columns (Date, Symbol, predicted_return)...")
        required_cols = ['Date', 'Symbol', 'predicted_return']

        # Check if predicted_return exists
        if 'predicted_return' not in symbol_data.columns:
            print(f"❌ Error: predicted_return column not found")
            print(f"   Available columns: {', '.join(symbol_data.columns)}")
            return None, 0

        # Keep only required columns
        symbol_data = symbol_data[required_cols].copy()

        # Remove rows with NaN predicted_return
        before_count = len(symbol_data)
        symbol_data = symbol_data[symbol_data['predicted_return'].notna()]
        after_count = len(symbol_data)

        if before_count != after_count:
            print(f"  ✓ Removed {before_count - after_count} rows with NaN predictions")

        print(f"  ✓ Kept 3 columns: Date, Symbol, predicted_return")

    # Generate output filename if not provided
    if output_path is None:
        input_stem = input_path.stem
        if symbol_upper == 'ALLSYMBOLS':
            suffix = '_all_symbols'
        else:
            suffix = f"_{symbol_upper}"

        if forecast_only:
            suffix += '_forecast_only'

        output_path = input_path.parent / f"{input_stem}{suffix}.csv"

    # Save the filtered data
    if symbol_upper == 'ALLSYMBOLS':
        print(f"\n💾 Saving all symbols data...")
    else:
        print(f"\n💾 Saving {symbol} data...")
    symbol_data.to_csv(output_path, index=False)

    # Display statistics
    if symbol_upper == 'ALLSYMBOLS':
        print(f"  ✓ Extracted {len(symbol_data):,} rows across {symbol_data['Symbol'].nunique()} symbols")
    else:
        print(f"  ✓ Extracted {len(symbol_data):,} rows for {symbol}")

    if 'Date' in symbol_data.columns:
        print(f"  ✓ Date range: {symbol_data['Date'].min()} to {symbol_data['Date'].max()}")

    if 'predicted_return' in symbol_data.columns:
        pred_stats = symbol_data['predicted_return'].describe()

        if symbol_upper == 'ALLSYMBOLS':
            print(f"\n📊 Predicted Return Statistics (All Symbols):")
        else:
            print(f"\n📊 Predicted Return Statistics for {symbol}:")

        print(f"  • Mean: {pred_stats['mean']:+.2f}%")
        print(f"  • Median: {symbol_data['predicted_return'].median():+.2f}%")
        print(f"  • Min: {pred_stats['min']:+.2f}%")
        print(f"  • Max: {pred_stats['max']:+.2f}%")
        print(f"  • Std: {pred_stats['std']:.2f}%")

    if 'forward_return' in symbol_data.columns and not forecast_only:
        # Show actual vs predicted if both available
        both_available = symbol_data[['forward_return', 'predicted_return']].notna().all(axis=1)
        if both_available.sum() > 0:
            corr = symbol_data.loc[both_available, ['forward_return', 'predicted_return']].corr().iloc[0, 1]

            if symbol_upper == 'ALLSYMBOLS':
                print(f"\n🎯 Actual vs Predicted (All Symbols):")
            else:
                print(f"\n🎯 Actual vs Predicted (for {symbol}):")

            print(f"  • Correlation: {corr:.4f} ({corr*100:.1f}%)")

    # Show sample data
    print(f"\n📋 Sample data (first 5 rows):")

    if forecast_only:
        # For forecast-only, show all 3 columns
        print(symbol_data.head().to_string(index=False))
    else:
        # Select key columns to display
        display_cols = []
        for col in ['Date', 'Symbol', 'RefPriceClose', 'CompanyMarketCap', 'predicted_return', 'forward_return']:
            if col in symbol_data.columns:
                display_cols.append(col)

        if display_cols:
            print(symbol_data[display_cols].head().to_string(index=False))
        else:
            # Show first few columns if key columns not found
            print(symbol_data.iloc[:5, :5].to_string(index=False))

    print(f"\n✅ Output saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path, len(symbol_data)


def main():
    """Main entry point for the symbol extraction utility."""
    parser = argparse.ArgumentParser(
        description='Extract data for a specific symbol from prediction output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract single symbol (all columns)
  python extract_symbol.py predictions.csv --symbol AAPL

  # Extract single symbol (forecast-only: Date, Symbol, predicted_return)
  python extract_symbol.py predictions.csv --symbol AAPL --forecast-only

  # Extract ALL symbols (entire dataset, all columns)
  python extract_symbol.py predictions.csv --symbol ALLSYMBOLS

  # Extract ALL symbols, forecast-only (lean file for trading systems)
  python extract_symbol.py predictions.csv --symbol ALLSYMBOLS --forecast-only

  # Custom output path
  python extract_symbol.py predictions.csv --symbol TSLA --output tesla_analysis.csv

  # Extract multiple individual symbols (run separately)
  python extract_symbol.py predictions.csv --symbol AAPL
  python extract_symbol.py predictions.csv --symbol MSFT
  python extract_symbol.py predictions.csv --symbol GOOGL

Use cases:
  - Verify predictions for specific stocks
  - Create lean forecast files for trading systems (--forecast-only)
  - Export all predictions in compact format (ALLSYMBOLS --forecast-only)
  - Create test datasets for strategy development
  - Analyze individual stock behavior
  - Validate output file correctness
        """
    )

    parser.add_argument('input',
                        help='Input CSV file (output from forecast_returns_ml.py)')
    parser.add_argument('--symbol', '-s', required=True,
                        help='Stock symbol to extract (e.g., AAPL, MSFT, GOOGL, or ALLSYMBOLS for all)')
    parser.add_argument('--output', '-o',
                        help='Output CSV file path (default: auto-generated)')
    parser.add_argument('--forecast-only', '-f', action='store_true',
                        help='Output only Date, Symbol, and predicted_return columns (lean file)')

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    # Set output path
    output_path = Path(args.output) if args.output else None

    # Print header
    print("=" * 70)
    print(f"     Symbol Data Extraction Utility")
    print("=" * 70)

    if args.symbol.upper() == 'ALLSYMBOLS':
        print(f"\n🎯 Extracting ALL symbols")
    else:
        print(f"\n🎯 Extracting data for symbol: {args.symbol.upper()}")

    if args.forecast_only:
        print(f"   Mode: Forecast-only (Date, Symbol, predicted_return)")

    print(f"   From file: {input_path.name}\n")

    # Extract the data
    output_file, row_count = extract_symbol_data(
        input_path,
        args.symbol,
        output_path,
        forecast_only=args.forecast_only
    )

    if output_file is None:
        return 1

    print("\n" + "=" * 70)
    print(f"✅ Extraction complete!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
