#!/usr/bin/env python3
"""
Quick script to convert CSV to Parquet for memory efficiency.

Usage:
    python convert_csv_to_parquet.py input.csv output.parquet
"""

import sys
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_csv_to_parquet.py input.csv output.parquet")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    # Verify input file exists
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)

    # Verify input is CSV
    if input_file.suffix.lower() != '.csv':
        print(f"⚠️  Warning: Input file doesn't have .csv extension")
        print(f"   File: {input_file.name}")
        print(f"   Will attempt to read as CSV anyway...")

    print(f"📂 Reading CSV: {input_file.name}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)

    print(f"  • Rows: {len(df):,}")
    print(f"  • Columns: {len(df.columns)}")
    print(f"  • CSV size: {input_file.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n💾 Writing Parquet: {output_file.name}...")
    try:
        df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    except Exception as e:
        print(f"❌ Error writing Parquet file: {e}")
        print(f"\nTip: Make sure PyArrow is installed: pip install pyarrow")
        sys.exit(1)

    print(f"  • Parquet size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  • Compression ratio: {input_file.stat().st_size / output_file.stat().st_size:.1f}x smaller")
    print(f"\n✅ Done!")
