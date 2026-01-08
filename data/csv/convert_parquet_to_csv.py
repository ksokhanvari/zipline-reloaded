#!/usr/bin/env python3
"""
Quick script to convert Parquet to CSV for compatibility and sharing.

Usage:
    python convert_parquet_to_csv.py input.parquet output.csv
"""

import sys
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_parquet_to_csv.py input.parquet output.csv")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    # Verify input file exists
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)

    # Verify input is Parquet
    if input_file.suffix.lower() not in ['.parquet', '.pq']:
        print(f"⚠️  Warning: Input file doesn't have .parquet extension")
        print(f"   File: {input_file.name}")
        print(f"   Will attempt to read as Parquet anyway...")

    print(f"📂 Reading Parquet: {input_file.name}...")
    try:
        df = pd.read_parquet(input_file, engine='pyarrow')
    except Exception as e:
        print(f"❌ Error reading Parquet file: {e}")
        print(f"\nTip: Make sure PyArrow is installed: pip install pyarrow")
        sys.exit(1)

    print(f"  • Rows: {len(df):,}")
    print(f"  • Columns: {len(df.columns)}")
    print(f"  • Parquet size: {input_file.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n💾 Writing CSV: {output_file.name}...")
    df.to_csv(output_file, index=False)
    print(f"  • CSV size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  • Expansion ratio: {output_file.stat().st_size / input_file.stat().st_size:.1f}x larger")
    print(f"\n✅ Done!")
