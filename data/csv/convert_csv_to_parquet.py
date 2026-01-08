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

    print(f"📂 Reading CSV: {input_file.name}...")
    df = pd.read_csv(input_file)
    print(f"  • Rows: {len(df):,}")
    print(f"  • Columns: {len(df.columns)}")
    print(f"  • CSV size: {input_file.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n💾 Writing Parquet: {output_file.name}...")
    df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    print(f"  • Parquet size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  • Compression ratio: {input_file.stat().st_size / output_file.stat().st_size:.1f}x smaller")
    print(f"\n✅ Done!")
