#!/usr/bin/env python3
"""
Check for duplicate (Sid, Date) combinations in fundamentals data
"""
import pandas as pd
import numpy as np

print("Loading fundamentals data...")
# Adjust this path to your CSV file location
csv_path = '/data/csv/REFE_fundamentals_updated.csv'  # Update this path

# Read CSV (adjust columns as needed)
df = pd.read_csv(csv_path)

print(f"Total rows loaded: {len(df):,}")
print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns

# Check if we have Sid and Date columns (or Symbol and Date)
if 'Sid' in df.columns and 'Date' in df.columns:
    key_cols = ['Sid', 'Date']
elif 'Symbol' in df.columns and 'Date' in df.columns:
    key_cols = ['Symbol', 'Date']
else:
    print("\nAvailable columns:")
    print(df.columns.tolist())
    print("\nPlease identify the Sid/Symbol and Date columns")
    exit()

print(f"\nChecking for duplicates on {key_cols}...")

# Find duplicates
duplicates = df[df.duplicated(subset=key_cols, keep=False)]

if len(duplicates) > 0:
    print(f"\n⚠️  Found {len(duplicates):,} duplicate rows!")

    # Show examples
    print("\nSample duplicates (first 20):")
    print(duplicates[key_cols].head(20))

    # Count duplicates per key
    dup_counts = duplicates.groupby(key_cols).size().reset_index(name='count')
    dup_counts = dup_counts.sort_values('count', ascending=False)

    print(f"\nTop 10 most duplicated (Sid, Date) pairs:")
    print(dup_counts.head(10))

    # Solution options
    print("\n" + "=" * 60)
    print("SOLUTION OPTIONS")
    print("=" * 60)
    print("\n1. Keep first occurrence:")
    print("   df_clean = df.drop_duplicates(subset=['Sid', 'Date'], keep='first')")

    print("\n2. Keep last occurrence (most recent data):")
    print("   df_clean = df.drop_duplicates(subset=['Sid', 'Date'], keep='last')")

    print("\n3. Aggregate duplicates (take mean of numeric columns):")
    print("   df_clean = df.groupby(['Sid', 'Date'], as_index=False).agg({")
    print("       'NumericCol': 'mean',  # Average numeric values")
    print("       'TextCol': 'first',    # Keep first text value")
    print("   })")

    print("\n4. Use INSERT OR REPLACE in database:")
    print("   Set UPDATE_MODE = 'replace' in the notebook")

else:
    print("\n✓ No duplicates found! Data is clean.")
    print(f"  Unique (Sid, Date) combinations: {len(df):,}")
