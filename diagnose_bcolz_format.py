#!/usr/bin/env python
"""
Diagnose bcolz 'day' column format to fix incremental ingestion.
"""

import pandas as pd
from pathlib import Path
import bcolz

# Find latest bundle
bundle_dir = Path.home() / '.zipline' / 'data' / 'sharadar'
existing_dirs = sorted(bundle_dir.glob('*/daily_equities.bcolz'))

if not existing_dirs:
    print("No bundle found!")
    exit(1)

latest_bcolz = existing_dirs[-1]
print(f"Examining: {latest_bcolz}")
print()

# Open bcolz
table = bcolz.open(str(latest_bcolz), mode='r')

print("Columns:", table.names)
print()

# Get first 10 rows of 'day' column
day_values = table['day'][:10]
print("First 10 'day' values:")
for i, val in enumerate(day_values):
    print(f"  {i}: {val} (type: {type(val).__name__})")

print()
print("Testing different conversions:")
print()

test_val = day_values[0]
print(f"Raw value: {test_val}")
print()

# Try as nanoseconds
try:
    as_ns = pd.to_datetime(test_val, unit='ns')
    print(f"As nanoseconds: {as_ns}")
except Exception as e:
    print(f"As nanoseconds: ERROR - {e}")

# Try as days since epoch
try:
    as_days = pd.to_datetime(test_val, unit='D', origin='unix')
    print(f"As days since epoch: {as_days}")
except Exception as e:
    print(f"As days since epoch: ERROR - {e}")

# Try as seconds
try:
    as_seconds = pd.to_datetime(test_val, unit='s')
    print(f"As seconds: {as_seconds}")
except Exception as e:
    print(f"As seconds: ERROR - {e}")

# Try as int64 timestamp
try:
    as_int64 = pd.Timestamp(test_val)
    print(f"As Timestamp(int): {as_int64}")
except Exception as e:
    print(f"As Timestamp(int): ERROR - {e}")

# Check if it's already a Timestamp
print(f"\nActual type: {type(test_val)}")
print(f"Value info: {test_val}")

# Sample more rows to understand the pattern
print("\n\nAnalyzing 100 rows:")
sample = table['day'][:100]
print(f"Min: {sample.min()}")
print(f"Max: {sample.max()}")
print(f"Dtype: {sample.dtype}")
