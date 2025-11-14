#!/usr/bin/env python3
"""
Load Fundamentals Data to SQLite Database

This script loads fundamental data from CSV files into a SQLite database
for use with Zipline's CustomSQLiteLoader.

Key Features:
- Handles duplicate (Sid, Date) pairs automatically
- Maps tickers to Zipline sids
- Supports incremental updates with INSERT OR REPLACE
- Batch processing for large datasets
- Progress reporting

Usage:
    python load_fundamentals_to_db.py
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from zipline.data.bundles import load as load_bundle

# Register Sharadar bundle (required)
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

register('sharadar', sharadar_bundle(tickers=None, incremental=True, include_funds=True))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database configuration
DATABASE_NAME = "refe-fundamentals"
DB_DIR = Path('/root/.zipline/data/custom')
DB_PATH = DB_DIR / f"{DATABASE_NAME}.sqlite"

# Data source (update this to match your CSV file)
CSV_PATH = '/data/csv/sample_20091231_20251111.csv'

# Processing configuration
BATCH_SIZE = 10000  # Rows per batch
UPDATE_MODE = 'replace'  # 'replace' or 'ignore'

# Text columns that need empty string for missing values
TEXT_COLUMNS = ['Symbol', 'CompanyCommonName', 'GICSSectorName', 'TradeDate']

print("=" * 60)
print("FUNDAMENTALS DATABASE LOADER")
print("=" * 60)
print(f"Database: {DB_PATH}")
print(f"CSV Source: {CSV_PATH}")
print(f"Update Mode: {UPDATE_MODE}")
print(f"Batch Size: {BATCH_SIZE:,}")
print("=" * 60)

# =============================================================================
# LOAD BUNDLE FOR SID MAPPING
# =============================================================================

print("\n1. Loading Zipline bundle for sid mapping...")
bundle_data = load_bundle('sharadar')
asset_finder = bundle_data.asset_finder

# Create ticker to sid mapping
all_assets = asset_finder.retrieve_all(asset_finder.sids)
ticker_to_sid = {asset.symbol: asset.sid for asset in all_assets if hasattr(asset, 'symbol')}
print(f"✓ Loaded {len(ticker_to_sid):,} ticker→sid mappings")

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("\n2. Loading CSV data...")
df = pd.read_csv(CSV_PATH)
print(f"✓ Loaded {len(df):,} rows")
print(f"  Columns: {len(df.columns)}")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  Unique symbols: {df['Symbol'].nunique():,}")

# Map symbols to sids
print("\n3. Mapping symbols to Zipline sids...")
df['Sid'] = df['Symbol'].map(ticker_to_sid)

# Report mapping results
unmapped = df['Sid'].isna().sum()
mapped = len(df) - unmapped
print(f"✓ Mapped: {mapped:,} rows ({mapped/len(df)*100:.1f}%)")
if unmapped > 0:
    print(f"⚠ Unmapped: {unmapped:,} rows ({unmapped/len(df)*100:.1f}%)")
    print(f"  Sample unmapped symbols: {df[df['Sid'].isna()]['Symbol'].unique()[:10].tolist()}")

# Remove unmapped rows
df = df.dropna(subset=['Sid'])
print(f"✓ After removing unmapped: {len(df):,} rows")

# Convert Sid to integer
df['Sid'] = df['Sid'].astype(int)

# =============================================================================
# DEDUPLICATE DATA
# =============================================================================

print("\n4. Checking for duplicates...")
duplicates_before = len(df)
df = df.drop_duplicates(subset=['Sid', 'Date'], keep='last')
duplicates_removed = duplicates_before - len(df)

if duplicates_removed > 0:
    print(f"⚠ Found and removed {duplicates_removed:,} duplicate (Sid, Date) pairs")
    print(f"  Kept 'last' occurrence for each duplicate")
else:
    print("✓ No duplicates found")

print(f"✓ Final dataset: {len(df):,} rows")

# =============================================================================
# PREPARE TEXT COLUMNS
# =============================================================================

print("\n5. Preparing text columns...")
for col in TEXT_COLUMNS:
    if col in df.columns:
        df[col] = df[col].fillna('')
        print(f"✓ {col}: filled NaN with empty string")

# =============================================================================
# CREATE DATABASE
# =============================================================================

print("\n6. Setting up database...")
DB_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get column types
column_types = {}
for col in df.columns:
    dtype = df[col].dtype
    if col in ['Sid']:
        column_types[col] = 'INTEGER'
    elif col in TEXT_COLUMNS:
        column_types[col] = 'TEXT'
    elif dtype in ['int64', 'int32']:
        column_types[col] = 'INTEGER'
    elif dtype in ['float64', 'float32']:
        column_types[col] = 'REAL'
    else:
        column_types[col] = 'TEXT'

# Create table if not exists
columns_def = ', '.join([f"{col} {dtype}" for col, dtype in column_types.items()])
create_sql = f"""
    CREATE TABLE IF NOT EXISTS Price (
        {columns_def},
        PRIMARY KEY (Sid, Date)
    )
"""
cursor.execute(create_sql)
print("✓ Table created/verified")

# Create index
cursor.execute("CREATE INDEX IF NOT EXISTS idx_date_sid ON Price(Date, Sid)")
print("✓ Index created/verified")

conn.commit()

# =============================================================================
# INSERT DATA IN BATCHES
# =============================================================================

print(f"\n7. Inserting data (mode: {UPDATE_MODE})...")

# Prepare INSERT statement
placeholders = ','.join(['?' for _ in df.columns])
if UPDATE_MODE == 'replace':
    insert_sql = f"INSERT OR REPLACE INTO Price VALUES ({placeholders})"
elif UPDATE_MODE == 'ignore':
    insert_sql = f"INSERT OR IGNORE INTO Price VALUES ({placeholders})"
else:
    insert_sql = f"INSERT INTO Price VALUES ({placeholders})"

# Insert in batches
total_rows = len(df)
num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, total_rows, BATCH_SIZE):
    batch_num = i // BATCH_SIZE + 1
    chunk = df.iloc[i:i+BATCH_SIZE]

    try:
        cursor.executemany(insert_sql, chunk.values.tolist())
        conn.commit()

        if batch_num % 10 == 0 or batch_num == num_batches:
            rows_so_far = min(i + BATCH_SIZE, total_rows)
            print(f"  Processed batch {batch_num}/{num_batches} ({rows_so_far:,} rows)...")

    except sqlite3.IntegrityError as e:
        print(f"⚠ Integrity error in batch {batch_num}: {e}")
        conn.rollback()
        continue

# =============================================================================
# VERIFY DATABASE
# =============================================================================

print("\n8. Verifying database...")
stats = pd.read_sql("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT Sid) as unique_sids,
        COUNT(DISTINCT Date) as unique_dates,
        MIN(Date) as min_date,
        MAX(Date) as max_date
    FROM Price
""", conn)

print("\n" + "=" * 60)
print("DATABASE STATISTICS")
print("=" * 60)
print(f"Total rows: {stats['total_rows'].iloc[0]:,}")
print(f"Unique sids: {stats['unique_sids'].iloc[0]:,}")
print(f"Unique dates: {stats['unique_dates'].iloc[0]:,}")
print(f"Date range: {stats['min_date'].iloc[0]} to {stats['max_date'].iloc[0]}")
print(f"Database size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
print("=" * 60)

# Sample data
print("\nSample data:")
sample = pd.read_sql("SELECT * FROM Price LIMIT 5", conn)
print(sample[['Symbol', 'Sid', 'Date', 'CompanyMarketCap', 'GICSSectorName']].to_string())

conn.close()

print("\n✓ Database loading complete!")
print(f"\nDatabase ready at: {DB_PATH}")
print("You can now run backtests using this data.")
