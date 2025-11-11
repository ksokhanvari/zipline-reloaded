#!/usr/bin/env python
"""Check the schema of the existing assets database"""
import sqlite3
from pathlib import Path

# Find latest bundle
bundle_dir = Path.home() / '.zipline' / 'data' / 'sharadar'
existing_dbs = sorted(bundle_dir.glob('*/assets-*.sqlite'))

if existing_dbs:
    latest_db = existing_dbs[-1]
    print(f"Checking database: {latest_db}")

    conn = sqlite3.connect(str(latest_db))
    cursor = conn.cursor()

    # Get table schema
    cursor.execute("PRAGMA table_info(equities)")
    columns = cursor.fetchall()

    print("\nEquities table columns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

    # Get a sample row
    cursor.execute("SELECT * FROM equities LIMIT 1")
    sample_row = cursor.fetchone()

    print("\nSample row:")
    for i, col in enumerate(columns):
        print(f"  {col[1]}: {sample_row[i] if sample_row else 'N/A'}")

    conn.close()
else:
    print("No existing database found")
