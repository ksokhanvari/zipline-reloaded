#!/usr/bin/env python
"""
Diagnostic script to check fundamentals database contents and data types.
"""

import pandas as pd
import sqlite3
from pathlib import Path
from zipline.data.custom import connect_db, describe_custom_db

DB_CODE = "fundamentals"


def main():
    print("=" * 80)
    print("FUNDAMENTALS DATABASE DIAGNOSTIC")
    print("=" * 80)
    print()

    # Get database info
    try:
        db_info = describe_custom_db(DB_CODE)
        print(f"Database: {db_info['db_path']}")
        print(f"Bar size: {db_info['bar_size']}")
        print(f"Columns: {', '.join(db_info['columns'].keys())}")
        print()
    except Exception as e:
        print(f"✗ Error getting database info: {e}")
        return 1

    # Connect and query
    conn = connect_db(DB_CODE)

    try:
        # Get table schema
        print("Table Schema:")
        print("-" * 80)
        schema = pd.read_sql_query("PRAGMA table_info(Price)", conn)
        print(schema[['name', 'type', 'notnull']].to_string(index=False))
        print()

        # Get row count
        count = pd.read_sql_query("SELECT COUNT(*) as count FROM Price", conn)
        print(f"Total rows: {count['count'][0]}")
        print()

        # Get sample data
        print("Sample Data (first 5 rows):")
        print("-" * 80)
        sample = pd.read_sql_query("SELECT * FROM Price LIMIT 5", conn)
        print(sample.to_string(index=False))
        print()

        # Check data types in the actual data
        print("Actual Data Types in DataFrame:")
        print("-" * 80)
        for col in sample.columns:
            print(f"  {col}: {sample[col].dtype}")
        print()

        # Check for unique SIDs
        sids = pd.read_sql_query("SELECT DISTINCT Sid FROM Price ORDER BY Sid", conn)
        print(f"Unique SIDs ({len(sids)}):")
        print(f"  {', '.join(map(str, sids['Sid'].tolist()))}")
        print()

        # Check for date range
        dates = pd.read_sql_query("SELECT MIN(Date) as min_date, MAX(Date) as max_date FROM Price", conn)
        print(f"Date range: {dates['min_date'][0]} to {dates['max_date'][0]}")
        print()

        # Check for specific numeric columns
        print("Sample Numeric Values:")
        print("-" * 80)
        numeric_cols = ['ROE', 'PERatio', 'DebtToEquity', 'EPS']
        for col in numeric_cols:
            if col in sample.columns:
                values = sample[col].head(3).tolist()
                dtype = sample[col].dtype
                print(f"  {col}: {values} (dtype: {dtype})")
        print()

        # Try to detect if values are stored as strings
        print("String Detection Test:")
        print("-" * 80)
        test_query = "SELECT ROE, PERatio, EPS FROM Price LIMIT 3"
        test_df = pd.read_sql_query(test_query, conn)
        for col in test_df.columns:
            first_val = test_df[col].iloc[0] if len(test_df) > 0 else None
            print(f"  {col}: type={type(first_val).__name__}, value={first_val}")
        print()

    except Exception as e:
        print(f"✗ Error querying database: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
