#!/usr/bin/env python3
"""
Quick script to check database Sid types and values
"""
import sqlite3
import pandas as pd

db_path = '/root/.zipline/data/custom/refe-fundamentals.sqlite'

conn = sqlite3.connect(db_path)

# Check table schema
print("=" * 60)
print("TABLE SCHEMA")
print("=" * 60)
schema = pd.read_sql("PRAGMA table_info(Price)", conn)
print(schema[['name', 'type']])

# Check sample data from 2020
print("\n" + "=" * 60)
print("SAMPLE DATA FROM 2020-10-01")
print("=" * 60)
sample_2020 = pd.read_sql("""
    SELECT Sid, Date, typeof(Sid) as sid_type
    FROM Price
    WHERE Date='2020-10-01'
    LIMIT 10
""", conn)
print(sample_2020)

# Check sample data from 2024
print("\n" + "=" * 60)
print("SAMPLE DATA FROM 2024-10-01")
print("=" * 60)
sample_2024 = pd.read_sql("""
    SELECT Sid, Date, typeof(Sid) as sid_type
    FROM Price
    WHERE Date='2024-10-01'
    LIMIT 10
""", conn)
print(sample_2024)

# Check date distribution
print("\n" + "=" * 60)
print("DATE DISTRIBUTION")
print("=" * 60)
date_stats = pd.read_sql("""
    SELECT
        MIN(Date) as min_date,
        MAX(Date) as max_date,
        COUNT(DISTINCT Date) as unique_dates,
        COUNT(DISTINCT Sid) as unique_sids,
        COUNT(*) as total_rows
    FROM Price
""", conn)
print(date_stats)

# Check Sid values - are they numeric?
print("\n" + "=" * 60)
print("SID VALUE SAMPLES")
print("=" * 60)
sid_samples = pd.read_sql("""
    SELECT DISTINCT Sid
    FROM Price
    ORDER BY Sid
    LIMIT 20
""", conn)
print("First 20 distinct Sids:")
print(sid_samples['Sid'].tolist())

conn.close()
print("\n✓ Database check complete")
