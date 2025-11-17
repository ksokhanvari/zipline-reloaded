#!/usr/bin/env python
"""
Check what SIDs exist in the LSEG database for our universe tickers.
"""
import sqlite3
from pathlib import Path

# Universe tickers and their expected SIDs (from Sharadar)
EXPECTED_SIDS = {
    'AAPL': 199059,
    'MSFT': 198508,
    'GOOGL': 195146,
    'AMZN': 197029,
    'NVDA': 196754,
    'META': 194817,
    'JPM': 199853,
    'V': 193960,
    'WMT': 199233,
    'XOM': 199739,
    'TSLA': 194897,
}

db_path = Path.home() / '.zipline' / 'data' / 'custom' / 'fundamentals.sqlite'

print("=" * 80)
print("CHECKING LSEG DATABASE SIDS")
print("=" * 80)
print(f"Database: {db_path}")
print()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Checking universe tickers in LSEG database:")
print("-" * 80)
print(f"{'Ticker':<10} {'Expected SID':<15} {'Actual SID':<15} {'Match?':<10} {'Row Count'}")
print("-" * 80)

for ticker, expected_sid in EXPECTED_SIDS.items():
    # Check what SID(s) exist for this symbol
    cursor.execute("""
        SELECT DISTINCT Sid, COUNT(*) as cnt
        FROM Price
        WHERE Symbol = ?
        GROUP BY Sid
    """, (ticker,))

    results = cursor.fetchall()

    if not results:
        print(f"{ticker:<10} {expected_sid:<15} {'NOT FOUND':<15} {'❌':<10} 0")
    else:
        for actual_sid, count in results:
            match = "✓" if actual_sid == expected_sid else "❌"
            print(f"{ticker:<10} {expected_sid:<15} {actual_sid:<15} {match:<10} {count:,}")

print()

# Also check date range
print("Date range in LSEG database:")
print("-" * 80)
cursor.execute("""
    SELECT
        MIN(Date) as min_date,
        MAX(Date) as max_date,
        COUNT(DISTINCT Date) as num_dates
    FROM Price
""")
result = cursor.fetchone()
if result:
    print(f"  Min Date: {result[0]}")
    print(f"  Max Date: {result[1]}")
    print(f"  Num Dates: {result[2]:,}")

print()

# Check a sample of data for AAPL with the expected SID
print("Sample data for AAPL (SID 199059) around 2023-01-01:")
print("-" * 80)
cursor.execute("""
    SELECT Date, Sid, Symbol, ReturnOnEquity_SmartEstimat, ForwardPEG_DailyTimeSeriesRatio_, Debt_Total
    FROM Price
    WHERE Sid = ? AND Date >= '2023-01-01' AND Date <= '2023-01-31'
    ORDER BY Date
    LIMIT 10
""", (199059,))

results = cursor.fetchall()
if results:
    print(f"{'Date':<12} {'SID':<10} {'Symbol':<8} {'ROE':<12} {'PEG':<12} {'Debt':<12}")
    print("-" * 80)
    for row in results:
        date, sid, symbol, roe, peg, debt = row
        roe_str = f"{roe:.4f}" if roe is not None else "NULL"
        peg_str = f"{peg:.4f}" if peg is not None else "NULL"
        debt_str = f"{debt:.0f}" if debt is not None else "NULL"
        print(f"{date:<12} {sid:<10} {symbol:<8} {roe_str:<12} {peg_str:<12} {debt_str:<12}")
else:
    print("  No data found for AAPL (SID 199059) in this date range")

print()
print("=" * 80)

conn.close()
