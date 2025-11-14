# Database Loading Guide

## Quick Start

To load fundamentals data into the database:

```bash
python /notebooks/load_fundamentals_to_db.py
```

## Overview

This guide covers loading custom fundamental data into SQLite for use with Zipline's custom fundamentals system.

## Common Issues and Solutions

### Issue 1: UNIQUE Constraint Failed

**Error:**
```
IntegrityError: UNIQUE constraint failed: Price.Sid, Price.Date
```

**Cause:**
- Duplicate (Sid, Date) combinations in source data
- Trying to insert data that already exists in the database

**Solutions:**

1. **Automatic deduplication** (recommended):
   - The script automatically deduplicates using `df.drop_duplicates(subset=['Sid', 'Date'], keep='last')`
   - Keeps the most recent data for each (Sid, Date) pair

2. **Use INSERT OR REPLACE**:
   - Set `UPDATE_MODE = 'replace'` in the script
   - Automatically overwrites existing records

3. **Use INSERT OR IGNORE**:
   - Set `UPDATE_MODE = 'ignore'` in the script
   - Skips records that already exist

### Issue 2: Multiple Symbols → Same Sid

**Cause:**
- Ticker changes (e.g., FB → META)
- Class shares (e.g., GOOGL and GOOG → same Sid)
- Corporate actions

**Solution:**
- **Always deduplicate on (Sid, Date) AFTER mapping symbols to sids**
- The script handles this automatically

### Issue 3: Unmapped Symbols

**Cause:**
- Symbol not in Zipline bundle
- Delisted stocks
- International stocks not in bundle

**Solution:**
- Script reports unmapped symbols
- They are automatically excluded from the database
- Check the bundle coverage if too many are unmapped

## Database Schema

```sql
CREATE TABLE Price (
    Symbol TEXT,
    Sid INTEGER NOT NULL,
    Date TEXT NOT NULL,
    -- Fundamentals columns...
    PRIMARY KEY (Sid, Date)
);

CREATE INDEX idx_date_sid ON Price(Date, Sid);
```

**Key Points:**
- Primary key on `(Sid, Date)` ensures uniqueness
- Index on `(Date, Sid)` for fast queries
- Sid is INTEGER (Zipline asset ID)
- Date is TEXT in 'YYYY-MM-DD' format

## Configuration Options

Edit these variables in `load_fundamentals_to_db.py`:

```python
# Database name
DATABASE_NAME = "refe-fundamentals"

# CSV source
CSV_PATH = '/data/csv/REFE_fundamentals_updated.csv'

# Insert mode
UPDATE_MODE = 'replace'  # or 'ignore'

# Batch size for large datasets
BATCH_SIZE = 10000

# Text columns (fill NaN with empty string)
TEXT_COLUMNS = ['Symbol', 'CompanyCommonName', 'GICSSectorName', 'TradeDate']
```

## Data Flow

```
CSV File
    ↓
Load & Clean
    ↓
Map Symbols → Sids (using Zipline bundle)
    ↓
Remove Unmapped
    ↓
Deduplicate on (Sid, Date) - keep='last'
    ↓
Fill Text Columns with ''
    ↓
Batch Insert to SQLite (with INSERT OR REPLACE)
    ↓
Database Ready
```

## Verification

After loading, the script shows:
- Total rows inserted
- Unique sids and dates
- Date range coverage
- Database file size
- Sample data

## Troubleshooting

### Database is Empty or Has Wrong Dates

**Check:**
```bash
python /notebooks/check_database.py
```

This shows:
- Actual date range in database
- Sid values
- Row counts

### Backtest Shows "No Data Found"

**Cause:** Backtest date range doesn't match database coverage

**Solution:**
1. Check database date range with `check_database.py`
2. Update backtest dates in `strategy_top5_roe.py` to match

### Too Many Duplicates

**Check source data:**
```bash
python /notebooks/check_duplicates.py
```

Shows:
- Number of duplicates
- Which (Symbol, Date) pairs are duplicated
- Suggestions for handling

## Best Practices

1. **Always deduplicate AFTER sid mapping**
   - Map symbols to sids first
   - Then deduplicate on (Sid, Date)

2. **Use INSERT OR REPLACE for updates**
   - Handles both new and updated records
   - No manual cleanup needed

3. **Fill text columns with empty string**
   - Not NaN or None
   - Prevents dtype errors in Pipeline

4. **Batch process large datasets**
   - Use BATCH_SIZE = 10000 or larger
   - Commit after each batch

5. **Verify after loading**
   - Check date range matches expectations
   - Verify row counts
   - Test with sample query

## Example: Update Workflow

```bash
# 1. Load new data (replaces duplicates automatically)
python /notebooks/load_fundamentals_to_db.py

# 2. Verify database
python /notebooks/check_database.py

# 3. Test backtest with new data
python /notebooks/strategy_top5_roe.py
```

## Docker Environment

**Paths:**
- Database: `/root/.zipline/data/custom/refe-fundamentals.sqlite`
- CSV data: `/data/csv/` (mounted volume)
- Scripts: `/notebooks/`

**Note:** The database is stored in the Docker volume and persists between container restarts.
