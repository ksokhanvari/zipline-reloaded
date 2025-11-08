# Custom Data Database Management Guide

This guide covers everything you need to know about managing your Zipline custom data databases: storage locations, querying, backup, maintenance, and troubleshooting.

## Table of Contents

- [Storage Locations](#storage-locations)
- [Database Structure](#database-structure)
- [Management Functions](#management-functions)
- [Querying Data](#querying-data)
- [Maintenance Operations](#maintenance-operations)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Storage Locations

### Overview

Custom fundamental databases are stored as SQLite files in a dedicated directory. The location depends on your environment setup.

### Docker Environment

**If you're running Zipline in Docker using docker-compose:**

| Aspect | Details |
|--------|---------|
| **Container Path** | `/data/custom_databases/` |
| **Host Path** | `./data/custom_databases/` (from project root) |
| **Environment Variable** | `ZIPLINE_CUSTOM_DATA_DIR=/data/custom_databases` |
| **Persistent** | ✅ Yes - survives container restarts |
| **Volume Mapping** | `./data:/data` in docker-compose.yml |

**To access your databases in Docker:**

```bash
# From your host machine (zipline-reloaded directory)
ls -lh ./data/custom_databases/

# Example output:
# quant_fundamentals.sqlite   2.5M
# quant_sentiment.sqlite       1.8M
```

**To check from inside the container:**

```bash
# Exec into container
docker exec -it zipline-reloaded-jupyter bash

# List databases
ls -lh /data/custom_databases/
```

### Local Environment

**If you're running Zipline locally (not in Docker):**

| Aspect | Details |
|--------|---------|
| **Default Path** | `~/.zipline/custom_data/` |
| **Environment Variable** | Set `ZIPLINE_CUSTOM_DATA_DIR` to customize |
| **Persistent** | ✅ Yes - stored alongside bundle data |

**To access your databases locally:**

```bash
ls -lh ~/.zipline/custom_data/

# Or check custom location
echo $ZIPLINE_CUSTOM_DATA_DIR
ls -lh $ZIPLINE_CUSTOM_DATA_DIR
```

### Programmatic Location Check

```python
import os
from zipline.data.custom.config import get_custom_data_dir

# Get the storage directory
storage_dir = get_custom_data_dir()
print(f"Custom data directory: {storage_dir}")
print(f"Directory exists: {storage_dir.exists()}")

# Check if running in Docker
env_var = os.environ.get('ZIPLINE_CUSTOM_DATA_DIR')
if env_var:
    print(f"Using custom location: {env_var}")
else:
    print("Using default location: ~/.zipline/custom_data/")
```

### Database Filename Format

All custom databases follow this naming convention:

```
quant_{db_code}.sqlite
```

**Examples:**
- `quant_fundamentals.sqlite` (db_code='fundamentals')
- `quant_sentiment.sqlite` (db_code='sentiment')
- `quant_earnings.sqlite` (db_code='earnings')

---

## Database Structure

### Tables

Each custom database contains two tables:

#### 1. ConfigBlob Table

Stores database metadata and configuration.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key (always 1) |
| config_json | TEXT | JSON configuration |

**Config JSON Structure:**
```json
{
  "db_code": "fundamentals",
  "bar_size": "1 quarter",
  "columns": {
    "Revenue": "float",
    "NetIncome": "float",
    "ROE": "float",
    ...
  },
  "version": "1.0"
}
```

#### 2. Price Table

Stores the actual data.

| Column | Type | Description |
|--------|------|-------------|
| Sid | VARCHAR(20) | Asset ID (string) |
| Date | DATETIME | Data timestamp |
| *YourColumn1* | varies | Your data column |
| *YourColumn2* | varies | Your data column |
| ... | varies | ... |

**Primary Key:** (Sid, Date) - ensures unique records per asset per date

**Indexes:**
- `idx_price_date` on `Date` column
- `idx_price_sid` on `Sid` column

### Schema Example

```sql
CREATE TABLE ConfigBlob (
    id INTEGER PRIMARY KEY,
    config_json TEXT NOT NULL
);

CREATE TABLE Price (
    Sid VARCHAR(20) NOT NULL,
    Date DATETIME NOT NULL,
    Revenue REAL,
    NetIncome REAL,
    ROE REAL,
    PERatio REAL,
    DebtToEquity REAL,
    Sector TEXT,
    PRIMARY KEY (Sid, Date)
);

CREATE INDEX idx_price_date ON Price(Date);
CREATE INDEX idx_price_sid ON Price(Sid);
```

---

## Management Functions

Zipline provides comprehensive functions for managing custom databases.

### 1. List All Databases

```python
from zipline.data.custom import list_custom_dbs

# Get list of all database codes
dbs = list_custom_dbs()
print(f"Available databases: {dbs}")
# Output: ['fundamentals', 'sentiment', 'ratings']
```

### 2. Get Database Path

```python
from zipline.data.custom import get_db_path

# Get path to a specific database
path = get_db_path('fundamentals')
print(f"Database location: {path}")
# Output: /data/custom_databases/quant_fundamentals.sqlite
```

### 3. Describe Database

Get comprehensive metadata about a database:

```python
from zipline.data.custom import describe_custom_db

info = describe_custom_db('fundamentals')

print(f"Database: {info['db_code']}")
print(f"Location: {info['db_path']}")
print(f"Frequency: {info['bar_size']}")
print(f"Rows: {info['row_count']}")
print(f"Assets: {info['num_sids']}")
print(f"Date range: {info['date_range']}")
print(f"Columns: {info['columns']}")
print(f"Sids: {info['sids']}")
```

**Example Output:**
```
Database: fundamentals
Location: /data/custom_databases/quant_fundamentals.sqlite
Frequency: 1 quarter
Rows: 44
Assets: 11
Date range: ('2023-07-31', '2024-04-30')
Columns: {'Revenue': 'int', 'NetIncome': 'int', 'ROE': 'float', ...}
Sids: ['24', '5061', '26090', '3766', '14328', ...]
```

### 4. Connect to Database

Get a SQLite connection for direct queries:

```python
from zipline.data.custom import connect_db

# Connect to database
conn = connect_db('fundamentals')

# Use the connection...
# (see Querying Data section)

# Always close when done
conn.close()
```

### 5. Get Database Size

```python
import os
from zipline.data.custom import get_db_path

path = get_db_path('fundamentals')
size_bytes = os.path.getsize(path)
size_mb = size_bytes / (1024 * 1024)
size_gb = size_mb / 1024

print(f"Database size: {size_mb:.2f} MB")
print(f"Database size: {size_gb:.4f} GB")
print(f"Exact bytes: {size_bytes:,}")
```

---

## Querying Data

### Using Zipline Functions

**Recommended for most use cases.**

```python
from zipline.data.custom import get_prices, get_latest_values

# Query all data for specific sids
df = get_prices(
    db_code='fundamentals',
    sids=['24', '5061'],  # AAPL, MSFT
    fields=['Revenue', 'NetIncome', 'ROE', 'PERatio']
)

# Query latest values as of a specific date
df = get_latest_values(
    db_code='fundamentals',
    as_of_date='2024-01-31',
    sids=['24', '5061'],
)
```

### Direct SQL Queries

**For advanced queries and data verification.**

#### Basic Query

```python
from zipline.data.custom import connect_db
import pandas as pd

conn = connect_db('fundamentals')

# Query specific asset
query = """
    SELECT Date, Sid, Revenue, NetIncome, ROE, PERatio
    FROM Price
    WHERE Sid = '24'
    ORDER BY Date DESC
    LIMIT 4
"""

df = pd.read_sql(query, conn)
print(df)

conn.close()
```

#### Aggregate Queries

```python
# Count records per asset
query = """
    SELECT
        Sid,
        COUNT(*) as num_quarters,
        MIN(Date) as first_date,
        MAX(Date) as last_date
    FROM Price
    GROUP BY Sid
    ORDER BY num_quarters DESC
"""

df = pd.read_sql(query, conn)
```

#### Data Quality Checks

```python
# Check for missing values
query = """
    SELECT
        COUNT(*) as total_rows,
        SUM(CASE WHEN Revenue IS NULL THEN 1 ELSE 0 END) as revenue_nulls,
        SUM(CASE WHEN NetIncome IS NULL THEN 1 ELSE 0 END) as netincome_nulls,
        SUM(CASE WHEN ROE IS NULL THEN 1 ELSE 0 END) as roe_nulls
    FROM Price
"""

df = pd.read_sql(query, conn)
```

#### Date Range Queries

```python
# Get data for specific date range
query = """
    SELECT *
    FROM Price
    WHERE Date BETWEEN '2023-01-01' AND '2023-12-31'
    ORDER BY Date, Sid
"""

df = pd.read_sql(query, conn)
```

#### Complex Queries

```python
# Calculate metrics across assets
query = """
    SELECT
        Date,
        AVG(ROE) as avg_roe,
        AVG(PERatio) as avg_pe,
        MIN(PERatio) as min_pe,
        MAX(PERatio) as max_pe,
        COUNT(DISTINCT Sid) as num_assets
    FROM Price
    WHERE Date >= '2023-01-01'
    GROUP BY Date
    ORDER BY Date
"""

df = pd.read_sql(query, conn)
```

### Using SQLite Command Line

You can also query databases using the `sqlite3` command-line tool:

```bash
# Connect to database
sqlite3 /data/custom_databases/quant_fundamentals.sqlite

# Run queries
sqlite> SELECT COUNT(*) FROM Price;
sqlite> SELECT DISTINCT Sid FROM Price ORDER BY Sid;
sqlite> SELECT * FROM Price WHERE Sid='24' LIMIT 5;

# Exit
sqlite> .quit
```

---

## Maintenance Operations

### Updating Data

To update existing records:

```python
from zipline.data.custom import load_csv_to_db
import pandas as pd

# Load new/updated data
result = load_csv_to_db(
    csv_path='updated_fundamentals.csv',
    db_code='fundamentals',
    sid_map=securities_df,
    id_col='Ticker',
    date_col='Date',
    on_duplicate='replace',  # ← Updates existing records
)

print(f"Rows updated: {result['rows_inserted']}")
```

**Options for `on_duplicate`:**
- `'replace'` - Update existing records with new values
- `'ignore'` - Skip existing records, only insert new ones
- `'fail'` - Raise error if duplicates are found

### Backing Up Databases

#### Simple File Copy

```bash
# Docker environment
cp ./data/custom_databases/quant_fundamentals.sqlite \
   ./data/custom_databases/quant_fundamentals_$(date +%Y%m%d).sqlite

# Local environment
cp ~/.zipline/custom_data/quant_fundamentals.sqlite \
   ~/.zipline/custom_data/quant_fundamentals_$(date +%Y%m%d).sqlite
```

#### Programmatic Backup

```python
import shutil
from datetime import datetime
from zipline.data.custom import get_db_path

# Source database
source = get_db_path('fundamentals')

# Create backup with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup = source.parent / f'quant_fundamentals_backup_{timestamp}.sqlite'

# Copy file
shutil.copy(source, backup)
print(f"Backup created: {backup}")
```

#### Automated Backup Script

```python
import shutil
from pathlib import Path
from datetime import datetime
from zipline.data.custom import list_custom_dbs, get_db_path

def backup_all_databases(backup_dir=None):
    """Backup all custom databases with timestamp."""
    if backup_dir is None:
        backup_dir = get_db_path('fundamentals').parent / 'backups'

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for db_code in list_custom_dbs():
        source = get_db_path(db_code)
        backup = backup_dir / f'quant_{db_code}_{timestamp}.sqlite'
        shutil.copy(source, backup)
        print(f"Backed up {db_code}: {backup}")

# Run backup
backup_all_databases()
```

### Vacuuming Database

SQLite databases can become fragmented over time. Use VACUUM to reclaim space:

```python
from zipline.data.custom import connect_db

conn = connect_db('fundamentals')
conn.execute("VACUUM")
conn.close()

print("Database vacuumed - freed unused space")
```

### Verifying Database Integrity

```python
from zipline.data.custom import connect_db

conn = connect_db('fundamentals')

# Check integrity
result = conn.execute("PRAGMA integrity_check").fetchone()

if result[0] == 'ok':
    print("✓ Database integrity check passed")
else:
    print(f"⚠ Database integrity issue: {result[0]}")

conn.close()
```

### Deleting a Database

**CAUTION: This is permanent!**

```python
import os
from zipline.data.custom import get_db_path

# Delete database file
db_path = get_db_path('old_database')
os.remove(db_path)
print(f"Deleted: {db_path}")
```

### Renaming a Database

```python
from zipline.data.custom import get_db_path

old_path = get_db_path('fundamentals')
new_path = old_path.parent / 'quant_fundamentals_old.sqlite'

old_path.rename(new_path)
print(f"Renamed to: {new_path}")
```

---

## Troubleshooting

### Database Not Found

**Error:**
```
FileNotFoundError: Database not found: /data/custom_databases/quant_fundamentals.sqlite
```

**Solutions:**

1. Check if database exists:
```python
from zipline.data.custom import list_custom_dbs
print(f"Available: {list_custom_dbs()}")
```

2. Verify storage location:
```python
from zipline.data.custom.config import get_custom_data_dir
print(f"Looking in: {get_custom_data_dir()}")
```

3. Create database if missing:
```python
from zipline.data.custom import create_custom_db

db_path = create_custom_db(
    db_code='fundamentals',
    bar_size='1 quarter',
    columns={'Revenue': 'int', 'NetIncome': 'int', ...}
)
```

### Permission Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied: '/data/custom_databases/quant_fundamentals.sqlite'
```

**Solutions:**

1. Check file permissions:
```bash
ls -l /data/custom_databases/
```

2. Fix permissions (Docker):
```bash
docker exec -it zipline-reloaded-jupyter bash
chmod 644 /data/custom_databases/*.sqlite
```

3. Fix permissions (Local):
```bash
chmod 644 ~/.zipline/custom_data/*.sqlite
```

### Database Locked

**Error:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**

1. Close all connections:
```python
# Always close connections when done
conn.close()
```

2. Use context manager:
```python
from zipline.data.custom import connect_db

# Automatically closes connection
with connect_db('fundamentals') as conn:
    df = pd.read_sql("SELECT * FROM Price", conn)
```

3. Check for other processes:
```bash
# Find processes using the database
lsof /data/custom_databases/quant_fundamentals.sqlite
```

### Corrupted Database

**Error:**
```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions:**

1. Try to recover:
```bash
sqlite3 corrupted.sqlite ".dump" | sqlite3 recovered.sqlite
```

2. Restore from backup:
```bash
cp quant_fundamentals_backup.sqlite quant_fundamentals.sqlite
```

3. Recreate and reload data:
```python
# Delete corrupted database
os.remove(get_db_path('fundamentals'))

# Recreate
create_custom_db('fundamentals', '1 quarter', columns_dict)

# Reload data
load_csv_to_db('data.csv', 'fundamentals', sid_map_df, ...)
```

### No Data Returned

**Problem:** Queries return empty results.

**Solutions:**

1. Check if data was loaded:
```python
info = describe_custom_db('fundamentals')
print(f"Rows: {info['row_count']}")
```

2. Verify Sids:
```python
# Check available Sids
print(f"Sids in database: {info['sids']}")

# Query specific Sid
conn = connect_db('fundamentals')
df = pd.read_sql("SELECT * FROM Price WHERE Sid='24'", conn)
print(f"Rows for Sid 24: {len(df)}")
```

3. Check date format:
```python
# Dates should be ISO format: YYYY-MM-DD
df = pd.read_sql("SELECT DISTINCT Date FROM Price ORDER BY Date", conn)
print(df)
```

---

## Best Practices

### 1. Always Close Connections

```python
# ✅ Good - using context manager
from zipline.data.custom import connect_db

with connect_db('fundamentals') as conn:
    df = pd.read_sql("SELECT * FROM Price", conn)
# Connection automatically closed

# ✅ Also good - explicit close
conn = connect_db('fundamentals')
try:
    df = pd.read_sql("SELECT * FROM Price", conn)
finally:
    conn.close()

# ❌ Bad - no close
conn = connect_db('fundamentals')
df = pd.read_sql("SELECT * FROM Price", conn)
# Connection left open!
```

### 2. Regular Backups

Create automated backups before major updates:

```python
def safe_update(csv_path, db_code, sid_map, **kwargs):
    """Update database with automatic backup."""
    # Backup first
    source = get_db_path(db_code)
    backup = source.parent / f'{source.stem}_backup{source.suffix}'
    shutil.copy(source, backup)

    try:
        # Load new data
        result = load_csv_to_db(csv_path, db_code, sid_map, **kwargs)
        print(f"✓ Update successful: {result['rows_inserted']} rows")
        return result
    except Exception as e:
        print(f"✗ Update failed: {e}")
        # Restore from backup
        shutil.copy(backup, source)
        print("✓ Restored from backup")
        raise
```

### 3. Validate Data Before Loading

```python
def validate_csv(csv_path, required_columns):
    """Validate CSV before loading into database."""
    df = pd.read_csv(csv_path)

    # Check required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check for duplicates
    dup_cols = ['Ticker', 'Date']
    duplicates = df[df.duplicated(subset=dup_cols, keep=False)]
    if len(duplicates) > 0:
        print(f"⚠ Warning: {len(duplicates)} duplicate records found")
        print(duplicates)

    # Check date format
    try:
        pd.to_datetime(df['Date'])
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    print("✓ CSV validation passed")
    return df

# Use before loading
df = validate_csv('fundamentals.csv', ['Ticker', 'Date', 'Revenue', ...])
result = load_csv_to_db(...)
```

### 4. Monitor Database Size

```python
def monitor_database_sizes():
    """Monitor all database sizes."""
    from zipline.data.custom import list_custom_dbs, get_db_path
    import os

    total_size = 0
    for db_code in list_custom_dbs():
        path = get_db_path(db_code)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        total_size += size_mb
        print(f"{db_code}: {size_mb:.2f} MB")

    print(f"\nTotal: {total_size:.2f} MB ({total_size/1024:.2f} GB)")

# Run periodically
monitor_database_sizes()
```

### 5. Use Appropriate Data Types

Choose the smallest data type that fits your data:

```python
# ✅ Good - saves space
columns = {
    'Revenue': 'int',        # 94,836,000,000
    'ROE': 'float',          # 47.7
    'Sector': 'text',        # 'Technology'
}

# ❌ Bad - wastes space
columns = {
    'Revenue': 'float',      # int would be enough
    'ROE': 'float',          # OK
    'Sector': 'float',       # Should be text!
}
```

### 6. Index Important Columns

The default indexes (on Sid and Date) are usually sufficient, but you can add more:

```python
conn = connect_db('fundamentals')

# Add index on frequently queried column
conn.execute("CREATE INDEX IF NOT EXISTS idx_sector ON Price(Sector)")

# Check existing indexes
indexes = conn.execute("""
    SELECT name FROM sqlite_master
    WHERE type='index' AND tbl_name='Price'
""").fetchall()

print(f"Indexes: {[idx[0] for idx in indexes]}")
conn.close()
```

### 7. Document Your Databases

Keep a metadata file:

```python
# Create metadata.json
import json

metadata = {
    'fundamentals': {
        'description': 'Quarterly fundamental data from SHARADAR',
        'source': 'Nasdaq Data Link',
        'frequency': '1 quarter',
        'assets': 1000,
        'date_range': '2010-01-01 to 2024-01-31',
        'last_updated': '2024-01-31',
        'columns': {
            'Revenue': 'Quarterly revenue in dollars',
            'NetIncome': 'Net income in dollars',
            'ROE': 'Return on equity (%)',
            ...
        }
    }
}

with open('database_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## Summary

You now have comprehensive knowledge of:

- ✅ Where custom databases are stored (Docker vs Local)
- ✅ How to list, inspect, and connect to databases
- ✅ How to query data using SQL and Zipline functions
- ✅ How to backup, update, and maintain databases
- ✅ How to troubleshoot common issues
- ✅ Best practices for database management

For more information, see:
- [README.md](README.md) - Quick start guide
- [research_with_fundamentals.ipynb](research_with_fundamentals.ipynb) - Interactive tutorial
- [DATABASE_CLASS_GUIDE.md](DATABASE_CLASS_GUIDE.md) - Database class pattern
- [BACKTEST_README.md](BACKTEST_README.md) - Backtest integration
