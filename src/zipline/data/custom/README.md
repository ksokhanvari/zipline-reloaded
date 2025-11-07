# Zipline Custom Data Module

A modular system for loading and using custom tabular data (fundamentals, alternative data, etc.) in Zipline backtests and Pipeline analyses.

## Overview

The custom data module allows you to:

- **Store custom data** in local SQLite databases with flexible schemas
- **Load CSV data** with automatic symbol-to-sid mapping
- **Query data** with flexible filtering by date range and assets
- **Integrate with Pipeline** using dynamically generated DataSet classes
- **Manage databases** via command-line interface

## Quick Start

### 1. Create a Database

```bash
zipline custom-data create-db fundamentals \
    --columns "Revenue:int,EPS:float,MarketCap:int,Currency:text"
```

This creates a SQLite database at `~/.zipline/custom_data/quant_fundamentals.sqlite`.

### 2. Load Data from CSV

```bash
zipline custom-data load-csv data.csv fundamentals \
    --securities-csv securities.csv \
    --id-col Symbol \
    --date-col Date
```

**Required CSV format:**
- Data CSV: Must have identifier column (e.g., `Symbol`) and date column (e.g., `Date`)
- Securities CSV: Must have `Symbol` and `Sid` columns for mapping

**Example data.csv:**
```csv
Symbol,Date,Revenue,EPS,MarketCap,Currency
AAPL,2020-01-01,91819000000,3.28,1301870000000,USD
MSFT,2020-01-01,36906000000,1.51,1225960000000,USD
```

**Example securities.csv:**
```csv
Symbol,Sid,Name,Exchange
AAPL,24,Apple Inc.,NASDAQ
MSFT,5061,Microsoft Corporation,NASDAQ
```

### 3. Use in Pipeline

```python
from zipline.data.custom import (
    make_custom_dataset_class,
    CustomSQLiteLoader,
    describe_custom_db,
)

# Get database info
info = describe_custom_db('fundamentals')

# Create DataSet class
FundamentalsDataSet = make_custom_dataset_class(
    db_code='fundamentals',
    columns=info['columns'],
)

# Create pipeline
from zipline.pipeline import Pipeline

pipeline = Pipeline(
    columns={
        'eps': FundamentalsDataSet.EPS.latest,
        'revenue': FundamentalsDataSet.Revenue.latest,
    }
)

# Create loader and register with engine
loader = CustomSQLiteLoader('fundamentals')
# (Register with SimplePipelineEngine - see examples/)
```

### 4. Query Data Directly

```python
from zipline.data.custom import get_prices

# Get all data for specific assets and date range
df = get_prices(
    db_code='fundamentals',
    start_date='2020-01-01',
    end_date='2020-12-31',
    sids=[24, 5061],  # AAPL, MSFT
)

print(df)
```

## Architecture

### Database Structure

Each custom database contains:

1. **ConfigBlob table**: Stores metadata (db_code, bar_size, columns, version)
2. **Price table**: Stores actual data
   - `Sid` (VARCHAR): Asset identifier
   - `Date` (DATETIME): Date/timestamp
   - Custom columns (defined by user)
   - Primary key: `(Sid, Date)`
   - Indices on `Sid` and `Date` for performance

### Module Structure

```
zipline/data/custom/
├── __init__.py              # Public API exports
├── config.py                # Configuration and validation
├── db_manager.py            # Database creation and introspection
├── insert_utils.py          # Insert strategies (replace/ignore/fail)
├── loader.py                # CSV loading with chunking
├── query.py                 # Data retrieval functions
├── pipeline_integration.py  # Pipeline DataSet and Loader
├── cli.py                   # Command-line interface
├── README.md                # This file
└── examples/
    ├── example_fundamentals.csv
    ├── securities.csv
    └── pipeline_example.py
```

## CLI Reference

### create-db

Create a new custom data database.

```bash
zipline custom-data create-db <db-code> --columns <cols> [options]
```

**Options:**
- `--bar-size TEXT`: Bar size/frequency (default: "1 day")
- `--columns TEXT`: Column definitions as "name:type" pairs (required)
- `--db-dir PATH`: Database directory (default: ~/.zipline/custom_data/)

**Supported column types:**
- `int`: Integer values (stored as INTEGER)
- `float`: Floating-point values (stored as REAL)
- `text`: Text strings (stored as TEXT)
- `date`: Dates (stored as TEXT in ISO 8601 format)
- `datetime`: Timestamps (stored as TEXT in ISO 8601 format)

**Example:**
```bash
zipline custom-data create-db fundamentals \
    --bar-size "1 quarter" \
    --columns "Revenue:int,EPS:float,PE_Ratio:float,Currency:text"
```

### load-csv

Load CSV data into a custom database.

```bash
zipline custom-data load-csv <csv-path> <db-code> [options]
```

**Options:**
- `--securities-csv PATH`: CSV with Symbol-to-Sid mapping
- `--id-col TEXT`: Identifier column name (default: "Symbol")
- `--date-col TEXT`: Date column name (default: "Date")
- `--date-format TEXT`: Date format (e.g., "%Y-%m-%d")
- `--tz TEXT`: Timezone (e.g., "UTC", "America/New_York")
- `--chunk-size INT`: Rows per batch (default: 100000)
- `--on-duplicate [replace|ignore|fail]`: Duplicate handling (default: replace)
- `--fail-on-unmapped`: Fail if identifiers can't be mapped
- `--db-dir PATH`: Database directory

**Examples:**
```bash
# Basic load
zipline custom-data load-csv data.csv fundamentals \
    --securities-csv securities.csv

# With custom date format and timezone
zipline custom-data load-csv data.csv fundamentals \
    --securities-csv securities.csv \
    --date-format "%m/%d/%Y" \
    --tz "America/New_York"

# Skip unmapped identifiers
zipline custom-data load-csv data.csv fundamentals \
    --securities-csv securities.csv \
    --skip-unmapped

# Ignore duplicates instead of replacing
zipline custom-data load-csv data.csv fundamentals \
    --securities-csv securities.csv \
    --on-duplicate ignore
```

### describe-db

Show database metadata and statistics.

```bash
zipline custom-data describe-db <db-code> [options]
```

**Options:**
- `--db-dir PATH`: Database directory

**Example:**
```bash
zipline custom-data describe-db fundamentals
```

**Output:**
```
Database: fundamentals
Path: /home/user/.zipline/custom_data/quant_fundamentals.sqlite
Bar Size: 1 quarter

Columns:
  - Revenue: int
  - EPS: float
  - MarketCap: int
  - Currency: text

Statistics:
  Total rows: 160
  Unique Sids: 10
  Date range: 2020-01-01 to 2020-12-31

Sample Sids:
  - 24
  - 5061
  - 26890
  ...
```

### list-dbs

List all available custom databases.

```bash
zipline custom-data list-dbs [options]
```

**Options:**
- `--db-dir PATH`: Database directory

**Example:**
```bash
zipline custom-data list-dbs
```

**Output:**
```
Found 2 custom database(s):

fundamentals
  Rows: 160
  Sids: 10
  Columns: Revenue, EPS, MarketCap, Currency
  Dates: 2020-01-01 to 2020-12-31

alternative_data
  Rows: 5000
  Sids: 50
  Columns: Sentiment, Volume, Mentions
  Dates: 2019-01-01 to 2021-12-31
```

### dump-db

Export database data to CSV.

```bash
zipline custom-data dump-db <db-code> <output-path> [options]
```

**Options:**
- `--start-date TEXT`: Start date (YYYY-MM-DD)
- `--end-date TEXT`: End date (YYYY-MM-DD)
- `--sids TEXT`: Comma-separated Sids to export
- `--db-dir PATH`: Database directory

**Examples:**
```bash
# Export all data
zipline custom-data dump-db fundamentals output.csv

# Export specific date range
zipline custom-data dump-db fundamentals output.csv \
    --start-date 2020-01-01 \
    --end-date 2020-06-30

# Export specific assets
zipline custom-data dump-db fundamentals output.csv \
    --sids 24,5061,26890
```

### run-example

Run a simple Pipeline example using custom data.

```bash
zipline custom-data run-example <db-code> --start-date <date> --end-date <date> [options]
```

**Options:**
- `--start-date TEXT`: Start date (required)
- `--end-date TEXT`: End date (required)
- `--db-dir PATH`: Database directory

**Example:**
```bash
zipline custom-data run-example fundamentals \
    --start-date 2020-01-01 \
    --end-date 2020-12-31
```

## Python API Reference

### Database Management

#### `create_custom_db(db_code, bar_size, columns, db_dir=None)`

Create a new custom data database.

**Parameters:**
- `db_code` (str): Database identifier (e.g., "fundamentals")
- `bar_size` (str): Data frequency (e.g., "1 day", "1 week", "1 quarter")
- `columns` (dict): Column definitions `{name: type}`
- `db_dir` (str/Path, optional): Database directory

**Returns:** Path to created database

**Example:**
```python
from zipline.data.custom import create_custom_db

db_path = create_custom_db(
    db_code='fundamentals',
    bar_size='1 quarter',
    columns={
        'Revenue': 'int',
        'EPS': 'float',
        'MarketCap': 'int',
    }
)
```

#### `describe_custom_db(db_code, db_dir=None)`

Get database metadata and statistics.

**Returns:** Dictionary with:
- `db_code`, `db_path`, `bar_size`
- `columns`: Column definitions
- `row_count`, `num_sids`
- `date_range`: (min_date, max_date)
- `sids`: List of unique Sids

#### `list_custom_dbs(db_dir=None)`

List all custom databases.

**Returns:** List of database codes

#### `connect_db(db_code, db_dir=None)`

Connect to a custom database.

**Returns:** `sqlite3.Connection`

### Data Loading

#### `load_csv_to_db(csv_path, db_code, sid_map=None, ...)`

Load CSV data into database.

**Parameters:**
- `csv_path` (str/Path): Path to CSV file
- `db_code` (str): Database code
- `sid_map` (dict/DataFrame, optional): Symbol-to-Sid mapping
- `id_col` (str): Identifier column name (default: "Symbol")
- `date_col` (str): Date column name (default: "Date")
- `date_format` (str, optional): Date format string
- `tz` (str, optional): Timezone
- `chunk_size` (int): Rows per batch (default: 100000)
- `on_duplicate` (str): "replace", "ignore", or "fail" (default: "replace")
- `fail_on_unmapped` (bool): Fail on unmapped identifiers (default: True)
- `db_dir` (str/Path, optional): Database directory

**Returns:** Dictionary with:
- `rows_inserted`: Number of rows added/updated
- `rows_skipped`: Number of rows skipped
- `unmapped_ids`: List of unmapped identifiers
- `errors`: List of error messages

**Example:**
```python
from zipline.data.custom import load_csv_to_db
import pandas as pd

# Load securities mapping
securities = pd.read_csv('securities.csv')

# Load data
result = load_csv_to_db(
    csv_path='fundamentals.csv',
    db_code='fundamentals',
    sid_map=securities,
    id_col='Symbol',
    date_col='Date',
    on_duplicate='replace',
)

print(f"Inserted: {result['rows_inserted']}")
print(f"Skipped: {result['rows_skipped']}")
```

### Data Querying

#### `get_prices(db_code, start_date=None, end_date=None, sids=None, fields=None, db_dir=None)`

Retrieve custom data with optional filters.

**Parameters:**
- `db_code` (str): Database code
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `sids` (list, optional): List of Sids to retrieve
- `fields` (list, optional): List of field names to retrieve
- `db_dir` (str/Path, optional): Database directory

**Returns:** DataFrame with columns `[Sid, Date, <custom columns>]`

**Example:**
```python
from zipline.data.custom import get_prices

# Get all data
df = get_prices('fundamentals')

# Get specific date range and assets
df = get_prices(
    db_code='fundamentals',
    start_date='2020-01-01',
    end_date='2020-12-31',
    sids=[24, 5061, 26890],
    fields=['Revenue', 'EPS'],
)
```

#### `get_prices_reindexed_like(db_code, template_df, fields=None, db_dir=None)`

Align custom data to template DataFrame shape.

**Parameters:**
- `db_code` (str): Database code
- `template_df` (DataFrame): Template with MultiIndex (Date, Sid)
- `fields` (list, optional): Fields to retrieve
- `db_dir` (str/Path, optional): Database directory

**Returns:** DataFrame aligned to template shape

#### `get_latest_values(db_code, as_of_date, sids=None, fields=None, db_dir=None)`

Get most recent values as of a specific date.

**Parameters:**
- `db_code` (str): Database code
- `as_of_date` (str): Reference date
- `sids` (list, optional): Sids to retrieve
- `fields` (list, optional): Fields to retrieve
- `db_dir` (str/Path, optional): Database directory

**Returns:** DataFrame with latest available values

### Pipeline Integration

#### `make_custom_dataset_class(db_code, columns, base_name=None)`

Create a DataSet class for Pipeline.

**Parameters:**
- `db_code` (str): Database code
- `columns` (dict): Column definitions `{name: type}`
- `base_name` (str, optional): DataSet class name prefix

**Returns:** DataSet subclass with Column attributes

**Example:**
```python
from zipline.data.custom import make_custom_dataset_class

FundamentalsDataSet = make_custom_dataset_class(
    db_code='fundamentals',
    columns={'Revenue': 'int', 'EPS': 'float'},
)

# Use in pipeline
from zipline.pipeline import Pipeline

pipeline = Pipeline(
    columns={
        'eps': FundamentalsDataSet.EPS.latest,
        'revenue': FundamentalsDataSet.Revenue.latest,
    }
)
```

#### `CustomSQLiteLoader(db_code, db_dir=None)`

PipelineLoader for custom data.

**Parameters:**
- `db_code` (str): Database code
- `db_dir` (str/Path, optional): Database directory

**Example:**
```python
from zipline.data.custom import CustomSQLiteLoader

loader = CustomSQLiteLoader('fundamentals')
```

#### `register_custom_loader(engine, dataset_class, loader)`

Register custom loader with Pipeline engine.

**Parameters:**
- `engine`: SimplePipelineEngine instance
- `dataset_class`: DataSet class (from `make_custom_dataset_class`)
- `loader`: CustomSQLiteLoader instance

**Example:**
```python
from zipline.data.custom import (
    make_custom_dataset_class,
    CustomSQLiteLoader,
    register_custom_loader,
)
from zipline.pipeline.engine import SimplePipelineEngine

# Create DataSet and loader
FundamentalsDataSet = make_custom_dataset_class('fundamentals', {...})
loader = CustomSQLiteLoader('fundamentals')

# Register with engine
engine = SimplePipelineEngine(...)
register_custom_loader(engine, FundamentalsDataSet, loader)
```

## Configuration

### Environment Variables

- `ZIPLINE_CUSTOM_DATA_DIR`: Override default database directory
  - Default: `~/.zipline/custom_data/`

**Example:**
```bash
export ZIPLINE_CUSTOM_DATA_DIR=/data/zipline_custom
```

## Best Practices

### 1. Identifier Mapping

Always maintain a securities CSV file with Symbol-to-Sid mappings:

```python
# Generate from bundle
from zipline.data.bundles import load
bundle_data = load('quandl')
assets = bundle_data.asset_finder.retrieve_all(
    bundle_data.asset_finder.sids
)

securities_df = pd.DataFrame({
    'Symbol': [a.symbol for a in assets],
    'Sid': [a.sid for a in assets],
    'Name': [a.asset_name for a in assets],
    'Exchange': [a.exchange for a in assets],
})

securities_df.to_csv('securities.csv', index=False)
```

### 2. Date Normalization

Ensure dates are normalized to midnight UTC for consistency:

```python
# loader.py handles this automatically
# But if you're querying directly:
df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
```

### 3. Chunked Loading

For large CSV files, use chunking to avoid memory issues:

```bash
zipline custom-data load-csv huge_file.csv fundamentals \
    --chunk-size 50000
```

### 4. Duplicate Handling

Choose appropriate strategy:
- `replace`: Update existing records (default, best for corrections)
- `ignore`: Keep existing records (best for append-only data)
- `fail`: Raise error on duplicates (best for validation)

### 5. Data Validation

Always check the result of data loading:

```python
result = load_csv_to_db(...)

if result['unmapped_ids']:
    print(f"Warning: {len(result['unmapped_ids'])} unmapped identifiers")

if result['errors']:
    print(f"Errors encountered: {result['errors']}")
```

## Examples

See `examples/` directory for complete working examples:

- `example_fundamentals.csv`: Sample fundamental data
- `securities.csv`: Sample symbol-to-sid mapping
- `pipeline_example.py`: Complete Pipeline integration example

## Troubleshooting

### "Database not found" error

**Solution:** Check available databases:
```bash
zipline custom-data list-dbs
```

### "Unmapped identifier" warnings

**Solution:** Verify securities.csv has all symbols:
```python
securities = pd.read_csv('securities.csv')
data = pd.read_csv('data.csv')
missing = set(data['Symbol']) - set(securities['Symbol'])
print(f"Missing symbols: {missing}")
```

### "Column not found" error

**Solution:** Check database schema:
```bash
zipline custom-data describe-db <db-code>
```

### Performance issues with large databases

**Solutions:**
1. Use chunked loading: `--chunk-size 50000`
2. Filter queries by date and Sid
3. Index columns used in queries
4. Consider splitting into multiple databases

## Limitations

- SQLite-based (single-writer, may not suit high-frequency updates)
- In-memory query processing (large result sets may require chunking)
- No built-in data versioning (use separate databases for versions)
- Point-in-time correctness requires careful date handling

## Future Enhancements

Potential improvements for future versions:
- PostgreSQL backend for production use
- Automatic data versioning and snapshots
- Built-in data validation rules
- Web UI for database management
- Direct integration with data providers (Quandl, Alpha Vantage, etc.)

## Support

For issues and questions:
- GitHub Issues: https://github.com/stefan-jansen/zipline-reloaded/issues
- Community: https://exchange.ml4trading.io

## License

Same as Zipline Reloaded (Apache 2.0)
