# LSEG Fundamentals + Sharadar Metadata Integration

This directory contains notebooks and scripts for loading LSEG fundamentals data and enriching it with Sharadar ticker metadata for universe filtering in Zipline strategies.

## Overview

The LSEG fundamentals data provides comprehensive financial metrics, but lacks metadata needed for universe filtering (exchange, category, ADR status, sector, etc.). This integration combines:

- **LSEG Fundamentals**: 38+ financial metrics from LSEG/Refinitiv
- **Sharadar Metadata**: Exchange, category, ADR status, sector, industry, market cap scale

## Directory Structure

```
examples/lseg_fundamentals/
├── README.md                                          # This file - comprehensive documentation
│
├── CSV Enrichment Tools (Recommended Workflow)
│   ├── add_sharadar_metadata_to_fundamentals.py     # CLI script for CSV enrichment
│   ├── add_sharadar_metadata_to_fundamentals.ipynb  # Interactive notebook with 9 charts
│   └── add_sharadar_metadata_to_fundamentals_full.ipynb  # Backup with embedded functions
│
├── Database Table Tools (Alternative Workflow)
│   ├── add_sharadar_tickers.ipynb                   # Create SharadarTickersDaily table
│   └── add_sharadar_tickers_fast.ipynb              # Optimized version for large datasets
│
├── Database Management Tools
│   ├── load_csv_fundamentals.ipynb                  # Load enriched CSV to fundamentals.sqlite
│   ├── create_fundamentals_db.py                    # Create fundamentals database schema
│   └── remap_fundamentals_sids.py                   # Remap SIDs after asset db changes
│
├── Research & Analysis
│   └── research_with_fundamentals.ipynb             # Example fundamentals research notebook
│
└── __pycache__/
    └── add_sharadar_metadata_to_fundamentals.cpython-311.pyc  # Compiled Python cache
```

### File Descriptions

#### **CSV Enrichment Tools** (Recommended)

1. **`add_sharadar_metadata_to_fundamentals.py`** (286 lines)
   - **Purpose**: Standalone CLI script to enrich LSEG CSV with Sharadar metadata
   - **Key Features**:
     - Loads Sharadar metadata from bundle's `tickers.h5`
     - **CRITICAL**: Enhanced deduplication handling (60,303 → 30,801 unique tickers)
     - Handles both active/delisted duplicates AND multiple active entries per ticker
     - Adds 9 metadata columns with `sharadar_` prefix
     - Reports matching statistics (95.6% match rate)
     - Optional preview mode
   - **Single Source of Truth**: Notebook imports functions from this script
   - **Usage**: See CLI examples below

2. **`add_sharadar_metadata_to_fundamentals.ipynb`** (32 cells)
   - **Purpose**: Interactive Jupyter notebook with full visualizations
   - **Key Features**:
     - Imports functions from `.py` script (no code duplication)
     - **9 chart cells**: exchange dist, category dist, ADR pie, sector dist, market cap scale, etc.
     - Comprehensive data analysis and statistics
     - Section 10: AAPL data example (last 10 rows with all 47 columns)
     - Live progress tracking and matching statistics
   - **Note**: This is the ACTIVE version with all charts and imports

3. **`add_sharadar_metadata_to_fundamentals_full.ipynb`** (31 cells)
   - **Purpose**: Backup version with embedded functions (pre-import pattern)
   - **Status**: Archived for reference
   - **Note**: Use main `.ipynb` instead (imports from script)

#### **Database Table Tools** (Alternative)

4. **`add_sharadar_tickers.ipynb`**
   - **Purpose**: Create `SharadarTickersDaily` table in fundamentals.sqlite
   - **Approach**: Symbol × Date cross-product (~241M rows for full universe)
   - **Features**:
     - Memory-efficient chunked processing
     - Creates indexes for fast queries (Symbol, Date, Symbol+Date)
     - Point-in-time correct (metadata repeated across all dates)
   - **Use Case**: When you want separate normalized table for joins

5. **`add_sharadar_tickers_fast.ipynb`**
   - **Purpose**: Optimized version with faster chunking
   - **Difference**: Better memory management for very large datasets

#### **Database Management Tools**

6. **`load_csv_fundamentals.ipynb`**
   - **Purpose**: Load LSEG fundamentals CSV into SQLite database
   - **Compatible With**: Both plain CSV and enriched CSV (with metadata columns)
   - **Creates Tables**:
     - `Price` - pricing data
     - `Valuation` - valuation metrics
     - `Income` - income statement
     - `BalanceSheet` - balance sheet
     - `CashFlow` - cash flow statement
   - **Features**: Data validation, indexing, integrity checks

7. **`create_fundamentals_db.py`**
   - **Purpose**: Create fundamentals database schema programmatically
   - **Use Case**: When you need to recreate DB structure from scratch

8. **`remap_fundamentals_sids.py`**
   - **Purpose**: Remap Security IDs after asset database changes
   - **Use Case**: When you re-ingest bundles and SIDs change

#### **Research & Analysis**

9. **`research_with_fundamentals.ipynb`**
   - **Purpose**: Example notebook showing how to query and analyze fundamentals
   - **Use Case**: Template for your own fundamentals-based research

## Contents

### 1. CSV Enrichment (Recommended for Simple Workflows)

**Purpose**: Add Sharadar metadata columns directly to LSEG fundamentals CSV before loading into database.

#### `add_sharadar_metadata_to_fundamentals.py`

Standalone Python script to enrich LSEG fundamentals CSV with Sharadar metadata.

```bash
# Usage (in Docker or local environment)
python add_sharadar_metadata_to_fundamentals.py \
    --input /data/csv/lseg_fundamentals.csv \
    --output /data/csv/lseg_fundamentals_enriched.csv \
    --sharadar-bundle sharadar \
    --preview
```

**Features**:
- Deduplicates Sharadar tickers (handles active/delisted entries)
- Adds 9 metadata columns with `sharadar_` prefix
- Reports matching statistics
- Preview mode shows sample data

**Added Columns**:
- `sharadar_exchange`: NYSE, NASDAQ, NYSEMKT, etc.
- `sharadar_category`: Domestic Common Stock, ADR, ETF, etc.
- `sharadar_is_adr`: 1=ADR, 0=Non-ADR
- `sharadar_location`: Company location
- `sharadar_sector`: Sharadar sector
- `sharadar_industry`: Sharadar industry
- `sharadar_sicsector`: SIC sector
- `sharadar_sicindustry`: SIC industry
- `sharadar_scalemarketcap`: 1-6 (Nano to Mega cap)

#### `add_sharadar_metadata_to_fundamentals.ipynb`

Interactive Jupyter notebook version with **full analysis and visualizations**.

**Features**:
- Imports functions from Python script (single source of truth)
- 31 cells with comprehensive data analysis
- 9 chart/plot cells:
  - Exchange distribution (bar chart)
  - Category distribution (bar chart)
  - ADR vs Non-ADR (pie chart)
  - Sector distribution (bar chart)
  - Market cap scale distribution (bar chart)
  - Exchange distribution in enriched data
  - Category distribution in enriched data
  - Sector distribution in enriched data
  - Metadata summary statistics

**Usage**:
```bash
# Open in JupyterLab (from Docker)
# Navigate to examples/lseg_fundamentals/
# Open add_sharadar_metadata_to_fundamentals.ipynb
```

### 2. Database Table Approach (For Pipeline Integration)

**Purpose**: Create `SharadarTickersDaily` table in fundamentals.sqlite for direct Pipeline access.

#### `add_sharadar_tickers.ipynb`

Loads Sharadar metadata into the fundamentals database as a time-series table (Symbol × Date cross-product).

**Features**:
- Creates `SharadarTickersDaily` table
- Memory-efficient chunked processing
- Indexed for fast queries (Symbol, Date, Symbol+Date)
- Point-in-time correct (repeats metadata across all dates)

**Table Structure**:
```sql
CREATE TABLE SharadarTickersDaily (
    Symbol TEXT,
    Date TEXT,
    exchange TEXT,
    category TEXT,
    is_adr INTEGER,
    location TEXT,
    sector TEXT,
    industry TEXT,
    sicsector TEXT,
    sicindustry TEXT,
    scalemarketcap TEXT
);
```

**Size**: ~241M rows for full universe (60K tickers × 4K dates)

#### `add_sharadar_tickers_fast.ipynb`

Fast version with optimized chunking for large datasets.

### 3. Loading Fundamentals into Database

#### `load_csv_fundamentals.ipynb`

Loads enriched LSEG fundamentals CSV (with Sharadar metadata) into `fundamentals.sqlite`.

**Features**:
- Loads Price, Valuation, Income, Balance Sheet, Cash Flow tables
- Validates data integrity
- Creates indexes for fast queries
- Compatible with enriched CSV (includes metadata columns)

## Workflows

### Workflow 1: CSV Enrichment → Database (Recommended)

Best for: Simple setup, all metadata in one table

```bash
# Step 1: Enrich CSV with metadata
python add_sharadar_metadata_to_fundamentals.py \
    --input /data/csv/lseg_fundamentals.csv \
    --output /data/csv/lseg_fundamentals_enriched.csv \
    --sharadar-bundle sharadar

# Step 2: Load enriched CSV into database
# Run load_csv_fundamentals.ipynb
# (Database tables now include sharadar_* columns)
```

**Pros**:
- Simpler: One table with all data
- Faster queries: No joins needed
- Easier Pipeline integration

**Cons**:
- Larger CSV file
- Metadata duplicated across time periods

### Workflow 2: Separate Metadata Table

Best for: Normalized database structure, flexible queries

```bash
# Step 1: Load LSEG fundamentals
# Run load_csv_fundamentals.ipynb

# Step 2: Add Sharadar metadata table
# Run add_sharadar_tickers.ipynb
```

**Pros**:
- Normalized: No data duplication
- Flexible: Join on Symbol, Date when needed

**Cons**:
- Requires joins for universe filtering
- More complex Pipeline setup

## Pipeline Integration

### Using Enriched CSV Metadata

```python
from zipline.pipeline import Pipeline
from zipline.pipeline.data import DataSet, Column

class CustomFundamentals(DataSet):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252

    # LSEG fundamentals
    RefPriceClose = Column(float)
    CompanyMarketCap = Column(float)

    # Sharadar metadata (from enriched CSV)
    sharadar_exchange = Column(object, missing_value='')
    sharadar_category = Column(object, missing_value='')
    sharadar_is_adr = Column(bool)
    sharadar_sector = Column(object, missing_value='')
    sharadar_scalemarketcap = Column(object, missing_value='')

# Create universe filter
exchange = CustomFundamentals.sharadar_exchange.latest
category = CustomFundamentals.sharadar_category.latest
is_adr = CustomFundamentals.sharadar_is_adr.latest

base_universe = (
    exchange.isin(['NYSE', 'NASDAQ', 'NYSEMKT']) &
    (category == 'Domestic Common Stock') &
    ~is_adr
)

# Use in pipeline
pipeline = Pipeline(
    columns={
        'market_cap': CustomFundamentals.CompanyMarketCap.latest,
        'exchange': exchange,
    },
    screen=base_universe
)
```

### Using Separate Metadata Table

See `examples/strategies/sharadar_filters.py` for custom filters using `SharadarTickers` DataSet.

## Important Notes

### Deduplication Issue (FIXED)

**Problem**: Sharadar `tickers.h5` contains duplicate entries per ticker, causing row doubling during merge (16.9M rows instead of 9.0M expected, and duplicate Date+Symbol rows in output).

**Root Cause**:
- Sharadar has 105,766 total ticker entries but only 60,303 unique tickers
- Two types of duplicates:
  1. Active vs delisted entries (different `isdelisted` values: True/False)
  2. Multiple active entries for same ticker (both `isdelisted='N'`)
- Example: AAPL had 2 entries, BOTH with `isdelisted='N'`

**Solution**: Enhanced deduplication logic in both script and notebook:

```python
if 'isdelisted' in tickers.columns:
    # Sort: non-delisted first (N before Y), then by index for stable ordering
    tickers = tickers.sort_values(['isdelisted', tickers.index.name or 'index'])
    tickers = tickers.drop_duplicates(subset='ticker', keep='first')
    print(f"After deduplication: {len(tickers)} unique tickers")
else:
    # Fallback: just deduplicate by ticker
    tickers = tickers.drop_duplicates(subset='ticker', keep='first')
    print(f"After deduplication: {len(tickers)} unique tickers")
```

**Result**: 60,303 raw entries → 30,801 unique tickers (~50% reduction). This ensures:
- Only one entry per ticker
- Prefers active (non-delisted) entries
- Stable ordering when multiple entries have same `isdelisted` value
- **NO duplicate Date+Symbol rows** in enriched output

### Sharadar Bundle Requirements

All tools require the Sharadar bundle to be ingested first:

```bash
# Ingest Sharadar bundle
docker exec zipline-reloaded-jupyter zipline ingest -b sharadar

# Verify tickers.h5 exists
docker exec zipline-reloaded-jupyter ls -lah /root/.zipline/data/sharadar/*/fundamentals/tickers.h5
```

## Files

| File | Purpose | Output |
|------|---------|--------|
| `add_sharadar_metadata_to_fundamentals.py` | CLI script for CSV enrichment | Enriched CSV file |
| `add_sharadar_metadata_to_fundamentals.ipynb` | Interactive notebook with charts | Enriched CSV + analysis |
| `add_sharadar_metadata_to_fundamentals_full.ipynb` | Backup with full implementation | Same as above |
| `add_sharadar_tickers.ipynb` | Load metadata to database table | `SharadarTickersDaily` table |
| `add_sharadar_tickers_fast.ipynb` | Optimized version for large datasets | Same as above |
| `load_csv_fundamentals.ipynb` | Load LSEG CSV to database | Multiple tables in fundamentals.sqlite |

## Data Sources

### LSEG Fundamentals

- **Source**: LSEG/Refinitiv via QuantRocket
- **Format**: CSV export
- **Coverage**: 4,440 symbols, 4,012 dates (2008-2025)
- **Size**: ~9M rows, 38 columns

### Sharadar Metadata

- **Source**: Sharadar bundle (`tickers.h5`)
- **Coverage**: 60,303 tickers total, 30,801 unique (after deduplication)
- **Fields**: exchange, category, location, sector, industry, scalemarketcap, etc.

## Matching Statistics

From the enrichment process:

```
Total rows: 9,010,487
Unique symbols: 4,440
Rows with metadata: 8,616,181 (95.6%)
```

**95.6% match rate** indicates excellent coverage between LSEG and Sharadar data.

## Related Documentation

- `/docs/SHARADAR_FUNDAMENTALS_GUIDE.md`: Sharadar fundamentals usage
- `/examples/strategies/SHARADAR_FILTERS_README.md`: Custom universe filters
- `/examples/strategies/sharadar_filters.py`: Ready-to-use filter implementations
- `/examples/strategies/fcf_yield_strategy.py`: Example strategy using filters

## Support

For issues or questions:
1. Check the notebook outputs for detailed error messages
2. Verify Sharadar bundle is properly ingested
3. Review the deduplication section if row counts don't match
4. Open an issue on GitHub with reproduction steps

---

**Author**: Kamran Sokhanvari / Hidden Point Capital
**Last Updated**: 2025-11-29
**Version**: 1.1.0 - Added comprehensive directory structure and enhanced deduplication
