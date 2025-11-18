# Multi-Source Data Integration for Zipline

A simple, clean architecture for combining multiple fundamental data sources (Sharadar, custom databases, etc.) in Zipline backtests.

## Quick Start

```python
from zipline import run_algorithm
from zipline.pipeline import multi_source as ms

# 1. Define your custom database
class CustomFundamentals(ms.Database):
    CODE = "fundamentals"  # Must match your SQLite database name
    LOOKBACK_WINDOW = 252

    # Define columns matching your database schema
    ROE = ms.Column(float)
    PEG = ms.Column(float)
    MarketCap = ms.Column(float)

# 2. Create a pipeline mixing Sharadar and custom data
def make_pipeline():
    # Sharadar data (note: ROE field is empty in some bundles, use FCF/PE instead)
    s_fcf = ms.SharadarFundamentals.fcf.latest
    s_marketcap = ms.SharadarFundamentals.marketcap.latest
    s_pe = ms.SharadarFundamentals.pe.latest

    # Custom data
    c_roe = CustomFundamentals.ROE.latest
    c_peg = CustomFundamentals.PEG.latest

    # Universe
    universe = s_marketcap.top(100)

    # Quality filters from each source
    sharadar_quality = (s_fcf > 0) & (s_pe > 0) & (s_pe < 30)
    custom_quality = (c_roe > 15) & (c_peg < 2.5)

    # Combined: both sources agree = higher confidence
    combined_quality = sharadar_quality & custom_quality

    # Selection
    selection = c_roe.top(10, mask=universe & combined_quality)

    return ms.Pipeline(
        columns={
            's_fcf': s_fcf,
            's_pe': s_pe,
            's_marketcap': s_marketcap,
            'c_roe': c_roe,
            'c_peg': c_peg,
        },
        screen=selection,
    )

# 3. Run backtest - that's it!
results = run_algorithm(
    start='2023-01-01',
    end='2024-01-01',
    initialize=initialize,
    bundle='sharadar',
    custom_loader=ms.setup_auto_loader(),  # Magic!
)
```

## Key Features

### Automatic Data Source Detection
The `AutoLoader` automatically:
- Detects Sharadar fundamentals columns
- Detects custom database columns
- Routes each column to the appropriate loader
- Handles SID translation transparently

### SID Translation
Custom databases use bundle SIDs, but `run_algorithm()` assigns different simulation SIDs internally. The AutoLoader automatically translates between them:

```
Pipeline Request (Simulation SID 103837)
    ↓
AutoLoader translates: 103837 → "AAPL" → Bundle SID 199059
    ↓
CustomSQLiteLoader queries database with Bundle SID 199059
    ↓
Results mapped back to Simulation SID 103837
```

No manual configuration needed - it just works!

### Lazy Loading
Loaders are created only when needed, reducing memory usage and startup time.

## Architecture

### Module Structure

```
zipline/
├── pipeline/
│   ├── multi_source.py          # Centralized imports and API
│   ├── loaders/
│   │   └── auto_loader.py       # AutoLoader implementation
│   └── data/
│       ├── sharadar.py          # Sharadar datasets
│       └── db.py                # Database and Column classes
└── data/
    └── custom/
        └── pipeline_integration.py  # CustomSQLiteLoader
```

### Import Pattern

Everything you need is available through the `multi_source` module:

```python
from zipline.pipeline import multi_source as ms

# Available components:
ms.Pipeline          # Pipeline class
ms.Database          # Base class for custom databases
ms.Column            # Column definition
ms.sharadar          # Sharadar datasets
ms.setup_auto_loader # One-line loader setup
ms.AutoLoader        # Advanced loader class
```

## Custom Database Setup

### 1. Create Database Schema

Define your database class:

```python
from zipline.pipeline import multi_source as ms

class MyFundamentals(ms.Database):
    # Required attributes
    CODE = "my_database"  # Must match SQLite filename
    LOOKBACK_WINDOW = 252  # Days to look back

    # Column definitions
    Revenue = ms.Column(float)
    EPS = ms.Column(float)
    Sector = ms.Column(object)  # For text columns
```

### 2. Database File Location

Your SQLite database should be at:
```
~/.zipline/data/custom/{CODE}.sqlite
```

For example, if `CODE = "fundamentals"`:
```
~/.zipline/data/custom/fundamentals.sqlite
```

### 3. Database Schema

Required columns:
- `Date` (TEXT): Date in YYYY-MM-DD format
- `Sid` (INTEGER): Security identifier (use bundle SIDs)
- Your custom columns (matching Column definitions)

Example schema:
```sql
CREATE TABLE Price (
    Date TEXT NOT NULL,
    Sid INTEGER NOT NULL,
    Revenue REAL,
    EPS REAL,
    Sector TEXT,
    PRIMARY KEY (Date, Sid)
);

CREATE INDEX idx_date ON Price(Date);
CREATE INDEX idx_sid ON Price(Sid);
```

## Using Sharadar Data

Access Sharadar SF1 fundamentals easily:

```python
from zipline.pipeline import multi_source as ms

# Get most recent quarterly data
fund = ms.sharadar.Fundamentals.slice('MRQ', period_offset=0)

# Access any field
roe = fund.ROE.latest
fcf = fund.FCF.latest
revenue = fund.REVENUE.latest
marketcap = fund.MARKETCAP.latest
pe = fund.PE.latest
```

### Dimension Types
- `'MRQ'`: Most Recent Quarter
- `'MRT'`: Most Recent Twelve months (TTM)
- `'ARQ'`: As Reported Quarterly
- `'ART'`: As Reported TTM

### Period Offset
- `period_offset=0`: Most recent period
- `period_offset=1`: Previous period
- `period_offset=2`: Two periods ago

### Common Fields

**Financial Metrics:**
- `fcf`, `fcfps` - Free cash flow (95%+ availability)
- `revenue`, `revenueusd` - Revenue (97%+ availability)
- `ebitda`, `ebitdausd` - EBITDA
- `eps`, `epsusd` - Earnings per share
- `netinc` - Net income (97%+ availability)

**Valuation:**
- `marketcap` - Market capitalization (96%+ availability)
- `pe`, `pe1` - Price to earnings (90%+ availability)
- `pb` - Price to book
- `ps`, `ps1` - Price to sales

**Balance Sheet:**
- `assets` - Total assets (100% availability)
- `cashnequsd` - Cash and equivalents (100% availability)
- `debt`, `debtusd` - Total debt (100% availability)
- `equity`, `equityusd` - Total equity (100% availability)

**⚠️ Important Note: ROE Field**
The `roe` field in Sharadar bundles may be **empty (0% availability)**. To calculate ROE manually:
```python
# Use netinc / equity instead of roe
s_netinc = ms.SharadarFundamentals.netinc.latest
s_equity = ms.SharadarFundamentals.equity.latest
calculated_roe = (s_netinc / s_equity) * 100
```

Or use your custom database's ROE field if available.

## Pipeline Patterns

### Consensus Scoring

Give bonus points when multiple sources agree:

```python
# Sharadar quality (using fields with high availability)
s_fcf = ms.SharadarFundamentals.fcf.latest
s_pe = ms.SharadarFundamentals.pe.latest
sharadar_quality = (s_fcf > 0) & (s_pe > 0) & (s_pe < 30)

# Custom database quality
c_roe = CustomFundamentals.ROE.latest
c_peg = CustomFundamentals.PEG.latest
custom_quality = (c_roe > 15) & (c_peg < 2.5)

# Both sources confirm quality = higher confidence
combined_quality = sharadar_quality & custom_quality

# Use in selection
selection = c_roe.top(20, mask=combined_quality)
```

### Multi-Metric Filtering

Combine metrics from different sources:

```python
# Sharadar: valuation and size
s_fund = ms.sharadar.Fundamentals.slice('MRQ')
s_marketcap = s_fund.MARKETCAP.latest
s_pe = s_fund.PE.latest

# Custom: quality metrics
c_roe = CustomFundamentals.ROE.latest
c_peg = CustomFundamentals.PEG.latest

# Universe: large caps
universe = s_marketcap.top(500)

# Multi-source quality screen
quality = (
    (s_pe > 0) & (s_pe < 25) &  # Sharadar valuation
    (c_roe > 15) & (c_peg < 2)   # Custom quality
)

selection = c_roe.top(20, mask=universe & quality)
```

### Divergence Detection

Find stocks where sources disagree:

```python
s_roe = ms.sharadar.Fundamentals.slice('MRQ').ROE.latest
c_roe = CustomFundamentals.ROE.latest

# Sharadar says good, LSEG says bad
divergence_up = (s_roe > 15) & (c_roe < 10)

# Sharadar says bad, LSEG says good
divergence_down = (s_roe < 10) & (c_roe > 15)

# Investigate these stocks further
investigate = divergence_up | divergence_down
```

## Complete Strategy Example

```python
from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    schedule_function,
    date_rules,
    time_rules,
)
from zipline.pipeline import multi_source as ms

# Define custom database
class LSEGFundamentals(ms.Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252

    ReturnOnEquity_SmartEstimat = ms.Column(float)
    ForwardPEG_DailyTimeSeriesRatio_ = ms.Column(float)
    CompanyMarketCap = ms.Column(float)

# Create pipeline
def make_pipeline():
    # Sharadar
    s_fund = ms.sharadar.Fundamentals.slice('MRQ')
    s_roe = s_fund.ROE.latest
    s_fcf = s_fund.FCF.latest
    s_marketcap = s_fund.MARKETCAP.latest

    # LSEG
    l_roe = LSEGFundamentals.ReturnOnEquity_SmartEstimat.latest
    l_peg = LSEGFundamentals.ForwardPEG_DailyTimeSeriesRatio_.latest

    # Universe: top 100 by market cap
    universe = s_marketcap.top(100)

    # Consensus quality
    both_quality = (s_roe > 15) & (l_roe > 15)

    # Selection: top 5 by Sharadar ROE with LSEG confirmation
    selection = s_roe.top(5, mask=universe & both_quality)

    return ms.Pipeline(
        columns={
            's_roe': s_roe,
            's_fcf': s_fcf,
            'l_roe': l_roe,
            'l_peg': l_peg,
            'both_quality': both_quality,
        },
        screen=selection,
    )

def initialize(context):
    attach_pipeline(make_pipeline(), 'multi_source')
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(hours=1)
    )

def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('multi_source')

def rebalance(context, data):
    if context.pipeline_data is None or context.pipeline_data.empty:
        return

    # Get tradeable stocks
    selected = [s for s in context.pipeline_data.index if data.can_trade(s)]

    if not selected:
        return

    # Equal weight
    target_weight = 1.0 / len(selected)

    # Sell positions no longer selected
    for stock in context.portfolio.positions:
        if stock not in selected and data.can_trade(stock):
            order_target_percent(stock, 0.0)

    # Buy/rebalance selected stocks
    for stock in selected:
        order_target_percent(stock, target_weight)

# Run backtest
if __name__ == '__main__':
    results = run_algorithm(
        start='2023-01-01',
        end='2024-11-01',
        initialize=initialize,
        before_trading_start=before_trading_start,
        capital_base=100000,
        bundle='sharadar',
        custom_loader=ms.setup_auto_loader(),
    )

    print(f"Final portfolio value: ${results['portfolio_value'].iloc[-1]:,.2f}")
```

## Advanced Configuration

### Custom Database Directory

```python
loader = ms.setup_auto_loader(
    bundle_name='sharadar',
    custom_db_dir='/path/to/custom/databases',
    enable_sid_translation=True,
)
```

### Disable SID Translation

If your database already uses simulation SIDs:

```python
loader = ms.setup_auto_loader(
    enable_sid_translation=False,
)
```

### Multiple Custom Databases

```python
class Fundamentals1(ms.Database):
    CODE = "fundamentals"
    # ...

class Fundamentals2(ms.Database):
    CODE = "alternative_data"
    # ...

# Both work with the same loader!
results = run_algorithm(
    ...,
    custom_loader=ms.setup_auto_loader(),
)
```

## Database Inspection Tools

Several scripts are provided to help you inspect and verify your data:

### Check Custom Database SIDs and Data

```bash
# Check LSEG/custom database
python examples/custom_data/check_lseg_db.py

# Or inside Docker
docker exec zipline-reloaded-jupyter python /app/examples/custom_data/check_lseg_db.py
```

**Output shows:**
- SID verification for major stocks (AAPL, MSFT, etc.)
- Date range coverage
- Sample data records
- Data availability statistics

### Inspect Sharadar Bundle Data

```bash
# Check data availability in Sharadar bundle
python examples/custom_data/check_sf1_data.py

# View sf1.h5 file structure
python examples/custom_data/inspect_sf1.py
```

**Useful for:**
- Identifying which Sharadar fields have data
- Checking availability percentages
- Debugging empty fields (like ROE)

### Test Pipeline Before Backtest

```bash
# Run pipeline for a single day to see what data is available
python examples/custom_data/debug_multi_source.py
```

**Shows:**
- How many stocks pass each filter
- Data availability from each source
- Top stocks selected by your criteria
- Combined quality confirmation counts

This is essential for debugging strategies that return 0 stocks!

### Test Sharadar Loader Directly

```bash
# Test if Sharadar loader can read data
python examples/custom_data/test_sharadar_loader.py
```

**Verifies:**
- Loader can read sf1.h5 file
- Field names are correct
- Data exists for test stocks

## Troubleshooting

### No data in custom database

**1. Check database location:**
```bash
ls -lh ~/.zipline/data/custom/
```

**2. Use inspection script:**
```bash
python examples/custom_data/check_lseg_db.py
```

This will show:
- If SIDs match between your database and bundle
- Date range coverage
- Sample data to verify correctness

**3. Verify SIDs match bundle:**
```python
from zipline.data.bundles import load as load_bundle
bundle_data = load_bundle('sharadar')
asset = bundle_data.asset_finder.lookup_symbol('AAPL', as_of_date=None)
print(f"Bundle SID for AAPL: {asset.sid}")
```

**4. Check database contents manually:**
```python
import sqlite3
import os
db_path = os.path.expanduser('~/.zipline/data/custom/fundamentals.sqlite')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check structure
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cursor.fetchall())

# Check data
cursor.execute("SELECT DISTINCT Sid FROM fundamentals LIMIT 10")
print("Sample SIDs:", cursor.fetchall())
```

### AutoLoader not finding custom columns

Ensure your Database class has a `CODE` attribute:
```python
class CustomFundamentals(ms.Database):
    CODE = "fundamentals"  # Required!
    # ...
```

### SID translation not working

Check that asset_finder is available:
```python
loader = ms.setup_auto_loader(enable_sid_translation=True)
# Check logs for "Asset finder loaded for SID translation"
```

### Pipeline returns 0 stocks

This is often due to empty data fields. **Use the debug script first:**

```bash
python examples/custom_data/debug_multi_source.py
```

**Common causes:**

1. **Empty Sharadar ROE field** - Use the inspection tool:
   ```bash
   python examples/custom_data/check_sf1_data.py
   ```

   If ROE shows 0% availability, use alternative fields:
   ```python
   # Instead of s_roe > 15
   s_fcf = ms.SharadarFundamentals.fcf.latest
   s_pe = ms.SharadarFundamentals.pe.latest
   quality = (s_fcf > 0) & (s_pe < 30)
   ```

2. **No data overlap between sources** - The debug script shows:
   ```
   Sharadar quality: 258 stocks
   LSEG quality: 154 stocks
   Combined quality: 0 stocks  ← Problem!
   ```

   Solution: Relax one of the filters or check date ranges

3. **Date mismatch** - Custom database may not have data for backtest dates.
   Use inspection script to verify date range.

## API Reference

### `multi_source.setup_auto_loader()`

Create an AutoLoader for multi-source backtests.

**Parameters:**
- `bundle_name` (str): Bundle name for Sharadar data. Default: 'sharadar'
- `custom_db_dir` (str or Path): Custom database directory. Default: ~/.zipline/data/custom
- `enable_sid_translation` (bool): Enable SID translation. Default: True

**Returns:**
- `AutoLoader`: Configured loader ready for use

### `multi_source.Database`

Base class for defining custom data schemas.

**Required Attributes:**
- `CODE` (str): Database identifier (matches SQLite filename)
- `LOOKBACK_WINDOW` (int): Days to look back when querying

**Example:**
```python
class MyData(ms.Database):
    CODE = "my_database"
    LOOKBACK_WINDOW = 252
    MyColumn = ms.Column(float)
```

### `multi_source.Column`

Column definition for Database classes.

**Parameters:**
- `dtype`: Column data type (float, int, object)

**Example:**
```python
Revenue = ms.Column(float)
Sector = ms.Column(object)
```

### `multi_source.Pipeline`

Pipeline for defining data queries and screens.

**Parameters:**
- `columns` (dict): Dictionary mapping column names to BoundColumns
- `screen` (Filter): Boolean filter for asset selection

### `multi_source.sharadar`

Sharadar fundamentals data module.

**Key Datasets:**
- `sharadar.Fundamentals.slice(dimension, period_offset)`: SF1 fundamentals

## Examples

Complete examples are available in:
- `examples/custom_data/simple_multi_source_example.py`
- `examples/notebooks/multi_source_fundamentals_example.ipynb`

## Help and Documentation

Access help from Python:
```python
from zipline.pipeline import multi_source as ms

# Print quick start guide
print(ms.help_quick_start())

# Print database definition guide
print(ms.help_database())

# Print Sharadar fundamentals guide
print(ms.help_sharadar())
```

Or use Python's built-in help:
```python
help(ms.setup_auto_loader)
help(ms.Database)
help(ms.AutoLoader)
```

## References

- [Zipline Documentation](https://zipline.ml4trading.io)
- [Sharadar SF1 Documentation](https://data.nasdaq.com/databases/SF1/documentation)
- [Exchange ML4Trading Community](https://exchange.ml4trading.io)
