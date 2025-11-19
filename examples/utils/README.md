# Utilities Documentation

This directory contains utility scripts for managing, testing, and debugging Zipline strategies and data.

---

## Table of Contents

1. [Bundle Management](#bundle-management)
2. [Data Verification](#data-verification)
3. [Backtest Helpers](#backtest-helpers)
4. [Strategy Runners](#strategy-runners)
5. [Debugging Tools](#debugging-tools)

---

## Bundle Management

### `register_bundles.py`

**Purpose:** Register Zipline data bundles (Sharadar, Yahoo, NASDAQ)

**Usage:**
```python
from utils.register_bundles import ensure_bundles_registered

# Register all available bundles
ensure_bundles_registered()
```

**Features:**
- Auto-registers Sharadar bundle with incremental updates
- Supports Yahoo Finance bundles (if available)
- Supports NASDAQ Data Link bundles (if available)
- Safe to call multiple times (idempotent)

**Example:**
```python
# In your strategy/notebook
from utils.register_bundles import ensure_bundles_registered
ensure_bundles_registered()

# Now you can use bundles
from zipline.data.bundles import load
bundle_data = load('sharadar')
```

---

## Data Verification

### `check_bundle_data.py`

**Purpose:** Verify Sharadar bundle data availability for specific tickers

**Usage:**
```bash
python check_bundle_data.py
```

**What it checks:**
- ✓ Bundle exists and is loaded
- ✓ Tickers are found in bundle
- ✓ Price data availability
- ✓ Date ranges for each ticker

**Example Output:**
```
Checking bundle data for: AAPL, MSFT, GOOGL
✓ AAPL: Found (SID 12345)
  Date range: 2010-01-01 to 2025-11-15
✓ MSFT: Found (SID 67890)
  Date range: 2010-01-01 to 2025-11-15
```

---

### `check_fundamentals_db.py`

**Purpose:** Verify custom fundamentals database structure and data

**Usage:**
```bash
python check_fundamentals_db.py
```

**What it checks:**
- ✓ Database file exists
- ✓ Schema is correct (Price table with UNIQUE constraint)
- ✓ Row counts
- ✓ Date ranges
- ✓ SID distribution
- ✓ Sample data for verification

**Example Output:**
```
Database: /root/.zipline/data/custom/fundamentals.sqlite
✓ Price table exists
  Rows: 8,947,234
  Date range: 2009-12-24 to 2025-11-11
  Unique SIDs: 11,234

Sample data for AAPL (SID 199059):
Date         ROE      P/E      Debt
2023-01-01   115.2    24.5     120000000000
```

---

### `check_lseg_db.py`

**Purpose:** Verify LSEG fundamentals data for specific tickers

**Usage:**
```bash
python check_lseg_db.py
```

**What it checks:**
- ✓ Expected SIDs match actual SIDs
- ✓ Data availability for universe tickers
- ✓ Row counts per ticker
- ✓ Date ranges
- ✓ Sample data display

**Example Output:**
```
Checking LSEG database SIDs:
Ticker    Expected SID    Actual SID    Match?    Row Count
AAPL      199059          199059        ✓         3,991
MSFT      198508          198508        ✓         3,991
META      194817          194817        ✓         970
```

---

### `check_sf1_data.py`

**Purpose:** Check Sharadar SF1 fundamentals data availability

**Usage:**
```bash
python check_sf1_data.py
```

**What it checks:**
- ✓ SF1 table exists
- ✓ Available metrics
- ✓ Data for specific tickers
- ✓ Date ranges

---

## Backtest Helpers

### `backtest_helpers.py`

**Purpose:** Comprehensive backtesting utilities and performance analysis

**Key Functions:**

#### `run_backtest()`
Run a complete backtest with custom parameters

```python
from utils.backtest_helpers import run_backtest

results = run_backtest(
    strategy_func=my_strategy,
    start_date='2020-01-01',
    end_date='2023-12-31',
    capital_base=100000,
    bundle='sharadar'
)
```

#### `analyze_performance()`
Analyze backtest results and generate metrics

```python
from utils.backtest_helpers import analyze_performance

metrics = analyze_performance(results)
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

#### `plot_results()`
Visualize backtest performance

```python
from utils.backtest_helpers import plot_results

plot_results(results, save_path='/data/backtest_results/plots.png')
```

**Features:**
- Performance metrics calculation
- Risk-adjusted returns
- Drawdown analysis
- Trade statistics
- Visualization utilities
- Export to CSV/pickle

---

## Strategy Runners

### `run_strategy.py`

**Purpose:** Command-line runner for executing strategies

**Usage:**
```bash
python run_strategy.py --strategy my_strategy.py --start 2020-01-01 --end 2023-12-31
```

**Arguments:**
- `--strategy`: Path to strategy file
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--capital`: Initial capital (default: 100000)
- `--bundle`: Bundle name (default: sharadar)
- `--output`: Output directory for results

**Features:**
- Load and validate strategy modules
- Set up custom data loaders
- Execute backtest
- Save results automatically
- Generate performance reports

---

## Debugging Tools

### `debug_sids.py`

**Purpose:** Debug SID mapping and asset lookup issues

**Usage:**
```bash
python debug_sids.py
```

**What it debugs:**
- ✓ Symbol → SID mapping
- ✓ Temporal lookups (symbol changes over time)
- ✓ Asset metadata
- ✓ Bundle vs database SID mismatches

**Example:**
```python
from utils.debug_sids import check_sid_mapping

check_sid_mapping('META', as_of_date='2022-01-01')
# Output: Symbol 'META' maps to SID 194817
#         Previously known as 'FB' before 2021-10-28
```

---

### `inspect_sf1.py`

**Purpose:** Inspect Sharadar SF1 fundamentals table structure and contents

**Usage:**
```bash
python inspect_sf1.py
```

**What it shows:**
- Table schema
- Column names and types
- Sample records
- Available dimensions (MRY, MRT, ARY, etc.)

---

### `test_sharadar_loader.py`

**Purpose:** Test custom SQLite loader functionality

**Usage:**
```bash
python test_sharadar_loader.py
```

**What it tests:**
- ✓ Loader initialization
- ✓ Column discovery
- ✓ Data loading
- ✓ Pipeline integration
- ✓ Domain handling

---

## Common Workflows

### Workflow 1: Verify Data Before Backtesting

```bash
# 1. Check bundle data
python check_bundle_data.py

# 2. Check custom fundamentals
python check_fundamentals_db.py

# 3. Verify specific tickers
python check_lseg_db.py
```

### Workflow 2: Debug Strategy Issues

```bash
# 1. Debug SID mapping
python debug_sids.py

# 2. Check data availability
python check_fundamentals_db.py

# 3. Test loader
python test_sharadar_loader.py
```

### Workflow 3: Run Complete Backtest

```python
from utils.register_bundles import ensure_bundles_registered
from utils.backtest_helpers import run_backtest, analyze_performance, plot_results

# Register bundles
ensure_bundles_registered()

# Run backtest
results = run_backtest(
    strategy_func=my_strategy,
    start_date='2020-01-01',
    end_date='2023-12-31',
)

# Analyze
metrics = analyze_performance(results)

# Visualize
plot_results(results)
```

---

## File Reference

| File | Purpose | Type | Usage |
|------|---------|------|-------|
| `register_bundles.py` | Bundle registration | Module | Import and call |
| `check_bundle_data.py` | Verify bundle data | Script | Run directly |
| `check_fundamentals_db.py` | Verify custom DB | Script | Run directly |
| `check_lseg_db.py` | Verify LSEG data | Script | Run directly |
| `check_sf1_data.py` | Check SF1 table | Script | Run directly |
| `backtest_helpers.py` | Backtest utilities | Module | Import functions |
| `run_strategy.py` | Strategy runner | Script | CLI with args |
| `debug_sids.py` | SID debugging | Script | Run directly |
| `inspect_sf1.py` | Inspect SF1 | Script | Run directly |
| `test_sharadar_loader.py` | Test loader | Script | Run directly |

---

## Tips

### Performance Optimization
- Use `check_*` scripts to verify data before running long backtests
- Check SID mappings with `debug_sids.py` if seeing unexpected results
- Use `backtest_helpers.py` for standardized performance metrics

### Debugging
- Start with `check_fundamentals_db.py` to verify data exists
- Use `debug_sids.py` to troubleshoot symbol mapping issues
- Check `test_sharadar_loader.py` if pipeline is not loading data

### Data Management
- Run `check_bundle_data.py` after bundle updates
- Verify LSEG data with `check_lseg_db.py` after loading CSV
- Use `inspect_sf1.py` to understand available Sharadar metrics

---

## Environment

All scripts are designed to run inside the Docker container:

```bash
# From inside container
cd /app/examples/utils
python check_fundamentals_db.py

# Or from host via docker exec
docker exec zipline-reloaded-jupyter python /app/examples/utils/check_fundamentals_db.py
```

**Data Locations:**
- Bundles: `/root/.zipline/bundles/`
- Custom DB: `/root/.zipline/data/custom/fundamentals.sqlite`
- Results: `/data/backtest_results/`

---

## Contributing

When adding new utilities:
1. Follow existing naming conventions (`check_*.py`, `test_*.py`)
2. Add comprehensive docstrings
3. Update this README with usage examples
4. Test inside Docker container
5. Document expected output format
