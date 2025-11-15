# Top 5 ROE Strategy - Documentation

## Overview

This is a quantitative trading strategy that selects high-quality stocks based on Return on Equity (ROE) from a universe of large-cap companies.

**Strategy Type**: Fundamental Factor-Based
**Rebalancing**: Weekly (configurable)
**Weighting**: Equal Weight
**Data Source**: Custom SQLite fundamentals database

## Strategy Logic

### Two-Stage Selection Process

1. **Universe Selection**
   Filter to top 100 stocks by market capitalization

2. **Factor Ranking**
   Select top 5 stocks by Return on Equity (ROE) from the universe

3. **Rebalancing**
   Weekly rebalancing at market open + 1 hour

4. **Position Sizing**
   Equal weight allocation across all selected stocks

## Key Features

### Clean Custom Data Integration

This strategy demonstrates Zipline's clean approach to custom data:

- ✅ **Auto-discovery**: Fundamental columns automatically discovered via introspection
- ✅ **No duplication**: Columns defined once in `CustomFundamentals` class
- ✅ **Domain-aware**: LoaderDict handles domain-aware column matching
- ✅ **No monkey-patching**: Clean loader map passed to `run_algorithm()`

### Configuration-Driven

All parameters are defined in a single configuration section:

```python
# Backtest Parameters
START_DATE = '2012-03-10'
END_DATE = '2025-11-11'
CAPITAL_BASE = 100000

# Strategy Parameters
UNIVERSE_SIZE = 100
SELECTION_SIZE = 5
REBALANCE_FREQ = 'weekly'

# Database Configuration
DB_NAME = "fundamentals"
DB_DIR = Path('/root/.zipline/data/custom')
```

### Comprehensive Documentation

- Detailed docstrings for all functions
- Usage examples
- Configuration guide
- Architecture explanation

## File Structure

```
notebooks/
├── strategy_top5_roe.py              # Main strategy file
├── load_csv_fundamentals.ipynb       # Database creation notebook
├── load_fundamentals_to_db.py        # Database loading script
├── backtest_results.csv              # Results (CSV format)
├── backtest_results.pkl              # Results (pickle format)
└── STRATEGY_README.md                # This file
```

## Usage

### Quick Start

```bash
# Run the strategy
python strategy_top5_roe.py
```

### Programmatic Usage

```python
from strategy_top5_roe import build_pipeline_loaders, make_pipeline
from zipline import run_algorithm

# Build loaders
custom_loader = build_pipeline_loaders()

# Create pipeline
pipeline = make_pipeline(universe_size=200, selection_size=10)

# Run backtest
results = run_algorithm(
    start='2020-01-01',
    end='2025-01-01',
    initialize=initialize,
    before_trading_start=before_trading_start,
    handle_data=handle_data,
    capital_base=100000,
    bundle='sharadar',
    custom_loader=custom_loader
)
```

## Configuration Options

### Backtest Parameters

- `START_DATE`: Backtest start date (string, YYYY-MM-DD)
- `END_DATE`: Backtest end date (string, YYYY-MM-DD)
- `CAPITAL_BASE`: Starting capital in USD (integer)

### Strategy Parameters

- `UNIVERSE_SIZE`: Number of stocks in universe (top N by market cap)
- `SELECTION_SIZE`: Number of stocks to hold (top M by ROE)
- `REBALANCE_FREQ`: Rebalancing frequency
  - `'weekly'`: Every Monday
  - `'monthly'`: First trading day of month
  - `'daily'`: Every trading day

### Database Configuration

- `DB_NAME`: Database identifier (must match `.sqlite` filename without extension)
- `DB_DIR`: Directory containing custom databases

### Output Configuration

- `RESULTS_DIR`: Directory to save results
- `SAVE_CSV`: Save results as CSV (boolean)
- `SAVE_PICKLE`: Save results as pickle (boolean)
- `SAVE_METADATA`: Save metadata as JSON (boolean)

### Logging Configuration

- `PROGRESS_BAR`: Progress bar frequency
  - `'D'`: Daily updates
  - `'W'`: Weekly updates
  - `'M'`: Monthly updates
  - `None`: No progress bar
- `LOG_PIPELINE_STATS`: Log daily pipeline statistics (boolean)
- `LOG_REBALANCE_DETAILS`: Detailed trade logging (boolean)

## Adding New Fundamental Columns

1. **Add to Database**
   Ensure the column exists in your SQLite database

2. **Add to CustomFundamentals**
   ```python
   class CustomFundamentals(Database):
       CODE = "fundamentals"
       LOOKBACK_WINDOW = 252

       # Add your new column here
       NewMetric = Column(float)
   ```

3. **Use in Pipeline**
   ```python
   def make_pipeline():
       new_metric = CustomFundamentals.NewMetric.latest
       # Use in your pipeline logic...
   ```

That's it! The column is automatically discovered and mapped.

## Architecture

### Auto-Discovery Pattern

The strategy uses Python introspection to automatically discover columns:

```python
for attr_name in dir(CustomFundamentals):
    if attr_name.startswith('_') or attr_name in ['CODE', 'LOOKBACK_WINDOW']:
        continue

    attr = getattr(CustomFundamentals, attr_name)
    if hasattr(attr, 'dataset'):  # It's a Column
        custom_loader[attr] = fundamentals_loader
```

**Benefits**:
- Define columns once (single source of truth)
- No manual mapping required
- Easy to add/remove columns
- Less error-prone

### Domain-Aware Column Matching

The `LoaderDict` class handles domain-aware lookups:

```python
# Column registered as: USEquityPricing.close
# Lookup as: USEquityPricing<US_EQUITIES>.close
# LoaderDict matches them automatically
```

This solves the common issue where Zipline adds domain suffixes during pipeline execution.

## Output

### Console Output

The strategy provides comprehensive progress logging during execution:

```
================================================================================
BUILDING PIPELINE LOADERS
================================================================================
✓ Pipeline loader map built (12 columns mapped)
  - Pricing columns: 5
  - Fundamental columns: 7 (auto-discovered)

================================================================================
BACKTEST CONFIGURATION
================================================================================
Strategy: Top 5 ROE from Top 100 by Market Cap
Period: 2012-03-10 to 2025-11-11 (13.7 years)
Capital: $100,000
Rebalancing: Weekly
Bundle: sharadar
Database: /root/.zipline/data/custom/fundamentals.sqlite
Progress: D
Pipeline logging: Enabled
Rebalance logging: Detailed
================================================================================

================================================================================
RUNNING BACKTEST
================================================================================
Start time: 2025-01-15 14:32:10

============================================================
ROE STRATEGY INITIALIZED
============================================================
Rebalancing: Weekly
Universe: Top 100 by market cap
Selection: Top 5 by ROE
Weighting: Equal weight
============================================================

2012-03-12 | Universe:  5 stocks | Avg ROE: 15.23% | Avg MCap: $250.3B
2012-03-13 | Universe:  5 stocks | Avg ROE: 15.41% | Avg MCap: $248.7B

======================================================================
REBALANCE #1
======================================================================
  Sell: 0 | Buy: 5 | Rebalance: 0 | Weight: 20.00%
    BUY:  AAPL (ROE: 18.45%, MCap: $500.2B)
    BUY:  MSFT (ROE: 16.23%, MCap: $420.1B)
    BUY:  GOOGL (ROE: 14.87%, MCap: $380.5B)
    BUY:  JPM (ROE: 13.92%, MCap: $310.3B)
    BUY:  WMT (ROE: 12.56%, MCap: $290.8B)
  Portfolio Summary:
    Holdings: AAPL, MSFT, GOOGL, JPM, WMT
    Avg ROE: 15.21%
    Avg Market Cap: $380.4B
======================================================================

...

================================================================================
BACKTEST COMPLETE
================================================================================
End time: 2025-01-15 14:45:23
Duration: 0:13:13

================================================================================
PERFORMANCE SUMMARY
================================================================================
Initial Capital:     $100,000.00
Final Value:         $245,230.00
Total Return:        145.23%
Sharpe Ratio:        0.87
Max Drawdown:        -23.45%
Win Rate:            54.2%
Total Trades:        720
Trading Days:        3,450
================================================================================

================================================================================
SAVING RESULTS
================================================================================
✓ CSV: /notebooks/backtest_results.csv
  Size: 1245.3 KB
  Rows: 3,450
✓ Pickle: /notebooks/backtest_results.pkl
  Size: 892.1 KB
✓ Metadata: /notebooks/backtest_results.json
  Size: 1.2 KB
================================================================================

✓ Backtest complete!
✓ Execution time: 0:13:13
```

### Logging Levels

**Minimal Logging** (`LOG_PIPELINE_STATS=False`, `LOG_REBALANCE_DETAILS=False`):
- Configuration summary only
- Simple rebalance notifications
- Performance summary

**Standard Logging** (`LOG_PIPELINE_STATS=True`, `LOG_REBALANCE_DETAILS=False`):
- Daily pipeline statistics
- Simple rebalance notifications
- Performance summary

**Detailed Logging** (`LOG_PIPELINE_STATS=True`, `LOG_REBALANCE_DETAILS=True`):
- Daily pipeline statistics
- Detailed BUY/SELL trade information with metrics
- Portfolio composition after each rebalance
- Full performance summary

### Saved Files

- **CSV**: `backtest_results.csv` - Portable, human-readable
  - Daily performance metrics
  - Portfolio value, positions, leverage
  - All custom `record()` metrics

- **Pickle**: `backtest_results.pkl` - Preserves full DataFrame state
  - Complete results with all data types
  - Faster to load for analysis

- **Metadata**: `backtest_results.json` - Configuration and performance summary
  - Strategy configuration
  - Date range and parameters
  - Performance metrics (return, Sharpe, drawdown)
  - Execution timestamp and duration
  - File paths

## Requirements

### System Requirements

- Python >= 3.9
- Zipline Reloaded with custom data support
- SQLite3

### Data Requirements

- Sharadar bundle (or equivalent)
- Custom fundamentals database at `/root/.zipline/data/custom/fundamentals.sqlite`
- See `load_csv_fundamentals.ipynb` for database creation

### Python Dependencies

```python
pandas >= 2.0
numpy >= 2.0
zipline-reloaded >= 3.0
```

## Customization Examples

### Monthly Rebalancing with Larger Universe

```python
# In configuration section
UNIVERSE_SIZE = 200
SELECTION_SIZE = 10
REBALANCE_FREQ = 'monthly'
```

### Different Time Period

```python
START_DATE = '2015-01-01'
END_DATE = '2020-12-31'
```

### Custom Database Location

```python
DB_DIR = Path('/custom/path/to/databases')
DB_NAME = 'my_fundamentals'
```

### Use Different Factor

```python
def make_pipeline(universe_size=100, selection_size=5):
    # Change from ROE to ROA
    roa = CustomFundamentals.ReturnOnAssets_SmartEstimate.latest
    market_cap = CustomFundamentals.CompanyMarketCap.latest

    universe = market_cap.top(universe_size)
    selection = roa.top(selection_size, mask=universe)

    return Pipeline(
        columns={'ROA': roa, 'Market_Cap': market_cap},
        screen=selection
    )
```

## Troubleshooting

### Database Not Found

```
FileNotFoundError: Database not found: /root/.zipline/data/custom/fundamentals.sqlite
```

**Solution**: Ensure `DB_NAME` matches your database filename (without `.sqlite` extension)

### Column Not Found

```
AttributeError: 'CustomFundamentals' object has no attribute 'ColumnName'
```

**Solution**:
1. Check column is defined in `CustomFundamentals` class
2. Verify column exists in SQLite database
3. Check column name spelling

### No Stocks in Pipeline

```
WARNING: No stocks in pipeline output
```

**Solution**:
1. Check database has data for backtest period
2. Verify date range is within database coverage
3. Ensure universe size is not too large

## Performance Considerations

- **Auto-discovery overhead**: Minimal (runs once at initialization)
- **Pipeline caching**: Zipline caches pipeline results automatically
- **Database queries**: CustomSQLiteLoader uses efficient SQL queries
- **Memory usage**: Results stored in memory; save large backtests to disk

## Related Files

- `load_csv_fundamentals.ipynb` - Create fundamentals database from CSV
- `load_fundamentals_to_db.py` - Standalone database loading script
- `DATABASE_LOADING_GUIDE.md` - Troubleshooting database issues
- `Claude.context.md` - Technical context and architecture notes

## License

Same as Zipline Reloaded (Apache 2.0)

## Support

For issues and questions:
- GitHub Issues: https://github.com/stefan-jansen/zipline-reloaded/issues
- Community: https://exchange.ml4trading.io
