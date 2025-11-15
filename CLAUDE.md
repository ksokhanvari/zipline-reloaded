# Zipline Reloaded - Project Context for Claude

## Overview
Zipline is a Pythonic algorithmic trading library for backtesting trading strategies. This is the "reloaded" fork maintained by Stefan Jansen after Quantopian closed in 2020.

## Key Information
- **Python Version**: >= 3.10
- **Main Dependencies**: pandas >= 2.0, SQLAlchemy >= 2.0, numpy >= 2.0, scikit-learn >= 1.0.0
- **Documentation**: https://zipline.ml4trading.io
- **Community**: https://exchange.ml4trading.io

## Project Structure
- `src/zipline/`: Main source code
  - `algorithm.py`: Core algorithm execution
  - `api.py`: Public API functions
  - `data/`: Data ingestion and handling
    - `custom/`: Custom fundamentals data integration
      - `pipeline_integration.py`: CustomSQLiteLoader implementation
      - `db_manager.py`: Database utilities
      - `config.py`: Configuration constants
  - `finance/`: Financial calculations and order execution
  - `pipeline/`: Factor-based screening system
- `notebooks/`: Strategy examples and data loading workflows
  - `load_csv_fundamentals.ipynb`: CSV → SQLite → Pipeline workflow
  - `strategy_top5_roe.py`: Full-featured production strategy example
  - `strategy_top5_roe_simple.py`: Minimal clean strategy example
  - `run_strategy.py`: Helper to run strategy files from notebooks
  - `run_backtest_example.ipynb`: Examples using run_strategy helper
  - `STRATEGY_README.md`: Comprehensive strategy documentation
- `tests/`: Test suite
- `docs/`: Documentation source

## Development Commands
```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_algorithm.py

# Build documentation
cd docs && make html

# Install in development mode
pip install -e .
```

## Testing Approach
- Unit tests use pytest
- Test data is stored in `tests/resources/`
- Mock trading environments for testing strategies

## Common Tasks
1. **Implementing new data bundles**: See `src/zipline/data/bundles/`
2. **Adding new pipeline factors**: See `src/zipline/pipeline/factors/`
3. **Modifying order execution**: See `src/zipline/finance/execution.py`
4. **Working with trading calendars**: Uses `exchange_calendars` library

## Important Notes
- The project uses Cython for performance-critical components
- Be careful with numpy/pandas API changes due to major version updates
- Trading calendars are handled by the external `exchange_calendars` package

---

# Custom Fundamentals System - Architecture & Implementation Guide

## Overview

This project includes a comprehensive custom fundamentals data integration system that enables users to load custom fundamental data into SQLite databases and use them seamlessly in Zipline Pipeline for algorithmic trading strategies.

**Target Audience**: LLM coding assistants, new developers, and maintainers who need to understand the complete architecture and implementation details.

**Docker Environment**: This setup is designed to run in Docker containers with paths configured for `/root/.zipline/data/custom` and `/notebooks`.

---

## System Architecture

### High-Level Design

```
┌─────────────────────┐
│   CSV Data Files    │
│  (Fundamentals)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Load & Transform   │  ← notebooks/load_csv_fundamentals.ipynb
│  (Pandas)           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  SQLite Database    │  ← /root/.zipline/data/custom/fundamentals.sqlite
│  (Custom Schema)    │     Schema: Price table with Sid, Date, columns
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ CustomSQLiteLoader  │  ← src/zipline/data/custom/pipeline_integration.py
│ (PipelineLoader)    │     Implements load_adjusted_array()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ SimplePipelineEngine│  ← Uses custom_loader parameter
│                     │     Routes columns to appropriate loaders
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Database Class     │  ← User-defined (e.g., CustomFundamentals)
│  (Dataset)          │     Defines columns with dtype, missing_value
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Pipeline Queries   │  ← Returns filtered DataFrames of stocks
│  (Factors/Filters)  │     e.g., .top(100), .latest, .rank()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Trading Strategy   │  ← Uses run_algorithm() or TradingAlgorithm
│  (Backtest)         │     Rebalances portfolio based on pipeline
└─────────────────────┘
```

### Core Principles

1. **Separation of Concerns**: Data loading, pipeline integration, and strategy logic are separate
2. **Extensibility**: Custom Database classes can define any columns with appropriate dtypes
3. **Performance**: SQLite indexing on (Date, Sid) for fast queries
4. **Type Safety**: Explicit dtype handling prevents mixed-type errors in Pipeline
5. **Compatibility**: Works seamlessly with existing Zipline bundles (e.g., sharadar)

---

## Key Files Deep Dive

### 📄 `src/zipline/data/custom/pipeline_integration.py`

**Purpose**: Core implementation of the custom data loader for Zipline Pipeline.

**Key Classes**:
- `CustomSQLiteLoader(PipelineLoader)`: Loads custom data from SQLite into Pipeline
  - `load_adjusted_array()`: Main entry point called by Pipeline engine
  - `_query_database()`: Executes SQL queries and returns 2D numpy arrays

**Critical Implementation Details**:
- **Dtype Handling**: Numeric columns (float64, int64) vs object columns (str) require different handling
- **Missing Values**:
  - Numeric: `np.nan`
  - Text/Object: `''` (empty string, NOT None or NaN)
- **AdjustedArray Creation**: Must use matching `missing_value` based on dtype
- **Type Conversion**: SQLite may return numeric data as strings → explicit `pd.to_numeric()` conversion

---

### 📄 `notebooks/strategy_top5_roe.py` - Full-Featured Production Strategy

**Purpose**: Production-ready strategy demonstrating custom fundamentals with full features.

**Key Features**:
- **Configuration via Constants**: All parameters at top of file
- **Progress Logging**: Zipline's built-in progress tracking
- **Auto-Discovery**: Automatically discovers all fundamental columns
- **Metadata Export**: Saves backtest configuration to JSON
- **Result Saving**: Exports to CSV and pickle
- **Comprehensive Logging**: Optional verbose logging (disabled by default)

**Architecture**:

1. **Database Definition with Auto-Discovery**
   ```python
   class CustomFundamentals(Database):
       """Custom Fundamentals database."""
       CODE = "fundamentals"  # Matches database file
       LOOKBACK_WINDOW = 252  # One year of trading days

       # Columns auto-discovered via introspection
       ReturnOnEquity_SmartEstimat = Column(float)
       CompanyMarketCap = Column(float)
       GICSSectorName = Column(str)
       # ... all other columns
   ```

2. **Build Pipeline Loader Map with LoaderDict**
   ```python
   def build_pipeline_loaders():
       """Build proper PipelineLoader map - no monkey-patching!"""

       # Custom dict that handles domain-aware columns
       class LoaderDict(dict):
           """Matches columns by dataset and column name, ignoring domain."""
           def get(self, key, default=None):
               # First try exact match
               if key in self:
                   return super().__getitem__(key)

               # Fuzzy match by dataset name and column name
               if hasattr(key, 'dataset') and hasattr(key, 'name'):
                   key_dataset_name = str(key.dataset).split('<')[0]
                   key_col_name = key.name

                   for registered_col, loader in self.items():
                       if hasattr(registered_col, 'dataset') and hasattr(registered_col, 'name'):
                           reg_dataset_name = str(registered_col.dataset).split('<')[0]
                           reg_col_name = registered_col.name

                           if key_dataset_name == reg_dataset_name and key_col_name == reg_col_name:
                               return loader
               return default

       # Auto-discover and map all fundamental columns
       custom_loader = LoaderDict()
       for attr_name in dir(CustomFundamentals):
           if attr_name.startswith('_') or attr_name in ['CODE', 'LOOKBACK_WINDOW']:
               continue
           attr = getattr(CustomFundamentals, attr_name)
           if hasattr(attr, 'dataset'):
               custom_loader[attr] = fundamentals_loader

       return custom_loader
   ```

   **Why LoaderDict**: The pipeline engine looks up domain-aware columns (e.g., `USEquityPricing<US_EQUITIES>.close`) but we register non-domain versions (`USEquityPricing.close`). LoaderDict matches by dataset name and column name, ignoring domain suffixes. The `get()` method is critical because the pipeline engine calls `custom_loader.get(column)`.

3. **Progress Logging Integration**
   ```python
   from zipline.utils.progress import enable_progress_logging
   from zipline.utils.flightlog_client import enable_flightlog

   enable_progress_logging(
       algo_name='Top5-ROE-Strategy',
       update_interval=PROGRESS_UPDATE_INTERVAL
   )
   enable_flightlog()
   ```

4. **Automatic Result Saving**
   ```python
   # Save results to CSV and pickle
   timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
   results.to_csv(f'backtest_results_{timestamp}.csv')
   results.to_pickle(f'backtest_results_{timestamp}.pkl')
   ```

---

### 📄 `notebooks/strategy_top5_roe_simple.py` - Minimal Clean Strategy

**Purpose**: Minimal version for quick testing and clean usage pattern.

**Key Features**:
- Clean `run_algorithm()` interface
- Inline parameter configuration
- Built-in analyze() function
- No external dependencies
- Perfect for notebook usage

**Usage**:
```python
# Direct execution
python strategy_top5_roe_simple.py

# From notebook
from run_strategy import run_strategy
results = run_strategy('strategy_top5_roe_simple.py', start='2020-01-01', end='2023-12-31')
```

---

### 📄 `notebooks/run_strategy.py` - Notebook Helper

**Purpose**: Helper function to run strategy files from Jupyter notebooks.

**Key Features**:
- Dynamic module loading using `importlib.util`
- Extracts initialize, handle_data, before_trading_start, analyze functions
- Auto-discovers and applies custom loaders
- Override parameters inline

**Usage**:
```python
from run_strategy import run_strategy

# Basic usage with defaults
results = run_strategy('strategy_top5_roe_simple.py')

# Custom dates and capital
results = run_strategy(
    'strategy_top5_roe_simple.py',
    start='2018-01-01',
    end='2023-12-31',
    capital_base=500000
)

# With progress logging
from zipline.utils.progress import enable_progress_logging
enable_progress_logging(algo_name='My-Strategy', update_interval=10)

results = run_strategy(
    'strategy_top5_roe.py',
    start='2021-01-01',
    end='2023-12-31'
)
```

**Implementation**:
```python
def load_module_from_file(filepath):
    """Load a Python module from a file path."""
    filepath = Path(filepath)
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[filepath.stem] = module
    spec.loader.exec_module(module)
    return module

def run_strategy(strategy_file, start='2020-01-01', end='2023-12-31',
                capital_base=100000, data_frequency='daily',
                bundle='sharadar', **kwargs):
    module = load_module_from_file(strategy_file)

    # Extract functions
    initialize = getattr(module, 'initialize', None)
    handle_data = getattr(module, 'handle_data', None)
    before_trading_start = getattr(module, 'before_trading_start', None)
    analyze = getattr(module, 'analyze', None)

    # Get custom loader if available
    if hasattr(module, 'build_pipeline_loaders'):
        custom_loader = module.build_pipeline_loaders()
        kwargs['custom_loader'] = custom_loader

    return run_algorithm(
        start=start, end=end,
        initialize=initialize,
        handle_data=handle_data,
        before_trading_start=before_trading_start,
        analyze=analyze,
        capital_base=capital_base,
        data_frequency=data_frequency,
        bundle=bundle,
        **kwargs
    )
```

---

### 📄 `notebooks/run_backtest_example.ipynb`

Example notebook showing multiple usage patterns for `run_strategy()`:

1. Quick backtest with defaults
2. Custom dates and capital
3. Full version with progress logging
4. Result analysis and visualization

---

## Database Schema

**Table Structure**: Single table named `Price` (historical convention)

```sql
CREATE TABLE Price (
    Sid INTEGER NOT NULL,           -- Zipline asset ID
    Date TEXT NOT NULL,             -- Format: 'YYYY-MM-DD'
    ReturnOnEquity_SmartEstimat REAL,
    CompanyMarketCap REAL,
    GICSSectorName TEXT,            -- Text columns for categorical data
    -- ... other columns
    PRIMARY KEY (Sid, Date)
);

CREATE INDEX idx_date_sid ON Price(Date, Sid);  -- Critical for performance
```

**Design Decisions**:
- **Sid (not Ticker)**: Zipline uses integer sids internally; more robust than tickers
- **Date as TEXT**: SQLite date handling; converted to datetime in Python
- **Denormalized**: All columns in one table for query performance
- **Indexed**: (Date, Sid) index enables fast range queries
- **Deduplication**: Handle duplicates on (Sid, Date) after symbol mapping

---

## Known Issues & Solutions

### Issue 1: "Categorical categories cannot be null"

**Root Cause**: Text columns contain NaN values, creating mixed str/float object arrays.

**Solution**: Apply in **two places**:

1. **Data Loading** (notebook):
   ```python
   for col in text_columns:
       custom_data[col].fillna('', inplace=True)
   ```

2. **Pipeline Integration** (already fixed in `pipeline_integration.py`):
   ```python
   if col_dtype == object:
       reindexed = reindexed.fillna('')
   ```

---

### Issue 2: "UNIQUE constraint failed: Price.Sid, Price.Date"

**Root Cause**: Duplicate (Sid, Date) combinations in source data.

**Solution**: Deduplicate AFTER mapping symbols to sids:
```python
df['Sid'] = df['Symbol'].map(ticker_to_sid)
df = df.drop_duplicates(subset=['Sid', 'Date'], keep='last')
```

---

### Issue 3: LoaderDict get() method required

**Root Cause**: Pipeline engine calls `custom_loader.get(column)`, not `custom_loader[column]`.

**Solution**: Implement both `get()` and `__getitem__()` methods in LoaderDict:
```python
class LoaderDict(dict):
    def get(self, key, default=None):
        # Fuzzy matching logic
        ...

    def __getitem__(self, key):
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result
```

---

## Key Learnings & Best Practices

1. **Text columns require special handling**: Empty string ('') not NaN or None
2. **Use custom_loader parameter**: `run_algorithm()` has built-in support - no monkey-patching!
3. **Auto-discovery pattern**: Use introspection to discover columns automatically
4. **LoaderDict is essential**: Handles domain-aware column matching
5. **Timezone handling is subtle**: Let Zipline handle TZ internally (no `tz='UTC'`)
6. **Version strategy**: Provide both full-featured and simple versions
7. **Progress logging**: Use Zipline's built-in `enable_progress_logging()`
8. **Module loading for notebooks**: Use `importlib.util` for dynamic loading
9. **Clean defaults**: Disable verbose logging by default (LOG_PIPELINE_STATS = False)
10. **scikit-learn integration**: sklearn >= 1.0.0 available for ML algorithms

---

## Strategy Development Workflow

### Option A: Using run_strategy() Helper (Recommended for Notebooks)

```python
from run_strategy import run_strategy

# Quick test
results = run_strategy('strategy_top5_roe_simple.py')

# Custom parameters
results = run_strategy(
    'strategy_top5_roe.py',
    start='2018-01-01',
    end='2023-12-31',
    capital_base=500000
)

# Analyze
total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0] - 1) * 100
print(f"Total Return: {total_return:.2f}%")
```

### Option B: Direct Execution

```bash
# Run from command line
python /notebooks/strategy_top5_roe.py

# Edit strategy logic directly
cp strategy_top5_roe_simple.py my_strategy.py
# ... edit ...
python my_strategy.py
```

---

## Version Comparison: Full vs Simple

| Feature | Full Version | Simple Version |
|---------|-------------|----------------|
| **File** | `strategy_top5_roe.py` | `strategy_top5_roe_simple.py` |
| **Configuration** | Constants at top | Inline parameters |
| **Progress Logging** | Yes (enable_progress_logging) | No |
| **Auto-Discovery** | Yes (introspection) | No (manual list) |
| **Result Saving** | Yes (CSV + pickle) | No |
| **Metadata Export** | Yes (JSON) | No |
| **Verbose Logging** | Configurable | Minimal |
| **Built-in Analyze** | Optional | Yes |
| **Use Case** | Production, long backtests | Quick testing, notebooks |
| **Lines of Code** | ~800 | ~220 |

---

## Recent Updates & Commits

### Latest Session Work

1. **Notebook Helper System** (commits ca409ee8, etc.)
   - Created `run_strategy.py` for running strategy files from notebooks
   - Created `run_backtest_example.ipynb` with usage examples
   - Dynamic module loading using `importlib.util`

2. **Simple Strategy Version** (commit 5bdc7517)
   - Created `strategy_top5_roe_simple.py` with minimal code
   - Clean `run_algorithm()` interface
   - Built-in analyze() function

3. **Documentation Updates** (commit 1449a75f)
   - Updated `STRATEGY_README.md` for both versions
   - Added version comparison table
   - Comprehensive usage examples

4. **Verbose Logging Disabled** (commit 927579ba)
   - Set LOG_PIPELINE_STATS = False by default
   - Set LOG_REBALANCE_DETAILS = False by default
   - Cleaner output for production use

5. **Progress Logging Integration** (commits 6b0017b8, 031aa3c0)
   - Added `enable_progress_logging()` imports
   - Fixed TypeError with progress parameter
   - FlightLog support

6. **Auto-Discovery Refactoring** (commit 2a5cfbef)
   - Implemented automatic column discovery
   - Eliminated manual column listing
   - More maintainable code

7. **Result Saving** (commit 9d6e6fb3)
   - Automatic CSV and pickle export
   - Timestamped filenames

8. **Database Alignment** (commit 5f60527a)
   - Changed CODE from 'refe-fundamentals' to 'fundamentals'
   - Matches notebook database name

9. **LoaderDict Implementation** (commits 7fa907e1, cc80898d)
   - Added `get()` method for domain-aware matching
   - Fixed AttributeError on custom_loader.get()

10. **scikit-learn Dependency** (commit 2c68df0a)
    - Added `scikit-learn >=1.0.0` to dependencies
    - Enables ML algorithms with sklearn imports

### Key Architecture Commits

- **5235d38**: Refactored to clean custom_loader approach (no monkey-patching)
- **71b556e**: Set working directory for strategy execution
- **bd93b2e**: Added Sharadar bundle registration
- **b120aec**: Removed verbose logging, fixed backtest dates
- **71949ca**: Fixed object dtype handling with empty strings
- **d488870**: Removed broken backtest cells from notebook

---

## Docker Environment

**Paths**:
```
/root/.zipline/data/custom/          # Custom database directory
/root/.zipline/data/custom/fundamentals.sqlite  # Database file
/notebooks/                          # Strategy files location
```

**Class Naming**:
- Database class: `CustomFundamentals` (generic, clean)
- Database CODE: `"fundamentals"` (matches file: fundamentals.sqlite)

---

## Testing

### Unit Tests
```bash
# All custom data tests
pytest tests/zipline/data/test_custom_pipeline.py -v

# Specific test
pytest tests/zipline/data/test_custom_pipeline.py::test_object_dtype_handling -v
```

### Validation Queries
```python
import sqlite3
conn = sqlite3.connect('/root/.zipline/data/custom/fundamentals.sqlite')

# Count rows
pd.read_sql('SELECT COUNT(*) FROM Price', conn)

# Date range
pd.read_sql('SELECT MIN(Date), MAX(Date) FROM Price', conn)

# Sample data
pd.read_sql('SELECT * FROM Price LIMIT 10', conn)
```

---

## Quick Reference

### Import Snippets

```python
# Basic Pipeline Setup
from zipline.pipeline import Pipeline
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.data.db import Database, Column
from zipline.data.custom import CustomSQLiteLoader
from zipline.data.bundles import load as load_bundle
from zipline.pipeline.domain import US_EQUITIES

# Progress Logging
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog

# Algorithm Execution
from zipline import run_algorithm

# Notebook Helper
from run_strategy import run_strategy
```

### Common Patterns

**Database Definition**:
```python
class CustomFundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252
    Revenue = Column(float)
    Sector = Column(str)
```

**Build Custom Loader**:
```python
custom_loader = build_pipeline_loaders()  # From strategy file
results = run_algorithm(
    start=start, end=end,
    initialize=initialize,
    custom_loader=custom_loader,  # Pass it here!
    bundle='sharadar'
)
```

**Run from Notebook**:
```python
results = run_strategy(
    'strategy_top5_roe_simple.py',
    start='2020-01-01',
    end='2023-12-31',
    capital_base=100000
)
```

---

## Summary

This Zipline Reloaded fork includes:

1. **Core Zipline**: Algorithmic trading backtesting library
2. **Custom Fundamentals System**: SQLite-based fundamental data integration
3. **Production Strategies**: Full-featured and simple strategy templates
4. **Notebook Helpers**: Tools for running strategies from Jupyter
5. **Auto-Discovery**: Automatic column mapping via introspection
6. **Progress Logging**: Built-in progress tracking and monitoring
7. **ML Support**: scikit-learn integration for machine learning algorithms

**Key Success Factors**:
- LoaderDict with `get()` method for domain-aware column matching
- Proper text column handling (empty strings, not NaN)
- Clean custom_loader parameter usage (no monkey-patching)
- Auto-discovery pattern for maintainability
- Dual versions (full/simple) for different use cases
- Notebook integration via run_strategy() helper

**Next Steps for New Sessions**:
- Review this document for full context
- Check `/root/.zipline/data/custom/fundamentals.sqlite` exists
- Test with: `python /notebooks/strategy_top5_roe_simple.py`
- Or use: `run_strategy('strategy_top5_roe_simple.py')` from notebook
- Extend with custom Database classes and strategies

---

**Document Version**: 4.0
**Last Updated**: 2025-11-14
**Branch**: claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA
**Maintainer**: Auto-generated from session context
