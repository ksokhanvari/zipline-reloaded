# Zipline Reloaded - Examples & Notebooks

Comprehensive collection of examples, notebooks, and utilities organized by functionality.

## 📂 Directory Structure

```
examples/
├── 1_getting_started/          # Start here! Basic examples
├── 2_strategies/               # Trading strategy examples
├── 3_analysis/                 # Performance analysis tools
├── 4_pipeline/                 # Pipeline API examples
├── 5_multi_source_data/        # Multi-source data integration
├── 6_custom_data/              # Custom data utilities and helpers
├── utils/                      # Shared utilities and helpers
└── deprecated/                 # Old examples (kept for reference)
```

## 🚀 Quick Start

### New to Zipline?

**Start with these in order:**

1. **Read**: `docs/GETTING_STARTED.md` - Setup guide
2. **Run**: `1_getting_started/simple_flightlog_demo.py` - First backtest
3. **Explore**: `notebooks/sharadar_data_explorer.ipynb` - Understand the data
4. **Learn**: `2_strategies/momentum_strategy_with_flightlog.py` - Real strategy

### Want to use custom data?

**Follow this path:**

1. **Read**: `docs/MULTI_SOURCE_QUICKREF.md` - Quick reference
2. **Study**: `5_multi_source_data/simple_multi_source_example.py` - Basic example
3. **Create**: `6_custom_data/create_fundamentals_db.py` - Your database
4. **Build**: Your own multi-source strategy!

## 📁 Detailed Contents

### 1️⃣ Getting Started (`1_getting_started/`)

Basic examples to learn Zipline fundamentals.

- **`simple_flightlog_demo.py`** - Simplest possible example: buy and hold AAPL with FlightLog monitoring
  - Perfect for: First-time users, quick testing
  - Features: Real-time logs, progress bars, performance summary

### 2️⃣ Strategies (`2_strategies/`)

Production-ready trading strategy examples.

- **`momentum_strategy_with_flightlog.py`** - Complete momentum trading strategy
  - Features: Multi-stock ranking, weekly rebalancing, error handling
  - Perfect for: Learning professional patterns, production use

### 3️⃣ Analysis (`3_analysis/`)

Tools for analyzing backtest results.

- **`plot_backtest_results.py`** - Visualize portfolio performance
  - Features: Returns charts, drawdown analysis, position heatmaps
  - Perfect for: Understanding backtest performance

### 4️⃣ Pipeline (`4_pipeline/`)

Examples using Zipline's Pipeline API for factor-based screening.

- Coming soon: Pipeline-specific examples will be organized here

### 5️⃣ Multi-Source Data (`5_multi_source_data/`)

Combine Sharadar and custom databases in your strategies.

- **`simple_multi_source_example.py`** - Complete working example
  - Combines: Sharadar fundamentals + LSEG custom data
  - Strategy: Quality factors from both sources
  - Perfect for: Learning multi-source patterns

- **`debug_multi_source.py`** - Test pipeline without running backtest
  - Features: Quick pipeline testing, field validation
  - Perfect for: Debugging screens before backtesting

- **`test_sharadar_loader.py`** - Test Sharadar loader directly
  - Features: Validate data loading, check field availability
  - Perfect for: Troubleshooting Sharadar issues

### 6️⃣ Custom Data (`6_custom_data/`)

Utilities for creating and managing custom databases.

#### Database Creation

- **`create_fundamentals_db.py`** - Create custom fundamentals database
  - Creates: SQLite database from CSV files
  - Location: `~/.zipline/data/custom/fundamentals.sqlite`

#### Database Inspection

- **`check_lseg_db.py`** - Inspect custom database contents
  - Shows: Available SIDs, date ranges, sample data
  - Perfect for: Verifying custom database structure

- **`check_sf1_data.py`** - Check Sharadar field availability
  - Shows: Which SF1 fields have data, availability percentages
  - Perfect for: Finding which Sharadar fields to use

- **`inspect_sf1.py`** - Inspect Sharadar HDF5 file structure
  - Shows: Available dimensions, tables, data structure
  - Perfect for: Understanding raw Sharadar format

#### Bundle Management

- **`check_bundle_data.py`** - Verify bundle data integrity
  - Shows: Assets, date ranges, data quality
  - Perfect for: Troubleshooting bundle issues

- **`debug_sids.py`** - Debug SID translation issues
  - Shows: Symbol-to-SID mapping, permatickers
  - Perfect for: Fixing SID mismatch errors

#### Testing & Utilities

- **`test_fundamentals_only.py`** - Test custom fundamentals loader
  - Perfect for: Validating custom database setup

- **`backtest_helpers.py`** - Shared utilities for backtesting
  - Features: Bundle registration, path management
  - Used by: Most example scripts

### 🛠️ Utils (`utils/`)

Shared utilities used across examples.

- **`register_bundles.py`** - Bundle registration utility
  - Ensures: Sharadar bundle is properly registered
  - Used by: All examples that need bundles

### 🗄️ Deprecated (`deprecated/`)

Old examples kept for reference. May not work with current version.

## 📓 Jupyter Notebooks

Located in `/notebooks` directory (mounted separately in Docker):

### Core Notebooks

- **`sharadar_data_explorer.ipynb`** - Interactive Sharadar data exploration
  - Features: Symbol lookup, price data, fundamentals, field availability
  - Perfect for: Understanding what data is available

- **`flightlog_elegant_example.ipynb`** - FlightLog in Jupyter
  - Features: Step-by-step setup, interactive cells, best practices
  - Perfect for: Jupyter users, interactive development

- **`multi_source_fundamentals_example.ipynb`** - Multi-source in Jupyter
  - Features: Notebook-based multi-source strategy
  - Perfect for: Interactive strategy development

### Reference Guides

- **`QUICK_REFERENCE.md`** - Pipeline API quick reference
- **`README.md`** - Notebooks overview and setup

## 🎯 Common Use Cases

### "I want to run my first backtest"

```bash
# Terminal 1: Start FlightLog
python scripts/flightlog.py

# Terminal 2: Run simple demo
python examples/1_getting_started/simple_flightlog_demo.py
```

### "I want to explore what data is available"

```bash
# Open Jupyter
# Navigate to: notebooks/sharadar_data_explorer.ipynb
# Run all cells to see available data
```

### "I want to use custom data with Sharadar"

1. Read: `docs/MULTI_SOURCE_QUICKREF.md`
2. Create database: `python examples/6_custom_data/create_fundamentals_db.py`
3. Check it worked: `python examples/6_custom_data/check_lseg_db.py`
4. Run example: `python examples/5_multi_source_data/simple_multi_source_example.py`

### "I want to check which Sharadar fields have data"

```bash
python examples/6_custom_data/check_sf1_data.py
```

### "My pipeline returns 0 stocks"

1. Test pipeline: `python examples/5_multi_source_data/debug_multi_source.py`
2. Check field availability: `python examples/6_custom_data/check_sf1_data.py`
3. Read troubleshooting: `docs/MULTI_SOURCE_DATA.md` (search "Pipeline returns 0 stocks")

## 🐳 Docker Volume Mounts

The examples directory is mounted at `/app/examples` inside the container:

```yaml
volumes:
  - ./examples:/app/examples
  - ./notebooks:/notebooks
```

**Inside container**: Use `/app/examples/`
**Outside container**: Use `./examples/`

## 📚 Documentation

- **Getting Started**: `docs/GETTING_STARTED.md`
- **Multi-Source Data**: `docs/MULTI_SOURCE_DATA.md`
- **Quick Reference**: `docs/MULTI_SOURCE_QUICKREF.md`
- **FlightLog Best Practices**: `docs/FLIGHTLOG_BEST_PRACTICES.md`
- **FlightLog Usage**: `docs/FLIGHTLOG_USAGE.md`

## ⚠️ Important Notes

### Import Paths

When importing from `examples/`, use:

```python
# From examples/utils/
from utils.register_bundles import ensure_bundles_registered

# From custom_data utilities
from custom_data.backtest_helpers import load_bundle_data
```

### Bundle Registration

Most examples need bundles registered:

```python
from utils.register_bundles import ensure_bundles_registered
ensure_bundles_registered()
```

### Database Locations

- **Bundles**: `~/.zipline/bundles/` (inside container: `/root/.zipline/bundles/`)
- **Custom databases**: `~/.zipline/data/custom/`
- **Configuration**: `~/.zipline/extension.py`

## 🔧 Troubleshooting

### Import errors

```python
# ❌ This will fail
from custom_data.register_bundles import ensure_bundles_registered

# ✅ Use this instead
from utils.register_bundles import ensure_bundles_registered
```

### Path issues

All scripts should be run from the project root:

```bash
# ✅ Correct
python examples/1_getting_started/simple_flightlog_demo.py

# ❌ Wrong
cd examples/1_getting_started && python simple_flightlog_demo.py
```

### Can't find bundle

```python
# Make sure to register bundles first
from utils.register_bundles import ensure_bundles_registered
ensure_bundles_registered()
```

## 🎓 Learning Path

**Beginner** → **Intermediate** → **Advanced**

1. **Beginner**
   - Run: `simple_flightlog_demo.py`
   - Read: `docs/GETTING_STARTED.md`
   - Explore: `sharadar_data_explorer.ipynb`

2. **Intermediate**
   - Study: `momentum_strategy_with_flightlog.py`
   - Learn: Pipeline API basics
   - Create: Simple custom strategy

3. **Advanced**
   - Master: `simple_multi_source_example.py`
   - Build: Custom databases
   - Develop: Production strategies with multiple data sources

## 🤝 Contributing

When adding new examples:

1. Place in appropriate numbered directory
2. Add description to this README
3. Include docstring with purpose and usage
4. Test with both Python scripts and Docker

## 📝 Summary

| Directory | Purpose | Start Here |
|-----------|---------|------------|
| `1_getting_started/` | Learn basics | `simple_flightlog_demo.py` |
| `2_strategies/` | Real strategies | `momentum_strategy_with_flightlog.py` |
| `3_analysis/` | Performance analysis | `plot_backtest_results.py` |
| `4_pipeline/` | Pipeline examples | Coming soon |
| `5_multi_source_data/` | Multiple data sources | `simple_multi_source_example.py` |
| `6_custom_data/` | Data management | `check_sf1_data.py` |
| `utils/` | Shared code | `register_bundles.py` |

Happy backtesting! 🚀
