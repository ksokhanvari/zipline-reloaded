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
    - `bundles/sharadar_bundle.py`: Sharadar data bundle with NaN fixes
    - `custom/`: Custom fundamentals data integration
      - `pipeline_integration.py`: CustomSQLiteLoader implementation
      - `db_manager.py`: Database utilities
      - `config.py`: Configuration constants
  - `pipeline/`: Factor-based screening system
    - `multi_source.py`: **Multi-source data integration (NEW)**
- `examples/`: **Reorganized examples directory (NEW)**
  - `1_getting_started/`: Basic examples and quickstart guides
  - `2_strategies/`: Trading strategy implementations
  - `3_analysis/`: Performance analysis and visualization
  - `4_pipeline/`: Pipeline API examples
  - `5_multi_source_data/`: **Multi-source integration examples (NEW)**
  - `6_custom_data/`: Custom database creation
  - `utils/`: Shared utilities and inspection tools
  - `custom_data/`: Legacy examples (backward compatibility)
- `notebooks/`: Jupyter notebooks
  - `sharadar_data_explorer.ipynb`: **Interactive Sharadar data exploration (NEW)**
  - `multi_source_fundamentals_example.ipynb`: Multi-source strategy notebook
- `docs/`: Documentation
  - `MULTI_SOURCE_DATA.md`: **Multi-source data comprehensive guide (NEW)**
  - `MULTI_SOURCE_QUICKREF.md`: **Quick reference guide (NEW)**
  - `GETTING_STARTED.md`: Getting started guide
  - `FLIGHTLOG_USAGE.md`: FlightLog monitoring guide
- `tests/`: Test suite

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

# Run simple backtest
python examples/1_getting_started/simple_flightlog_demo.py

# Check Sharadar field availability
python examples/utils/check_sf1_data.py

# Check custom database
python examples/utils/check_lseg_db.py
```

## Docker Environment

### BuildKit Configuration
Docker BuildKit is **enabled** for faster builds with caching:

**`.env` file includes:**
```bash
DOCKER_BUILDKIT=1
COMPOSE_DOCKER_CLI_BUILD=1
```

**Benefits**: 5-10x faster rebuilds with `--mount=type=cache` directives in Dockerfile

### Volume Mounts
```yaml
volumes:
  - ./examples:/app/examples
  - ./notebooks:/notebooks
  - zipline-data:/root/.zipline
```

### Important Paths
```
/root/.zipline/bundles/sharadar/      # Sharadar bundle data
/root/.zipline/data/custom/           # Custom databases
/app/examples/                        # Examples directory
/notebooks/                           # Jupyter notebooks
```

---

# Multi-Source Data System (NEW)

## Overview

The **Multi-Source Data System** allows combining Sharadar fundamentals with custom databases in a single Pipeline query. This enables sophisticated strategies that leverage multiple data sources.

**Key Innovation**: Use different data sources for different parts of your strategy without complex workarounds.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│  Sharadar Bundle    │     │  Custom SQLite DB   │
│  (SF1 Fundamentals) │     │  (LSEG/Custom Data) │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           │                           │
           ▼                           ▼
    ┌────────────────────────────────────────┐
    │      AutoLoader (Multi-Source)         │
    │  - Routes columns to correct loader    │
    │  - Handles SID translation             │
    │  - Manages domain awareness            │
    └──────────────┬─────────────────────────┘
                   │
                   ▼
    ┌────────────────────────────────────────┐
    │       SimplePipelineEngine             │
    │  - Executes combined queries           │
    │  - Returns unified DataFrame           │
    └──────────────┬─────────────────────────┘
                   │
                   ▼
    ┌────────────────────────────────────────┐
    │          Your Strategy                 │
    │  - Use both sources in filters         │
    │  - Combine metrics from each           │
    │  - Build consensus signals             │
    └────────────────────────────────────────┘
```

## Key Files

### Core Implementation
- `src/zipline/pipeline/multi_source.py`: Main implementation
  - `AutoLoader`: Intelligent multi-source routing
  - `Database`: Base class for custom databases
  - `Column`: Column definition with type safety
  - `setup_auto_loader()`: One-line setup function
  - `SharadarFundamentals`: Pre-configured Sharadar dataset

### Documentation
- `docs/MULTI_SOURCE_DATA.md`: Comprehensive guide with examples
- `docs/MULTI_SOURCE_QUICKREF.md`: Quick reference for common patterns
- `examples/README.md`: Reorganized examples index

### Examples
- `examples/5_multi_source_data/simple_multi_source_example.py`: Complete working example
- `examples/5_multi_source_data/debug_multi_source.py`: Test pipeline without backtest
- `notebooks/multi_source_fundamentals_example.ipynb`: Jupyter notebook version

### Utilities
- `examples/utils/check_sf1_data.py`: Check Sharadar field availability
- `examples/utils/check_lseg_db.py`: Inspect custom database
- `examples/utils/inspect_sf1.py`: Inspect HDF5 structure
- `examples/utils/test_sharadar_loader.py`: Test Sharadar loader directly

## Quick Start

### 1. Define Custom Database
```python
from zipline.pipeline import multi_source as ms

class CustomFundamentals(ms.Database):
    CODE = "fundamentals"  # Must match SQLite filename
    LOOKBACK_WINDOW = 252

    # Define columns with types
    ROE = ms.Column(float)
    PEG = ms.Column(float)
    MarketCap = ms.Column(float)
```

### 2. Create Pipeline
```python
def make_pipeline():
    # Sharadar data (high availability fields)
    s_fcf = ms.SharadarFundamentals.fcf.latest
    s_pe = ms.SharadarFundamentals.pe.latest

    # Custom data
    c_roe = CustomFundamentals.ROE.latest
    c_peg = CustomFundamentals.PEG.latest

    # Combine both sources
    sharadar_quality = (s_fcf > 0) & (s_pe < 30)
    custom_quality = (c_roe > 15) & (c_peg < 2.5)
    combined = sharadar_quality & custom_quality

    return ms.Pipeline(
        columns={'s_fcf': s_fcf, 's_pe': s_pe, 'c_roe': c_roe},
        screen=combined,
    )
```

### 3. Run Backtest
```python
from zipline import run_algorithm

results = run_algorithm(
    start='2023-01-01',
    end='2024-01-01',
    initialize=initialize,
    bundle='sharadar',
    custom_loader=ms.setup_auto_loader(),  # That's it!
)
```

## Sharadar Field Availability

**CRITICAL**: Many Sharadar fields are **empty**. Always check availability before using.

### High Availability Fields (95%+)
```python
# Cash Flow
fcf = ms.SharadarFundamentals.fcf.latest
fcfps = ms.SharadarFundamentals.fcfps.latest

# Valuation
marketcap = ms.SharadarFundamentals.marketcap.latest  # 96%
pe = ms.SharadarFundamentals.pe.latest                # 90%
pb = ms.SharadarFundamentals.pb.latest
ps = ms.SharadarFundamentals.ps.latest

# Income Statement
revenue = ms.SharadarFundamentals.revenue.latest
ebitda = ms.SharadarFundamentals.ebitda.latest
netinc = ms.SharadarFundamentals.netinc.latest
eps = ms.SharadarFundamentals.eps.latest

# Balance Sheet (100%)
assets = ms.SharadarFundamentals.assets.latest
cashnequsd = ms.SharadarFundamentals.cashnequsd.latest
debt = ms.SharadarFundamentals.debt.latest
equity = ms.SharadarFundamentals.equity.latest
```

### Empty Fields (0% availability)
```python
# ⚠️ DO NOT USE - These fields are empty
roe = ms.SharadarFundamentals.roe.latest  # 0% availability!
```

**Solution**: Use custom database for ROE or calculate manually:
```python
calculated_roe = (s_netinc / s_equity) * 100
```

### Check Availability
```bash
# Run this to see all field availability percentages
python examples/utils/check_sf1_data.py
```

## Common Patterns

### Consensus Scoring
```python
# Both sources agree = higher confidence
consensus = (s_roe > 15) & (c_roe > 15)
selection = s_roe.top(20, mask=consensus)
```

### Multi-Source Universe
```python
# Sharadar: size and valuation
universe = s_marketcap.top(500)
value = (s_pe > 0) & (s_pe < 25)

# Custom: quality
quality = (c_roe > 15) & (c_peg < 2)

# Combined
selection = c_roe.top(20, mask=universe & value & quality)
```

### Divergence Detection
```python
# Find disagreements between sources
divergence = ((s_roe > 15) & (c_roe < 10)) | ((s_roe < 10) & (c_roe > 15))
```

---

# Sharadar Bundle Fixes

## NaN Permaticker Fix

**Issue**: Some assets have NaN permaticker values causing ingest to fail with:
```
IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
```

**Fix Applied** (in `src/zipline/data/bundles/sharadar_bundle.py`):
```python
# Filter out any rows with NaN permaticker before converting to SID
if metadata['permaticker'].isna().any():
    na_count = metadata['permaticker'].isna().sum()
    na_symbols = metadata[metadata['permaticker'].isna()]['symbol'].tolist()
    print(f"   ⚠️  Dropping {na_count} assets with missing permaticker")
    metadata = metadata[metadata['permaticker'].notna()].copy()

# Use permaticker as SID
metadata['sid'] = metadata['permaticker'].astype(int)
```

**Status**: Fixed and tested ✅

## Dimension Information

**Available Dimension**: `ARQ` (As Reported Quarterly)
**NOT Available**: `MRQ` (Most Recent Quarter)

**Correct Usage**:
```python
# ✅ Correct
aapl_quarterly = aapl_fund[aapl_fund['dimension'] == 'ARQ'].copy()

# ❌ Wrong - will return empty
aapl_quarterly = aapl_fund[aapl_fund['dimension'] == 'MRQ'].copy()
```

---

# Directory Reorganization (NEW)

## New Structure

All examples are now organized by functionality under `examples/`:

```
examples/
├── README.md                   # Comprehensive index and guides
├── 1_getting_started/          # Start here!
│   ├── simple_flightlog_demo.py
│   └── quickstart notebooks
├── 2_strategies/               # Real strategies
│   ├── momentum_strategy_with_flightlog.py
│   └── other strategy examples
├── 3_analysis/                 # Performance analysis
│   └── pyfolio notebooks
├── 4_pipeline/                 # Pipeline examples
│   └── sharadar_data_explorer.ipynb
├── 5_multi_source_data/        # Multi-source integration
│   ├── simple_multi_source_example.py
│   ├── debug_multi_source.py
│   └── test_sharadar_loader.py
├── 6_custom_data/              # Database creation
│   └── create_fundamentals_db.py
└── utils/                      # Shared utilities
    ├── register_bundles.py     # Bundle registration
    ├── check_sf1_data.py       # Check Sharadar fields
    ├── check_lseg_db.py        # Inspect custom DB
    ├── inspect_sf1.py          # Inspect HDF5
    └── backtest_helpers.py     # Shared helpers
```

## Key Features

1. **Numbered directories** for clear learning path
2. **Comprehensive README.md** with:
   - Directory structure overview
   - File descriptions
   - Common use cases
   - Quick start guides
   - Learning path recommendations
3. **Unified imports**: All utilities in `examples/utils/`
4. **Backward compatible**: Old paths still work

## Import Pattern

All scripts use this pattern for imports:
```python
import sys
from pathlib import Path

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.register_bundles import ensure_bundles_registered
```

---

# Database Schema & Custom Data

## Custom Database Structure

**Table**: `Price` (historical convention)

```sql
CREATE TABLE Price (
    Date TEXT NOT NULL,      -- YYYY-MM-DD format
    Sid INTEGER NOT NULL,    -- Use bundle SIDs
    [YourColumns...],
    PRIMARY KEY (Date, Sid)
);

CREATE INDEX idx_date ON Price(Date);
CREATE INDEX idx_sid ON Price(Sid);
```

## Database Location

```
~/.zipline/data/custom/{CODE}.sqlite
```

Example:
- Database CODE: `"fundamentals"`
- Location: `~/.zipline/data/custom/fundamentals.sqlite`

## Column Types

```python
Revenue = ms.Column(float)   # Numeric data
Count = ms.Column(int)       # Integer data
Sector = ms.Column(object)   # Text data
```

**IMPORTANT**: Text columns must use empty strings (`''`) not NaN for missing values.

---

# Known Issues & Solutions

## Issue 1: Pipeline Returns 0 Stocks

**Symptoms**: Pipeline executes but returns empty DataFrame

**Causes**:
1. Using empty Sharadar fields (e.g., `roe`)
2. Wrong dimension (`MRQ` instead of `ARQ`)
3. Filters too restrictive

**Solutions**:
1. Check field availability:
   ```bash
   python examples/utils/check_sf1_data.py
   ```
2. Use high-availability fields (fcf, pe, marketcap)
3. Test pipeline first:
   ```bash
   python examples/5_multi_source_data/debug_multi_source.py
   ```

## Issue 2: Categorical Categories Cannot Be Null

**Cause**: Text columns contain NaN values

**Solution**: Fill NaN with empty strings in both places:
```python
# 1. In data loading
for col in text_columns:
    df[col].fillna('', inplace=True)

# 2. In loader (already fixed in multi_source.py)
if col_dtype == object:
    reindexed = reindexed.fillna('')
```

## Issue 3: Import Errors After Reorganization

**Old Import** (now fails):
```python
from custom_data.register_bundles import ensure_bundles_registered
```

**New Import** (correct):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.register_bundles import ensure_bundles_registered
```

---

# Testing & Validation

## Unit Tests
```bash
# All custom data tests
pytest tests/zipline/data/test_custom_pipeline.py -v

# Multi-source tests
pytest tests/zipline/pipeline/test_multi_source.py -v
```

## Data Inspection

### Check Sharadar Fields
```bash
# Inside Docker
docker exec zipline-reloaded-jupyter python /app/examples/utils/check_sf1_data.py
```

### Check Custom Database
```bash
# Inside Docker
docker exec zipline-reloaded-jupyter python /app/examples/utils/check_lseg_db.py
```

### Explore Data Interactively
```bash
# Open Jupyter
# Navigate to: notebooks/sharadar_data_explorer.ipynb
# Run cells to explore bundle data, fundamentals, price history
```

## Pipeline Testing

### Test Without Backtest
```python
# Run this for quick pipeline validation
python examples/5_multi_source_data/debug_multi_source.py
```

**Output**:
```
Pipeline returned 102 stocks passing quality filters
Top 10 by ROE:
  Symbol    ROE    FCF      PE
  AAPL     85.2   123.4B   28.5
  ...
```

---

# FlightLog Integration

## Setup
```python
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog

# Enable FlightLog
enable_flightlog(host='localhost', port=9020)

# Enable progress logging
enable_progress_logging(
    algo_name='My-Strategy',
    update_interval=5  # Log every 5 days
)
```

## Running with FlightLog

**Terminal 1** (FlightLog server):
```bash
python scripts/flightlog.py --host 0.0.0.0 --level INFO
```

**Terminal 2** (Backtest):
```bash
python examples/1_getting_started/simple_flightlog_demo.py
```

**Result**: Real-time colored logs appear in Terminal 1!

---

# Recent Updates & Commits

## Multi-Source Data System (2025-11-17)
- **Commit baecd08f**: Directory reorganization
- **Commit f224fd84**: Fixed USEquityPricingLoader initialization
- **Commit 266237a4**: Updated documentation for multi-source
- **Commit 5235d380**: Replaced monkey-patching with clean custom_loader
- **Commit 71b556eb**: Set working directory for strategy execution
- **Commit bd93b2e9**: Added Sharadar bundle registration

**Key Features Added**:
1. Multi-source data integration via `multi_source.py`
2. AutoLoader for intelligent routing
3. SID translation between bundles
4. Comprehensive documentation
5. Multiple working examples
6. Inspection utilities

## Sharadar Bundle Fixes (2025-11-17)
- Fixed NaN permaticker handling in daily ingest
- Documented dimension usage (ARQ not MRQ)
- Created sharadar_data_explorer.ipynb
- Fixed timezone handling in price data queries

## Directory Reorganization (2025-11-17)
- Reorganized examples into numbered categories
- Moved register_bundles.py to utils/
- Updated all import paths
- Created comprehensive examples/README.md
- Backward compatible with old structure

## BuildKit Configuration (2025-11-17)
- Added DOCKER_BUILDKIT=1 to .env
- Created .env.example with BuildKit settings
- 5-10x faster Docker rebuilds

---

# Quick Reference

## One-Line Import
```python
from zipline.pipeline import multi_source as ms
```

This gives you everything:
- `ms.Pipeline` - Pipeline class
- `ms.Database` - Custom database base class
- `ms.Column` - Column definition
- `ms.SharadarFundamentals` - Sharadar datasets
- `ms.setup_auto_loader()` - Automatic loader setup

## Common Commands

```bash
# Check field availability
python examples/utils/check_sf1_data.py

# Check custom database
python examples/utils/check_lseg_db.py

# Test pipeline
python examples/5_multi_source_data/debug_multi_source.py

# Run simple backtest
python examples/1_getting_started/simple_flightlog_demo.py

# Run multi-source example
python examples/5_multi_source_data/simple_multi_source_example.py
```

## Configuration Options

### Custom Database Directory
```python
loader = ms.setup_auto_loader(
    custom_db_dir='/path/to/databases',
)
```

### Different Bundle
```python
loader = ms.setup_auto_loader(
    bundle_name='my_bundle',
)
```

### Disable SID Translation
```python
loader = ms.setup_auto_loader(
    enable_sid_translation=False,
)
```

---

# Best Practices

## Data Source Selection

1. **Use Sharadar for**: Size, valuation, liquidity filters
   - Market cap, P/E ratio, volume
   - High availability (90-100%)

2. **Use Custom for**: Proprietary metrics, quality factors
   - Custom ROE calculations
   - Analyst estimates
   - Alternative data

3. **Combine for**: Consensus signals
   - Both agree = higher confidence
   - Divergence = potential opportunity

## Pipeline Design

1. **Always check field availability** before using Sharadar fields
2. **Test pipeline first** with debug_multi_source.py
3. **Use high-availability fields** (fcf, pe, marketcap)
4. **Start with loose filters** then tighten based on results
5. **Log pipeline statistics** during development

## Performance

1. **Use indexed SQLite queries** (Date, Sid primary key)
2. **Limit lookback window** to what you need
3. **Filter universe early** (e.g., top 500 by market cap)
4. **Cache results** when appropriate
5. **Enable BuildKit** for faster Docker builds

---

# Troubleshooting Guide

## "Module 'zipline' has no attribute 'pipeline'"

**Cause**: Running outside Docker without zipline installed

**Solution**: Run inside Docker:
```bash
docker exec zipline-reloaded-jupyter python /app/examples/...
```

## "No such file: fundamentals.sqlite"

**Cause**: Custom database not created

**Solution**:
```bash
python examples/6_custom_data/create_fundamentals_db.py
```

## "Pipeline returned 0 stocks"

**Cause**: Empty Sharadar fields or wrong dimension

**Solution**:
1. Check availability: `python examples/utils/check_sf1_data.py`
2. Use high-availability fields (fcf, pe, marketcap)
3. Test pipeline: `python examples/5_multi_source_data/debug_multi_source.py`

## "ImportError: cannot import name 'multi_source'"

**Cause**: Old branch or zipline not rebuilt

**Solution**:
```bash
# Rebuild Docker image
docker compose build

# Or pull latest code
git pull origin your-branch
```

---

# Summary

This Zipline Reloaded fork includes:

1. **Core Zipline**: Algorithmic trading backtesting library
2. **Multi-Source Data System**: Combine Sharadar + custom databases (NEW)
3. **Sharadar Bundle**: Fixed NaN handling, documented field availability
4. **Custom Fundamentals**: SQLite-based integration
5. **Reorganized Examples**: Numbered directories for easy navigation (NEW)
6. **Inspection Tools**: Check data availability, debug pipelines
7. **FlightLog Integration**: Real-time monitoring
8. **BuildKit Support**: Faster Docker builds
9. **Comprehensive Docs**: MULTI_SOURCE_DATA.md, QUICKREF.md, examples/README.md

**Key Success Factors**:
- AutoLoader handles multi-source routing automatically
- SID translation works seamlessly
- High-availability Sharadar fields documented
- Clean examples structure
- Extensive inspection utilities
- Backward compatible reorganization

**Next Steps for New Sessions**:
1. Review `docs/MULTI_SOURCE_QUICKREF.md` for quick patterns
2. Check examples/README.md for learning path
3. Test with: `python examples/5_multi_source_data/simple_multi_source_example.py`
4. Explore data: Open `notebooks/sharadar_data_explorer.ipynb`
5. Build custom strategies combining both sources

---

**Document Version**: 5.0
**Last Updated**: 2025-11-17
**Branch**: claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA
**Key Features**: Multi-source data, reorganized examples, Sharadar fixes, BuildKit
