# Hidden Point Capital - Zipline Reloaded

## Project Context for Claude

This document serves as the central repository of project architecture and information. A new Claude session should read this file first to understand the project structure, functionality, and conventions.

---

## Overview

**Zipline-Reloaded** is a professional-grade algorithmic trading backtesting platform built on the original Quantopian Zipline framework. This fork by Hidden Point Capital provides:

- **Institutional-quality data integration** with Sharadar (NASDAQ) and LSEG (London Stock Exchange Group)
- **Multi-source fundamental analysis** combining multiple data providers in a single pipeline
- **Real-time monitoring** via FlightLog streaming system
- **Dockerized environment** for reproducible results

### Target Audience
- Institutional Quantitative Traders
- Quantitative Researchers
- Portfolio Managers
- Hedge Fund Analysts

---

## Technical Stack

### Core Technologies
- **Python**: >= 3.10
- **pandas**: >= 2.0
- **SQLAlchemy**: >= 2.0
- **numpy**: >= 2.0
- **scikit-learn**: >= 1.0.0

### Infrastructure
- **Docker**: Containerized environment
- **Jupyter**: Interactive notebook server
- **SQLite**: Custom fundamentals storage
- **HDF5**: Time-series data storage

---

## Project Structure

```
zipline-reloaded/
├── src/zipline/                    # Main source code
│   ├── algorithm.py                # Core algorithm execution engine
│   ├── api.py                      # Public API (order, symbol, etc.)
│   ├── data/
│   │   ├── bundles/
│   │   │   ├── sharadar_bundle.py  # Sharadar data bundle with incremental updates
│   │   │   └── core.py             # Bundle registry and management
│   │   └── custom/
│   │       ├── pipeline_integration.py  # CustomSQLiteLoader
│   │       ├── db_manager.py       # Database utilities
│   │       └── config.py           # Configuration constants
│   ├── pipeline/
│   │   ├── multi_source.py         # Multi-source data integration
│   │   ├── engine.py               # Pipeline execution engine
│   │   └── data/
│   │       └── sharadar.py         # SharadarFundamentals (66+ columns)
│   ├── finance/
│   │   ├── execution.py            # Order execution
│   │   └── trading.py              # Trading mechanics
│   └── utils/
│       ├── flightlog_client.py     # FlightLog TCP client
│       └── progress.py             # Progress logging
│
├── examples/
│   ├── custom_data/                # Custom database examples
│   │   ├── backtest_helpers.py     # Shared backtest utilities
│   │   └── create_fundamentals_db.py
│   ├── lseg_fundamentals/          # LSEG data loading
│   │   └── load_csv_fundamentals.ipynb
│   └── strategies/                 # Strategy examples
│
├── notebooks/                      # Jupyter notebooks
│   └── (mounted at /notebooks in container)
│
├── docs/                           # Documentation
│   ├── HPC-toplevel-zipline-documentation.html  # Main HTML documentation
│   ├── README.md                   # Documentation index
│   ├── QUICK_START_DATA.md         # Quick start guide (5 min)
│   ├── DATA_MANAGEMENT.md          # Bundle management, updates, cleanup
│   ├── SHARADAR_FUNDAMENTALS_GUIDE.md
│   ├── SYMBOL_MAPPING.md           # FB→META ticker changes
│   ├── MULTI_SOURCE_DATA.md        # Combining data sources
│   ├── FLIGHTLOG.md                # Real-time monitoring
│   └── DEVELOPMENT.md              # Docker build optimization
│
├── scripts/
│   └── flightlog.py                # FlightLog TCP server
│
├── tests/                          # Test suite
├── data/                           # Output directory (persists)
├── docker-compose.yml              # Docker services configuration
├── Dockerfile                      # Container build instructions
├── .env                            # Environment variables (API keys)
└── CLAUDE.md                       # This file
```

---

## Architecture

### Data Flow
```
┌─────────────────────┐     ┌─────────────────────┐
│  Sharadar Bundle    │     │  Custom SQLite DB   │
│  (SF1 Fundamentals) │     │  (LSEG/Custom Data) │
└──────────┬──────────┘     └──────────┬──────────┘
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
    │          Trading Algorithm             │
    │  - Use both sources in filters         │
    │  - Build consensus signals             │
    └────────────────────────────────────────┘
```

### Key Components

#### 1. Sharadar Bundle (`src/zipline/data/bundles/sharadar_bundle.py`)
- Downloads price and fundamental data from NASDAQ Data Link
- Supports incremental updates (10-30 seconds vs 10-20 minutes)
- Handles NaN permaticker values
- Uses ARQ (As Reported Quarterly) dimension

#### 2. Multi-Source System (`src/zipline/pipeline/multi_source.py`)
- `AutoLoader`: Intelligent routing between data sources
- `Database`: Base class for custom databases
- `Column`: Type-safe column definitions
- `setup_auto_loader()`: One-line configuration

#### 3. Pipeline Engine (`src/zipline/pipeline/`)
- Factor-based stock screening
- Combines Sharadar + custom data
- Returns filtered DataFrame for trading

#### 4. FlightLog (`scripts/flightlog.py`, `src/zipline/utils/`)
- Real-time log streaming via TCP
- Color-coded log levels
- Progress bars and portfolio metrics

---

## Data Sources

### Sharadar Fundamentals

**Provider**: NASDAQ Data Link
**Coverage**: 8,000+ US equities (NYSE, NASDAQ, AMEX)
**History**: 1998 to present
**Metrics**: 66+ fundamental columns
**Update**: Daily after market close

#### Field Categories
- **Income Statement**: revenue, netinc, ebitda, grossprofit, eps
- **Balance Sheet**: assets, debt, equity, cashneq, receivables
- **Cash Flow**: fcf, ncfo, ncfi, ncff, capex
- **Valuation**: pe, pb, ps, marketcap, ev
- **Profitability**: roe, roa, roic, grossmargin
- **Per-Share**: eps, bvps, dps, fcfps

#### High Availability Fields (95%+)
```python
from zipline.pipeline.data.sharadar import SharadarFundamentals

fcf = SharadarFundamentals.fcf.latest           # Free cash flow
marketcap = SharadarFundamentals.marketcap.latest
pe = SharadarFundamentals.pe.latest
revenue = SharadarFundamentals.revenue.latest
netinc = SharadarFundamentals.netinc.latest
assets = SharadarFundamentals.assets.latest
equity = SharadarFundamentals.equity.latest
```

**Important**: Some fields like `roe` have 0% availability - calculate manually or use LSEG.

### LSEG Data

**Provider**: London Stock Exchange Group (formerly Refinitiv)
**Coverage**: 4,440 symbols, 4,012 dates (2009-2025), ~9M rows
**Database**: `fundamentals.sqlite` (enriched with Sharadar metadata)

#### Available LSEG Fundamental Columns (36 fields)

**Price & Volume (2):**
- `RefPriceClose`, `RefVolume`

**Company Information (3):**
- `CompanyCommonName`, `CompanyMarketCap`, `GICSSectorName`

**Valuation Metrics (9):**
- `EnterpriseValue_DailyTimeSeries_`, `EnterpriseValueToEBIT_DailyTimeSeriesRatio_`
- `EnterpriseValueToEBITDA_DailyTimeSeriesRatio_`, `EnterpriseValueToSales_DailyTimeSeriesRatio_`
- `ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_`
- `ForwardPEG_DailyTimeSeriesRatio_`, `PriceEarningsToGrowthRatio_SmartEstimate_`
- `ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_`
- `ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_`

**Cash Flow (2):**
- `FOCFExDividends_Discrete`, `CashCashEquivalents_Total`

**Debt & Interest (2):**
- `Debt_Total`, `InterestExpense_NetofCapitalizedInterest`

**Earnings Metrics (4):**
- `EarningsPerShare_Actual`, `EarningsPerShare_SmartEstimate_current_Q`
- `EarningsPerShare_SmartEstimate_prev_Q`, `EarningsPerShare_ActualSurprise`

**Growth & Estimates (4):**
- `LongTermGrowth_Mean`, `Estpricegrowth_percent`
- `PriceTarget_Median`, `Dividend_Per_Share_SmartEstimate`

**Profitability Ratios (3):**
- `ReturnOnEquity_SmartEstimat`, `ReturnOnAssets_SmartEstimate`
- `GrossProfitMargin_ActualSurprise`

**Rankings & Quality (4):**
- `CombinedAlphaModelRegionRank`, `CombinedAlphaModelSectorRank`
- `CombinedAlphaModelSectorRankChange`, `EarningsQualityRegionRank_Current`

**Analyst Recommendations (1):**
- `Recommendation_Median_1_5_`

**Custom Signals (2):**
- `pred` (VIX prediction signal), `bc1` (BC signal)

**Additional Metadata**: 9 Sharadar metadata columns (`sharadar_exchange`, `sharadar_category`, `sharadar_is_adr`, etc.) merged during CSV enrichment for universe filtering.

#### Integration Pattern
```python
from zipline.pipeline import multi_source as ms

class CustomFundamentals(ms.Database):
    CODE = "fundamentals"  # Matches database filename
    LOOKBACK_WINDOW = 252

    # Example columns
    ReturnOnEquity_SmartEstimat = ms.Column(float)
    CombinedAlphaModelSectorRank = ms.Column(float)
    ForwardPEG_DailyTimeSeriesRatio_ = ms.Column(float)
    GICSSectorName = ms.Column(str)
    CompanyMarketCap = ms.Column(float)
    pred = ms.Column(float)  # VIX signal
    bc1 = ms.Column(float)   # BC signal

    # Sharadar metadata (for universe filtering)
    sharadar_exchange = ms.Column(str)
    sharadar_category = ms.Column(str)
    sharadar_is_adr = ms.Column(bool)
```

**Database Locations**:
- Docker: `/data/custom_databases/fundamentals.sqlite`
- Local: `~/.zipline/data/custom/fundamentals.sqlite`

---

## Docker Environment

### System Requirements
| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 16 GB | 64 GB |
| Disk Space | 50 GB | 100 GB |
| CPU | 4 cores | 8+ cores |

### Services
```yaml
services:
  zipline-jupyter:       # Main Jupyter server
  flightlog:             # Log streaming server
```

### Volume Mounts
```yaml
volumes:
  - ./data:/data                    # Strategy output
  - zipline-data:/root/.zipline     # Bundles and data
  - pip-cache:/root/.cache/pip      # Package cache
```

### Important Container Paths
```
/root/.zipline/bundles/sharadar/    # Sharadar bundle data
/root/.zipline/data/custom/          # Custom SQLite databases
/app/src/zipline/                    # Source code
/app/examples/                       # Examples
/notebooks/                          # Jupyter notebooks
```

### BuildKit Configuration
For faster builds (5-10x improvement):
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

---

## API Reference

### Multi-Source Import
```python
from zipline.pipeline import multi_source as ms

# Available:
ms.Pipeline           # Pipeline class
ms.Database           # Custom database base class
ms.Column             # Column definition
ms.SharadarFundamentals  # Sharadar dataset
ms.setup_auto_loader()   # AutoLoader setup
```

### Backtest Pattern
```python
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol
from zipline.pipeline import multi_source as ms

def initialize(context):
    attach_pipeline(make_pipeline(), 'my_pipeline')

def before_trading_start(context, data):
    context.output = pipeline_output('my_pipeline')

def rebalance(context, data):
    for stock in context.output.index:
        order_target_percent(stock, 0.05)

results = run_algorithm(
    start=pd.Timestamp('2020-01-01', tz='UTC'),
    end=pd.Timestamp('2024-01-01', tz='UTC'),
    initialize=initialize,
    before_trading_start=before_trading_start,
    capital_base=1000000,
    bundle='sharadar',
    custom_loader=ms.setup_auto_loader(),
)
```

### FlightLog Integration
```python
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
from zipline.utils.progress import enable_progress_logging

enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='My-Strategy', update_interval=5)

def initialize(context):
    log_to_flightlog('Strategy initialized', level='INFO')
```

### Custom Database Schema
```sql
CREATE TABLE Price (
    Date TEXT NOT NULL,      -- YYYY-MM-DD format
    Sid INTEGER NOT NULL,    -- Bundle SIDs
    [Column1] REAL,
    [Column2] REAL,
    PRIMARY KEY (Date, Sid)
);

CREATE INDEX idx_date ON Price(Date);
CREATE INDEX idx_sid ON Price(Sid);
```

---

## Common Commands

### Data Management
```bash
# List bundles
docker compose exec zipline-jupyter zipline bundles

# Ingest/update data (incremental)
docker compose exec zipline-jupyter zipline ingest -b sharadar

# Clean old ingestions
docker compose exec zipline-jupyter zipline clean -b sharadar --keep-last 2
```

### Docker Operations
```bash
# Start services
docker compose up -d

# Rebuild image
docker compose build zipline-jupyter

# Restart container
docker compose restart zipline-jupyter

# View logs
docker compose logs zipline-jupyter

# Enter container shell
docker exec -it zipline-reloaded-jupyter bash
```

### FlightLog
```bash
# Start FlightLog (foreground with colors)
docker compose run --rm flightlog

# Filter progress bars
docker compose run --rm flightlog python /app/scripts/flightlog.py --filter-progress
```

---

## Known Issues & Solutions

### 1. 504 Gateway Timeout on Ingest
**Cause**: NASDAQ Data Link server overload
**Solution**: Wait 5-10 minutes and retry

### 2. Pipeline Returns 0 Stocks
**Causes**:
- Using empty Sharadar fields (e.g., `roe`)
- Wrong dimension (MRQ instead of ARQ)
- Filters too restrictive

**Solutions**:
- Use high-availability fields (fcf, pe, marketcap)
- Use `ARQ` dimension
- Test pipeline first without backtest

### 3. NaN Permaticker Error
**Cause**: Some assets have missing permaticker values
**Status**: Fixed in sharadar_bundle.py (drops affected rows)

### 4. Import Errors Outside Docker
**Solution**: Run inside Docker container:
```bash
docker exec zipline-reloaded-jupyter python /app/examples/...
```

### 5. "No such file: fundamentals.sqlite"
**Solution**: Create the database first:
```bash
python examples/custom_data/create_fundamentals_db.py
```

---

## Symbol Mapping

Handles ticker changes (FB→META, GOOG→GOOGL) with temporal lookups:

```python
from temporal_sid_mapper import TemporalSIDMapper

mapper = TemporalSIDMapper(asset_finder)
custom_data['Sid'] = mapper.map_dataframe_auto(
    custom_data,
    symbol_col='Symbol',
    date_col='Date',
)
```

---

## Git Conventions

### Commit Message Format
```
type: Short description

Longer description if needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### Current Branch
`claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA`

### Main Branch
`main`

---

## Documentation

### Primary Documentation
- **HTML**: `docs/HPC-toplevel-zipline-documentation.html` (comprehensive, dark theme)

### Markdown Guides
| Guide | Purpose | Time |
|-------|---------|------|
| QUICK_START_DATA.md | Get started fast | 5 min |
| DATA_MANAGEMENT.md | Bundle management | 15 min |
| SHARADAR_FUNDAMENTALS_GUIDE.md | Fundamentals in pipelines | 20 min |
| SYMBOL_MAPPING.md | Handle symbol changes | 10 min |
| MULTI_SOURCE_DATA.md | Combine data sources | 15 min |
| FLIGHTLOG.md | Real-time monitoring | 10 min |
| DEVELOPMENT.md | Development workflow | 10 min |

---

## Resource Requirements

| Operation | Time | Storage | RAM |
|-----------|------|---------|-----|
| Initial download | 10-20 min | 10 GB | 16 GB |
| Daily update | 10-30 sec | +50 MB | 4 GB |
| Typical backtest | 1-5 min | - | 8 GB |

---

## External Resources

- **Zipline Docs**: https://zipline.ml4trading.io
- **Community Forum**: https://exchange.ml4trading.io
- **Sharadar Data**: https://data.nasdaq.com/databases/SFA
- **LSEG Data**: https://www.lseg.com/en/data-analytics
- **NASDAQ Support**: connect@data.nasdaq.com

---

## Session Continuation Notes

When continuing a session:

1. **Check git status**: `git status` to see any pending changes
2. **Check current branch**: Ensure on correct feature branch
3. **Review recent commits**: `git log --oneline -10`
4. **Check running containers**: `docker compose ps`
5. **Review this file**: For project context and conventions

---

## Recent Session: LSEG Fundamentals + Sharadar Metadata Integration (2025-11-27 to 2025-11-29)

### Summary

Completed comprehensive integration of LSEG fundamentals with Sharadar ticker metadata for universe filtering in Zipline strategies. The work involved creating a CSV enrichment workflow and fixing critical data quality issues.

### Key Accomplishments

1. **Created CSV Enrichment Tools**:
   - `add_sharadar_metadata_to_fundamentals.py` - CLI script (286 lines)
   - `add_sharadar_metadata_to_fundamentals.ipynb` - Interactive notebook with 9 visualizations (32 cells)
   - Single source of truth pattern: notebook imports from script

2. **Fixed Critical Deduplication Issue**:
   - **Problem**: Sharadar tickers.h5 had 105,766 entries but only 60,303 unique tickers
   - **Cause**: Two types of duplicates:
     - Active vs delisted entries (different `isdelisted` values)
     - Multiple active entries for same ticker (both `isdelisted='N'`)
   - **Result**: Merge created 16.9M rows instead of 9.0M, with duplicate Date+Symbol combinations
   - **Solution**: Enhanced deduplication sorting by `['isdelisted', index]`
   - **Outcome**: 60,303 → 30,801 unique tickers, NO duplicate rows

3. **Added 9 Metadata Columns**:
   - `sharadar_exchange` - NYSE, NASDAQ, NYSEMKT, etc.
   - `sharadar_category` - Domestic Common Stock, ADR, ETF, etc.
   - `sharadar_is_adr` - 1=ADR, 0=Non-ADR
   - `sharadar_location` - Company location
   - `sharadar_sector` - Sharadar sector classification
   - `sharadar_industry` - Sharadar industry classification
   - `sharadar_sicsector` - SIC sector
   - `sharadar_sicindustry` - SIC industry
   - `sharadar_scalemarketcap` - 1-6 (Nano to Mega cap)

4. **Comprehensive Documentation**:
   - Updated `examples/lseg_fundamentals/README.md` with:
     - Complete directory structure and file descriptions
     - Two workflows: CSV enrichment (recommended) vs database table approach
     - Pipeline integration examples
     - Detailed deduplication explanation
     - Troubleshooting section
   - All files committed and pushed to GitHub

### Files Modified

**Primary Files**:
- `examples/lseg_fundamentals/add_sharadar_metadata_to_fundamentals.py` - Script with deduplication fix
- `examples/lseg_fundamentals/add_sharadar_metadata_to_fundamentals.ipynb` - Notebook with 9 charts, imports from script
- `examples/lseg_fundamentals/add_sharadar_metadata_to_fundamentals_full.ipynb` - Backup with embedded functions
- `examples/lseg_fundamentals/README.md` - Comprehensive documentation with directory structure

**Key Code Pattern** (add_sharadar_metadata_to_fundamentals.py:86-99):
```python
# CRITICAL: Enhanced deduplication handles both types of duplicates
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

### Data Quality Statistics

- **LSEG Fundamentals**: 9,010,487 rows, 4,440 symbols, 4,012 dates (2008-2025)
- **Sharadar Metadata**: 60,303 raw entries → 30,801 unique tickers after deduplication
- **Match Rate**: 95.6% (8,616,181 of 9,010,487 rows matched)
- **Output**: Enriched CSV with 47 columns (38 original + 9 metadata)

### Technical Decisions

1. **Single Source of Truth**: Notebook imports functions from Python script to avoid code duplication and ensure fixes propagate
2. **Stable Deduplication**: Sort by `['isdelisted', index]` ensures reproducible results when multiple entries have same delisted status
3. **Preserved Visualizations**: Kept all 9 chart cells in notebook for data analysis
4. **CSV Enrichment Over Database Table**: Recommended workflow adds metadata columns to CSV before database load (simpler, no joins needed)

### Important Notes for Future Sessions

1. **Deduplication is Critical**: Always use the enhanced deduplication when loading Sharadar tickers
2. **Verification**: After enrichment, check for duplicate Date+Symbol rows (should be none)
3. **Match Rate**: Expect ~95.6% match rate between LSEG and Sharadar data
4. **Docker Sync**: Remember to copy updated notebooks to Docker container
5. **Import Pattern**: Notebook cell 5 imports from script - update script, not notebook functions

### Git Commits (Branch: claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA)

- `34b870fb` - fix: Improve deduplication to handle multiple active ticker entries
- `f224fd84` - fix: Use USEquityPricingLoader.without_fx() for proper initialization
- `266237a4` - docs: Update documentation to reflect proper custom_loader approach
- Earlier commits on LS-ZR strategy porting and FlightLog integration

### Next Steps (if continuing this work)

1. Test enriched CSV loading into fundamentals.sqlite with `load_csv_fundamentals.ipynb`
2. Verify Pipeline integration with metadata columns
3. Create example strategy using `sharadar_exchange`, `sharadar_category`, `sharadar_is_adr` filters
4. Consider extending to other data sources (QuantRocket entities, ownership data)

---

## Recent Session: MRQ Configuration, LS-ZR Fixes, and Auto-Detection Features (2025-12-09)

### Summary

Completed MRQ bundle configuration, fixed critical LS-ZR-ported strategy issues (IBM handling, missing data), and added auto-detection features to LSEG fundamentals workflows for improved usability.

### Key Accomplishments

1. **MRQ Dimension Configuration**:
   - Modified `sharadar_bundle.py` to filter for MRQ dimension at API level
   - Added dimension filters at lines 1064 (nasdaqdatalink API) and 1264 (bulk export API)
   - Created `SESSION_SUMMARY_2025-12-02.md` documenting ARQ vs MRQ differences
   - Identified 9 critical differences between QuantRocket and Zipline implementations

2. **LS-ZR-ported Strategy Fixes**:
   - **IBM VIX Flag Handling**: Extract IBM row before market cap filtering, add back if filtered out
   - **Missing Quarterly Data**: Fill missing fcf/int with 0 instead of dropping stocks
   - **Removed Zero Filter**: Allow cash_return=0 (stocks without earnings data)
   - **Debug Output**: Added comprehensive diagnostics for data coverage issues
   - Fixed KeyError and empty portfolio crashes when IBM filtered out or data missing

3. **Auto-Detection Features for LSEG Fundamentals**:
   - Added `find_newest_csv()` function to detect newest CSV by date in filename
   - Added `generate_output_filename()` to auto-generate output names
   - Made `--input` and `--output` CLI arguments optional
   - Updated both enrichment and load notebooks to auto-detect newest files
   - Pattern: `YYYYMMDD_YYYYMMDD.csv` → selects file with most recent end date

4. **Directory Organization**:
   - Created `deprecated/` directory in `examples/lseg_fundamentals/`
   - Moved 3 deprecated notebooks (`_full.ipynb`, `add_sharadar_tickers*.ipynb`)
   - Created `deprecated/README.md` explaining deprecations
   - Updated main README to reflect new structure

5. **Sharadar TICKERS API Fix**:
   - Bundle no longer creates `tickers.h5` file
   - Updated `load_sharadar_tickers()` to download from NASDAQ Data Link API
   - Downloads SHARADAR/TICKERS table with all metadata columns
   - Requires `NASDAQ_DATA_LINK_API_KEY` environment variable

### Files Modified

**Strategy Files**:
- `examples/strategies/LS-ZR-ported.py` - IBM handling, missing data fixes, debug output
- `examples/strategies/SESSION_SUMMARY_2025-12-02.md` - Performance delta analysis

**LSEG Fundamentals**:
- `examples/lseg_fundamentals/add_sharadar_metadata_to_fundamentals.py` - Auto-detection, API download
- `examples/lseg_fundamentals/add_sharadar_metadata_to_fundamentals.ipynb` - Auto-detection, cell reordering
- `examples/lseg_fundamentals/load_csv_fundamentals.ipynb` - Auto-detection helper function
- `examples/lseg_fundamentals/README.md` - Updated structure, deprecated directory
- `examples/lseg_fundamentals/deprecated/README.md` - Deprecation explanations

**Bundle Configuration**:
- `src/zipline/data/bundles/sharadar_bundle.py` - MRQ dimension filters

### Key Technical Decisions

1. **Preserve IBM for VIX Signals**: Even if not in top 1500 by market cap, needed for strategy signals
2. **Fill vs Drop Missing Data**: Better to fill with 0 and let ranking handle it than drop all stocks
3. **Auto-Detection by Date**: Always process most recent data without manual configuration
4. **API Download for TICKERS**: More reliable than depending on cached bundle files
5. **Deprecated Directory**: Keep old files for reference but reduce confusion

### Git Commits (Branch: claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA)

- `0230e6e8` - feat: Add auto-detection of newest CSV by date in load_csv_fundamentals
- `13132385` - fix: Download Sharadar TICKERS directly from API
- `3af1e790` - fix: Import auto-detection functions before using them
- `c7ea0669` - feat: Add auto-detection of newest CSV file for LSEG enrichment
- `c4c49f06` - refactor: Move deprecated LSEG notebooks to deprecated/ directory
- `96622c4e` - docs: Add session summary for MRQ configuration
- `d3481a0c` - fix: Handle missing data and preserve IBM for VIX signals

### Known Issues

1. **LSEG Enterprise Value Coverage**: Only ~2 stocks have entval data in current database
   - Solution: Reload LSEG fundamentals with full dataset
   - Temporary workaround: Use Sharadar enterprise value instead

2. **MRQ Data Recency**: MRQ dimension may have gaps for most recent quarters
   - Solution: Use ARQ for point-in-time accuracy if needed

### Next Steps

1. Reload LSEG fundamentals database with full enterprise value coverage
2. Test LS-ZR-ported strategy with complete LSEG data
3. Compare backtest results with QuantRocket implementation
4. Apply remaining fixes from 9-item comparison (money flow factors, Ichimoku signal, etc.)

---

**Document Version**: 9.0
**Last Updated**: 2025-12-09
**Key Features**: Hidden Point Capital branding, Sharadar + LSEG integration, multi-source pipelines, FlightLog monitoring, MRQ configuration, auto-detection workflows, comprehensive strategy debugging
   2. CombinedAlphaModelRegionRank
   3. CombinedAlphaModelSectorRank
   4. CombinedAlphaModelSectorRankChange
   5. CompanyCommonName
   6. CompanyMarketCap
   7. Debt_Total
   8. Dividend_Per_Share_SmartEstimate
   9. EarningsPerShare_Actual
  10. EarningsPerShare_ActualSurprise
  11. EarningsPerShare_SmartEstimate_current_Q
  12. EarningsPerShare_SmartEstimate_prev_Q
  13. EarningsQualityRegionRank_Current
  14. EnterpriseValueToEBITDA_DailyTimeSeriesRatio_
  15. EnterpriseValueToEBIT_DailyTimeSeriesRatio_
  16. EnterpriseValueToSales_DailyTimeSeriesRatio_
  17. EnterpriseValue_DailyTimeSeries_
  18. Estpricegrowth_percent
  19. FOCFExDividends_Discrete
  20. ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_
  21. ForwardPEG_DailyTimeSeriesRatio_
  22. ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_
  23. ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_
  24. GICSSectorName
  25. GrossProfitMargin_ActualSurprise
  26. InterestExpense_NetofCapitalizedInterest
  27. LongTermGrowth_Mean
  28. PriceEarningsToGrowthRatio_SmartEstimate_
  29. PriceTarget_Median
  30. Recommendation_Median_1_5_
  31. RefPriceClose
  32. RefVolume
  33. ReturnOnAssets_SmartEstimate
  34. ReturnOnEquity_SmartEstimat
  35. bc1
  36. pred"