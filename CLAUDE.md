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
**Coverage**: Millions of entities across 250 markets

#### Available Data Types
- **Entity Data**: Corporate structures, legal entity identifiers
- **Ownership**: Investment manager holdings, insider data
- **Smart Estimates**: Consensus analyst estimates (ROE, ROA, EPS)
- **Alpha Models**: Combined factor scores, sector rankings
- **Forward Metrics**: Forward PE, PEG ratios, target prices

#### Integration Pattern
```python
from zipline.pipeline import multi_source as ms

class LSEGFundamentals(ms.Database):
    CODE = "lseg_fundamentals"  # Matches filename
    LOOKBACK_WINDOW = 252

    ReturnOnEquity_SmartEstimate = ms.Column(float)
    CombinedAlphaModelSectorRank = ms.Column(float)
    ForwardPEG = ms.Column(float)
```

**Database Location**: `~/.zipline/data/custom/lseg_fundamentals.sqlite`

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

**Document Version**: 8.0
**Last Updated**: 2025-11-29
**Key Features**: Hidden Point Capital branding, Sharadar + LSEG integration, multi-source pipelines, FlightLog monitoring, comprehensive HTML documentation, LSEG fundamentals enrichment with Sharadar metadata
- redo using these " 1. CashCashEquivalents_Total
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