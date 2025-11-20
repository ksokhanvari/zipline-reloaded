# Hidden Point Capital - Zipline Reloaded

## Project Context for Claude

## Overview
Zipline-Reloaded is a professional-grade algorithmic trading backtesting platform built on the original Quantopian Zipline framework. This fork provides institutional-quality data integration with Sharadar and LSEG, multi-source fundamental analysis, and real-time monitoring capabilities.

## Key Information
- **Python Version**: >= 3.10
- **Main Dependencies**: pandas >= 2.0, SQLAlchemy >= 2.0, numpy >= 2.0, scikit-learn >= 1.0.0
- **Primary Documentation**: `docs/HPC-toplevel-zipline-documentation.html`
- **External Resources**:
  - Zipline Docs: https://zipline.ml4trading.io
  - LSEG Data: https://www.lseg.com/en/data-analytics
  - Sharadar: https://data.nasdaq.com/databases/SFA

## Project Structure
- `src/zipline/`: Main source code
  - `algorithm.py`: Core algorithm execution
  - `api.py`: Public API functions
  - `data/`: Data ingestion and handling
    - `bundles/sharadar_bundle.py`: Sharadar data bundle with incremental updates
    - `custom/`: Custom fundamentals data integration
  - `pipeline/`: Factor-based screening system
    - `multi_source.py`: Multi-source data integration
    - `data/sharadar.py`: Sharadar fundamentals (66+ columns)
- `examples/`: Example notebooks and strategies
  - `custom_data/`: Custom database examples
  - `lseg_fundamentals/`: LSEG data loading examples
- `notebooks/`: Jupyter notebooks
- `docs/`: Documentation
  - `HPC-toplevel-zipline-documentation.html`: **Main HTML documentation**
  - `README.md`: Documentation index
  - `QUICK_START_DATA.md`: Quick start guide
  - `DATA_MANAGEMENT.md`: Bundle management, updates, cleanup
  - `SHARADAR_FUNDAMENTALS_GUIDE.md`: Fundamentals in pipelines
  - `SYMBOL_MAPPING.md`: Handle FB→META symbol changes
  - `MULTI_SOURCE_DATA.md`: Combine Sharadar + LSEG data
  - `FLIGHTLOG.md`: Real-time monitoring
  - `DEVELOPMENT.md`: Docker build, development workflow
- `tests/`: Test suite
- `data/`: Output directory (persists)

## Docker Environment

### System Requirements
| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 16 GB | 64 GB |
| Disk Space | 50 GB | 100 GB |
| CPU | 4 cores | 8+ cores |

### BuildKit Configuration
Docker BuildKit is **enabled** for faster builds:

```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### Volume Mounts
```yaml
volumes:
  - ./data:/data                    # Strategy output
  - zipline-data:/root/.zipline     # Bundles
  - pip-cache:/root/.cache/pip      # Package cache
```

### Important Paths
```
/root/.zipline/bundles/sharadar/      # Sharadar bundle data
/root/.zipline/data/custom/           # Custom databases
/app/examples/                        # Examples directory
/notebooks/                           # Jupyter notebooks
```

---

## Data Sources

### Sharadar Fundamentals
- **Coverage**: 8,000+ US equities (NYSE, NASDAQ, AMEX)
- **History**: 1998 to present
- **Metrics**: 66+ fundamental columns
- **Update**: Daily after market close (10-30 seconds incremental)

**High Availability Fields (95%+)**:
```python
from zipline.pipeline.data.sharadar import SharadarFundamentals

# Cash Flow
fcf = SharadarFundamentals.fcf.latest
fcfps = SharadarFundamentals.fcfps.latest

# Valuation
marketcap = SharadarFundamentals.marketcap.latest
pe = SharadarFundamentals.pe.latest
pb = SharadarFundamentals.pb.latest

# Income Statement
revenue = SharadarFundamentals.revenue.latest
ebitda = SharadarFundamentals.ebitda.latest
netinc = SharadarFundamentals.netinc.latest
eps = SharadarFundamentals.eps.latest

# Balance Sheet
assets = SharadarFundamentals.assets.latest
debt = SharadarFundamentals.debt.latest
equity = SharadarFundamentals.equity.latest
```

### LSEG Data
- **Entity Data**: Millions of entities across 250 markets
- **Ownership**: Investment manager holdings, insider data
- **Smart Estimates**: Consensus analyst estimates
- **Alpha Models**: Combined factor scores and sector rankings
- **Delivery**: DataScope Select with API/SFTP options

**Integration via Custom Database**:
```python
from zipline.pipeline import multi_source as ms

class LSEGFundamentals(ms.Database):
    CODE = "lseg_fundamentals"
    LOOKBACK_WINDOW = 252

    ReturnOnEquity_SmartEstimate = ms.Column(float)
    CombinedAlphaModelSectorRank = ms.Column(float)
    ForwardPEG = ms.Column(float)
```

---

## Multi-Source Data System

### Quick Start
```python
from zipline.pipeline import multi_source as ms

# 1. Define Custom Database
class CustomFundamentals(ms.Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252
    ROE = ms.Column(float)
    PEG = ms.Column(float)

# 2. Create Pipeline mixing sources
def make_pipeline():
    s_fcf = ms.SharadarFundamentals.fcf.latest
    c_roe = CustomFundamentals.ROE.latest

    return ms.Pipeline(
        columns={'s_fcf': s_fcf, 'c_roe': c_roe},
        screen=(s_fcf > 0) & (c_roe > 15),
    )

# 3. Run Backtest
results = run_algorithm(
    ...,
    custom_loader=ms.setup_auto_loader(),
)
```

### Database Schema
```sql
CREATE TABLE Price (
    Date TEXT NOT NULL,      -- YYYY-MM-DD format
    Sid INTEGER NOT NULL,    -- Use bundle SIDs
    [YourColumns...],
    PRIMARY KEY (Date, Sid)
);
```

**Location**: `~/.zipline/data/custom/{CODE}.sqlite`

---

## Common Commands

### Data Management
```bash
# List bundles
docker compose exec zipline-jupyter zipline bundles

# Ingest/update data
docker compose exec zipline-jupyter zipline ingest -b sharadar

# Clean old ingestions
docker compose exec zipline-jupyter zipline clean -b sharadar --keep-last 2
```

### Docker
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
# Start FlightLog (foreground, shows colors)
docker compose run --rm flightlog

# Filter progress bars
docker compose run --rm flightlog python /app/scripts/flightlog.py --filter-progress
```

---

## FlightLog Integration

```python
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog

# Enable FlightLog
enable_flightlog(host='localhost', port=9020)

# Enable progress logging
enable_progress_logging(algo_name='My-Strategy', update_interval=5)

# Log events in strategy
def initialize(context):
    log_to_flightlog('Strategy initialized', level='INFO')
```

---

## Known Issues & Solutions

### Issue 1: 504 Gateway Timeout on Ingest
**Cause**: NASDAQ Data Link server overload
**Solution**: Wait 5-10 minutes and retry

### Issue 2: Pipeline Returns 0 Stocks
**Cause**: Using empty Sharadar fields or wrong dimension
**Solution**:
- Use high-availability fields (fcf, pe, marketcap)
- Use `ARQ` dimension (not `MRQ`)

### Issue 3: NaN Permaticker Error
**Cause**: Some assets have missing permaticker values
**Status**: Fixed in sharadar_bundle.py

### Issue 4: Import Errors
**Solution**: Run inside Docker:
```bash
docker exec zipline-reloaded-jupyter python /app/examples/...
```

---

## Symbol Mapping

Handles ticker changes (FB→META, GOOG→GOOGL) automatically:

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

## Git Workflow

### Commit Message Format
```
type: Short description

Longer description if needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### Current Branch
`claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA`

---

## Target Audience

- **Institutional Quantitative Traders**: Professional trading desks requiring production-grade backtesting
- **Quantitative Researchers**: Testing factor-based strategies with point-in-time data
- **Portfolio Managers**: Backtesting portfolio construction and risk management
- **Hedge Fund Analysts**: Developing systematic alpha signals across multiple data sources

---

## Quick Reference

### One-Line Import
```python
from zipline.pipeline import multi_source as ms
```

This gives you:
- `ms.Pipeline` - Pipeline class
- `ms.Database` - Custom database base class
- `ms.Column` - Column definition
- `ms.SharadarFundamentals` - Sharadar datasets
- `ms.setup_auto_loader()` - Automatic loader setup

### Key Documentation Files
1. **Main HTML Doc**: `docs/HPC-toplevel-zipline-documentation.html`
2. **Quick Start**: `docs/QUICK_START_DATA.md`
3. **Multi-Source**: `docs/MULTI_SOURCE_DATA.md`
4. **FlightLog**: `docs/FLIGHTLOG.md`

### Resource Requirements
| Operation | Time | Storage | RAM |
|-----------|------|---------|-----|
| Initial download | 10-20 min | 10 GB | 16 GB |
| Daily update | 10-30 sec | +50 MB | 4 GB |

---

**Document Version**: 6.0
**Last Updated**: 2025-11-19
**Key Features**: Hidden Point Capital branding, Sharadar + LSEG integration, multi-source pipelines, FlightLog monitoring, comprehensive HTML documentation
