<p align="center">
<a href="https://zipline.ml4trading.io">
<img src="https://i.imgur.com/DDetr8I.png" width="25%">
</a>
</p>

# Zipline Reloaded - Enhanced Professional Fork

> **Maintained by [Kamran Sokhanvari](https://github.com/ksokhanvari) at [Hidden Point Capital](https://www.hiddenpointcapital.com)**

> **This is a professional fork of [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) with Sharadar data integration, LSEG fundamentals support, FlightLog monitoring, and enhanced strategy development tools.**

| Version Info        | [![Python](https://img.shields.io/pypi/pyversions/zipline-reloaded.svg?cacheSeconds=2592000)](https://pypi.python.org/pypi/zipline-reloaded) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![Pandas 2.0+](https://img.shields.io/badge/pandas-2.0%2B-blue) ![NumPy 2.0+](https://img.shields.io/badge/numpy-2.0%2B-blue) |
| ------------------- | ---------- |
| **Author**          | [Kamran Sokhanvari](https://github.com/ksokhanvari) - [Hidden Point Capital](https://www.hiddenpointcapital.com) |
| **Upstream**        | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) |
| **Community**       | [![Discourse](https://img.shields.io/discourse/topics?server=https%3A%2F%2Fexchange.ml4trading.io%2F)](https://exchange.ml4trading.io) [![ML4T](https://img.shields.io/badge/Powered%20by-ML4Trading-blue)](https://ml4trading.io) |

## What's New in This Fork

This fork extends zipline-reloaded with **professional-grade enhancements** for institutional-quality algorithmic trading research:

### 🤖 ML-Based Return Forecasting (v3.1.0 - NEW!)

- **Production-Ready ML System**: Predict stock returns (10-day, 90-day, custom horizons) using Histogram-based Gradient Boosting
- **Zero Look-Ahead Bias**: Walk-forward validation ensures production safety for live trading
- **Data-Agnostic**: Automatically processes ANY fundamental columns (LSEG, FMP, Sharadar, custom)
- **Flexible Features**: Scales from 70 to 290+ features based on your input data
- **80%+ Correlation**: Excellent predictive power on 90-180 day return horizons
- **Checkpoint/Resume**: Incremental training for large datasets with resume capability
- **Comprehensive Docs**: 1,124-line README with production deployment guide

**Key Scripts**:
- `data/csv/forecast_returns_ml_walk_forward.py` - Production walk-forward (recommended)
- `data/csv/forecast_returns_ml.py` - Single model for exploration
- `data/csv/README.md` - Complete documentation

**Example**:
```python
# Production-safe ML forecasting (zero look-ahead bias)
python data/csv/forecast_returns_ml_walk_forward.py \
    your_data.csv \
    --forecast-days 10 \
    --target-return-days 90 \
    --resume
```

### 📊 LSEG (Refinitiv) Fundamentals Integration

- **Institutional-Grade Data**: Complete integration with LSEG World-Check fundamentals data
- **Comprehensive Coverage**: 40+ fundamental metrics including financial ratios, cash flow, earnings, and estimates
- **Auto-Discovery System**: Automatic column mapping from CSV to Pipeline DataSet
- **SQLite Backend**: High-performance fundamentals.sqlite database with optimized indexing
- **Pipeline Integration**: Seamless use in Zipline Pipeline alongside price data
- **Custom Loader**: Efficient auto_loader handles Sharadar pricing + LSEG fundamentals

**Available Metrics**:
- **Valuation**: Enterprise Value, Market Cap, P/E, P/B, EV/EBIT, EV/EBITDA, EV/Sales
- **Cash Flow**: Free Operating Cash Flow (FOCF), Forward EV/OCF
- **Profitability**: ROE, ROA, Gross Profit Margin
- **Estimates**: EPS (Actual, Smart Estimate, Surprise), PEG Ratio, Price Target
- **Alpha Signals**: Combined Alpha Model Ranks, Earnings Quality Ranks
- **And more**: Total Debt, Cash, Forward P/E, Forward P/S, P/CF, Dividend estimates

**Example**:
```python
from zipline.pipeline.data import DataSet, Column

class CustomFundamentals(DataSet):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252

    # Automatically loaded from fundamentals.sqlite
    FOCFExDividends_Discrete = Column(float)
    InterestExpense_NetofCapitalizedInterest = Column(float)
    EnterpriseValue_DailyTimeSeries_ = Column(float)
    CompanyMarketCap = Column(float)
    ReturnOnEquity_SmartEstimat = Column(float)
    # ... 35+ more columns

# Use in Pipeline
def make_pipeline():
    fcf = CustomFundamentals.FOCFExDividends_Discrete.latest
    ev = CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest
    mcap = CustomFundamentals.CompanyMarketCap.latest

    # FCF Yield strategy
    fcf_yield = fcf / ev
    universe = mcap.top(1500)

    return Pipeline(
        columns={'fcf_yield': fcf_yield},
        screen=fcf_yield.top(100, mask=universe)
    )
```

### 📈 Sharadar Data Integration

- **Premium Data Bundle**: Full integration with NASDAQ Sharadar (institutional-grade US equity data)
- **Incremental Updates**: Download only new data since last ingestion
- **Flexible Ticker Selection**: Download all tickers or specify custom lists
- **Total Return Pricing**: Uses `closeadj` for accurate returns (price + dividends)
- **Corporate Actions**: Automatic handling of splits, dividends
- **Point-in-Time Accuracy**: Prevents look-ahead bias
- **Metadata Support**: Exchange, category, ADR status, sector, market cap scale

**Example**:
```bash
# Setup Sharadar bundle
docker exec zipline-reloaded-jupyter python /app/scripts/manage_data.py setup \
    --source sharadar \
    --tickers AAPL,MSFT,GOOGL \
    --name sharadar-test

# Run strategy with Sharadar + LSEG fundamentals
results = run_algorithm(
    bundle='sharadar-test',  # Sharadar pricing
    custom_loader=auto_loader  # LSEG fundamentals
)
```

### 🎯 Advanced Strategy Development

- **FCF Yield Strategy**: Production-ready free cash flow yield long-only strategy
- **Sharadar Filters**: Exchange, category, ADR, sector, market cap filters (CustomFilter-based)
- **Pipeline Integration**: Combine Sharadar pricing with LSEG fundamentals
- **Backtest Helpers**: Simplified backtest execution with auto-export
- **FlightLog Integration**: Real-time logging with dual-channel support (LOG + PRINT)

**Strategy Features**:
```python
# fcf_yield_strategy.py
- Top 1500 by market cap filtering
- FCF Yield = (FOCF - Interest) / Enterprise Value
- Long top 100 by FCF Yield
- Monthly rebalancing
- FlightLog monitoring
- Auto-export results to CSV/pickle
```

### 📡 FlightLog Real-Time Monitoring

- **Dual-Channel System**: Separate LOG (9020) and PRINT (9021) channels
- **Live Streaming**: Watch backtest logs in real-time
- **Color-Coded Levels**: INFO (green), WARNING (yellow), ERROR (red), DEBUG (blue)
- **IPython Magic**: `%flightlog` commands for Jupyter notebooks
- **Automatic Integration**: backtest_helpers automatically connects
- **Zero Performance Impact**: Async logging

**Quick Start**:
```bash
# Terminal 1: Start LOG channel
docker exec zipline-reloaded-jupyter python /app/scripts/flightlog.py --port 9020

# Terminal 2: Start PRINT channel
docker exec zipline-reloaded-jupyter python /app/scripts/flightlog.py --port 9021

# Terminal 3: Run backtest (auto-connects to both channels)
from backtest_helpers import backtest
results = backtest(algo_filename='fcf_yield_strategy.py', ...)
```

### 🚀 Auto Loader System

- **Unified Data Access**: Single loader for Sharadar pricing + custom fundamentals
- **Auto-Discovery**: Finds fundamentals.sqlite automatically
- **SID Translation**: Handles symbol → SID mapping across data sources
- **Efficient Queries**: Optimized SQL with proper indexing
- **Domain-Aware**: Respects US_EQUITIES domain for column matching

### 📚 Comprehensive Examples

**Trading Strategies**:
- **`fcf_yield_strategy.py`** - Production FCF Yield long-only strategy
- **`sharadar_filters.py`** - CustomFilter classes for universe filtering
- **`backtest_helpers.py`** - Simplified backtest execution

**ML Forecasting & Data Processing** (in `data/csv/`):
- **`forecast_returns_ml_walk_forward.py`** - Production ML return forecasting with walk-forward validation
- **`forecast_returns_ml.py`** - Single-model ML forecasting for exploration
- **`data/csv/README.md`** - Comprehensive ML forecasting documentation (1,124 lines)

**Data Loading**:
- **`load_csv_fundamentals.ipynb`** - LSEG data loading workflow
- **`01_quickstart_sharadar_flightlog.ipynb`** - Complete quick start guide

## Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) NASDAQ Data Link API key for Sharadar
- (Optional) LSEG fundamentals CSV data

### Installation

```bash
# Clone this fork
git clone https://github.com/ksokhanvari/zipline-reloaded.git
cd zipline-reloaded

# Build and start
docker-compose up -d

# Access Jupyter at http://localhost:8888
```

### Option 1: Complete Setup (Sharadar + LSEG Fundamentals)

```bash
# 1. Setup Sharadar data
echo "NASDAQ_DATA_LINK_API_KEY=your_key" >> .env
docker-compose restart

docker exec zipline-reloaded-jupyter python /app/scripts/manage_data.py setup \
    --source sharadar \
    --tickers AAPL,MSFT,GOOGL,AMZN,TSLA \
    --name sharadar-test

# 2. Load LSEG fundamentals (via Jupyter notebook)
# Navigate to examples/lseg_fundamentals/load_csv_fundamentals.ipynb
# Follow the notebook to load your CSV data

# 3. Run FCF Yield strategy
docker exec zipline-reloaded-jupyter python3 /app/examples/strategies/fcf_yield_strategy.py
```

### Option 2: Quick Test (No Data Setup)

```bash
# Use pre-loaded example data
docker exec zipline-reloaded-jupyter jupyter notebook \
    examples/getting_started/01_quickstart_sharadar_flightlog.ipynb
```

## FCF Yield Strategy Usage

The included FCF Yield strategy demonstrates institutional-quality factor investing:

```python
from backtest_helpers import backtest

# Run with auto-configuration
results = backtest(
    algo_filename='/app/examples/strategies/fcf_yield_strategy.py',
    name='fcf-yield-test',
    start_date='2023-01-01',
    end_date='2024-01-01',
    capital_base=1000000,
    bundle='sharadar'  # Uses Sharadar pricing + LSEG fundamentals
)

# Results auto-exported to:
# - /data/backtest_results/fcf-yield-test_YYYYMMDD_HHMMSS.csv
# - /data/backtest_results/fcf-yield-test_YYYYMMDD_HHMMSS.pkl
```

**Strategy Logic**:
1. Filter to top 1500 stocks by market cap
2. Calculate FCF Yield = (Free Operating Cash Flow - Interest Expense) / Enterprise Value
3. Go long top 100 stocks by FCF Yield
4. Rebalance monthly
5. Log to FlightLog for real-time monitoring

## LSEG Fundamentals Setup

### Loading Your Data

```python
# In Jupyter: examples/lseg_fundamentals/load_csv_fundamentals.ipynb

import pandas as pd
import sqlite3

# 1. Load your CSV
df = pd.read_csv('your_lseg_data.csv')

# 2. Transform to Pipeline format (Symbol, Date, metrics...)
# See notebook for full transformation code

# 3. Save to SQLite
conn = sqlite3.connect('/data/custom_databases/fundamentals.sqlite')
df.to_sql('Price', conn, if_exists='replace', index=False)

# 4. Create indexes for performance
conn.execute('CREATE INDEX idx_price_date_symbol ON Price(Date, Symbol)')
conn.execute('CREATE INDEX idx_price_sid ON Price(Sid)')
conn.commit()
```

### Available Columns

The system automatically discovers all columns in your fundamentals.sqlite database. Standard LSEG columns include:

- `CompanyMarketCap` - Market capitalization
- `FOCFExDividends_Discrete` - Free operating cash flow
- `InterestExpense_NetofCapitalizedInterest` - Interest expense
- `EnterpriseValue_DailyTimeSeries_` - Enterprise value
- `ReturnOnEquity_SmartEstimat` - ROE (Smart Estimate)
- `ReturnOnAssets_SmartEstimate` - ROA
- `EarningsPerShare_Actual` - Actual EPS
- `EarningsPerShare_SmartEstimate_current_Q` - Est. EPS
- `PriceTarget_Median` - Analyst price target
- `EnterpriseValueToEBIT_DailyTimeSeriesRatio_` - EV/EBIT
- And 30+ more...

## Sharadar Metadata Filtering

### Available Filters (CustomFilter-based)

```python
from sharadar_filters import (
    ExchangeFilter,      # Filter by exchange (NYSE, NASDAQ, NYSEMKT)
    CategoryFilter,      # Domestic Common Stock only
    ADRFilter,          # Exclude ADRs
    SectorFilter,       # Filter by sector
    ScaleMarketCapFilter,  # Market cap scale (1-6: Nano to Mega)
    create_sharadar_universe  # Helper function
)

# Example: Standard US equity universe
universe = create_sharadar_universe(
    exchanges=['NYSE', 'NASDAQ', 'NYSEMKT'],
    include_adrs=False,
    sectors=['Technology', 'Healthcare'],
    min_market_cap_scale=4  # Mid-cap and above
)

# Use in Pipeline
market_cap = CustomFundamentals.CompanyMarketCap.latest
top_stocks = market_cap.top(500, mask=universe)
```

**Note**: Sharadar metadata filtering requires a date-expanded SharadarTickers table. See `examples/strategies/SHARADAR_FILTERS_README.md` for setup.

## FlightLog Monitoring

### Basic Setup

```bash
# Terminal 1: LOG channel (INFO, WARNING, ERROR, DEBUG)
docker exec zipline-reloaded-jupyter python /app/scripts/flightlog.py --port 9020

# Terminal 2: PRINT channel (print() statements)
docker exec zipline-reloaded-jupyter python /app/scripts/flightlog.py --port 9021
```

### Using in Strategies

```python
from zipline.utils.flightlog_client import log_to_flightlog

def rebalance(context, data):
    log_to_flightlog(f"Rebalancing {len(context.longs)} positions", level='INFO')
    # ... trading logic
```

### IPython Magic Commands

```python
# In Jupyter notebooks
%flightlog both   # Connect to both channels
%flightlog log    # Connect to LOG only
%flightlog print  # Connect to PRINT only
%flightlog off    # Disconnect all
```

## Docker Environment

### Directory Structure

```
/data/
├── custom_databases/
│   └── fundamentals.sqlite          # LSEG fundamentals
├── backtest_results/                # Auto-exported results
│   ├── *.csv
│   └── *.pkl
└── logs/                            # FlightLog output

/app/
├── examples/
│   ├── strategies/
│   │   ├── fcf_yield_strategy.py
│   │   └── sharadar_filters.py
│   ├── lseg_fundamentals/
│   │   └── load_csv_fundamentals.ipynb
│   └── utils/
│       └── backtest_helpers.py
└── scripts/
    ├── flightlog.py
    └── manage_data.py
```

### Key Paths

```bash
# Run strategy
docker exec zipline-reloaded-jupyter python3 /app/examples/strategies/fcf_yield_strategy.py

# Access fundamentals DB
docker exec zipline-reloaded-jupyter python3 -c "
import sqlite3
conn = sqlite3.connect('/data/custom_databases/fundamentals.sqlite')
print(pd.read_sql('SELECT * FROM Price LIMIT 5', conn))
"

# Check backtest results
ls /Users/kamran/Documents/Code/Docker/zipline-reloaded/data/backtest_results/
```

## Key Differences from Upstream

This fork adds:

1. **LSEG Fundamentals Integration** - 40+ institutional-grade fundamental metrics
2. **Auto Loader System** - Unified Sharadar + fundamentals data access
3. **FCF Yield Strategy** - Production-ready fundamental factor strategy
4. **Sharadar Metadata Filters** - Exchange, ADR, sector, market cap filtering
5. **Dual-Channel FlightLog** - Separate LOG and PRINT monitoring
6. **Backtest Helpers** - Simplified strategy execution and auto-export
7. **IPython Magic** - `%flightlog` commands for Jupyter
8. **CustomFilter Pattern** - Proper Pipeline filter implementation
9. **Comprehensive Examples** - Ready-to-use strategy templates
10. **Enhanced Documentation** - Complete guides for all features

All changes maintain **backward compatibility** with upstream zipline-reloaded.

## Documentation

### This Fork's Features
- **[examples/strategies/SHARADAR_FILTERS_README.md](examples/strategies/SHARADAR_FILTERS_README.md)** - Metadata filtering guide
- **[docs/FLIGHTLOG.md](docs/FLIGHTLOG.md)** - FlightLog complete documentation
- **[docs/HPC-toplevel-zipline-documentation.html](docs/HPC-toplevel-zipline-documentation.html)** - Visual guide
- **[CLAUDE.md](CLAUDE.md)** - Architecture documentation
- **[examples/lseg_fundamentals/](examples/lseg_fundamentals/)** - Data loading notebooks

### Upstream Documentation
- **[Upstream Docs](https://zipline.ml4trading.io)** - Original documentation
- **[Installation Guide](https://zipline.ml4trading.io/install.html)** - Setup instructions
- **[Beginner Tutorial](https://zipline.ml4trading.io/beginner-tutorial)** - Getting started

## Example: Complete Strategy

```python
# fcf_yield_strategy.py - Complete implementation

import sys
sys.path.insert(0, '/app/examples/strategies')

from zipline.pipeline import Pipeline
from zipline.pipeline.data import DataSet, Column
from zipline.api import attach_pipeline, pipeline_output, order_target_percent
from sharadar_filters import create_sharadar_universe

class CustomFundamentals(DataSet):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252

    FOCFExDividends_Discrete = Column(float)
    InterestExpense_NetofCapitalizedInterest = Column(float)
    EnterpriseValue_DailyTimeSeries_ = Column(float)
    CompanyMarketCap = Column(float)

def make_pipeline():
    # Universe: Top 1500 by market cap
    market_cap = CustomFundamentals.CompanyMarketCap.latest
    universe = market_cap.top(1500)

    # Alpha: FCF Yield
    fcf = CustomFundamentals.FOCFExDividends_Discrete.latest
    interest = CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest
    ev = CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest
    fcf_yield = (fcf - interest) / ev

    # Screen: Top 100 by FCF Yield
    return Pipeline(
        columns={'fcf_yield': fcf_yield},
        screen=fcf_yield.top(100, mask=universe)
    )

def initialize(context):
    attach_pipeline(make_pipeline(), 'fcf_yield')
    schedule_function(rebalance, date_rules.month_start())

def rebalance(context, data):
    weight = 0.95 / len(context.longs)
    for stock in context.longs:
        order_target_percent(stock, weight)
```

## Performance Features

- **Optimized Queries**: Indexed SQLite queries for fundamentals
- **Efficient Loading**: Auto-loader handles symbol → SID mapping
- **Memory Management**: Chunked data loading for large universes
- **Fast Backtests**: 1-year backtest completes in ~30 seconds

## Contributing

Maintained by [Kamran Sokhanvari](https://github.com/ksokhanvari) at [Hidden Point Capital](https://www.hiddenpointcapital.com).

For issues:
- **LSEG fundamentals**: Open issue in this repo
- **Sharadar integration**: Open issue in this repo
- **FlightLog**: Open issue in this repo
- **Core Zipline**: Consider [upstream repo](https://github.com/stefan-jansen/zipline-reloaded)

## License

Apache 2.0 License - see [LICENSE](LICENSE)

## Acknowledgments

### Fork Development
- **[Kamran Sokhanvari](https://github.com/ksokhanvari)** - Enhanced fork author and maintainer
- **[Hidden Point Capital](https://www.hiddenpointcapital.com)** - Quantitative research sponsor

### Upstream & Community
- **Stefan Jansen** - zipline-reloaded maintainer
- **Quantopian Team** - Original Zipline creators
- **NASDAQ / LSEG** - Premium data providers
- **ML4Trading Community** - Continued development support

## Resources

- [NASDAQ Sharadar](https://data.nasdaq.com/databases/SFA) - Premium equity data
- [LSEG World-Check](https://www.lseg.com/) - Fundamentals data
- [ML4Trading Book](https://www.ml4trading.io/) - Excellent Zipline resource
- [Zipline Docs](https://zipline.ml4trading.io) - Official documentation
- [Community Forum](https://exchange.ml4trading.io) - Get help and share ideas

---

**Built for institutional-quality quantitative research. Production-ready strategies with professional-grade data.**
