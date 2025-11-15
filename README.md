<p align="center">
<a href="https://zipline.ml4trading.io">
<img src="https://i.imgur.com/DDetr8I.png" width="25%">
</a>
</p>

# Zipline Reloaded - Enhanced Professional Fork

> **This is a professional fork of [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) with Sharadar data integration, custom fundamentals support, FlightLog monitoring, and enhanced strategy development tools.**

| Version Info        | [![Python](https://img.shields.io/pypi/pyversions/zipline-reloaded.svg?cacheSeconds=2592000)](https://pypi.python.org/pypi/zipline-reloaded) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![Pandas 2.0+](https://img.shields.io/badge/pandas-2.0%2B-blue) ![NumPy 2.0+](https://img.shields.io/badge/numpy-2.0%2B-blue) |
| ------------------- | ---------- |
| **Upstream**        | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) |
| **Community**       | [![Discourse](https://img.shields.io/discourse/topics?server=https%3A%2F%2Fexchange.ml4trading.io%2F)](https://exchange.ml4trading.io) [![ML4T](https://img.shields.io/badge/Powered%20by-ML4Trading-blue)](https://ml4trading.io) |

## What's New in This Fork

This fork extends zipline-reloaded with **professional-grade enhancements** for serious algorithmic trading research and development:

### 📈 Sharadar Data Integration

- **Premium Data Bundle**: Full integration with NASDAQ Sharadar (institutional-grade US equity data)
- **Incremental Updates**: Download only new data since last ingestion (seconds instead of minutes)
- **Flexible Ticker Selection**: Download all tickers or specify custom lists
- **Total Return Pricing**: Uses `closeadj` for accurate portfolio returns (price + dividends)
- **Corporate Actions**: Automatic handling of splits, dividends, and other actions
- **Point-in-Time Accuracy**: Prevents look-ahead bias with as-known-then data
- **Management Scripts**: Easy-to-use commands for setup, updates, and verification

**Example**:
```bash
# Setup Sharadar bundle with specific tickers
docker exec -it zipline python /scripts/manage_data.py setup \
    --source sharadar \
    --tickers AAPL,MSFT,GOOGL,AMZN,TSLA \
    --name sharadar-tech

# Run strategy with Sharadar data
results = run_algorithm(
    start='2020-01-01',
    end='2023-12-31',
    initialize=initialize,
    capital_base=100000,
    bundle='sharadar-tech'  # Use Sharadar data!
)
```

### 🎯 Custom Fundamentals System

- **SQLite-Based Fundamentals**: Load custom fundamental data (ROE, P/E, market cap, etc.) from CSV
- **Seamless Pipeline Integration**: Use custom fundamentals alongside price data in Pipeline queries
- **Auto-Discovery Pattern**: Automatically discovers and maps all fundamental columns via introspection
- **LoaderDict Implementation**: Handles domain-aware column matching for clean pipeline integration
- **Type-Safe**: Proper dtype handling for numeric and categorical data

**Example**:
```python
# Define custom fundamentals
class CustomFundamentals(Database):
    CODE = "fundamentals"
    ReturnOnEquity = Column(float)
    MarketCap = Column(float)
    Sector = Column(str)

# Use in Pipeline
roe = CustomFundamentals.ReturnOnEquity.latest
market_cap = CustomFundamentals.MarketCap.latest
top_5_roe = roe.top(5, mask=market_cap.top(100))
```

### 📡 FlightLog Real-Time Monitoring

- **Live Log Streaming**: Watch backtest logs in real-time in a separate terminal
- **Color-Coded Levels**: INFO (green), WARNING (yellow), ERROR (red), DEBUG (blue)
- **Multiple Channels**: Support for multiple simultaneous backtests
- **Zero Performance Impact**: Logging happens asynchronously
- **Automatic Saving**: All logs saved to timestamped files
- **QuantRocket-Style Interface**: Professional monitoring UI

**Example**:
```bash
# Terminal 1: Start FlightLog server
python scripts/flightlog.py --host 0.0.0.0 --level INFO

# Terminal 2: Run backtest with FlightLog
from zipline.utils.flightlog_client import enable_flightlog
enable_flightlog(host='localhost', port=9020)
results = run_algorithm(...)  # Logs stream to Terminal 1!
```

### 📊 Progress Logging

- **Real-Time Progress Bars**: QuantRocket-style progress visualization
- **Live Metrics**: Returns, Sharpe ratio, drawdown, P&L updated during backtest
- **Configurable Updates**: Daily, weekly, or custom intervals
- **Zero Configuration**: One line to enable

**Example**:
```python
from zipline.utils.progress import enable_progress_logging

enable_progress_logging(algo_name='My-Strategy', update_interval=5)
results = run_algorithm(...)

# Output:
# [My-Strategy]  Progress      Pct    Date          Cum Returns      Sharpe      Max DD      Cum PNL
# [My-Strategy]  ██--------     20%   2020-03-15            2.5%        1.89         -5%       $2,500
# [My-Strategy]  ██████████    100%   2020-12-31           15.2%        2.15        -12%      $15,200
```

### 🚀 Strategy Development Tools

- **Dual Strategy Templates**:
  - `strategy_top5_roe.py` - Full-featured production strategy with all features
  - `strategy_top5_roe_simple.py` - Minimal clean version for quick testing
- **Notebook Helper**: `run_strategy.py` - Run strategy files from Jupyter with inline parameters
- **Auto-Export**: Automatic CSV and pickle export of backtest results
- **Metadata Tracking**: Save backtest configuration to JSON

### 📊 Machine Learning Support

- **scikit-learn Integration**: Full sklearn library support for ML-based strategies
- **Supported Modules**: SVM, Random Forests, PCA, Ridge/Lasso, KNN, GMM, and more

### 📚 Comprehensive Documentation

- **[docs/SHARADAR_GUIDE.md](docs/SHARADAR_GUIDE.md)** - Complete Sharadar setup and usage guide
- **[docs/FLIGHTLOG_USAGE.md](docs/FLIGHTLOG_USAGE.md)** - FlightLog monitoring documentation
- **[CLAUDE.md](CLAUDE.md)** - Architecture and custom fundamentals documentation
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Quick start guide for all features
- **[notebooks/STRATEGY_README.md](notebooks/STRATEGY_README.md)** - Strategy development guide
- **Multiple example notebooks** - Ready-to-use examples for all features

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- (Optional) NASDAQ Data Link API key for Sharadar data

### Installation

```bash
# Clone this fork
git clone https://github.com/ksokhanvari/zipline-reloaded.git
cd zipline-reloaded

# Build Docker container
docker-compose build

# Start container
docker-compose up -d

# Access Jupyter
# Navigate to http://localhost:8888
```

### Option 1: Quick Start with Sharadar (Recommended)

```bash
# 1. Add your API key to .env file
echo "NASDAQ_DATA_LINK_API_KEY=your_key_here" >> .env
docker-compose restart

# 2. Setup Sharadar bundle (specific tickers for testing)
docker exec -it zipline-reloaded-jupyter python /scripts/manage_data.py setup \
    --source sharadar \
    --tickers AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,NFLX \
    --name sharadar-test

# 3. Run example strategy
docker exec -it zipline-reloaded-jupyter python /notebooks/01_quickstart_sharadar_flightlog.ipynb

# 4. (Optional) Enable FlightLog monitoring
# Terminal 1:
docker exec -it zipline-reloaded-jupyter python scripts/flightlog.py

# Terminal 2:
docker exec -it zipline-reloaded-jupyter python notebooks/test_flightlog_backtest.py
```

### Option 2: Quick Start with Free Data

```bash
# Use Yahoo Finance data (free, no API key needed)
docker exec -it zipline-reloaded-jupyter zipline ingest -b yahoo-direct

# Run dual moving average example
docker exec -it zipline-reloaded-jupyter zipline run \
    -f dual_moving_average.py \
    --start 2020-1-1 \
    --end 2023-1-1 \
    -o dma.pickle \
    --no-benchmark
```

### Option 3: Custom Fundamentals Workflow

```bash
# 1. Load your CSV fundamentals into SQLite (use provided notebook)
docker exec -it zipline-reloaded-jupyter jupyter notebook notebooks/load_csv_fundamentals.ipynb

# 2. Run strategy using custom fundamentals
from run_strategy import run_strategy
results = run_strategy('strategy_top5_roe_simple.py', start='2020-01-01', end='2023-12-31')

# Or direct execution
docker exec -it zipline-reloaded-jupyter python /notebooks/strategy_top5_roe_simple.py
```

## Sharadar Setup Guide

### Get Sharadar Subscription

1. Visit [NASDAQ Sharadar](https://data.nasdaq.com/databases/SFA)
2. Subscribe (typically $200-500/month for individuals)
3. Get API key from Account Settings

### Configuration

```bash
# Add API key to .env
echo "NASDAQ_DATA_LINK_API_KEY=your_actual_key_here" >> .env

# Restart container
docker-compose restart

# Verify API key works
docker exec -it zipline-reloaded-jupyter python scripts/verify_setup.py
```

### Download Data

**Small Dataset (Testing)**:
```bash
docker exec -it zipline-reloaded-jupyter python /scripts/manage_data.py setup \
    --source sharadar \
    --tickers AAPL,MSFT,GOOGL,AMZN \
    --name sharadar-test
```

**Pre-configured Sets**:
```bash
# Tech stocks (15 tickers)
docker exec -it zipline-reloaded-jupyter python /scripts/manage_data.py setup \
    --source sharadar \
    --name sharadar-tech \
    --tickers AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,ADBE,CRM,INTC,AMD,ORCL,CSCO,AVGO

# S&P 500 sample (30 tickers)
docker exec -it zipline-reloaded-jupyter python /scripts/manage_data.py setup \
    --source sharadar \
    --name sharadar-sp500 \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,BRK.B,V,UNH,JPM,JNJ,WMT,MA,PG,XOM,CVX,HD,MRK,ABBV,PFE,KO,PEP,COST,AVGO,TMO,DIS,CSCO,ABT,ACN
```

**All US Equities (Production)**:
```bash
# WARNING: 8,000+ tickers, 10-20 GB, 10-30 minutes
docker exec -it zipline-reloaded-jupyter python /scripts/manage_data.py setup \
    --source sharadar \
    --name sharadar-all
```

### Daily Updates

```bash
# Manual update
docker exec -it zipline-reloaded-jupyter zipline ingest -b sharadar-test

# Automated (add to crontab)
0 18 * * 1-5 docker exec zipline-reloaded-jupyter python /scripts/manage_data.py update --bundle sharadar-test
```

## FlightLog Monitoring

### Basic Usage

**Terminal 1 - Start Server**:
```bash
docker exec -it zipline-reloaded-jupyter python scripts/flightlog.py --host 0.0.0.0
```

**Terminal 2 - Run Backtest**:
```python
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog

enable_flightlog(host='localhost', port=9020)
log_to_flightlog("Starting backtest...", level='INFO')

results = run_algorithm(...)  # Logs stream to Terminal 1!
```

### Advanced Options

```bash
# Custom log level
python scripts/flightlog.py --level DEBUG

# Save to custom file
python scripts/flightlog.py --file logs/backtest_$(date +%Y%m%d).log

# Different port
python scripts/flightlog.py --port 9021

# No colors (for piping)
python scripts/flightlog.py --no-color > output.log
```

## Example Strategy with All Features

```python
import pandas as pd
from zipline import run_algorithm
from zipline.api import *
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog

# Enable monitoring
enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='Multi-Feature-Strategy', update_interval=5)

def make_pipeline():
    # Use Sharadar price data
    close = USEquityPricing.close.latest
    volume = USEquityPricing.volume.latest

    # Custom fundamentals
    roe = CustomFundamentals.ReturnOnEquity.latest
    market_cap = CustomFundamentals.MarketCap.latest

    # Universe: Large cap, high ROE
    universe = market_cap.top(200) & (roe > 15)

    return Pipeline(
        columns={
            'close': close,
            'volume': volume,
            'roe': roe,
            'market_cap': market_cap
        },
        screen=universe
    )

def initialize(context):
    log_to_flightlog("Strategy initialized with Sharadar + Custom Fundamentals", level='INFO')
    attach_pipeline(make_pipeline(), 'combo')
    schedule_function(rebalance, date_rules.week_start())

def before_trading_start(context, data):
    context.stocks = pipeline_output('combo').index
    log_to_flightlog(f"Pipeline returned {len(context.stocks)} stocks", level='INFO')

def rebalance(context, data):
    weight = 1.0 / len(context.stocks)
    for stock in context.stocks:
        order_target_percent(stock, weight)
    log_to_flightlog(f"Rebalanced: {len(context.stocks)} positions", level='INFO')

# Build custom loader for fundamentals
custom_loader = build_pipeline_loaders()

# Run with Sharadar bundle and custom fundamentals!
results = run_algorithm(
    start=pd.Timestamp('2020-01-01'),
    end=pd.Timestamp('2023-12-31'),
    initialize=initialize,
    before_trading_start=before_trading_start,
    capital_base=100000,
    bundle='sharadar-test',  # Sharadar data
    custom_loader=custom_loader  # Custom fundamentals
)

# Results auto-exported to CSV and pickle
print(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
```

---

# Original Zipline Reloaded Documentation

Zipline is a Pythonic event-driven system for backtesting, developed and used as the backtesting and live-trading engine by [crowd-sourced investment fund Quantopian](https://www.bizjournals.com/boston/news/2020/11/10/quantopian-shuts-down-cofounders-head-elsewhere.html). Since it closed late 2020, the domain that had hosted these docs expired. The library is used extensively in the book [Machine Learning for Algorithmic Trading](https://ml4trading.io)
by [Stefan Jansen](https://www.linkedin.com/in/applied-ai/) who is trying to keep the library up to date and available to his readers and the wider Python algotrading community.

- [Join our Community!](https://exchange.ml4trading.io)
- [Documentation](https://zipline.ml4trading.io)

## Features

- **Ease of Use:** Zipline tries to get out of your way so that you can focus on algorithm development.
- **Batteries Included:** Many common statistics like moving average and linear regression can be readily accessed from within a user-written algorithm.
- **PyData Integration:** Input of historical data and output of performance statistics are based on Pandas DataFrames to integrate nicely into the existing PyData ecosystem.
- **Statistics and Machine Learning Libraries:** You can use libraries like matplotlib, scipy, statsmodels, and scikit-learn to support development, analysis, and visualization of state-of-the-art trading systems.

> **Note:** Release 3.05 makes Zipline compatible with NumPy 2.0, which requires Pandas 2.2.2 or higher. If you are using an older version of Pandas, you will need to upgrade it. Other packages may also still take more time to catch up with the latest NumPy release.

> **Note:** Release 3.0 updates Zipline to use [pandas](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v2.0.0.html) >= 2.0 and [SQLAlchemy](https://docs.sqlalchemy.org/en/20/) > 2.0. These are major version updates that may break existing code; please review the linked docs.

> **Note:** Release 2.4 updates Zipline to use [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars) >= 4.2. This is a major version update and may break existing code (which we have tried to avoid but cannot guarantee). Please review the changes [here](https://github.com/gerrymanoim/exchange_calendars/issues/61).

## Installation

Zipline supports Python >= 3.10 and is compatible with current versions of the relevant [NumFOCUS](https://numfocus.org/sponsored-projects?_sft_project_category=python-interface) libraries, including [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html).

### Using `pip`

If your system meets the pre-requisites described in the [installation instructions](https://zipline.ml4trading.io/install.html), you can install Zipline using `pip` by running:

```bash
pip install zipline-reloaded
```

### Using `conda`

If you are using the [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) distributions, you install `zipline-reloaded` from the channel `conda-forge` like so:

```bash
conda install -c conda-forge zipline-reloaded
```

You can also [enable](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html) `conda-forge` by listing it in your `.condarc`.

In case you are installing `zipline-reloaded` alongside other packages and encounter [conflict errors](https://github.com/conda/conda/issues/9707), consider using [mamba](https://github.com/mamba-org/mamba) instead.

### Installing This Fork

To use all the enhanced features in this fork:

```bash
git clone https://github.com/ksokhanvari/zipline-reloaded.git
cd zipline-reloaded
pip install -e .
```

Or use the Docker setup (recommended):

```bash
git clone https://github.com/ksokhanvari/zipline-reloaded.git
cd zipline-reloaded
docker-compose up -d
```

## Quickstart

See our [getting started tutorial](https://zipline.ml4trading.io/beginner-tutorial).

The following code implements a simple dual moving average algorithm:

```python
from zipline.api import order_target, record, symbol


def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')


def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return

    # Compute averages
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    short_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=300, frequency="1d").mean()

    # Trading logic
    if short_mavg > long_mavg:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        order_target(context.asset, 100)
    elif short_mavg < long_mavg:
        order_target(context.asset, 0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)
```

You can then run this algorithm using the Zipline CLI. But first, you need to download some market data with historical prices and trading volumes.

This will download asset pricing data from [NASDAQ](https://data.nasdaq.com/databases/WIKIP) (formerly [Quandl](https://www.nasdaq.com/about/press-center/nasdaq-acquires-quandl-advance-use-alternative-data)).

> This requires an API key, which you can get for free by signing up at [NASDAQ Data Link](https://data.nasdaq.com).

```bash
$ export QUANDL_API_KEY="your_key_here"
$ zipline ingest -b quandl
```

The following will:
- Stream through the algorithm over the specified time range
- Save the resulting performance DataFrame as `dma.pickle`, which you can load and analyze from Python using, e.g., [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded)

```bash
$ zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark
```

You can find other examples in the [zipline/examples](https://github.com/stefan-jansen/zipline-reloaded/tree/main/src/zipline/examples) directory.

## Key Differences from Upstream

This fork adds:

1. **Sharadar Data Integration** - Complete NASDAQ Sharadar bundle with incremental updates
2. **FlightLog Monitoring** - Real-time log streaming with color-coded levels
3. **Progress Logging** - QuantRocket-style progress bars with live metrics
4. **Custom Fundamentals System** - SQLite-based fundamental data integration
5. **LoaderDict Pattern** - Clean domain-aware column matching
6. **Auto-Discovery** - Automatic column mapping via introspection
7. **Strategy Templates** - Production-ready and simple strategy examples
8. **Notebook Integration** - `run_strategy()` helper for Jupyter workflows
9. **scikit-learn Support** - Full sklearn integration for ML algorithms
10. **Enhanced Documentation** - Comprehensive guides for all features

All changes are designed to be **backward compatible** with upstream zipline-reloaded.

## Documentation

### This Fork's Features
- **[docs/SHARADAR_GUIDE.md](docs/SHARADAR_GUIDE.md)** - Complete Sharadar setup and usage
- **[docs/FLIGHTLOG_USAGE.md](docs/FLIGHTLOG_USAGE.md)** - FlightLog monitoring guide
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Quick start for all features
- **[CLAUDE.md](CLAUDE.md)** - Complete architecture and custom fundamentals documentation
- **[notebooks/STRATEGY_README.md](notebooks/STRATEGY_README.md)** - Strategy development guide
- **[Claude.context.md](Claude.context.md)** - Detailed implementation guide for developers

### Upstream Documentation
- **[Upstream Docs](https://zipline.ml4trading.io)** - Original zipline-reloaded documentation
- **[Installation Guide](https://zipline.ml4trading.io/install.html)** - System requirements and setup
- **[Beginner Tutorial](https://zipline.ml4trading.io/beginner-tutorial)** - Getting started tutorial

## File Structure

```
zipline-reloaded/
├── src/zipline/
│   ├── data/
│   │   ├── bundles/
│   │   │   └── sharadar_bundle.py       # Sharadar integration
│   │   └── custom/                      # Custom fundamentals system
│   │       ├── pipeline_integration.py   # CustomSQLiteLoader
│   │       ├── db_manager.py             # Database utilities
│   │       └── config.py                 # Configuration
│   └── utils/
│       ├── progress.py                   # Progress logging
│       └── flightlog_client.py           # FlightLog client
├── scripts/
│   ├── flightlog.py                      # FlightLog server
│   ├── manage_data.py                    # Data management CLI
│   └── verify_setup.py                   # Setup verification
├── notebooks/
│   ├── 01_quickstart_sharadar_flightlog.ipynb  # Quick start
│   ├── load_csv_fundamentals.ipynb       # Data loading workflow
│   ├── strategy_top5_roe.py              # Full-featured strategy
│   ├── strategy_top5_roe_simple.py       # Minimal strategy
│   ├── run_strategy.py                   # Notebook helper
│   └── run_backtest_example.ipynb        # Usage examples
├── docs/
│   ├── SHARADAR_GUIDE.md                 # Sharadar documentation
│   ├── FLIGHTLOG_USAGE.md                # FlightLog documentation
│   └── DATA_MANAGEMENT.md                # Data management guide
├── CLAUDE.md                             # Complete architecture docs
├── GETTING_STARTED.md                    # Quick start guide
└── README.md                             # This file
```

## Docker Environment

This fork is optimized for Docker deployment:

```bash
# Database locations
/root/.zipline/data/custom/fundamentals.sqlite  # Custom fundamentals
/root/.zipline/data/sharadar*/                  # Sharadar bundles

# Strategy files
/notebooks/

# Scripts
/scripts/

# Example: Run in container
docker exec -it zipline-reloaded-jupyter python /notebooks/strategy_top5_roe_simple.py
```

## Contributing

This fork is maintained by [Kamran Sokhanvari](https://github.com/ksokhanvari). Contributions are welcome!

For issues or feature requests related to:
- **Sharadar integration**: Open an issue in this repository
- **FlightLog monitoring**: Open an issue in this repository
- **Custom fundamentals system**: Open an issue in this repository
- **Core Zipline functionality**: Consider opening an issue in the [upstream repository](https://github.com/stefan-jansen/zipline-reloaded)

## Questions, Suggestions, Bugs?

If you find a bug or have questions about this fork's enhanced features, feel free to [open an issue](https://github.com/ksokhanvari/zipline-reloaded/issues/new).

For general Zipline questions, visit the [ML4Trading Community](https://exchange.ml4trading.io).

## License

This project is licensed under the Apache 2.0 License - see the upstream [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Stefan Jansen** - For maintaining zipline-reloaded and keeping it alive for the community
- **Quantopian Team** - For creating the original Zipline library
- **NASDAQ Data Link** - For providing the Sharadar dataset
- **ML4Trading Community** - For continued support and development

## Resources

- [Sharadar Product Page](https://data.nasdaq.com/databases/SFA)
- [Sharadar Documentation](https://data.nasdaq.com/databases/SFA/documentation)
- [ML4Trading Book](https://www.ml4trading.io/) - Excellent resource using Sharadar data
- [Zipline Documentation](https://zipline.ml4trading.io)
- [ML4Trading Community](https://exchange.ml4trading.io)
