<p align="center">
<a href="https://zipline.ml4trading.io">
<img src="https://i.imgur.com/DDetr8I.png" width="25%">
</a>
</p>

# Zipline Reloaded - Custom Fundamentals Fork

> **This is a fork of [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) with enhanced custom fundamentals support and strategy development tools.**

| Version Info        | [![Python](https://img.shields.io/pypi/pyversions/zipline-reloaded.svg?cacheSeconds=2592000)](https://pypi.python.org/pypi/zipline-reloaded) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![Pandas 2.0+](https://img.shields.io/badge/pandas-2.0%2B-blue) ![NumPy 2.0+](https://img.shields.io/badge/numpy-2.0%2B-blue) |
| ------------------- | ---------- |
| **Upstream**        | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) |
| **Community**       | [![Discourse](https://img.shields.io/discourse/topics?server=https%3A%2F%2Fexchange.ml4trading.io%2F)](https://exchange.ml4trading.io) [![ML4T](https://img.shields.io/badge/Powered%20by-ML4Trading-blue)](https://ml4trading.io) |

## What's New in This Fork

This fork extends zipline-reloaded with a complete **custom fundamentals data integration system** and enhanced strategy development tools:

### 🎯 Custom Fundamentals System

- **SQLite-Based Fundamentals**: Load custom fundamental data (ROE, P/E, market cap, etc.) from CSV into SQLite databases
- **Seamless Pipeline Integration**: Use custom fundamentals alongside price data in Zipline Pipeline queries
- **Auto-Discovery Pattern**: Automatically discovers and maps all fundamental columns via introspection
- **LoaderDict Implementation**: Handles domain-aware column matching for clean pipeline integration
- **Type-Safe**: Proper dtype handling for numeric and categorical data

### 🚀 Strategy Development Tools

- **Dual Strategy Templates**:
  - `strategy_top5_roe.py` - Full-featured production strategy with progress logging, auto-discovery, and result export
  - `strategy_top5_roe_simple.py` - Minimal clean version for quick testing
- **Notebook Helper**: `run_strategy.py` - Run strategy files from Jupyter notebooks with inline parameter overrides
- **Progress Logging**: Built-in progress tracking using Zipline's `enable_progress_logging()`
- **Auto-Export**: Automatic CSV and pickle export of backtest results

### 📊 Machine Learning Support

- **scikit-learn Integration**: Full sklearn library support for ML-based strategies
- **Supported Modules**:
  - `sklearn.svm` (SVR, LinearSVR)
  - `sklearn.mixture` (GaussianMixture)
  - `sklearn.linear_model` (Lasso, Ridge, ElasticNet, etc.)
  - `sklearn.neighbors` (KNeighborsRegressor)
  - `sklearn.decomposition` (PCA)
  - `sklearn.ensemble` (RandomForestRegressor)
  - And more!

### 📚 Documentation

- **CLAUDE.md**: Comprehensive architecture documentation for the custom fundamentals system
- **STRATEGY_README.md**: Detailed strategy usage guide with examples
- **Claude.context.md**: Complete implementation guide for developers
- **Example Notebooks**: Ready-to-use Jupyter notebooks demonstrating all features

## Quick Start with Custom Fundamentals

### 1. Load Your Fundamental Data

```python
# In Jupyter notebook: load_csv_fundamentals.ipynb
import pandas as pd
from zipline.data.custom import create_custom_database

# Load your CSV with fundamentals
df = pd.read_csv('fundamentals.csv')

# Transform and load into SQLite
# Handles ticker → sid mapping, deduplication, dtype handling
create_custom_database(df, db_path='/root/.zipline/data/custom/fundamentals.sqlite')
```

### 2. Define Your Database Class

```python
from zipline.pipeline.data.db import Database, Column

class CustomFundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252

    # Define your fundamental metrics
    ReturnOnEquity = Column(float)
    MarketCap = Column(float)
    Sector = Column(str)
    # ... add more columns as needed
```

### 3. Run a Strategy

**Option A: Using the Notebook Helper**

```python
from run_strategy import run_strategy

results = run_strategy(
    'strategy_top5_roe_simple.py',
    start='2020-01-01',
    end='2023-12-31',
    capital_base=100000
)

# Analyze results
total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0] - 1) * 100
print(f"Total Return: {total_return:.2f}%")
```

**Option B: Direct Execution**

```bash
python /notebooks/strategy_top5_roe_simple.py
```

### 4. Example Strategy: Top 5 ROE

```python
from zipline import run_algorithm
from zipline.pipeline import Pipeline
from zipline.api import attach_pipeline, pipeline_output, order_target_percent

def make_pipeline():
    roe = CustomFundamentals.ReturnOnEquity.latest
    market_cap = CustomFundamentals.MarketCap.latest

    # Screen: top 100 by market cap
    universe = market_cap.top(100)

    # Select top 5 by ROE
    top_5_roe = roe.top(5, mask=universe)

    return Pipeline(
        columns={'ROE': roe, 'Market_Cap': market_cap},
        screen=top_5_roe
    )

def initialize(context):
    attach_pipeline(make_pipeline(), 'top_roe')
    schedule_function(rebalance, date_rules.week_start())

def before_trading_start(context, data):
    context.stocks = pipeline_output('top_roe').index

def rebalance(context, data):
    weight = 1.0 / len(context.stocks)
    for stock in context.stocks:
        order_target_percent(stock, weight)

# Build custom loader and run
custom_loader = build_pipeline_loaders()
results = run_algorithm(
    start='2020-01-01',
    end='2023-12-31',
    initialize=initialize,
    before_trading_start=before_trading_start,
    capital_base=100000,
    bundle='sharadar',
    custom_loader=custom_loader
)
```

## Architecture Highlights

### Custom SQLite Loader

```python
class CustomSQLiteLoader(PipelineLoader):
    """Loads custom fundamental data from SQLite into Pipeline."""

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        # Query SQLite for date/sid range
        # Convert to 2D numpy arrays
        # Handle dtype conversions (numeric vs categorical)
        # Return AdjustedArray with proper missing values
```

### LoaderDict Pattern

```python
class LoaderDict(dict):
    """Handles domain-aware column matching."""

    def get(self, key, default=None):
        # Matches columns by dataset and column name
        # Ignores domain suffixes (e.g., <US_EQUITIES>)
        # Enables clean integration without monkey-patching
```

### Auto-Discovery

```python
# Automatically discover all fundamental columns
for attr_name in dir(CustomFundamentals):
    if not attr_name.startswith('_'):
        attr = getattr(CustomFundamentals, attr_name)
        if hasattr(attr, 'dataset'):
            custom_loader[attr] = fundamentals_loader
```

## File Structure

```
zipline-reloaded/
├── src/zipline/
│   └── data/custom/              # Custom fundamentals system
│       ├── pipeline_integration.py  # CustomSQLiteLoader
│       ├── db_manager.py            # Database utilities
│       └── config.py                # Configuration
├── notebooks/
│   ├── load_csv_fundamentals.ipynb  # Data loading workflow
│   ├── strategy_top5_roe.py         # Full-featured strategy
│   ├── strategy_top5_roe_simple.py  # Minimal strategy
│   ├── run_strategy.py              # Notebook helper
│   ├── run_backtest_example.ipynb   # Usage examples
│   └── STRATEGY_README.md           # Strategy documentation
├── CLAUDE.md                    # Complete architecture docs
├── Claude.context.md            # Developer implementation guide
└── README.md                    # This file
```

## Docker Environment

This fork is designed to work in Docker containers:

```bash
# Database location
/root/.zipline/data/custom/fundamentals.sqlite

# Strategy files
/notebooks/

# Example: Run in container
docker exec -it zipline python /notebooks/strategy_top5_roe_simple.py
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

To use the custom fundamentals features in this fork:

```bash
git clone https://github.com/ksokhanvari/zipline-reloaded.git
cd zipline-reloaded
pip install -e .
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

1. **Custom Fundamentals System** - Complete SQLite-based fundamental data integration
2. **LoaderDict Pattern** - Clean domain-aware column matching
3. **Auto-Discovery** - Automatic column mapping via introspection
4. **Strategy Templates** - Production-ready and simple strategy examples
5. **Notebook Integration** - `run_strategy()` helper for Jupyter workflows
6. **Progress Logging** - Built-in progress tracking for long backtests
7. **scikit-learn Support** - Full sklearn integration for ML algorithms
8. **Enhanced Documentation** - Comprehensive guides and examples

All changes are designed to be **backward compatible** with upstream zipline-reloaded.

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete architecture and implementation documentation
- **[notebooks/STRATEGY_README.md](notebooks/STRATEGY_README.md)** - Strategy development guide
- **[Claude.context.md](Claude.context.md)** - Detailed implementation guide for developers
- **[Upstream Docs](https://zipline.ml4trading.io)** - Original zipline-reloaded documentation

## Contributing

This fork is maintained by [Kamran Sokhanvari](https://github.com/ksokhanvari). Contributions are welcome!

For issues or feature requests related to:
- **Custom fundamentals system**: Open an issue in this repository
- **Core Zipline functionality**: Consider opening an issue in the [upstream repository](https://github.com/stefan-jansen/zipline-reloaded)

## Questions, Suggestions, Bugs?

If you find a bug or have questions about this fork's custom fundamentals features, feel free to [open an issue](https://github.com/ksokhanvari/zipline-reloaded/issues/new).

For general Zipline questions, visit the [ML4Trading Community](https://exchange.ml4trading.io).

## License

This project is licensed under the Apache 2.0 License - see the upstream [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Stefan Jansen** - For maintaining zipline-reloaded and keeping it alive for the community
- **Quantopian Team** - For creating the original Zipline library
- **ML4Trading Community** - For continued support and development
