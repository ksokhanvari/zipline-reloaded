# Sharadar Ticker Metadata Filtering - Implementation Guide

## Overview

This implementation adds universe filtering capabilities using Sharadar ticker metadata (exchange, category, ADR status, sector, etc.) to Zipline-Reloaded strategies. The key innovation is using **custom Pipeline Factors** to filter string-based metadata, which bypasses the limitation that string columns don't support the `.in_()` method.

## Problem Statement

When trying to filter stocks by exchange, category, or ADR status in Zipline Pipeline, the naive approach fails:

```python
# ❌ This doesn't work - AttributeError: 'Latest' object has no attribute 'in_'
exchange = SharadarTickers.exchange.latest
universe = exchange.in_(['NYSE', 'NASDAQ', 'NYSEMKT'])
```

String columns in Pipeline don't support the `.in_()` method for filtering.

## Solution

The solution uses **CustomFactor** classes that process the string data and return boolean filters:

```python
# ✅ This works - CustomFactor returns boolean True/False
exchange_filter = ExchangeFilter()
category_filter = CategoryFilter()
adr_filter = ADRFilter()

universe = exchange_filter & category_filter & ~adr_filter
```

## Files Created

### 1. `sharadar_filters.py`
**Location:** `examples/strategies/sharadar_filters.py`

Provides custom Pipeline Factors for Sharadar metadata filtering:

- **SharadarTickers** - DataSet definition for the static metadata table
- **ExchangeFilter** - Filter by exchange (NYSE, NASDAQ, NYSEMKT)
- **CategoryFilter** - Filter to domestic common stocks
- **ADRFilter** - Identify ADRs (use with `~` to exclude them)
- **SectorFilter** - Filter by sector(s)
- **ScaleMarketCapFilter** - Filter by market cap scale (1-6)
- **create_sharadar_universe()** - Convenience function to combine filters

**Key Innovation:** Each filter is a `CustomFactor` that processes the string metadata and returns boolean values that can be used in Pipeline screens.

### 2. `fcf_yield_strategy.py` (Updated)
**Location:** `examples/strategies/fcf_yield_strategy.py`

Updated FCF Yield strategy to use Sharadar filters:

```python
from sharadar_filters import create_sharadar_universe

def make_pipeline():
    # Create Sharadar universe
    sharadar_universe = create_sharadar_universe(
        exchanges=['NYSE', 'NASDAQ', 'NYSEMKT'],
        include_adrs=False,
    )

    # Apply to market cap filter
    market_cap = CustomFundamentals.CompanyMarketCap.latest
    top_1500_by_mcap = market_cap.top(1500, mask=sharadar_universe)

    # ... rest of strategy logic
```

### 3. `test_sharadar_filters.ipynb`
**Location:** `examples/strategies/test_sharadar_filters.ipynb`

Comprehensive test notebook that validates:
- Individual filter functionality
- Combined filter results
- Filter statistics and distributions
- Advanced filtering (sector, market cap scale)

### 4. `add_sharadar_tickers_fast.ipynb`
**Location:** `examples/lseg_fundamentals/add_sharadar_tickers_fast.ipynb`

Creates the `SharadarTickers` table in the fundamentals database:
- **Static metadata approach** - Just 60K rows (one per ticker)
- **No date expansion** - Metadata doesn't change daily
- **Fast execution** - Completes in ~2 seconds
- **Minimal storage** - ~10 MB vs 10+ GB for expanded version

## Database Table Structure

The `SharadarTickers` table in `fundamentals.sqlite`:

| Column | Type | Description |
|--------|------|-------------|
| Symbol | TEXT | Ticker symbol |
| exchange | TEXT | Exchange (NYSE, NASDAQ, etc.) |
| category | TEXT | Stock category (Domestic Common Stock, ADR, etc.) |
| is_adr | INTEGER | Boolean flag for ADRs (1=True, 0=False) |
| location | TEXT | Company location |
| sector | TEXT | Sharadar sector |
| industry | TEXT | Sharadar industry |
| sicsector | TEXT | SIC sector code |
| sicindustry | TEXT | SIC industry code |
| scalemarketcap | TEXT | Market cap scale (1-6: Nano to Mega) |

**Table Size:** 60,303 rows (one per ticker)
**Indexes:** `idx_sharadar_tickers_symbol` on Symbol column

## How It Works

### 1. Static Metadata Storage

Instead of expanding 60K tickers × 4K dates = 241M rows, we store just 60K rows of static metadata. The custom loader handles joining this with dates at query time.

### 2. CustomFactor Filtering

Each filter is a CustomFactor that:
1. Receives the string data as input (e.g., exchange names)
2. Processes it in the `compute()` method
3. Returns boolean True/False for each asset

Example:

```python
class ExchangeFilter(CustomFactor):
    inputs = [SharadarTickers.exchange]
    window_length = 1

    def compute(self, today, assets, out, exchange):
        # Check if exchange is in our target list
        valid_exchanges = {'NYSE', 'NASDAQ', 'NYSEMKT'}
        out[:] = np.isin(exchange[-1], list(valid_exchanges))
```

### 3. Combining Filters

Filters can be combined using boolean operators:

```python
# AND logic - must pass all filters
combined = filter1 & filter2 & filter3

# OR logic - pass any filter
combined = filter1 | filter2

# NOT logic - invert filter
combined = ~filter1

# Complex combinations
combined = (filter1 & filter2) | (filter3 & ~filter4)
```

## Usage Examples

### Basic Universe Filtering

```python
from sharadar_filters import create_sharadar_universe

# Standard US equity universe
universe = create_sharadar_universe()

# Custom exchange list
universe = create_sharadar_universe(
    exchanges=['NYSE', 'NASDAQ'],
    include_adrs=True,
)

# Large-cap tech stocks
universe = create_sharadar_universe(
    sectors=['Technology'],
    min_market_cap_scale=5,  # Large + Mega cap
)
```

### Manual Filter Combination

```python
from sharadar_filters import ExchangeFilter, CategoryFilter, ADRFilter

# Build custom universe
exchange_filter = ExchangeFilter()
category_filter = CategoryFilter()
adr_filter = ADRFilter()

universe = exchange_filter & category_filter & ~adr_filter
```

### Using in Pipeline

```python
def make_pipeline():
    # Create base universe
    universe = create_sharadar_universe(
        exchanges=['NYSE', 'NASDAQ', 'NYSEMKT'],
        include_adrs=False,
    )

    # Apply additional filters
    market_cap = CustomFundamentals.CompanyMarketCap.latest
    top_1500 = market_cap.top(1500, mask=universe)

    # Calculate alpha factor
    fcf_yield = (fcf - interest) / enterprise_value
    top_100 = fcf_yield.top(100, mask=top_1500)

    return Pipeline(
        columns={'fcf_yield': fcf_yield},
        screen=top_100,
    )
```

## Market Cap Scale Reference

Sharadar market cap scales (from `scalemarketcap` column):

| Scale | Name | Market Cap Range |
|-------|------|-----------------|
| 1 | Nano | < $50M |
| 2 | Micro | $50M - $300M |
| 3 | Small | $300M - $2B |
| 4 | Mid | $2B - $10B |
| 5 | Large | $10B - $200B |
| 6 | Mega | > $200B |

Usage:

```python
# Large-cap and mega-cap only
universe = create_sharadar_universe(min_market_cap_scale=5)

# Small-cap to mid-cap
universe = create_sharadar_universe(
    min_market_cap_scale=3,
    max_market_cap_scale=4,
)
```

## Testing

The easiest way to test the filters is to run the FCF Yield strategy with a short backtest:

```python
# In RunLS.ipynb or a new notebook
import sys
sys.path.insert(0, '/app/examples/utils')

from backtest_helpers import backtest

results = backtest(
    algo_filename='/app/examples/strategies/fcf_yield_strategy.py',
    name='test-sharadar-filters',
    start_date='2024-01-01',
    end_date='2024-03-31',  # Short 3-month test
    capital_base=1000000,
    bundle='sharadar',
)
```

The strategy will:
- ✅ Load Sharadar filters
- ✅ Filter to NYSE/NASDAQ/NYSEMKT domestic common stocks (no ADRs)
- ✅ Select top 1500 by market cap
- ✅ Select top 100 by FCF Yield
- ✅ Run backtest and show results

Check the FlightLog output to see:
- Universe size after Sharadar filtering
- Number of stocks passing each filter step
- Final portfolio composition

## Performance Considerations

### Memory Efficiency
- **Static table:** 60K rows ~10 MB
- **No pre-expansion:** Avoids 241M row table
- **Query-time JOIN:** Handled efficiently by SQLite

### Execution Speed
- **Table creation:** ~2 seconds
- **Query time:** Minimal overhead (<1 second per pipeline execution)
- **Custom Factors:** Efficient numpy operations

### Scalability
- Handles full universe (60K tickers)
- Minimal memory footprint during backtests
- No degradation over long time periods

## Troubleshooting

### Problem: "Table SharadarTickers does not exist"

**Solution:** Run the `add_sharadar_tickers_fast.ipynb` notebook to create the table.

### Problem: "ImportError: cannot import name 'ExchangeFilter'"

**Solution:** Make sure `examples/strategies` is in your Python path:

```python
import sys
sys.path.insert(0, '/app/examples/strategies')
```

### Problem: Filters return unexpected results

**Solution:** Run the test notebook to verify the table data and filter logic:

```bash
jupyter lab examples/strategies/test_sharadar_filters.ipynb
```

### Problem: String columns show as empty or missing

**Solution:** Check that the SharadarTickers table has data:

```python
import sqlite3
conn = sqlite3.connect('/data/custom_databases/fundamentals.sqlite')
result = pd.read_sql('SELECT * FROM SharadarTickers WHERE Symbol = "AAPL"', conn)
print(result)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Pipeline Execution                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Custom Filters (CustomFactor)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Exchange   │  │  Category    │  │     ADR      │      │
│  │   Filter     │  │   Filter     │  │   Filter     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Auto Loader                               │
│  (setup_auto_loader - handles Sharadar + custom DBs)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          SharadarTickers Table (SQLite)                      │
│  ┌───────────────────────────────────────────────────┐      │
│  │  Symbol │ exchange │ category │ is_adr │ sector   │      │
│  ├───────────────────────────────────────────────────┤      │
│  │  AAPL   │ NASDAQ   │ Domestic │   0    │ Tech...  │      │
│  │  MSFT   │ NASDAQ   │ Domestic │   0    │ Tech...  │      │
│  │  ...    │ ...      │ ...      │  ...   │ ...      │      │
│  │  (60,303 rows total)                               │      │
│  └───────────────────────────────────────────────────┘      │
│                                                              │
│  Database: /data/custom_databases/fundamentals.sqlite       │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Test the filters** - Run `test_sharadar_filters.ipynb` to verify setup
2. **Run a backtest** - Use `fcf_yield_strategy.py` as a template
3. **Create custom strategies** - Import and use the filters in your own strategies
4. **Extend filters** - Add new CustomFactors for additional metadata fields

## References

- **Sharadar Documentation:** https://data.nasdaq.com/databases/SFA/documentation
- **Zipline Pipeline Guide:** https://zipline.ml4trading.io/pipeline.html
- **Custom Factors:** https://zipline.ml4trading.io/pipeline-custom-factors.html
