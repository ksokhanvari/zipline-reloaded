# Multi-Source Data - Quick Reference

## One-Line Import

```python
from zipline.pipeline import multi_source as ms
```

This gives you everything you need:
- `ms.Pipeline` - Pipeline class
- `ms.Database` - Custom database base class
- `ms.Column` - Column definition
- `ms.sharadar` - Sharadar datasets
- `ms.setup_auto_loader()` - Automatic loader setup

## Three-Step Pattern

### 1. Define Custom Database

```python
class CustomFundamentals(ms.Database):
    CODE = "fundamentals"  # Must match SQLite filename
    LOOKBACK_WINDOW = 252

    # Define your columns
    ROE = ms.Column(float)
    PEG = ms.Column(float)
    MarketCap = ms.Column(float)
```

### 2. Create Pipeline

```python
def make_pipeline():
    # Sharadar data
    s_roe = ms.sharadar.Fundamentals.slice('MRQ').ROE.latest

    # Custom data
    c_roe = CustomFundamentals.ROE.latest

    # Mix them!
    both_quality = (s_roe > 15) & (c_roe > 15)

    return ms.Pipeline(
        columns={'s_roe': s_roe, 'c_roe': c_roe},
        screen=both_quality,
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
# Find disagreements
divergence = ((s_roe > 15) & (c_roe < 10)) | ((s_roe < 10) & (c_roe > 15))
```

## Sharadar Quick Reference

### Dimension Types
```python
ms.sharadar.Fundamentals.slice('MRQ', period_offset=0)  # Most Recent Quarter
ms.sharadar.Fundamentals.slice('MRT', period_offset=0)  # Most Recent TTM
ms.sharadar.Fundamentals.slice('ARQ', period_offset=0)  # As Reported Quarterly
ms.sharadar.Fundamentals.slice('ART', period_offset=0)  # As Reported TTM
```

### Common Fields
```python
fund = ms.sharadar.Fundamentals.slice('MRQ')

# Returns
roe = fund.ROE.latest
roa = fund.ROA.latest
roic = fund.ROIC.latest

# Cash Flow
fcf = fund.FCF.latest
fcfps = fund.FCFPS.latest

# Valuation
marketcap = fund.MARKETCAP.latest
pe = fund.PE.latest
pb = fund.PB.latest
ps = fund.PS.latest

# Income Statement
revenue = fund.REVENUE.latest
ebitda = fund.EBITDA.latest
eps = fund.EPS.latest

# Balance Sheet
assets = fund.ASSETS.latest
cash = fund.CASHNEQUSD.latest
debt = fund.DEBTUSD.latest
equity = fund.EQUITY.latest
```

## Column Types

```python
Revenue = ms.Column(float)   # Numeric data
Count = ms.Column(int)       # Integer data
Sector = ms.Column(object)   # Text data
```

## Database Location

```
~/.zipline/data/custom/{CODE}.sqlite
```

Example:
```python
CODE = "fundamentals"
# Location: ~/.zipline/data/custom/fundamentals.sqlite
```

## Database Schema

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

## Help Functions

```python
# Print guides
print(ms.help_quick_start())
print(ms.help_database())
print(ms.help_sharadar())

# Or use Python help
help(ms.setup_auto_loader)
help(ms.Database)
help(ms.AutoLoader)
```

## Complete Example

```python
from zipline import run_algorithm
from zipline.api import attach_pipeline, pipeline_output, order_target_percent
from zipline.pipeline import multi_source as ms

# 1. Define database
class MyFundamentals(ms.Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252
    ROE = ms.Column(float)
    PEG = ms.Column(float)

# 2. Create pipeline
def make_pipeline():
    s_roe = ms.sharadar.Fundamentals.slice('MRQ').ROE.latest
    s_marketcap = ms.sharadar.Fundamentals.slice('MRQ').MARKETCAP.latest
    c_roe = MyFundamentals.ROE.latest

    universe = s_marketcap.top(100)
    consensus = (s_roe > 15) & (c_roe > 15)
    selection = s_roe.top(10, mask=universe & consensus)

    return ms.Pipeline(
        columns={'s_roe': s_roe, 'c_roe': c_roe},
        screen=selection,
    )

# 3. Strategy
def initialize(context):
    attach_pipeline(make_pipeline(), 'my_pipeline')

def before_trading_start(context, data):
    context.output = pipeline_output('my_pipeline')

def rebalance(context, data):
    selected = [s for s in context.output.index if data.can_trade(s)]
    weight = 1.0 / len(selected) if selected else 0

    for stock in context.portfolio.positions:
        if stock not in selected and data.can_trade(stock):
            order_target_percent(stock, 0.0)

    for stock in selected:
        order_target_percent(stock, weight)

# 4. Run
results = run_algorithm(
    start='2023-01-01',
    end='2024-01-01',
    initialize=initialize,
    before_trading_start=before_trading_start,
    capital_base=100000,
    bundle='sharadar',
    custom_loader=ms.setup_auto_loader(),
)
```

## Documentation

- Full Guide: `docs/MULTI_SOURCE_DATA.md`
- Examples: `examples/custom_data/simple_multi_source_example.py`
- Notebook: `examples/notebooks/multi_source_fundamentals_example.ipynb`
