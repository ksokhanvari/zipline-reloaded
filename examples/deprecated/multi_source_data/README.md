# Deprecated Multi-Source Data Files

This directory contains multi-source data integration files that are no longer actively used or have been superseded.

## Deprecated Files

### debug_multi_source.py
- **Status**: Deprecated (Dec 2025)
- **Reason**: Debug/test file no longer needed
- **Original Purpose**: Debugging multi-source data integration issues
- **Resolution**: Multi-source integration now stable and well-documented

## Current Multi-Source Data Workflow

For current multi-source data examples, see:

### Active Files
- `examples/multi_source_data/backtest_with_fundamentals.py` - Complete backtest example
- `examples/multi_source_data/multi_source_strategy.ipynb` - Interactive strategy development
- `examples/multi_source_data/simple_multi_source_example.py` - Simple example

### Documentation
- `docs/MULTI_SOURCE_DATA.md` - Comprehensive multi-source guide
- `docs/MULTI_SOURCE_QUICKREF.md` - Quick reference
- `CLAUDE.md` - Architecture overview with multi-source data flow

## Multi-Source Architecture

Current implementation uses `AutoLoader` pattern:

```python
from zipline.pipeline import multi_source as ms

# Define custom database
class CustomFundamentals(ms.Database):
    CODE = "fundamentals"  # Matches database filename
    LOOKBACK_WINDOW = 252

    # LSEG columns
    ReturnOnEquity_SmartEstimat = ms.Column(float)
    EnterpriseValue_DailyTimeSeries_ = ms.Column(float)

    # Sharadar metadata
    sharadar_exchange = ms.Column(str)
    sharadar_category = ms.Column(str)

# Use in pipeline
def make_pipeline():
    roe = CustomFundamentals.ReturnOnEquity_SmartEstimat.latest
    exchange = CustomFundamentals.sharadar_exchange.latest

    # Combine Sharadar SF1 and custom data
    fcf_sharadar = ms.SharadarFundamentals.fcf.latest

    return ms.Pipeline(
        columns={'ROE': roe, 'FCF': fcf_sharadar, 'Exchange': exchange},
        screen=(exchange == 'NASDAQ')
    )

# Setup and run
results = run_algorithm(
    ...
    custom_loader=ms.setup_auto_loader(),
)
```

## Key Features

- ✅ Automatic routing between Sharadar and custom data sources
- ✅ Transparent SID translation
- ✅ Domain-aware pipeline execution
- ✅ Combines price, fundamentals, and custom signals

---

**Last Updated**: 2025-12-10
**Deprecation Date**: 2025-12-10
