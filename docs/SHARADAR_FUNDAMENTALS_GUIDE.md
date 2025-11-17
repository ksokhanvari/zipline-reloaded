# Sharadar Fundamentals Guide

## Overview

The Sharadar Fundamentals feature provides access to 150+ quarterly fundamental metrics from Sharadar's SF1 table through Zipline's Pipeline API. This enables sophisticated factor-based strategies that screen stocks based on financial fundamentals.

**Key Features:**
- **Point-in-Time Correct**: Uses filing dates (`datekey`) to prevent look-ahead bias
- **150+ Metrics**: Revenue, earnings, ratios, per-share metrics, cash flow, and more
- **Automatic Forward-Filling**: Quarterly data persists until next quarter is filed
- **Pipeline Integration**: Seamlessly works with Zipline's factor library
- **Zero Performance Impact**: Fundamentals are optional - pricing/backtests unaffected

## Table of Contents

1. [Quick Start](#quick-start)
2. [Bundle Ingestion](#bundle-ingestion)
3. [Available Metrics](#available-metrics)
4. [Pipeline Usage](#pipeline-usage)
5. [Point-in-Time Correctness](#point-in-time-correctness)
6. [Example Strategies](#example-strategies)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Ingest Bundle with Fundamentals

```bash
# In Docker container
docker exec zipline-reloaded-jupyter zipline ingest -b sharadar
```

Fundamentals are automatically included by default (`include_fundamentals=True`).

### 2. Use in Pipeline

```python
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals
from zipline.pipeline.factors import Latest

# Create pipeline with fundamentals
pipeline = Pipeline(
    columns={
        'revenue': SharadarFundamentals.revenue.latest,
        'net_income': SharadarFundamentals.netinc.latest,
        'roe': SharadarFundamentals.roe.latest,
        'pe_ratio': SharadarFundamentals.pe.latest,
    }
)

# Run pipeline in your strategy
def initialize(context):
    attach_pipeline(pipeline, 'fundamentals')

def before_trading_start(context, data):
    context.fundamentals = pipeline_output('fundamentals')
    # Use context.fundamentals DataFrame in your trading logic
```

---

## Bundle Ingestion

### Including Fundamentals (Default)

By default, fundamentals are downloaded and stored during bundle ingestion:

```python
# In extension.py or registration script
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

# Default - includes fundamentals
register('sharadar', sharadar_bundle())
```

### Excluding Fundamentals

If you only need pricing data and want to save download time:

```python
# Exclude fundamentals (pricing only)
register('sharadar', sharadar_bundle(include_fundamentals=False))
```

### Bundle Storage

Fundamentals are stored in the bundle directory:

```
~/.zipline/data/sharadar/
└── 2025-11-16T12;34;56.789012/
    ├── assets-7.sqlite          # Asset metadata
    ├── daily_equities.bcolz/    # Pricing data
    ├── adjustments/             # Split/dividend adjustments
    └── fundamentals/            # NEW: Fundamentals data
        └── sf1.h5               # SF1 metrics (HDF5 format)
```

**File Size**: ~10-50 MB for typical bundles (e.g., tech stocks, SP500 sample)
**Compression**: HDF5 format with level-9 compression

---

## Available Metrics

### Income Statement

| Metric | Column Name | Description |
|--------|-------------|-------------|
| Revenue | `revenue` | Total revenue (sales) |
| Net Income | `netinc` | Bottom-line profit |
| EBITDA | `ebitda` | Earnings before interest, taxes, depreciation, amortization |
| EBIT | `ebit` | Operating income |
| Gross Profit | `grossprofit` | Revenue minus cost of revenue |
| Operating Expenses | `opex` | Operating expenses |
| R&D | `rnd` | Research & development expenses |
| SG&A | `sgna` | Selling, general & administrative expenses |

### Balance Sheet

| Metric | Column Name | Description |
|--------|-------------|-------------|
| Total Assets | `assets` | Total assets |
| Total Equity | `equity` | Shareholders' equity |
| Total Debt | `debt` | Short-term + long-term debt |
| Cash & Equivalents | `cashneq` | Cash and cash equivalents |
| Working Capital | `workingcapital` | Current assets - current liabilities |
| PP&E | `ppnenet` | Property, plant & equipment (net) |

### Cash Flow

| Metric | Column Name | Description |
|--------|-------------|-------------|
| Operating Cash Flow | `ncfo` | Cash from operations |
| Free Cash Flow | `fcf` | Operating cash flow - capex |
| Capital Expenditures | `capex` | Investments in fixed assets |
| Cash from Financing | `ncff` | Cash from debt/equity issuance |
| Cash from Investing | `ncfi` | Cash from investments |

### Ratios

| Metric | Column Name | Description |
|--------|-------------|-------------|
| ROE | `roe` | Return on equity (%) |
| ROA | `roa` | Return on assets (%) |
| P/E Ratio | `pe` | Price to earnings |
| P/B Ratio | `pb` | Price to book |
| P/S Ratio | `ps` | Price to sales |
| EV/EBITDA | `evebitda` | Enterprise value / EBITDA |
| Debt/Equity | `de` | Debt to equity ratio |
| Current Ratio | `currentratio` | Current assets / current liabilities |
| Gross Margin | `grossmargin` | Gross profit margin (%) |
| Net Margin | `netmargin` | Net profit margin (%) |

### Per-Share Metrics

| Metric | Column Name | Description |
|--------|-------------|-------------|
| EPS | `eps` | Earnings per share (basic) |
| EPS Diluted | `epsdil` | Earnings per share (diluted) |
| Book Value per Share | `bvps` | Book value per share |
| Sales per Share | `sps` | Revenue per share |
| FCF per Share | `fcfps` | Free cash flow per share |
| Dividends per Share | `dps` | Dividends per share |

### Growth Metrics

| Metric | Column Name | Description |
|--------|-------------|-------------|
| Revenue Growth | `revenuegrowth` | Year-over-year revenue growth (%) |
| Net Income Growth | `netincgrowth` | Year-over-year net income growth (%) |
| EPS Growth | `epsgrowth` | Year-over-year EPS growth (%) |

**Complete List**: See `src/zipline/pipeline/data/sharadar.py` for all 80+ available columns.

---

## Pipeline Usage

### Basic Usage: Latest Values

Get the most recent fundamental value for each stock:

```python
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals

pipeline = Pipeline(
    columns={
        'revenue': SharadarFundamentals.revenue.latest,
        'roe': SharadarFundamentals.roe.latest,
    }
)
```

### Creating Factors

Use fundamentals in custom factors:

```python
from zipline.pipeline.factors import CustomFactor

class QualityScore(CustomFactor):
    """Quality score based on ROE, ROA, and margins."""

    inputs = [
        SharadarFundamentals.roe,
        SharadarFundamentals.roa,
        SharadarFundamentals.grossmargin,
        SharadarFundamentals.netmargin,
    ]
    window_length = 1  # Latest values

    def compute(self, today, assets, out, roe, roa, gross_margin, net_margin):
        # Normalize each metric (0-100 scale)
        roe_score = np.clip(roe[-1] / 30 * 100, 0, 100)
        roa_score = np.clip(roa[-1] / 15 * 100, 0, 100)
        gm_score = np.clip(gross_margin[-1] / 50 * 100, 0, 100)
        nm_score = np.clip(net_margin[-1] / 20 * 100, 0, 100)

        # Average score
        out[:] = (roe_score + roa_score + gm_score + nm_score) / 4

# Use in pipeline
pipeline = Pipeline(
    columns={'quality': QualityScore()},
    screen=QualityScore() > 60,  # Only high-quality stocks
)
```

### Screening

Filter stocks based on fundamental criteria:

```python
from zipline.pipeline.filters import StaticAssets

# Value screen: low P/E, low P/B, high dividend yield
value_screen = (
    (SharadarFundamentals.pe.latest < 15) &
    (SharadarFundamentals.pb.latest < 2) &
    (SharadarFundamentals.dps.latest > 1.0)
)

# Growth screen: revenue growth > 20%, positive earnings growth
growth_screen = (
    (SharadarFundamentals.revenuegrowth.latest > 20) &
    (SharadarFundamentals.netincgrowth.latest > 0)
)

# Quality screen: high ROE, low debt
quality_screen = (
    (SharadarFundamentals.roe.latest > 15) &
    (SharadarFundamentals.de.latest < 1.0) &
    (SharadarFundamentals.currentratio.latest > 1.5)
)

pipeline = Pipeline(
    columns={
        'pe': SharadarFundamentals.pe.latest,
        'revenue_growth': SharadarFundamentals.revenuegrowth.latest,
        'roe': SharadarFundamentals.roe.latest,
    },
    screen=value_screen | growth_screen | quality_screen,
)
```

### Combining with Technical Factors

Blend fundamentals with price-based signals:

```python
from zipline.pipeline.data import EquityPricing
from zipline.pipeline.factors import Returns, SimpleMovingAverage

# Fundamental factor: Value
value_factor = -SharadarFundamentals.pe.latest  # Negative because lower is better

# Technical factor: Momentum
momentum_factor = Returns(window_length=252)  # 1-year momentum

# Blend: 50% value, 50% momentum
combined_factor = 0.5 * value_factor.zscore() + 0.5 * momentum_factor.zscore()

pipeline = Pipeline(
    columns={
        'combined_score': combined_factor,
        'pe': SharadarFundamentals.pe.latest,
        'momentum': momentum_factor,
    },
    screen=combined_factor.top(50),  # Top 50 stocks by combined score
)
```

---

## Point-in-Time Correctness

### Why It Matters

**Problem**: Companies report earnings ~45 days after quarter-end. Using the quarter-end date would introduce look-ahead bias.

**Example**:
- Apple's Q4 2022 ends: **December 31, 2022**
- Apple files 10-K: **February 15, 2023** (`datekey`)

If we use December 31 as the availability date, backtests could "know" Q4 earnings in January/February 2023, which is impossible in real trading.

### How We Handle It

Sharadar's SF1 table includes a `datekey` field representing the **filing date** when data became public. Our loader uses this for point-in-time correctness:

```python
# Behind the scenes in SharadarFundamentalsLoader:
#
# For a request on 2023-02-10 (before filing):
#   → Returns NaN or previous quarter's data
#
# For a request on 2023-02-15 (filing date):
#   → Returns Q4 2022 data (now available)
#
# For a request on 2023-03-01 (after filing):
#   → Returns Q4 2022 data (still latest)
```

### Forward-Filling

Quarterly data is automatically forward-filled until the next quarter is filed:

```
Q3 2022 filed: 2022-11-01  ───┐
                              ├─ Q3 data used
Q4 2022 filed: 2023-02-15  ───┤
                              ├─ Q4 data used
Q1 2023 filed: 2023-05-01  ───┘
```

Between filings (e.g., Feb 15 - Apr 30), the Q4 2022 data is the "latest" available.

### Verifying Point-in-Time

```python
# Check when data becomes available
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals

# Run pipeline on consecutive days around a filing
pipeline = Pipeline(
    columns={'revenue': SharadarFundamentals.revenue.latest}
)

# If you see revenue change on a specific date,
# that's likely when the new quarterly data was filed
```

---

## Example Strategies

### 1. Magic Formula (Greenblatt)

Classic value + quality strategy:

```python
from zipline.api import *
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals

def initialize(context):
    # Magic Formula: High ROC (earnings yield) + High ROIC
    earnings_yield = SharadarFundamentals.ebit.latest / SharadarFundamentals.ev.latest
    roic = SharadarFundamentals.roic.latest

    # Rank and combine
    ey_rank = earnings_yield.rank(ascending=False)
    roic_rank = roic.rank(ascending=False)
    magic_formula_rank = ey_rank + roic_rank

    pipeline = Pipeline(
        columns={
            'earnings_yield': earnings_yield,
            'roic': roic,
            'magic_rank': magic_formula_rank,
        },
        screen=(
            magic_formula_rank.top(30) &  # Top 30 stocks
            (SharadarFundamentals.marketcap.latest > 1e9)  # Large cap only
        )
    )

    attach_pipeline(pipeline, 'magic_formula')
    schedule_function(rebalance, date_rules.month_start())

def rebalance(context, data):
    stocks = pipeline_output('magic_formula').index

    # Equal weight portfolio
    for stock in stocks:
        order_target_percent(stock, 1.0 / len(stocks))

    # Close positions not in top 30
    for stock in context.portfolio.positions:
        if stock not in stocks:
            order_target_percent(stock, 0)
```

### 2. Piotroski F-Score

Quality screen using 9 fundamental signals:

```python
from zipline.pipeline.factors import CustomFactor
import numpy as np

class PiotroskiFScore(CustomFactor):
    """
    Piotroski F-Score: 9-point quality score.

    Profitability (4 points):
    - Positive net income
    - Positive operating cash flow
    - Increasing ROA
    - Quality of earnings (CFO > NI)

    Leverage (3 points):
    - Decreasing debt ratio
    - Increasing current ratio
    - No new shares issued

    Operating Efficiency (2 points):
    - Increasing gross margin
    - Increasing asset turnover
    """

    inputs = [
        SharadarFundamentals.netinc,
        SharadarFundamentals.ncfo,
        SharadarFundamentals.roa,
        SharadarFundamentals.de,
        SharadarFundamentals.currentratio,
        SharadarFundamentals.grossmargin,
        SharadarFundamentals.assetturnover,
    ]

    window_length = 2  # Current and previous quarter

    def compute(self, today, assets, out, ni, cfo, roa, de, cr, gm, at):
        # Point 1: Positive net income
        p1 = ni[-1] > 0

        # Point 2: Positive operating cash flow
        p2 = cfo[-1] > 0

        # Point 3: Increasing ROA
        p3 = roa[-1] > roa[0]

        # Point 4: Quality of earnings (CFO > NI)
        p4 = cfo[-1] > ni[-1]

        # Point 5: Decreasing leverage
        p5 = de[-1] < de[0]

        # Point 6: Increasing liquidity
        p6 = cr[-1] > cr[0]

        # Point 7: Increasing gross margin
        p7 = gm[-1] > gm[0]

        # Point 8: Increasing asset turnover
        p8 = at[-1] > at[0]

        # Sum up points (each is 0 or 1)
        out[:] = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8

# Use in strategy
pipeline = Pipeline(
    columns={'f_score': PiotroskiFScore()},
    screen=PiotroskiFScore() >= 7,  # Only stocks with 7-9 points
)
```

### 3. Dividend Aristocrats

High-quality dividend payers:

```python
def initialize(context):
    # Screen for dividend aristocrats
    screen = (
        (SharadarFundamentals.dps.latest > 2.0) &  # Dividend > $2/share
        (SharadarFundamentals.de.latest < 0.5) &   # Low debt
        (SharadarFundamentals.roe.latest > 15) &   # High ROE
        (SharadarFundamentals.currentratio.latest > 1.5)  # Healthy liquidity
    )

    pipeline = Pipeline(
        columns={
            'div_yield': SharadarFundamentals.dps.latest / EquityPricing.close.latest,
            'roe': SharadarFundamentals.roe.latest,
        },
        screen=screen
    )

    attach_pipeline(pipeline, 'dividend_aristocrats')
```

---

## Advanced Usage

### Custom Loader Configuration

If you need to customize the loader (e.g., for a specific bundle):

```python
from zipline.pipeline.loaders.sharadar_fundamentals import SharadarFundamentalsLoader

# Point to a specific bundle ingestion
loader = SharadarFundamentalsLoader(
    bundle_path='/root/.zipline/data/sharadar/2025-11-16T12;34;56.789012'
)

# Use in Pipeline
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals

pipeline = Pipeline(
    columns={'revenue': SharadarFundamentals.revenue.latest},
    loader_overrides={SharadarFundamentals: loader}
)
```

### Multi-Bundle Support

If you have multiple bundles with fundamentals:

```python
from zipline.pipeline.loaders.sharadar_fundamentals import make_sharadar_fundamentals_loader

# Load from different bundles
tech_loader = make_sharadar_fundamentals_loader('sharadar-tech')
sp500_loader = make_sharadar_fundamentals_loader('sharadar-sp500')

# Use in separate pipelines
tech_pipeline = Pipeline(
    columns={'revenue': SharadarFundamentals.revenue.latest},
    loader_overrides={SharadarFundamentals: tech_loader}
)

sp500_pipeline = Pipeline(
    columns={'revenue': SharadarFundamentals.revenue.latest},
    loader_overrides={SharadarFundamentals: sp500_loader}
)
```

---

## Troubleshooting

### "Fundamentals data not found"

**Error**:
```
FileNotFoundError: Fundamentals data not found at /root/.zipline/data/sharadar/.../fundamentals/sf1.h5
```

**Solution**:
Re-ingest the bundle with `include_fundamentals=True` (default):

```bash
docker exec zipline-reloaded-jupyter zipline ingest -b sharadar
```

Verify fundamentals were downloaded:
```bash
docker exec zipline-reloaded-jupyter find /root/.zipline/data/sharadar -name "sf1.h5"
```

### Missing Metrics (NaN Values)

**Issue**: Some stocks return NaN for certain metrics.

**Reasons**:
1. **Not yet filed**: Company hasn't filed earnings for the requested quarter
2. **Non-applicable**: Metric not relevant (e.g., R&D for non-tech companies)
3. **Data gap**: Sharadar doesn't have data for that stock/quarter

**Solution**:
```python
# Handle NaN values in your strategy
fundamentals = pipeline_output('fundamentals')
fundamentals = fundamentals.fillna(0)  # Or use median, drop, etc.
```

### Slow Pipeline Performance

**Issue**: Pipeline takes too long to run.

**Solutions**:

1. **Reduce date range**: Test with shorter periods first
```python
run_algorithm(start='2023-01-01', end='2023-03-31', ...)  # 3 months
```

2. **Limit universe**: Use fewer stocks
```python
screen = StaticAssets([symbol('AAPL'), symbol('MSFT'), ...])
```

3. **Cache pipeline results**: Run once, save to CSV
```python
pipeline_output('fundamentals').to_csv('cached_fundamentals.csv')
```

### Version Compatibility

**Pandas deprecation warnings**:

If you see warnings about `method='ffill'`, it's a Pandas 2.x deprecation. This is handled automatically in the loader.

### HDF5 Errors

**Error**: `tables.exceptions.HDF5ExtError`

**Solution**: Reinstall tables/pytables:
```bash
pip install --upgrade tables
```

---

## Performance Considerations

### Storage

- **Typical bundle**: 10-50 MB fundamentals data
- **Full universe (8,000+ stocks)**: 200-500 MB
- **Format**: HDF5 with level-9 compression

### Speed

- **First Pipeline run**: 1-5 seconds (loads HDF5)
- **Subsequent runs**: <1 second (uses cache)
- **Memory**: ~50-200 MB for fundamentals DataFrame

### Best Practices

1. **Use screens early**: Filter before loading fundamentals
```python
# Good: Screen first
screen = EquityPricing.close.latest > 10
pipeline = Pipeline(
    columns={'roe': SharadarFundamentals.roe.latest},
    screen=screen
)

# Avoid: Load all fundamentals then filter in Python
```

2. **Request only needed columns**: Don't load all 150+ metrics
```python
# Good: Only what you need
pipeline = Pipeline(columns={'roe': SharadarFundamentals.roe.latest})

# Avoid: Loading many unused columns
```

3. **Reuse loaders**: Create loader once, use multiple times
```python
# Good: Reuse loader
loader = make_sharadar_fundamentals_loader('sharadar')
pipeline1 = Pipeline(..., loader_overrides={SharadarFundamentals: loader})
pipeline2 = Pipeline(..., loader_overrides={SharadarFundamentals: loader})
```

---

## Next Steps

- **Phase 2**: LSEG fundamentals integration (coming soon)
- **Phase 3**: Multi-source fundamentals comparison and consensus
- **Advanced**: Custom fundamental factors and screening

## Related Documentation

- [MULTI_SOURCE_FUNDAMENTALS_PLAN.md](MULTI_SOURCE_FUNDAMENTALS_PLAN.md) - Implementation roadmap
- [SHARADAR_GUIDE.md](SHARADAR_GUIDE.md) - Sharadar bundle setup
- [BUNDLE_CLEANUP.md](BUNDLE_CLEANUP.md) - Managing bundle disk space

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section above
2. Review the implementation plan: `docs/MULTI_SOURCE_FUNDAMENTALS_PLAN.md`
3. Open an issue on GitHub with reproduction steps

---

**Version**: 1.0.0 (Phase 1 Complete)
**Last Updated**: 2025-11-16
**Author**: Kamran Sokhanvari, Hidden Point Capital
