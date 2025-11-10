# Database Class Approach for Custom Data

## Overview

The **Database class pattern** is the recommended way to integrate custom data with Zipline Pipeline. It's cleaner, more Pythonic, and more powerful than the `make_custom_dataset_class()` function approach.

## Comparison

### ❌ Old Approach (Functional)

```python
from zipline.data.custom import make_custom_dataset_class

# Create dataset from schema dictionary
Fundamentals = make_custom_dataset_class(
    db_code='fundamentals',
    columns={
        'ROE': 'float',
        'PERatio': 'float',
        'DebtToEquity': 'float',
        'Sector': 'text',
    },
    base_name='Fundamentals'
)

# Usage
roe = Fundamentals.ROE.latest
```

###  ✅ New Approach (Class-Based)

```python
from zipline.pipeline.data.db import Database, Column

# Define database as a class
class Fundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240

    ROE = Column(float)
    PERatio = Column(float)
    DebtToEquity = Column(float)
    Sector = Column(str)

# Usage (same!)
roe = Fundamentals.ROE.latest
```

## Why Database Class is Better

### 1. **Type Safety & IDE Support**

```python
class Fundamentals(Database):
    ROE = Column(float)  # IDE knows this is a Column
    Sector = Column(str)  # Auto-complete works!
```

- Better IDE autocomplete
- Type hints work correctly
- Easier to spot typos

### 2. **Self-Documenting**

```python
class Fundamentals(Database):
    """Custom fundamental data from XYZ source."""

    CODE = "fundamentals"  # Clear configuration
    LOOKBACK_WINDOW = 240  # Explicit lookback

    # Clearly organized columns
    Revenue = Column(float)
    NetIncome = Column(float)
    ROE = Column(float)
```

### 3. **Easier to Extend**

```python
class Fundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240

    ROE = Column(float)
    PERatio = Column(float)

    # Add custom properties
    @property
    def quality_columns(self):
        return [self.ROE, self.PERatio, self.DebtToEquity]

    # Add class methods
    @classmethod
    def get_value_metrics(cls):
        return [cls.PERatio, cls.PriceToBook, cls.DividendYield]
```

### 4. **Multiple Databases**

Easy to define multiple data sources:

```python
class Fundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240

    ROE = Column(float)
    EPS = Column(float)


class SentimentData(Database):
    CODE = "sentiment"
    LOOKBACK_WINDOW = 60

    BullishPercent = Column(float)
    SentimentScore = Column(float)


class AlternativeData(Database):
    CODE = "alternative"
    LOOKBACK_WINDOW = 120

    WebTraffic = Column(float)
    AppDownloads = Column(float)


# Use all together in pipeline:
pipe = Pipeline(
    columns={
        'roe': Fundamentals.ROE.latest,
        'sentiment': SentimentData.SentimentScore.latest,
        'traffic': AlternativeData.WebTraffic.latest,
    }
)
```

## Complete Example

```python
from zipline.pipeline.data.db import Database, Column
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import EquityPricing
from zipline.pipeline.factors import Returns
import numpy as np


# ============================================================================
# DEFINE DATABASE
# ============================================================================

class Fundamentals(Database):
    """
    Quarterly fundamental metrics from custom database.

    Data should be stored in SQLite using the custom data loader.
    """

    CODE = "fundamentals"  # Must match database code
    LOOKBACK_WINDOW = 240  # 240 trading days ≈ 1 year

    # Income Statement
    Revenue = Column(float)
    NetIncome = Column(float)
    OperatingIncome = Column(float)

    # Balance Sheet
    TotalAssets = Column(float)
    TotalEquity = Column(float)
    TotalDebt = Column(float)

    # Cash Flow
    FreeCashFlow = Column(float)
    OperatingCashFlow = Column(float)

    # Per-Share Metrics
    EPS = Column(float)
    BookValuePerShare = Column(float)
    CashPerShare = Column(float)

    # Ratios
    ROE = Column(float)  # Return on Equity
    ROA = Column(float)  # Return on Assets
    ROIC = Column(float)  # Return on Invested Capital
    DebtToEquity = Column(float)
    CurrentRatio = Column(float)
    QuickRatio = Column(float)

    # Valuation
    PERatio = Column(float)
    PBRatio = Column(float)  # Price-to-Book
    PSRatio = Column(float)  # Price-to-Sales
    EVToEBITDA = Column(float)
    DividendYield = Column(float)

    # Growth Metrics
    RevenueGrowthYoY = Column(float)
    EarningsGrowthYoY = Column(float)
    EPSGrowth5Y = Column(float)

    # Metadata
    Sector = Column(str)
    Industry = Column(str)


# ============================================================================
# DEFINE CUSTOM FACTORS
# ============================================================================

class QualityScore(CustomFactor):
    """
    Composite quality score based on profitability and financial health.

    Combines:
    - ROE (profitability)
    - ROIC (capital efficiency)
    - Current Ratio (liquidity)
    - Debt/Equity (leverage)
    """

    inputs = [
        Fundamentals.ROE,
        Fundamentals.ROIC,
        Fundamentals.CurrentRatio,
        Fundamentals.DebtToEquity,
    ]
    window_length = 1

    def compute(self, today, assets, out, roe, roic, current, debt):
        # Get latest values
        roe_latest = roe[0]
        roic_latest = roic[0]
        current_latest = current[0]
        debt_latest = debt[0]

        # Normalize each metric (0-1 scale)
        def normalize(arr):
            min_val, max_val = np.nanmin(arr), np.nanmax(arr)
            if max_val > min_val:
                return (arr - min_val) / (max_val - min_val)
            return np.full_like(arr, 0.5)

        # Score each dimension (higher is better)
        profitability_score = (normalize(roe_latest) + normalize(roic_latest)) / 2
        financial_health_score = (normalize(current_latest) + normalize(1/debt_latest)) / 2

        # Composite score
        out[:] = (profitability_score + financial_health_score) / 2


class ValueScore(CustomFactor):
    """
    Value score based on multiple valuation ratios.

    Lower ratios = better value, so we invert the normalized scores.
    """

    inputs = [
        Fundamentals.PERatio,
        Fundamentals.PBRatio,
        Fundamentals.PSRatio,
        Fundamentals.EVToEBITDA,
    ]
    window_length = 1

    def compute(self, today, assets, out, pe, pb, ps, ev_ebitda):
        def normalize_inverted(arr):
            """Normalize and invert (lower values get higher scores)"""
            min_val, max_val = np.nanmin(arr), np.nanmax(arr)
            if max_val > min_val:
                normalized = (arr - min_val) / (max_val - min_val)
                return 1 - normalized  # Invert
            return np.full_like(arr, 0.5)

        # Score each valuation metric
        pe_score = normalize_inverted(pe[0])
        pb_score = normalize_inverted(pb[0])
        ps_score = normalize_inverted(ps[0])
        ev_score = normalize_inverted(ev_ebitda[0])

        # Average value score
        out[:] = (pe_score + pb_score + ps_score + ev_score) / 4


# ============================================================================
# CREATE PIPELINE
# ============================================================================

def make_fundamental_pipeline():
    """
    Pipeline combining fundamental data with pricing data.
    """

    # FUNDAMENTAL DATA
    roe = Fundamentals.ROE.latest
    roic = Fundamentals.ROIC.latest
    pe_ratio = Fundamentals.PERatio.latest
    debt_to_equity = Fundamentals.DebtToEquity.latest
    current_ratio = Fundamentals.CurrentRatio.latest
    sector = Fundamentals.Sector.latest
    revenue_growth = Fundamentals.RevenueGrowthYoY.latest

    # CUSTOM FACTORS
    quality_score = QualityScore()
    value_score = ValueScore()

    # PRICING DATA
    close = EquityPricing.close.latest
    volume = EquityPricing.volume.latest
    returns_60d = Returns(window_length=60)

    # COMBINED SCORE
    combined_score = (quality_score + value_score) / 2

    # SCREENING
    # Fundamental filters
    profitable = (roe > 5.0)
    reasonable_valuation = (pe_ratio < 50.0)
    financial_stability = (debt_to_equity < 3.0) & (current_ratio > 1.0)

    # Liquidity filters
    avg_dollar_volume = (close * volume).mavg(20)
    liquid = (avg_dollar_volume > 1e6)  # $1M daily volume

    # Combined universe
    universe = profitable & reasonable_valuation & financial_stability & liquid

    # RANKING
    combined_rank = combined_score.rank(mask=universe, ascending=False)

    return Pipeline(
        columns={
            # Scores
            'quality_score': quality_score,
            'value_score': value_score,
            'combined_score': combined_score,
            'combined_rank': combined_rank,

            # Fundamentals
            'roe': roe,
            'roic': roic,
            'pe_ratio': pe_ratio,
            'debt_to_equity': debt_to_equity,
            'revenue_growth': revenue_growth,
            'sector': sector,

            # Pricing
            'close': close,
            'returns_60d': returns_60d,
            'avg_dollar_volume': avg_dollar_volume,
        },
        screen=universe
    )


# ============================================================================
# USAGE IN BACKTEST
# ============================================================================

"""
def initialize(context):
    # Attach pipeline
    attach_pipeline(make_fundamental_pipeline(), 'fundamentals')

    # Schedule monthly rebalancing
    schedule_function(
        rebalance,
        date_rules.month_start(),
        time_rules.market_open(hours=1)
    )

    context.top_n = 20


def before_trading_start(context, data):
    # Get pipeline output
    context.output = pipeline_output('fundamentals')

    # Log stats
    print(f"Universe: {len(context.output)} stocks")
    print(f"Avg Quality Score: {context.output['quality_score'].mean():.3f}")
    print(f"Avg ROE: {context.output['roe'].mean():.1f}%")


def rebalance(context, data):
    # Select top N by combined score
    top_stocks = context.output.nsmallest(context.top_n, 'combined_rank')

    # Equal weight
    target_weight = 1.0 / len(top_stocks)

    # Rebalance
    for asset in context.portfolio.positions:
        if asset not in top_stocks.index:
            order_target_percent(asset, 0)

    for asset in top_stocks.index:
        order_target_percent(asset, target_weight)
"""
```

## All Pipeline Operations Work

With the Database class, you get full Pipeline functionality:

```python
# Latest value
roe = Fundamentals.ROE.latest

# Shifted (N days ago)
roe_yesterday = Fundamentals.ROE.latest.shift()
roe_5days_ago = Fundamentals.ROE.latest.shift(5)

# Ranking
roe_rank = Fundamentals.ROE.latest.rank()
roe_rank_desc = Fundamentals.ROE.latest.rank(ascending=False)

# Z-score normalization
roe_zscore = Fundamentals.ROE.latest.zscore()

# Winsorization (clip outliers)
roe_winsorized = Fundamentals.ROE.latest.winsorize(min_percentile=0.05, max_percentile=0.95)

# Demean by group
roe_sector_neutral = Fundamentals.ROE.latest.demean(groupby=Fundamentals.Sector.latest)

# Top/bottom N
top_roe_filter = Fundamentals.ROE.latest.top(100)
bottom_pe_filter = Fundamentals.PERatio.latest.bottom(50)

# Percentile filters
mid_cap_filter = Fundamentals.MarketCap.latest.percentile_between(25, 75)

# Boolean operations
high_quality = (Fundamentals.ROE.latest > 15.0) & (Fundamentals.PERatio.latest < 30.0)

# Null checks
has_roe = Fundamentals.ROE.latest.notnull()
```

## Integration with CustomFactors

Use Database columns directly as CustomFactor inputs:

```python
class MyCashReturn(CustomFactor):
    """Free cash flow to enterprise value ratio"""

    inputs = [
        Fundamentals.FreeCashFlow,
        Fundamentals.TotalDebt,
        EquityPricing.close,
        Fundamentals.SharesOutstanding,
    ]
    window_length = 1

    def compute(self, today, assets, out, fcf, debt, price, shares):
        # Enterprise value = market cap + debt
        market_cap = price[0] * shares[0]
        enterprise_value = market_cap + debt[0]

        # Cash return ratio
        out[:] = fcf[0] / enterprise_value


# Use in pipeline
cash_return = MyCashReturn()
```

## Migration Checklist

Migrating from `make_custom_dataset_class()` to Database class:

- [ ] Import `Database` and `Column`: `from zipline.pipeline.data.db import Database, Column`
- [ ] Create class inheriting from `Database`
- [ ] Set `CODE` attribute (your database code)
- [ ] Set `LOOKBACK_WINDOW` attribute (days to look back)
- [ ] Convert column dictionary to Column attributes
- [ ] Update type mapping: `'int'` → `float`, `'text'` → `str`
- [ ] Test pipeline still works
- [ ] Remove old `make_custom_dataset_class()` call
- [ ] Update documentation

## Type Mapping

| Old (dict) | New (Column) |
|------------|--------------|
| `'int'` | `Column(float)` or `Column(int)` |
| `'float'` | `Column(float)` |
| `'text'` | `Column(str)` |
| `'date'` | `Column(pd.Timestamp)` |
| `'datetime'` | `Column(pd.Timestamp)` |
| `'bool'` | `Column(bool)` |

**Note**: Use `float` for numeric data even if stored as int in database.

## Best Practices

### 1. Organize Columns by Category

```python
class Fundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240

    # === INCOME STATEMENT ===
    Revenue = Column(float)
    NetIncome = Column(float)
    OperatingIncome = Column(float)

    # === BALANCE SHEET ===
    TotalAssets = Column(float)
    TotalEquity = Column(float)
    TotalDebt = Column(float)

    # === RATIOS ===
    ROE = Column(float)
    ROA = Column(float)
    CurrentRatio = Column(float)

    # === METADATA ===
    Sector = Column(str)
    Industry = Column(str)
```

### 2. Use Docstrings

```python
class Fundamentals(Database):
    """
    Quarterly fundamental data from SEC filings.

    Data is updated quarterly, approximately 45 days after quarter end.
    All per-share metrics are adjusted for splits.
    """

    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240

    ROE = Column(float)  # Return on Equity (%)
    PERatio = Column(float)  # Price-to-Earnings ratio
```

### 3. Set Appropriate LOOKBACK_WINDOW

```python
# For daily data
class DailySentiment(Database):
    CODE = "sentiment"
    LOOKBACK_WINDOW = 60  # 60 days

# For quarterly data
class QuarterlyFundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240  # ~1 year of trading days

# For high-frequency data
class MinuteData(Database):
    CODE = "minute"
    LOOKBACK_WINDOW = 5  # Just a few days
```

### 4. Multiple Databases for Different Sources

```python
class Fundamentals(Database):
    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240
    # ... columns ...


class EarningsEstimates(Database):
    CODE = "earnings-estimates"
    LOOKBACK_WINDOW = 120
    # ... columns ...


class SocialSentiment(Database):
    CODE = "social-sentiment"
    LOOKBACK_WINDOW = 30
    # ... columns ...


# Use all in one pipeline
pipe = Pipeline(
    columns={
        'roe': Fundamentals.ROE.latest,
        'eps_estimate': EarningsEstimates.ConsensusEPS.latest,
        'sentiment': SocialSentiment.BullishPercent.latest,
    }
)
```

## Conclusion

The **Database class approach** is:
- ✅ More Pythonic
- ✅ Better type safety
- ✅ Easier to maintain
- ✅ Self-documenting
- ✅ Fully compatible with all Pipeline operations

**Recommendation**: Use Database class for all new projects and migrate existing code when convenient.
