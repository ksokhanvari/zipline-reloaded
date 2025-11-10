# Backtesting with Custom Fundamental Data

This directory contains a complete example of running Zipline backtests with custom fundamental data from a SQLite database combined with pricing data from the Sharadar bundle.

## Files

- **`backtest_with_fundamentals.py`** - Complete backtest script
- **`plot_backtest_results.py`** - Visualization script for results
- **`research_with_fundamentals.ipynb`** - Interactive tutorial notebook

## Prerequisites

1. **Custom fundamentals database created**:
   ```bash
   # Run the notebook cells in Part 1-3 to create the database
   jupyter notebook research_with_fundamentals.ipynb
   ```

2. **Sharadar bundle ingested**:
   ```bash
   zipline ingest -b sharadar
   ```

## Quick Start

### 1. Run the Backtest

```bash
cd examples/custom_data
python backtest_with_fundamentals.py
```

This will:
- Load fundamental data from your custom database
- Load pricing data from the Sharadar bundle
- Run a quality-factor strategy (monthly rebalancing)
- Save results to `backtest_results.csv`

### 2. Visualize Results

```bash
python plot_backtest_results.py
```

This creates `backtest_performance.png` with comprehensive charts.

## Strategy Overview

**Quality Factor Strategy**:
- **Universe**: Stocks passing fundamental + liquidity screens
- **Ranking**: Composite quality score (ROE, P/E, Debt/Equity)
- **Portfolio**: Top 10 stocks, equal-weighted
- **Rebalancing**: Monthly (first trading day)

**Screening Criteria**:

| Filter Type | Metric | Threshold | Source |
|-------------|--------|-----------|---------|
| Fundamental | ROE | > 5% | Custom DB |
| Fundamental | P/E Ratio | < 50 | Custom DB |
| Fundamental | Debt/Equity | < 5.0 | Custom DB |
| Liquidity | 20-day Avg Volume | > 100,000 | Bundle |
| Price | Close Price | > $1.00 | Bundle |

## Customization

Edit `backtest_with_fundamentals.py` to customize:

### Strategy Parameters

```python
TOP_N_STOCKS = 10  # Number of stocks to hold
REBALANCE_FREQUENCY = 'monthly'  # 'monthly' or 'weekly'
START_DATE = '2023-04-01'
END_DATE = '2024-01-31'
INITIAL_CAPITAL = 100000.0
```

### Screening Thresholds

```python
# In make_pipeline() function:
high_roe = (roe > 5.0)  # Change to 10.0 for stricter criteria
reasonable_pe = (pe_ratio < 50.0)  # Lower for value stocks
manageable_debt = (debt_to_equity < 5.0)  # Lower for conservative
```

### Factor Weights

```python
# In QualityScore.compute():
out[:] = (
    roe_score * 0.5 +      # 50% weight on ROE
    pe_score * 0.3 +       # 30% weight on P/E
    debt_score * 0.2       # 20% weight on Debt
)
```

## Pipeline Architecture

### Data Sources

```
┌─────────────────────────────────────────────────────┐
│                   PIPELINE                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────┐    ┌──────────────────┐     │
│  │  Custom Database │    │  Sharadar Bundle │     │
│  │  ───────────────  │    │  ──────────────  │     │
│  │  • ROE           │    │  • Close Price   │     │
│  │  • P/E Ratio     │    │  • Volume        │     │
│  │  • Debt/Equity   │    │  • OHLC Data     │     │
│  │  • EPS           │    │  • Dividends     │     │
│  │  • Sector        │    │  • Splits        │     │
│  └──────────────────┘    └──────────────────┘     │
│           │                       │                │
│           └───────┬───────────────┘                │
│                   ▼                                │
│         ┌──────────────────┐                       │
│         │  Combined Data   │                       │
│         │  ─────────────   │                       │
│         │  Quality Score   │                       │
│         │  + Liquidity     │                       │
│         │  + Price         │                       │
│         └──────────────────┘                       │
│                   │                                │
│                   ▼                                │
│         ┌──────────────────┐                       │
│         │   Top N Stocks   │                       │
│         └──────────────────┘                       │
└─────────────────────────────────────────────────────┘
```

### Pipeline Definition

```python
def make_pipeline():
    # FUNDAMENTALS (from custom database)
    roe = Fundamentals.ROE.latest
    pe_ratio = Fundamentals.PERatio.latest
    debt_to_equity = Fundamentals.DebtToEquity.latest
    quality_score = QualityScore()

    # PRICING (from sharadar bundle)
    close_price = EquityPricing.close.latest
    volume = EquityPricing.volume.latest
    avg_volume = volume.mavg(20)

    # SCREENING
    quality_universe = (
        (roe > 5.0) &
        (pe_ratio < 50.0) &
        (debt_to_equity < 5.0) &
        (avg_volume > 100000) &
        (close_price > 1.0)
    )

    return Pipeline(
        columns={
            'quality_score': quality_score,
            'roe': roe,
            'pe_ratio': pe_ratio,
            'close': close_price,
            'volume': avg_volume,
        },
        screen=quality_universe
    )
```

## Algorithm Flow

```
START BACKTEST
    │
    ▼
INITIALIZE
    │
    ├─ Attach Pipeline
    ├─ Schedule Rebalancing (monthly/weekly)
    └─ Set Parameters (top_n, etc.)
    │
    ▼
┌───────────────────────────┐
│   TRADING LOOP (daily)    │
├───────────────────────────┤
│                           │
│  BEFORE_TRADING_START     │
│     │                     │
│     ├─ Run Pipeline       │
│     ├─ Get Universe       │
│     └─ Calculate Metrics  │
│     │                     │
│  ┌──▼───────────────┐     │
│  │ Rebalance Day?   │     │
│  └──┬───────────────┘     │
│     │                     │
│     │ YES (monthly)       │
│     ▼                     │
│  REBALANCE                │
│     │                     │
│     ├─ Select Top N       │
│     ├─ Sell Exits         │
│     ├─ Buy Entries        │
│     └─ Adjust Weights     │
│                           │
└───────────────────────────┘
         │
         │ (repeat daily)
         ▼
    ANALYZE
         │
         ├─ Calculate Metrics
         ├─ Print Summary
         └─ Save Results
         │
         ▼
    END BACKTEST
```

## Output Files

### backtest_results.csv

Detailed daily performance data:

| Column | Description |
|--------|-------------|
| `portfolio_value` | Total portfolio value |
| `returns` | Daily returns |
| `positions` | JSON of positions |
| `transactions` | Trade details |
| `orders` | Order details |
| `pnl` | Profit & loss |

### backtest_performance.png

Visualization with 5 charts:
1. Portfolio Value Over Time
2. Cumulative Returns
3. Drawdown
4. Daily Returns Distribution
5. Monthly Returns Heatmap

## Performance Metrics

The backtest calculates:

- **Total Return**: Overall percentage gain/loss
- **Annualized Return**: Return extrapolated to 1 year
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns
- **Win Rate**: Percentage of profitable days

## Troubleshooting

### "No module named 'zipline'"
```bash
pip install -e .
```

### "Bundle 'sharadar' not found"
```bash
zipline ingest -b sharadar
```

### "Database 'fundamentals' not found"
Run the notebook cells in Part 1-3 to create the database.

### "No data in universe"
Your screening criteria might be too strict. Try:
- Lowering ROE threshold
- Increasing P/E threshold
- Increasing Debt/Equity threshold
- Using different dates with more data

### "ImportError: cannot import CustomFactor"
Make sure you're using the latest zipline-reloaded:
```bash
git pull
pip install -e .
```

## Advanced Usage

### Multiple Strategies

Compare different strategies by modifying the pipeline:

```python
# Value strategy
value_pipeline = make_pipeline(
    roe_threshold=10.0,
    pe_threshold=15.0,  # Lower P/E for value
    top_n=20
)

# Growth strategy
growth_pipeline = make_pipeline(
    roe_threshold=15.0,  # Higher ROE for growth
    pe_threshold=100.0,
    top_n=10
)
```

### Adding More Factors

```python
# Add momentum
from zipline.pipeline.factors import Returns

def make_enhanced_pipeline():
    # ... existing code ...

    # Add momentum
    returns_1m = Returns(window_length=21)  # 1-month returns
    returns_3m = Returns(window_length=63)  # 3-month returns

    # Combine with quality
    combined_score = (quality_score + returns_1m.zscore()) / 2

    return Pipeline(columns={'combined_score': combined_score, ...})
```

### Walk-Forward Analysis

Test strategy robustness with rolling windows:

```python
periods = [
    ('2023-01-01', '2023-06-30'),  # Train
    ('2023-07-01', '2023-12-31'),  # Test
    ('2023-04-01', '2023-09-30'),  # Train
    ('2023-10-01', '2024-03-31'),  # Test
]

for train_start, test_end in periods:
    results = run_algorithm(start=train_start, end=test_end, ...)
```

## Next Steps

1. **Optimize Parameters**: Use walk-forward analysis to find optimal thresholds
2. **Add Risk Management**: Implement stop-losses or position sizing
3. **Combine Factors**: Add momentum, value, or other factors
4. **Sector Neutrality**: Equal-weight across sectors
5. **Transaction Costs**: Add slippage and commission models

## Resources

- **Zipline Documentation**: https://zipline.ml4trading.io
- **Pipeline Tutorial**: https://zipline.ml4trading.io/pipeline.html
- **Custom Data Guide**: See METHODOLOGY.md in this directory
- **Community Forum**: https://exchange.ml4trading.io

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the METHODOLOGY.md file
3. Open an issue on GitHub
4. Ask on the ML4Trading forum
