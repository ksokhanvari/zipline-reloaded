# Walk-Forward Window Strategy Testing Framework

## Overview

This testing framework compares different walk-forward training methodologies **without modifying your production forecasting scripts**. It's a pure analysis tool to help you make data-driven decisions.

## What It Tests

### Window Strategies

1. **Expanding Window** (Current implementation)
   - Uses ALL historical data up to prediction date
   - Example: To predict 2015-08, trains on 2009-2015-07

2. **Rolling Window** (Various sizes)
   - Uses only last N months of data
   - Default tests: 6, 12, 24, 36 months
   - Example: To predict 2015-08 with 12M window, trains on 2014-08 to 2015-07

### Metrics Calculated

For each month and each strategy:

1. **Correlation**: Linear relationship between predicted and actual returns
2. **IC (Information Coefficient)**: Rank correlation (industry standard)
3. **RMSE**: Root mean squared error
4. **MAE**: Mean absolute error
5. **Hit Rate**: % of correct direction predictions
6. **Top Quintile Return**: Average return of stocks predicted to perform best
7. **Bottom Quintile Return**: Average return of stocks predicted to perform worst
8. **Long/Short Spread**: Top quintile - Bottom quintile (profitability measure)

## Quick Start

### Basic Test (All Defaults)

```bash
python test_window_strategies.py your_data.csv
```

This tests:
- Expanding window
- Rolling 6 months
- Rolling 12 months
- Rolling 24 months
- Rolling 36 months

### Fast Test (For Quick Iteration)

```bash
# Process every 3 months instead of every month (3x faster)
python test_window_strategies.py your_data.csv --sample-every 3
```

### Custom Window Sizes

```bash
# Test only 12 and 24 month windows
python test_window_strategies.py your_data.csv --windows 12 24
```

### Custom Forecast Parameters

```bash
# Match your production settings
python test_window_strategies.py your_data.csv --forecast-days 10 --target-return-days 90
```

## Expected Runtime

For 40K rows, 16 years of data:

| Configuration | Runtime |
|--------------|---------|
| Full test (all months) | ~30-60 minutes |
| Sample every 2 months | ~15-30 minutes |
| Sample every 3 months | ~10-20 minutes |

**Recommendation**: Start with `--sample-every 3` for quick results, then run full test overnight for final decision.

## Output Files

### 1. `*_window_comparison.csv`
Detailed monthly metrics for every strategy:

```csv
month,n_train,n_predict,correlation,ic,rmse,mae,hit_rate,top_quintile_return,bottom_quintile_return,long_short_spread,strategy
2010-01,1585,29739,0.0234,0.0189,8.45,6.23,0.512,2.34,-1.23,3.57,Expanding
2010-01,0,0,,,,,,,,,Rolling_6M
...
```

### 2. `*_window_comparison_summary.txt`
Human-readable summary with:
- Overall statistics for each strategy
- Comparison table ranked by performance
- Interpretation guide

Example:
```
Expanding:
  Months evaluated: 185
  Correlation:     0.0876 ± 0.0234
  IC (Spearman):   0.0712 ± 0.0198
  Hit Rate:        54.23%
  Long/Short Spread: 3.45%

Rolling_12M:
  Months evaluated: 173
  Correlation:     0.0654 ± 0.0312
  IC (Spearman):   0.0598 ± 0.0287
  Hit Rate:        52.89%
  Long/Short Spread: 2.87%
```

### 3. `*_window_comparison_plots.png` (if matplotlib available)
Visual comparison with 4 plots:
1. Correlation over time
2. IC over time
3. Long/Short spread over time
4. Average performance comparison bar chart

## Interpreting Results

### Which Strategy Is Better?

Look at these key metrics in order:

1. **IC (Information Coefficient)** - Most important for institutional quant
   - Higher average IC = better
   - Lower IC std dev = more stable
   - **Target**: IC > 0.05 is excellent, > 0.03 is good

2. **Correlation** - Secondary metric
   - Higher correlation = stronger predictive power
   - **Target**: > 0.10 is strong, > 0.05 is moderate

3. **Long/Short Spread** - Profitability measure
   - Larger spread = better stock selection
   - Directly translates to strategy returns
   - **Target**: > 3% monthly spread is excellent

4. **Hit Rate** - Directional accuracy
   - > 55% is strong
   - < 50% is worse than random
   - **Target**: > 52% is good

### Common Patterns

**Expanding wins on average**:
```
Expanding:     IC = 0.0876 ± 0.0234  ✅ Higher, more stable
Rolling_12M:   IC = 0.0654 ± 0.0312  ❌ Lower, more volatile
```
→ Use expanding window

**Rolling adapts better in some regimes**:
- Check correlation plot over time
- If rolling outperforms during 2020 COVID, 2022 inflation → consider hybrid

**Rolling performs similarly with less compute**:
```
Expanding:   IC = 0.0712, Time = 45 min
Rolling_24M: IC = 0.0698, Time = 12 min  ✅ 95% performance, 4x faster
```
→ Consider rolling for production speed

## Advanced Analysis

### Regime Analysis

Look at the CSV file and filter by time periods:

```python
import pandas as pd

df = pd.read_csv('your_data_window_comparison.csv')

# Compare during COVID crash
covid = df[(df['month'] >= '2020-02') & (df['month'] <= '2020-04')]
print(covid.groupby('strategy')['correlation'].mean())

# Compare during bull market
bull = df[(df['month'] >= '2017-01') & (df['month'] <= '2019-12')]
print(bull.groupby('strategy')['correlation'].mean())
```

### Optimal Window Size

If rolling windows perform well, find the sweet spot:

```bash
# Test many window sizes
python test_window_strategies.py data.csv --windows 6 9 12 18 24 30 36
```

Plot IC vs window size to find optimal lookback.

### Sector-Specific Analysis

If you have sector data:

1. Run test on full dataset
2. Filter CSV by sector (using original data)
3. Compare: Does tech benefit from shorter windows? Financials from longer?

## What To Do With Results

### Scenario 1: Expanding Clearly Wins
**Action**: Keep using `forecast_returns_ml_walk_forward.py` with default expanding window

### Scenario 2: Rolling 12M or 24M Wins
**Action**: Modify walk-forward script to add `--rolling-window 12` option

### Scenario 3: Close Call
**Action**:
- Use expanding for development (more stable)
- Consider ensemble: average predictions from both
- Or use expanding but with sample weighting (recent data weighted more)

### Scenario 4: Rolling Wins During Regime Changes
**Action**: Implement adaptive strategy:
- Use rolling during high volatility (VIX > 25)
- Use expanding during normal markets
- Or implement regime detection

## Troubleshooting

### "Not enough training data"
- Increase `--sample-every` or
- Use longer rolling windows (36M instead of 6M)

### "Takes too long"
- Use `--sample-every 3` for 3x speedup
- Test on subset of data first

### "All strategies show low correlation"
- This is normal! IC > 0.03 is actually good
- Focus on relative performance, not absolute
- Check if your fundamentals data is quality

### "Plots not generated"
Install matplotlib:
```bash
pip install matplotlib
```

## Example Workflow

```bash
# 1. Quick test to see if rolling is worth exploring
python test_window_strategies.py data.csv --sample-every 3 --windows 12 24

# 2. If rolling looks promising, full test
python test_window_strategies.py data.csv --windows 6 12 18 24 36

# 3. Analyze results
cat data_window_comparison_summary.txt

# 4. Make decision and implement in production script
```

## Technical Notes

### Why Not Change Production Scripts?

- Test first, implement later
- Multiple experiments without breaking working code
- Easy rollback if tests show expanding is better

### Feature Engineering

This tester uses a **simplified feature set** (momentum, volatility, volume) to run fast. Your production script has more features, so:

- Absolute performance may differ
- **Relative** ranking between strategies should hold

### Statistical Significance

With ~180 months of data, differences of:
- IC > 0.01 are likely significant
- Correlation > 0.015 are likely significant

But visual inspection of time series is more informative than t-tests here.

## Next Steps After Testing

Based on your results:

1. **Expanding wins**: Document findings, keep current implementation
2. **Rolling wins**: Add `--rolling-window N` parameter to production script
3. **Mixed results**: Implement ensemble or adaptive strategy
4. **Need more data**: Test on different time periods, sectors, market caps

---

**Created**: 2024-12-16
**For questions**: Check ML_FORECASTING_VERSIONS.md for production script comparison
