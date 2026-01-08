# ML Return Forecasting - Script Versions

This directory contains two versions of the ML return forecasting script. Both produce predictions, but they differ in how they handle look-ahead bias.

## Quick Comparison

| Feature | forecast_returns_ml.py | forecast_returns_ml_walk_forward.py |
|---------|------------------------|-------------------------------------|
| **Training approach** | Single model on all data | Monthly expanding window |
| **Look-ahead bias** | ⚠️ YES (predictions use future data) | ✅ NO (uses only past data) |
| **Speed** | ⚡ FAST (~5 seconds) | 🐢 SLOWER (~5-20 minutes) |
| **Use case** | Quick analysis, experimentation | Backtesting, realistic validation |
| **Recommended for** | Fast iteration | Production strategies |

---

## forecast_returns_ml.py (Original Version)

### How it works
1. Trains ONE model on ALL available data (2009-2025)
2. Uses that model to predict for all dates

### The Problem
For predictions in 2010, the model was trained on data from 2011-2025. This means:
- Historical predictions "know the future"
- Overly optimistic backtest results
- Won't match real trading performance

### When to use
- ✅ Quick exploratory analysis
- ✅ Feature engineering experiments
- ✅ Fast iteration during development
- ❌ NOT for backtesting strategies
- ❌ NOT for validating trading performance

### Usage
```bash
# Fast prediction (5 seconds for 40K rows)
python forecast_returns_ml.py data.csv

# Custom parameters
python forecast_returns_ml.py data.csv --forecast-days 10 --target-return-days 90
```

---

## forecast_returns_ml_walk_forward.py (Walk-Forward Version)

### How it works
1. Groups data by month
2. For each month:
   - Trains model on all data BEFORE that month
   - Predicts for that month only
   - Moves to next month
3. Expanding window: Model sees more data over time

### Example
```
2010-01: Train on [2009 data] → Predict Jan 2010
2010-02: Train on [2009 + Jan 2010] → Predict Feb 2010
2015-08: Train on [2009-2015 Jul] → Predict Aug 2015
2025-12: Train on [2009-2025 Nov] → Predict Dec 2025
```

### Benefits
- ✅ NO look-ahead bias
- ✅ Each prediction uses only data available at that time
- ✅ Realistic backtest results
- ✅ Matches what you could achieve in live trading
- ✅ Valid for strategy validation

### When to use
- ✅ Backtesting trading strategies
- ✅ Validating model performance
- ✅ Estimating realistic returns
- ✅ Final production predictions

### Usage
```bash
# Walk-forward prediction (default, takes longer)
python forecast_returns_ml_walk_forward.py data.csv

# Custom parameters (same as original)
python forecast_returns_ml_walk_forward.py data.csv --forecast-days 10 --target-return-days 90

# Disable walk-forward for quick test (has look-ahead bias)
python forecast_returns_ml_walk_forward.py data.csv --no-walk-forward

# Skip feature importance calculation (saves 1-3 minutes)
python forecast_returns_ml_walk_forward.py data.csv --skip-feature-importance
```

### Performance
- **Training time**: ~5-20 minutes for 16 years of data (including feature importance)
- **Feature importance**: Adds 1-3 minutes (skip with --skip-feature-importance)
- **Models trained**: ~192 (one per month)
- **Memory**: Same as single model
- **Output**: Identical format to original version

---

## Which Should You Use?

### For Development & Experimentation
Use **forecast_returns_ml.py**:
- Fast iterations
- Quick feature testing
- Exploratory analysis

### For Backtesting & Production
Use **forecast_returns_ml_walk_forward.py**:
- Realistic performance estimates
- Valid strategy backtesting
- No look-ahead bias
- Honest correlation metrics

---

## Output Files

Both scripts generate the same output format:

### With default filename
```bash
# Original version
forecast_returns_ml.py data.csv
→ data_predictions_NO_LOOKAHEAD.csv  # Misleading name!

# Walk-forward version
forecast_returns_ml_walk_forward.py data.csv
→ data_predictions_WALK_FORWARD.csv  # Accurate!
```

### Output columns
Both add 2 columns to your original data:
- `forward_return` - Actual returns (for validation)
- `predicted_return` - ML forecasted returns

---

## Technical Details

### Look-Ahead Bias Explained

**Original version:**
```python
# Trains ONE model
model.train(all_data_2009_to_2025)

# Predicts for all dates
predictions[2010] = model.predict(data_2010)  # ❌ Used 2011-2025 data!
predictions[2015] = model.predict(data_2015)  # ❌ Used 2016-2025 data!
predictions[2025] = model.predict(data_2025)  # ✅ OK for recent dates
```

**Walk-forward version:**
```python
# Trains multiple models
for each month:
    model.train(data_before_this_month)
    predictions[this_month] = model.predict(this_month_data)

# Example:
predictions[2010-01] = model_trained_on_2009.predict(2010_01_data)  # ✅ Valid!
predictions[2015-08] = model_trained_on_2009_2015_07.predict(2015_08_data)  # ✅ Valid!
predictions[2025-12] = model_trained_on_2009_2025_11.predict(2025_12_data)  # ✅ Valid!
```

### Feature Engineering

Both versions use **identical features**:
- **~76 engineered features** per row
- **1-day lag** on ALL fundamentals (prevents look-ahead bias)
- **Rolling windows**: 5, 10, 20 days for momentum and volatility
- **Market cap weighting**: Focused training on large-cap stocks
- **Histogram-based Gradient Boosting**: Fast and accurate

**The only difference is WHEN the model is trained, not WHAT features it uses.**

For complete details on feature engineering (what columns are used, how they're transformed, etc.), see the **"Feature Engineering Pipeline (Detailed)"** section in [README.md](README.md#-feature-engineering-pipeline-detailed).

---

## FAQ

### Q: Why keep the original version if it has look-ahead bias?
**A:** It's useful for fast development and testing. Just don't use it for final backtesting.

### Q: Is the walk-forward version slower?
**A:** Yes, 5-20 minutes vs 5 seconds. But it's a one-time cost for realistic results.

### Q: Can I use walk-forward for live trading?
**A:** Yes! The most recent predictions (last month) use all available historical data, perfect for trading.

### Q: Will the predictions be different?
**A:** Yes, significantly. Walk-forward predictions are typically LESS optimistic but MORE realistic.

### Q: Can I make walk-forward faster?
**A:** Use `--no-walk-forward` flag for testing, but remember it has look-ahead bias.

---

## Recommendation

**For production use, ALWAYS use forecast_returns_ml_walk_forward.py**

The speed difference is negligible compared to the importance of valid backtest results. Walk-forward is the industry standard for realistic machine learning backtesting.

---

**Created:** 2024-12-16
**Version:** 1.0
