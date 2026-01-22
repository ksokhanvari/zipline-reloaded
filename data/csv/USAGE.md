# ML Return Forecasting - Usage Guide

Quick reference for running `forecast_returns_ml_walk_forward.py` with all available options.

## 📋 Table of Contents

- [Basic Usage](#basic-usage)
- [All Command-Line Flags](#all-command-line-flags)
- [Common Use Cases](#common-use-cases)
- [Input/Output](#inputoutput)
- [Model Parameters](#model-parameters)
- [Training Options](#training-options)
- [Performance Tuning](#performance-tuning)
- [Diagnostics & Debugging](#diagnostics--debugging)

---

## Basic Usage

### Minimal Command (Default Settings)
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions.parquet
```

**What this does:**
- Trains walk-forward models (one per month)
- Predicts 10-day forward, 10-day return period
- Uses all default parameters (max_depth=6, min_samples_leaf=100, L2=0.2)
- Auto-lags ALL fundamental columns by T-1
- Outputs Parquet file with predictions

---

## All Command-Line Flags

### Required Arguments

| Flag | Type | Description |
|------|------|-------------|
| `--input-file PATH` | str | Input CSV file with fundamentals |
| `--output PATH` | str | Output file path (.csv or .parquet) |

**Note**: Before v3.2.0, these were positional arguments. Now they're explicit flags.

---

### Forecast Horizon

| Flag | Default | Description |
|------|---------|-------------|
| `--forecast-days N` | 10 | Days ahead to start prediction window |
| `--target-return-days N` | 10 | Return period length (days) |

**Example**: Predict 90-day returns starting 10 days from now:
```bash
--forecast-days 10 --target-return-days 90
```

**Result**: On date T, predicts return from T+10 to T+100.

---

### Model Parameters

| Flag | Default | Description | Recommended Range |
|------|---------|-------------|-------------------|
| `--n-estimators N` | 400 | Number of trees (boosting rounds) | 200-600 |
| `--learning-rate X` | 0.05 | Learning rate (shrinkage) | 0.01-0.1 |
| `--max-depth N` | 6 | Maximum tree depth | 5-7 |
| `--num-leaves N` | 31 | Maximum leaf nodes per tree | 31-127 |

**Hardcoded (not configurable via CLI)**:
- `min_samples_leaf=100` - Minimum samples per leaf (0.0125% of 800K rows)
- `l2_regularization=0.2` - L2 penalty on leaf weights
- `max_bins=255` - Histogram bins for feature discretization
- `random_state=42` - Reproducibility seed

**Why some are hardcoded**: Core stability parameters optimized for stock returns with 800K rows and 300 features. Changing them requires understanding their interaction effects.

---

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--no-walk-forward` | False | Disable walk-forward (trains ONE model on all data) ⚠️ Has look-ahead bias |
| `--lookback-months N` | None | Use rolling N-month window instead of expanding window |
| `--sample-fraction X` | 1.0 | Use X% of training data (e.g., 0.1 = 10% for 10x speedup) |

**Walk-forward (default)**:
```
2020-01: Train on [2009-2019] → Predict Jan 2020
2020-02: Train on [2009-Jan 2020] → Predict Feb 2020
2025-12: Train on [2009-Nov 2025] → Predict Dec 2025
```

**Rolling window** (--lookback-months 36):
```
2020-01: Train on [2017-2019] → Predict Jan 2020
2020-02: Train on [2017-Jan 2020] → Predict Feb 2020
2025-12: Train on [2022-Nov 2025] → Predict Dec 2025
```

---

### Feature Engineering

| Flag | Default | Description |
|------|---------|-------------|
| `--no-lag` | False | Skip auto-lagging (assumes input is pre-lagged) ⚠️ Advanced users only |
| `--pca N` | None | Reduce to N principal components (exploration only, not production) |

**⚠️ Warning on --no-lag**:
- Only use if you've manually lagged ALL fundamentals by T-1
- Price, Volume, MarketCap are ALWAYS lagged (safety override)
- Default (auto-lag) is safer for most users

**⚠️ Warning on --pca**:
- PCA is disabled in walk-forward mode (would cause look-ahead bias)
- Only works with `--no-walk-forward`
- Not recommended for production

---

### Resume & Incremental Updates

| Flag | Default | Description |
|------|---------|-------------|
| `--resume-file PATH` | None | Resume from previous predictions file |
| `--overwrite-months N` | 1 | Re-train last N months (for data revisions) |
| `--preserve-existing` | False | 🔒 **Never overwrite existing predictions** (freeze historical forecasts) |

**Resume workflow**:
```bash
# Initial run (2009-2025)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2025.csv \
    --output predictions_2025.parquet

# Weekly update (add new week of data)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_week1.csv \
    --output predictions_2026_week1.parquet \
    --resume-file predictions_2025.parquet

# Monthly update with re-training last 2 months
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --overwrite-months 2

# 🔒 FREEZE historical forecasts (recommended for backtesting stability)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --preserve-existing
```

**What resume does**:
1. Loads previous predictions from `--resume-file`
2. Finds last prediction date
3. Skips months already predicted
4. Trains only for new months

**What --preserve-existing does** (⭐ RECOMMENDED for backtesting stability):
1. Loads previous predictions
2. **Never overwrites existing predictions** (keeps historical forecasts frozen)
3. Only computes predictions for rows with NaN
4. Ensures forecast stability when adding new data
5. Prevents historical predictions from changing due to new training data

---

### ⚠️ Why Predictions Change When Re-Running (Technical Deep Dive)

**THE PROBLEM**: When you add new data and re-run training, historical predictions change even when using the **same training window**.

**Example with --lookback-months 12**:

```bash
# First run (December 2025)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2025.csv \
    --output predictions_2025.parquet \
    --lookback-months 12

# Result: Dec 2025 prediction = 9.772%

# Second run (adding January 2026 data)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --lookback-months 12
# (default --overwrite-months 1)

# Result: Dec 2025 prediction = 10.981%  ← CHANGED by +1.2%!
```

**WHY?** The training window is the **same time period** (Dec 2024 - Nov 2025), but predictions change due to **three factors**:

#### 1. 🎯 Different Stock Universe (Main Reason)

When you load new data, the historical months have a different stock universe:

```
data_2025.csv (Dec 2025 snapshot):
  Dec 2024: 4,440 stocks
  Nov 2025: 4,440 stocks
  Dec 2025: 4,440 stocks

data_2026_jan.csv (Jan 2026 snapshot):
  Dec 2024: 4,445 stocks  ← 5 new stocks with historical data added!
  Nov 2025: 4,445 stocks
  Dec 2025: 4,445 stocks
  Jan 2026: 4,450 stocks
```

**Why new stocks appear in historical data:**
- IPOs in Jan 2026 that have backfilled historical fundamentals
- Previously delisted stocks now included
- Data provider expanded coverage
- New stocks from data merges/acquisitions

#### 2. 📊 Cross-Sectional Rankings Recalculated

The model uses **percentile rankings** (not raw values) for many features. Rankings are computed **within each date**:

```python
# For each date, rank all stocks by market cap (0.0 to 1.0)
rank = df.groupby('Date')['CompanyMarketCap'].rank(pct=True)
```

**First run (4,440 stocks on 2024-12-31)**:
```
AAPL: $3.0T → rank = 0.9998 (4439/4440)
Stock XYZ: $500B → rank = 0.8500 (3774/4440)
```

**Second run (4,445 stocks on 2024-12-31)**:
```
AAPL: $3.0T → rank = 0.9998 (4444/4445)
Stock XYZ: $500B → rank = 0.8475 (3768/4445)  ← Changed!
```

Even though XYZ's **raw market cap** is unchanged ($500B), its **percentile rank** drops from 0.8500 to 0.8475 because 5 new mid-cap stocks were added!

**Features affected by cross-sectional rankings**:
- `CompanyMarketCap_rank`
- `return_20d_rank`
- `volatility_20d_rank`
- `roe_rank`, `roa_rank`
- `ev_to_ebitda_rank`
- And ~10-15 other ranking features

#### 3. 📝 Data Revisions

Your data provider may revise historical values:
- **Earnings restatements**: Companies revise quarterly earnings
- **Adjusted fundamentals**: Balance sheet corrections
- **Corporate actions**: Stock splits, dividends adjusted retroactively
- **Error corrections**: Provider fixes data quality issues

**Example**:
```
data_2025.csv:
  AAPL 2025-11-30: revenue = $100.5B

data_2026_jan.csv:
  AAPL 2025-11-30: revenue = $100.8B  ← Revised up by $300M
```

---

### 📊 Impact Magnitude

| Factor | Typical Impact on Predictions | Frequency |
|--------|------------------------------|-----------|
| **Stock universe changes** | 1-5% prediction change | Every data update |
| **Cross-sectional rankings** | 0.5-3% prediction change | Every data update |
| **Data revisions** | 0-1% prediction change | Occasional |
| **Combined effect** | 1-7% prediction change | Every data update |

**Your observed changes**:
- Dec 2025: 9.772% → 10.981% (+1.2%) ✅ Within expected range
- Jan 2026: 27.581% → 32.303% (+4.7%) ✅ Within expected range

---

### ✅ Solution: Always Use --preserve-existing for Production

**Without --preserve-existing** (default behavior):
```
Every time you add new data:
✗ Historical predictions change by 1-7%
✗ Backtest results change
✗ Performance metrics drift
✗ Hard to track alpha decay vs. forecast drift
```

**With --preserve-existing** (recommended):
```
When you add new data:
✓ Historical predictions frozen (immutable)
✓ Backtest results stable
✓ Performance metrics consistent
✓ Only new months get predictions
```

**Production command**:
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --lookback-months 12 \
    --preserve-existing
```

**What happens**:
1. Loads Dec 2025 prediction: **9.772%** (from previous run)
2. Checks: Dec 2025 has prediction → **SKIP** (frozen)
3. Trains for Jan 2026: Train on Jan 2025 - Dec 2025 (12 months)
4. Predicts Jan 2026: **New prediction**
5. Output: Dec 2025 = **9.772%** (unchanged), Jan 2026 = **new value**

---

### 🔄 When to Use --overwrite-months Instead

| Scenario | Command | Reason |
|----------|---------|--------|
| **Data revisions** | `--overwrite-months 3` | Provider corrected last 3 months |
| **Bug fix in data pipeline** | `--overwrite-months 6` | Fix affected 6 months |
| **Symbol mapping error** | `--overwrite-months 12` | Need to recompute full year |
| **Normal production update** | `--preserve-existing` | Keep history stable ✅ |

**Example - Handle data revisions**:
```bash
# Provider announced they revised Q3 2025 earnings data
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --lookback-months 12 \
    --overwrite-months 3  # Recompute Oct, Nov, Dec 2025
```

**Note**: This will change historical predictions! Only use when you **intentionally** want to recompute due to data corrections.

---

### Diagnostics & Debugging

| Flag | Default | Description |
|------|---------|-------------|
| `--temporal-diagnostics` | False | Run ACF, Ljung-Box, stability tests (analyzes last 6 months) |
| `--skip-feature-importance` | False | Skip feature importance calculation (saves 1-3 minutes) |
| `--log-features` | False | Log top 20 feature importances per month to CSV in logs/ directory |

**Temporal diagnostics** checks for look-ahead bias:
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions.parquet \
    --temporal-diagnostics
```

**Output**:
- Residual autocorrelation (ACF): Should be near zero
- Ljung-Box test: p-value > 0.05 = good (no autocorrelation)
- Temporal stability: Residuals stable over time
- Time period analyzed: Last 6 months only (fast)

---

## Common Use Cases

### 1. Weekly Production Update (Fastest)
```bash
# Add new week of data to existing predictions
python forecast_returns_ml_walk_forward.py \
    --input-file data_latest.parquet \
    --output predictions_latest.parquet \
    --resume-file predictions_previous.parquet
```

**Time**: ~30 seconds to 2 minutes (only trains new months)

---

### 2. Monthly Re-training (Recommended)
```bash
# Re-train last 2 months (for data revisions)
python forecast_returns_ml_walk_forward.py \
    --input-file data_latest.parquet \
    --output predictions_latest.parquet \
    --resume-file predictions_previous.parquet \
    --overwrite-months 2
```

**Time**: ~2-5 minutes (re-trains 2 months + new data)

---

### 3. Custom Forecast Horizon (90-Day Returns)
```bash
# Predict 90-day returns starting 10 days ahead
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions_90d.parquet \
    --forecast-days 10 \
    --target-return-days 90
```

**Use case**: Longer-term position trading

---

### 4. Faster Training (Development/Testing)
```bash
# Use 10% of data for 10x speedup
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions_fast.parquet \
    --sample-fraction 0.1
```

**Time**: ~1-2 minutes (full run on large dataset)
**⚠️ Warning**: Lower accuracy, only for testing

---

### 5. Improved Accuracy (More Trees + Leaves)
```bash
# Best accuracy settings (slower)
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions_best.parquet \
    --n-estimators 600 \
    --num-leaves 127
```

**Time**: ~15-30 minutes (full run)
**Benefit**: 2-5% better correlation

---

### 6. Rolling Window (Last 3 Years Only)
```bash
# Train on last 36 months only (not expanding window)
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions_rolling.parquet \
    --lookback-months 36
```

**Use case**: Adapt faster to regime changes, ignore distant history

---

### 7. Full Diagnostics (Verify No Look-Ahead Bias)
```bash
# Run all diagnostics (temporal tests)
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions.parquet \
    --temporal-diagnostics
```

**Time**: +2-3 minutes for diagnostics
**When**: Monthly verification, after code changes

---

### 8. Pre-Lagged Input Data
```bash
# You've already lagged fundamentals in your pipeline
python forecast_returns_ml_walk_forward.py \
    --input-file data_pre_lagged.csv \
    --output predictions.parquet \
    --no-lag
```

**⚠️ Critical**: Verify your input is lagged by T-1. Price/Volume/MarketCap are ALWAYS lagged (safety override).

---

## Input/Output

### Input File Requirements

**Required columns**:
- `Date` - Format: YYYY-MM-DD
- `Symbol` - Stock ticker (e.g., AAPL, MSFT)
- Any fundamental columns (e.g., revenue, pe, marketcap, etc.)

**Supported formats**:
- CSV (`.csv`)
- Parquet (`.parquet`, `.pq`) - **Recommended** (5-10x faster I/O, 10x smaller)

**Example CSV**:
```csv
Date,Symbol,RefPriceClose,CompanyMarketCap,revenue,netinc,pe,roe
2020-01-15,AAPL,293.65,1250000000000,260174000000,55256000000,24.5,0.61
2020-01-15,MSFT,165.04,1260000000000,143015000000,44281000000,30.2,0.42
```

---

### Output File Format

**Columns added**:
- `forward_return` - Actual return (for validation)
- `predicted_return` - ML forecast

**Formats**:
- CSV: Same as input, human-readable
- Parquet: **10x smaller files**, 5-10x faster I/O

**Additional outputs**:
```
predictions.parquet                          # Main output
predictions_20091231_20260106.parquet        # Auto-generated from date range
predictions_20091231_20260106_forecast_only.csv  # Lean file (Date, Symbol, predicted_return)
logs/forecast_ml_YYYYMMDD_HHMMSS.log        # Detailed log
```

**Forecast-only file**:
- Automatically generated
- Only 3 columns: Date, Symbol, predicted_return
- Removes NaN predictions
- Perfect for trading systems

**Console output includes**:
```
🏆 TOP 10 PREDICTED GAINERS (most recent data):
  - Highest 10 predictions (may show same symbol multiple times)
  - Shows absolute maximum predicted returns
  - Useful for identifying extreme opportunities

🎯 TOP 10 UNIQUE SYMBOLS (most recent prediction per symbol):
  - Top 10 different stocks (one per symbol)
  - Uses most recent prediction for each symbol
  - Better for diversified portfolio construction
```

---

## Model Parameters

### Default Production Stack (v3.3.12)

```python
# Optimized for 800K rows, 300 features, stock returns
n_estimators=400          # 400 trees (boosting rounds)
learning_rate=0.05        # 5% shrinkage per tree
max_depth=6               # Shallow trees (prevents overfitting on noise)
num_leaves=31             # 31 leaf nodes max per tree
min_samples_leaf=100      # 100 samples min per leaf (0.0125% of 800K)
l2_regularization=0.2     # Moderate L2 penalty
max_bins=255              # 255 histogram bins
random_state=42           # Reproducibility
```

### When to Adjust

**Increase n_estimators** (400 → 600):
- Want better accuracy (+2-5% correlation)
- Don't mind longer training time (+50%)
```bash
--n-estimators 600
```

**Increase num_leaves** (31 → 127):
- Large dataset (>1M rows)
- Complex patterns
- Best combined with more trees
```bash
--num-leaves 127 --n-estimators 500
```

**Decrease learning_rate** (0.05 → 0.01):
- Want more stable training
- Must increase n_estimators to compensate
```bash
--learning-rate 0.01 --n-estimators 1000
```

**Use rolling window** (lookback-months):
- Recent data more relevant than distant history
- Faster regime adaptation
```bash
--lookback-months 36  # Last 3 years only
```

---

## Performance Tuning

### Speed Optimizations

| Technique | Flag | Speedup | Accuracy Impact |
|-----------|------|---------|-----------------|
| **Parquet I/O** | Use `.parquet` extension | 5-10x faster I/O | None |
| **Sample fraction** | `--sample-fraction 0.1` | 10x faster training | -5 to -10% correlation |
| **Skip feature importance** | `--skip-feature-importance` | Saves 1-3 min | None (just logging) |
| **Resume mode** | `--resume-file` | Only train new months | None |
| **Fewer trees** | `--n-estimators 200` | 2x faster | -2 to -3% correlation |

**Recommended for production**:
```bash
# Weekly updates (30 seconds to 2 minutes)
--resume-file previous.parquet  # Only train new data
--skip-feature-importance       # Don't need every week

# Use Parquet format
--input-file data.parquet --output predictions.parquet
```

---

### Memory Optimizations

**Large datasets (>2M rows)**:
- Use Parquet format (lower memory footprint)
- Consider `--sample-fraction 0.5` (50% sampling, minimal accuracy loss)
- Use `--lookback-months` to limit training window

**Example** (10M rows):
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file huge_data.parquet \
    --output predictions.parquet \
    --lookback-months 60 \
    --sample-fraction 0.5
```

---

## Diagnostics & Debugging

### Temporal Diagnostics (Look-Ahead Bias Check)

**What it does**:
- Autocorrelation Function (ACF) on residuals
- Ljung-Box test for autocorrelation
- Temporal stability analysis
- Analyzes last 6 months (fast, sufficient)

**How to interpret**:
```
✅ GOOD:
  • ACF near zero at all lags
  • Ljung-Box p-value > 0.05
  • Residuals stable over time

⚠️ BAD:
  • ACF significantly positive (look-ahead bias suspected)
  • Ljung-Box p-value < 0.05 (autocorrelation detected)
  • Residuals trend over time (data leakage)
```

**When to run**:
- After code changes to feature engineering
- Monthly verification
- When correlation seems "too good" (>90%)

---

### Feature Importance

**Enabled by default** (disable with `--skip-feature-importance`):
- Shows top 30 most important features
- Logged to console and log file
- Takes 1-3 minutes to calculate

**Example output**:
```
📊 FEATURE IMPORTANCE (Top 30):
   1. CompanyMarketCap_lag1                  0.0845
   2. RefPriceClose_lag1                     0.0712
   3. return_20d                             0.0623
   4. ev_to_ebitda                           0.0511
   5. volatility_20d                         0.0498
```

**When to skip**:
- Weekly production updates (you know what's important)
- Fast iterations during development
- When using `--sample-fraction` (less reliable importance)

---

### Feature Importance Logging (--log-features)

**Purpose**: Track how feature importance changes over time during walk-forward training.

**What it does**:
- Saves top 20 features for EACH month to CSV
- Creates timestamped file: `logs/feature_importance_YYYYMMDD_HHMMSS.csv`
- CSV format allows easy plotting with pandas/matplotlib

**Example usage**:
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions.parquet \
    --log-features
```

**CSV output format**:
```
Month,Rank,Feature,Importance,Std
2020-01,1,CompanyMarketCap_lag1,0.084523,0.002341
2020-01,2,RefPriceClose_lag1,0.071234,0.001987
...
2020-02,1,CompanyMarketCap_lag1,0.082145,0.002198
2020-02,2,return_20d,0.069876,0.001823
```

**Plot feature importance over time**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Read logged features
df = pd.read_csv('logs/feature_importance_20260120_143052.csv')

# Plot top 5 features over time
top_features = df[df['Rank'] <= 5]
for feature in top_features['Feature'].unique():
    subset = top_features[top_features['Feature'] == feature]
    plt.plot(subset['Month'], subset['Importance'], label=feature)

plt.xlabel('Month')
plt.ylabel('Importance')
plt.legend()
plt.title('Feature Importance Over Time (Top 5)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Performance impact**:
- Adds ~5-10 seconds per month
- Uses smaller sample (5,000 rows) for speed
- 3 permutation repeats (vs 5 for final importance)

**When to use**:
- Research: Understand regime changes in feature importance
- Production: Monitor if model is adapting correctly
- Debugging: Detect sudden shifts in feature rankings

---

### Log Files

**Automatically created**:
```
logs/forecast_ml_20260116_143052.log
```

**Contains**:
- All console output
- Detailed progress
- Feature importance
- Error messages
- Performance metrics

**Log rotation**: Not implemented, manually delete old logs if needed.

---

## Examples by Use Case

### Research & Development
```bash
# Fast iterations (10% sampling)
python forecast_returns_ml_walk_forward.py \
    --input-file test_data.csv \
    --output test_predictions.parquet \
    --sample-fraction 0.1 \
    --skip-feature-importance
```

### Production Weekly Update
```bash
# Resume from last week, skip diagnostics
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_week3.parquet \
    --output predictions_2026_week3.parquet \
    --resume-file predictions_2026_week2.parquet \
    --skip-feature-importance
```

### Production Monthly Re-training
```bash
# Re-train last 2 months, full diagnostics
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.parquet \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025_dec.parquet \
    --overwrite-months 2 \
    --temporal-diagnostics
```

### Long-Term Returns (180 days)
```bash
# Predict 180-day returns starting 10 days ahead
python forecast_returns_ml_walk_forward.py \
    --input-file data.parquet \
    --output predictions_180d.parquet \
    --forecast-days 10 \
    --target-return-days 180 \
    --n-estimators 500 \
    --num-leaves 127
```

### High-Frequency Updates (Rolling 2 Years)
```bash
# Adapt quickly to recent regimes
python forecast_returns_ml_walk_forward.py \
    --input-file data.parquet \
    --output predictions_rolling.parquet \
    --lookback-months 24 \
    --resume-file predictions_previous.parquet
```

---

## Quick Reference Card

```bash
# BASIC USAGE
python forecast_returns_ml_walk_forward.py --input-file data.csv --output predictions.parquet

# FORECAST HORIZON
--forecast-days 10           # Start prediction 10 days ahead
--target-return-days 90      # Return period: 90 days

# MODEL TUNING
--n-estimators 600           # More trees (better accuracy, slower)
--num-leaves 127             # More leaves (more capacity)
--learning-rate 0.01         # Lower learning rate (needs more trees)

# TRAINING MODE
--lookback-months 36         # Rolling 3-year window
--sample-fraction 0.1        # Use 10% of data (10x faster)
--no-walk-forward            # Single model (⚠️ look-ahead bias)

# RESUME & UPDATE
--resume-file prev.parquet   # Resume from previous run
--overwrite-months 2         # Re-train last 2 months

# DIAGNOSTICS
--temporal-diagnostics       # Check for look-ahead bias
--skip-feature-importance    # Skip importance calculation

# ADVANCED
--no-lag                     # Skip auto-lagging (⚠️ advanced)
--pca 50                     # Reduce to 50 components (⚠️ exploration only)
```

---

## Getting Help

**See also**:
- [README.md](README.md) - Feature overview, what's new
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [LOOK_AHEAD_BIAS_AUDIT.md](LOOK_AHEAD_BIAS_AUDIT.md) - Safety verification
- [Docs/INDEX.md](Docs/INDEX.md) - Technical deep dives

**For bugs or questions**:
- Check log files in `logs/` directory
- Review LOOK_AHEAD_BIAS_AUDIT.md for safety concerns
- Verify input data format (Date, Symbol columns required)

---

**Last Updated**: 2026-01-16
**Version**: v3.3.12
**Script**: `forecast_returns_ml_walk_forward.py`
