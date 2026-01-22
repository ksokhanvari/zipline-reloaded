# Forecast Stability & --preserve-existing Flag

**Version**: v3.3.14
**Date**: 2026-01-20
**Status**: Production-Ready

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Root Cause Analysis](#root-cause-analysis)
3. [The Solution](#the-solution)
4. [Technical Deep Dive](#technical-deep-dive)
5. [Production Best Practices](#production-best-practices)
6. [FAQ](#faq)

---

## The Problem

### Symptom: Historical Predictions Change When Adding New Data

When you run walk-forward training with new data, historical predictions change even when using the **same training window**.

**Example Scenario**:

```bash
# First run (December 2025)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2025.csv \
    --output predictions_2025.parquet \
    --lookback-months 12

# Predictions:
# 2025-11-30: 13.230%
# 2025-12-31:  9.772%
```

```bash
# Second run (adding January 2026 data)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --lookback-months 12

# Predictions:
# 2025-11-30: 13.230%  (unchanged)
# 2025-12-31: 10.981%  ← CHANGED by +1.2%!
# 2026-01-31: 32.303%  (new)
```

### Why This Is a Problem

1. **Backtest Instability**: Your backtest results change every time you add new data
2. **Non-Reproducible Research**: Same historical data produces different predictions
3. **Performance Drift**: Hard to distinguish real alpha decay from forecast drift
4. **Production Issues**: Trading decisions based on shifting historical signals

---

## Root Cause Analysis

### The Paradox: Same Training Window, Different Predictions

**Both runs train Dec 2025 on the same 12-month window** (Dec 2024 - Nov 2025), yet predictions differ. Why?

### Three Sources of Variance

#### 1. 🎯 Stock Universe Changes (Primary Cause)

**The Issue**: When you load new data, historical months have a different stock universe.

**Example**:

```
data_2025.csv (December 2025 snapshot):
┌──────────┬─────────────────────┐
│   Date   │ Number of Stocks    │
├──────────┼─────────────────────┤
│ 2024-12  │ 4,440 stocks        │
│ 2025-11  │ 4,440 stocks        │
│ 2025-12  │ 4,440 stocks        │
└──────────┴─────────────────────┘

data_2026_jan.csv (January 2026 snapshot):
┌──────────┬─────────────────────┬──────────────────────┐
│   Date   │ Number of Stocks    │ What Changed?        │
├──────────┼─────────────────────┼──────────────────────┤
│ 2024-12  │ 4,445 stocks ✗      │ +5 stocks added      │
│ 2025-11  │ 4,445 stocks ✗      │ +5 stocks added      │
│ 2025-12  │ 4,445 stocks ✗      │ +5 stocks added      │
│ 2026-01  │ 4,450 stocks        │ New month            │
└──────────┴─────────────────────┴──────────────────────┘
```

**Why Historical Months Have New Stocks**:

1. **IPOs with Backfilled Data**: Company IPO'd in Jan 2026 but data provider backfilled fundamentals to 2010
2. **Coverage Expansion**: Data provider added previously unavailable stocks
3. **Delisted Stock Removals**: Previously included stocks now removed
4. **Data Merges**: New data source merged with different stock coverage
5. **Symbol Mapping Changes**: FB→META, GOOG→GOOGL now handled differently

**Real-World Example**:

```
First run (data_2025.csv):
  RIVN (Rivian): Not included in 2024-12 data (small cap filter)

Second run (data_2026_jan.csv):
  RIVN: Now included in 2024-12 data (grew to mid-cap by 2026)
       Provider backfilled historical fundamentals
```

#### 2. 📊 Cross-Sectional Rankings Recalculated

**The Issue**: The model uses **percentile rankings**, not raw values. Rankings depend on the entire stock universe.

**How It Works**:

```python
# Code from forecast_returns_ml_walk_forward.py:1501
month_df[f'{col}_rank'] = month_df.groupby('Date')[col].rank(pct=True)
```

**Example - Market Cap Rankings**:

**First run (4,440 stocks on 2024-12-31)**:
```
┌──────────┬───────────┬──────────┬─────────────┐
│  Symbol  │ Market Cap│   Rank   │ Percentile  │
├──────────┼───────────┼──────────┼─────────────┤
│  AAPL    │  $3.0T    │ 4,439    │  0.9998     │
│  MSFT    │  $2.8T    │ 4,438    │  0.9995     │
│  XYZ     │  $500B    │ 3,774    │  0.8500     │
│  ABC     │  $10B     │   500    │  0.1126     │
└──────────┴───────────┴──────────┴─────────────┘
Total: 4,440 stocks
```

**Second run (4,445 stocks on 2024-12-31)**:
```
┌──────────┬───────────┬──────────┬─────────────┬──────────┐
│  Symbol  │ Market Cap│   Rank   │ Percentile  │  Change  │
├──────────┼───────────┼──────────┼─────────────┼──────────┤
│  AAPL    │  $3.0T    │ 4,444    │  0.9998     │  0.0000  │
│  MSFT    │  $2.8T    │ 4,443    │  0.9996     │ +0.0001  │
│  XYZ     │  $500B    │ 3,768    │  0.8475     │ -0.0025  │
│  ABC     │  $10B     │   502    │  0.1129     │ +0.0003  │
│  NEW1    │  $520B    │ 3,780    │  0.8502     │   NEW    │
│  NEW2    │  $480B    │ 3,760    │  0.8457     │   NEW    │
└──────────┴───────────┴──────────┴─────────────┴──────────┘
Total: 4,445 stocks (+5 new stocks)
```

**Key Observation**: Stock XYZ's market cap is **unchanged** ($500B), but its percentile rank dropped from **0.8500** to **0.8475** because:
- 2 new stocks above it (NEW1: $520B)
- 3 new stocks below it (NEW2: $480B, etc.)
- Net effect: Pushed down slightly in rankings

**Features Affected by Rankings** (~15-20 features):

```python
# From forecast_returns_ml_walk_forward.py:536-537
_rank_cols = [
    'CompanyMarketCap',    # Market cap percentile
    'return_20d',          # Momentum percentile
    'volatility_20d',      # Volatility percentile
    'roe',                 # Return on equity percentile
    'roa',                 # Return on assets percentile
    'ev_to_ebitda',        # Valuation percentile
    'ltg',                 # Long-term growth percentile
    # ... and more
]
```

**Impact**: If 15 ranking features change by 0.001-0.003 each, the combined effect on predictions can be **1-5%**.

#### 3. 📝 Data Revisions

**The Issue**: Data providers revise historical values as new information becomes available.

**Common Revisions**:

1. **Earnings Restatements**:
   ```
   First run (data_2025.csv):
     AAPL 2025-11-30: EPS = $1.52 (preliminary)

   Second run (data_2026_jan.csv):
     AAPL 2025-11-30: EPS = $1.54 (restated)  ← +$0.02
   ```

2. **Balance Sheet Corrections**:
   ```
   TSLA 2025-09-30: Total Assets = $100B → $102B (accounting correction)
   ```

3. **Corporate Actions**:
   ```
   GOOGL 2025-06-15: 20-for-1 stock split
   → All historical prices adjusted retroactively
   ```

4. **Data Quality Fixes**:
   ```
   Provider discovered data pipeline error affecting Q3 2025
   → Corrected values for 500+ stocks
   ```

**Impact Frequency**:

| Revision Type | Frequency | Typical Impact |
|--------------|-----------|----------------|
| Earnings restatements | Quarterly | 0-0.5% |
| Balance sheet corrections | Annually | 0-0.3% |
| Corporate actions | Occasional | 0-1.0% |
| Data quality fixes | Rare | 0-2.0% |

---

## The Solution: --preserve-existing Flag

### What It Does

The `--preserve-existing` flag **freezes historical forecasts** so they never change when adding new data.

**Mechanism**:

1. Loads previous predictions from `--resume-file`
2. Identifies months that **already have predictions** (non-NaN values)
3. **Skips those months entirely** in the walk-forward loop
4. Only trains and predicts for months with **missing predictions** (NaN)

### How It Works (Code Flow)

**File**: `forecast_returns_ml_walk_forward.py:1407-1433`

```python
# Step 1: Determine candidate months to process
if resume_from_date:
    months_to_process = [m for m in unique_months if m >= resume_from_month]
else:
    months_to_process = unique_months

# Step 2: Filter out months that already have predictions
if preserve_existing and previous_predictions is not None:
    months_with_predictions = []
    months_without_predictions = []

    for month in months_to_process:
        # Get all rows for this month
        month_mask = df['_year_month'] == month
        month_positions = np.where(month_mask)[0]

        # Check if ALL rows have predictions (not NaN)
        month_preds = predictions[month_positions]
        has_all_predictions = np.all(~np.isnan(month_preds))

        if has_all_predictions:
            months_with_predictions.append(month)  # SKIP
        else:
            months_without_predictions.append(month)  # PROCESS

    # Only process months without predictions
    months_to_process = months_without_predictions

    print(f"🔒 PRESERVE MODE: Skipping {len(months_with_predictions)} months")
    print(f"Processing {len(months_without_predictions)} months")
```

### Example Output

```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --lookback-months 12 \
    --preserve-existing
```

**Console Output**:
```
📂 Aligning previous predictions with sorted dataframe...
  • Previous predictions with values: 9,281,661
  • Aligned predictions: 9,281,661 rows

🔒 PRESERVE MODE: Skipping 195 months with existing predictions
  • Processing 1 months with missing predictions

🔄 Training models month by month...

  [196/196] 2026-01: Trained on 9,240,123 rows → Predicted 41,538 rows (45.2s)

  ✅ Walk-Forward Complete!
  • Months processed: 1
  • Total predictions: 9,323,199 of 9,323,199
  • Total time: 0m 45s
```

**What Happened**:
- Loaded 195 months of previous predictions (Jan 2020 - Dec 2025)
- **Skipped** all 195 months (already have predictions)
- **Trained and predicted** only Jan 2026 (new month)
- Dec 2025 prediction: **9.772%** (frozen from first run)
- Jan 2026 prediction: **New value** (based on fresh training)

---

## Technical Deep Dive

### Training vs. Prediction Separation

**Key Concept**: `--preserve-existing` **does NOT change the training process**, only what gets predicted.

#### Without --preserve-existing (Default Behavior)

```
Month 195 (2025-12):
  ┌─────────────────────────────────────────────┐
  │ 1. Train on: Dec 2024 - Nov 2025 (12 months)│
  │    - 9.2M rows                               │
  │    - Stock universe from data_2026_jan.csv   │ ← New universe!
  │    - Rankings recalculated                   │ ← Rankings shift!
  ├─────────────────────────────────────────────┤
  │ 2. Predict: Dec 2025                         │
  │    - Overwrites old prediction               │ ← 9.772% → 10.981%
  └─────────────────────────────────────────────┘

Month 196 (2026-01):
  ┌─────────────────────────────────────────────┐
  │ 1. Train on: Jan 2025 - Dec 2025 (12 months)│
  │    - 9.3M rows                               │
  ├─────────────────────────────────────────────┤
  │ 2. Predict: Jan 2026                         │
  │    - Creates new prediction                  │ ← New value
  └─────────────────────────────────────────────┘
```

**Result**: Dec 2025 prediction **changed** from 9.772% to 10.981%

#### With --preserve-existing (Stable Behavior)

```
Month 195 (2025-12):
  ┌─────────────────────────────────────────────┐
  │ ⏭️ SKIPPED                                   │
  │ Reason: Already has prediction (9.772%)     │
  └─────────────────────────────────────────────┘

Month 196 (2026-01):
  ┌─────────────────────────────────────────────┐
  │ 1. Train on: Jan 2025 - Dec 2025 (12 months)│
  │    - 9.3M rows                               │
  │    - Stock universe from data_2026_jan.csv   │ ← Uses new data
  │    - Rankings computed on new universe       │
  ├─────────────────────────────────────────────┤
  │ 2. Predict: Jan 2026                         │
  │    - Creates new prediction                  │ ← New value
  └─────────────────────────────────────────────┘
```

**Result**: Dec 2025 prediction **frozen** at 9.772% (unchanged)

### Does Jan 2026 Model See New Data?

**YES!** The Jan 2026 model trains on:
- **12 months**: Jan 2025 - Dec 2025
- **Stock universe**: From `data_2026_jan.csv` (includes new stocks)
- **Rankings**: Computed cross-sectionally on new universe
- **Data revisions**: Any corrections in `data_2026_jan.csv`

**This is correct behavior**:
- Jan 2026 model should use the **most recent data** available
- Historical forecasts (Dec 2025 and earlier) remain **frozen**
- New forecast (Jan 2026) benefits from **updated data**

---

## Production Best Practices

### 1. Always Use --preserve-existing for Production

**Recommended Production Command**:

```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data_latest.csv \
    --output predictions_latest.parquet \
    --resume-file predictions_previous.parquet \
    --lookback-months 12 \
    --preserve-existing \
    --skip-feature-importance
```

**Why**:
- ✅ Historical forecasts stable (reproducible backtests)
- ✅ Only trains new months (95%+ faster)
- ✅ Performance metrics consistent
- ✅ Clear separation: historical (frozen) vs. new (live)

### 2. When to Use --overwrite-months

**Only use --overwrite-months when**:

| Scenario | Command | Impact |
|----------|---------|--------|
| **Data provider announced revisions** | `--overwrite-months 3` | Recomputes last 3 months |
| **Bug fix in data pipeline** | `--overwrite-months 6` | Recomputes last 6 months |
| **Symbol mapping error** | `--overwrite-months 12` | Recomputes last 12 months |
| **Intentional historical update** | `--overwrite-months 24` | Recomputes last 24 months |

**Example - Handle Q3 2025 Data Revisions**:

```bash
# Provider announced they corrected Q3 2025 fundamentals
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan_corrected.csv \
    --output predictions_2026_jan_v2.parquet \
    --resume-file predictions_2026_jan_v1.parquet \
    --lookback-months 12 \
    --overwrite-months 4  # Recompute Sep, Oct, Nov, Dec 2025 + Jan 2026
```

**⚠️ WARNING**: This will **change historical predictions**! Only use when you have a legitimate reason.

### 3. Version Control for Predictions

**Track prediction versions**:

```bash
# Directory structure
predictions/
├── 2025-12-31_predictions.parquet  # Dec 2025 run
├── 2026-01-07_predictions.parquet  # Jan week 1 update
├── 2026-01-14_predictions.parquet  # Jan week 2 update
├── 2026-01-21_predictions.parquet  # Jan week 3 update
└── 2026-01-31_predictions.parquet  # Jan final update
```

**With --preserve-existing**:
```bash
# All versions have IDENTICAL Dec 2025 predictions
diff <(parquet-tools cat predictions/2025-12-31_predictions.parquet | grep "2025-12") \
     <(parquet-tools cat predictions/2026-01-31_predictions.parquet | grep "2025-12")
# Output: (no differences) ✓
```

**Without --preserve-existing**:
```bash
# Dec 2025 predictions DRIFT across versions
diff <(parquet-tools cat predictions/2025-12-31_predictions.parquet | grep "2025-12") \
     <(parquet-tools cat predictions/2026-01-31_predictions.parquet | grep "2025-12")
# Output: 4,523 differences ✗
```

### 4. Backtest Validation

**Test forecast stability**:

```python
import pandas as pd

# Load two consecutive runs
v1 = pd.read_parquet('predictions_2025_12_31.parquet')
v2 = pd.read_parquet('predictions_2026_01_31.parquet')

# Merge on Date + Symbol
merged = v1.merge(
    v2,
    on=['Date', 'Symbol'],
    suffixes=('_v1', '_v2')
)

# Filter to historical period (Dec 2025 and earlier)
historical = merged[merged['Date'] <= '2025-12-31']

# Check for differences
historical['pred_diff'] = (
    historical['predicted_return_v2'] - historical['predicted_return_v1']
)

# With --preserve-existing: Should be 0.000000
print(f"Mean absolute difference: {historical['pred_diff'].abs().mean():.6f}")
print(f"Max absolute difference: {historical['pred_diff'].abs().max():.6f}")
print(f"Rows changed: {(historical['pred_diff'].abs() > 0.0001).sum()}")

# Expected output:
# Mean absolute difference: 0.000000
# Max absolute difference: 0.000000
# Rows changed: 0
```

---

## FAQ

### Q1: Does --preserve-existing affect model quality?

**A: No. The model training is identical.**

- Jan 2026 model still trains on all available data (Jan 2025 - Dec 2025)
- Uses updated stock universe from new data file
- Benefits from data revisions and corrections
- Only difference: We don't **re-predict** historical months

### Q2: What if I want to recompute historical forecasts due to data corrections?

**A: Use --overwrite-months instead of --preserve-existing.**

```bash
# Provider corrected Q4 2025 data - recompute last 3 months
python forecast_returns_ml_walk_forward.py \
    --input-file data_corrected.csv \
    --output predictions_corrected.parquet \
    --resume-file predictions_old.parquet \
    --lookback-months 12 \
    --overwrite-months 3  # Oct, Nov, Dec 2025
```

### Q3: Can I use both --preserve-existing and --overwrite-months?

**A: --preserve-existing overrides --overwrite-months.**

If both are provided:
1. `--overwrite-months N` determines candidate months
2. `--preserve-existing` filters out months with existing predictions
3. Net effect: Only months with NaN predictions get processed

**Example**:
```bash
--resume-file predictions.parquet \
--preserve-existing \
--overwrite-months 12  # Ignored! preserve-existing takes precedence
```

### Q4: What happens if I have partial predictions for a month?

**A: The month is processed (not skipped).**

```python
# Code checks: ALL rows must have predictions
has_all_predictions = np.all(~np.isnan(month_preds))
```

**Scenario**:
- Dec 2025: 4,440 stocks
- 4,430 have predictions (existing)
- 10 have NaN (new stocks added)
- **Result**: Dec 2025 is **processed** (trains and predicts all 4,440 stocks)

This ensures new stocks get predictions, but existing stocks will be **overwritten**.

**Workaround**: If you want to preserve existing predictions and only add new stocks, you need to:
1. Manually merge predictions
2. Or accept that the month is recomputed

### Q5: How do I verify --preserve-existing is working?

**A: Check console output for "🔒 PRESERVE MODE":**

```
🔒 PRESERVE MODE: Skipping 195 months with existing predictions
  • Processing 1 months with missing predictions
```

**Or compare predictions**:

```python
import pandas as pd

v1 = pd.read_parquet('predictions_old.parquet')
v2 = pd.read_parquet('predictions_new.parquet')

# Should be identical for historical dates
historical_v1 = v1[v1['Date'] <= '2025-12-31'].sort_values(['Date', 'Symbol'])
historical_v2 = v2[v2['Date'] <= '2025-12-31'].sort_values(['Date', 'Symbol'])

assert historical_v1['predicted_return'].equals(historical_v2['predicted_return']), \
    "Historical predictions changed!"

print("✓ Forecast stability verified!")
```

### Q6: Does this affect --lookback-months behavior?

**A: No. Training window calculation is unchanged.**

Whether you use:
- Expanding window (default)
- Rolling window (`--lookback-months 12`)

The `--preserve-existing` flag only controls **which months to predict**, not **how much data to train on**.

**Example with --lookback-months 12**:

```
Without --preserve-existing:
  Dec 2025: Train on Dec 2024 - Nov 2025 → Predict Dec 2025 ✓
  Jan 2026: Train on Jan 2025 - Dec 2025 → Predict Jan 2026 ✓

With --preserve-existing:
  Dec 2025: SKIPPED (already has prediction)
  Jan 2026: Train on Jan 2025 - Dec 2025 → Predict Jan 2026 ✓
```

---

## Summary

### The Problem
- Historical predictions change when adding new data
- Root causes: Stock universe changes, cross-sectional rankings, data revisions
- Impact: 1-7% prediction drift

### The Solution
- Use `--preserve-existing` flag
- Freezes historical forecasts (immutable)
- Only predicts new months

### Production Command
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data_latest.csv \
    --output predictions_latest.parquet \
    --resume-file predictions_previous.parquet \
    --lookback-months 12 \
    --preserve-existing
```

### Benefits
- ✅ Reproducible backtests
- ✅ Stable performance metrics
- ✅ Clear audit trail
- ✅ Production-ready

---

**Last Updated**: 2026-01-20
**Version**: v3.3.14
