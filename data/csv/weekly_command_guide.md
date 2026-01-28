# Weekly Command Guide: When to Use Each Flag

This guide explains the difference between `--preserve-existing` and `--overwrite-months` flags, and when to use each one for production ML forecasting.

---

## Quick Reference

| Scenario | Flag to Use | Why |
|----------|------------|-----|
| **Weekly/Monthly production update** | `--preserve-existing` ✅ | Keep historical forecasts frozen |
| **Data provider revised past data** | `--overwrite-months N` | Recompute affected months |
| **You want to repredict with new features** | `--overwrite-months N` | Recalculate with updated model |
| **Bug fix in data pipeline** | `--overwrite-months N` | Correct affected period |

---

## `--preserve-existing` (For Stable Backtesting)

### When to Use:
- ✅ **Normal weekly/monthly updates** (adding new data)
- ✅ **Production backtesting** (need stable historical forecasts)
- ✅ **Research reproducibility** (same data = same predictions)
- ✅ **Live trading** (don't want historical signals to drift)

### Command Example:
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --preserve-existing  # 🔒 FREEZE all historical predictions
```

### What It Does:

**Efficient High Water Mark Approach**:
- Checks months backwards from most recent
- **Last 2 months**: Requires 100% strict matching (ALL rows must have predictions)
- **Older months**: Allows 95% threshold (tolerates minor data provider changes)
- Finds the "high water mark" (last complete month)
- Skips all months up to and including that month
- Only processes months AFTER the high water mark

**Behavior**:
```
Previous run had predictions through Dec 2025:
  Nov 2025: 13.230%  (100% complete)
  Dec 2025:  9.772%  (100% complete)

New run with --preserve-existing:
  Checks Dec 2025: 100% complete → High water mark found!
  Nov 2025: 13.230% ← SKIPPED (below high water mark)
  Dec 2025:  9.772% ← SKIPPED (high water mark)
  Jan 2026: 27.500% ← NEW (computed - after high water mark)
```

### Why This Matters:

**Problem without `--preserve-existing`**:
```
Week 1: Dec 2025 prediction = 9.772%
Week 2: Add new data → Dec 2025 prediction = 10.981% (+1.2%)
Week 3: Add new data → Dec 2025 prediction = 11.150% (+1.4%)
Week 4: Add new data → Dec 2025 prediction = 10.500% (-0.6%)
```

❌ Historical predictions keep changing! This makes backtesting unstable and performance tracking impossible.

**Solution with `--preserve-existing`**:
```
Week 1: Dec 2025 prediction = 9.772%
Week 2: Add new data → Dec 2025 prediction = 9.772% (frozen)
Week 3: Add new data → Dec 2025 prediction = 9.772% (frozen)
Week 4: Add new data → Dec 2025 prediction = 9.772% (frozen)
```

✅ Historical predictions **never change** - stable backtests, consistent performance tracking.

### Console Output:
```
📂 RESUME MODE
  • 🔒 PRESERVE MODE: Will skip months with existing predictions
  • 🔒 PRESERVE MODE: Found predictions through 2025-12
  • Skipping 195 months with existing predictions
  • Processing 1 months with missing predictions

  [196/196] 2026-01: Trained on 9,240,123 rows → Predicted 41,538 rows (45.2s)
```

**Key Features**:
- ✅ **100% strict** for last 2 months (your latest work)
- ✅ **95% threshold** for older months (robust to minor changes)
- ✅ **Efficient** - finds high water mark in 1-5 iterations
- ✅ **Only processes new months** - skips all historical data

---

## `--overwrite-months N` (For Corrections & Updates)

### When to Use:
- ❌ **Data provider revised past fundamentals** (earnings restatements)
- ❌ **Bug in data pipeline** (need to fix affected period)
- ❌ **New features added to model** (want to repredict with new features)
- ❌ **Symbol mapping errors** (ticker changes affected past data)
- ❌ **Model parameter changes** (want to see impact on recent predictions)

### Command Example:
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_2026_jan.parquet \
    --resume-file predictions_2025.parquet \
    --overwrite-months 3  # Recompute last 3 months
```

### What It Does:

**Behavior**:
```
Previous run had predictions through Dec 2025:
  Oct 2025: 14.500%
  Nov 2025: 13.230%
  Dec 2025:  9.772%

New run with --overwrite-months 3:
  Oct 2025: 15.100% ← RECOMPUTED (training + new data)
  Nov 2025: 14.050% ← RECOMPUTED (training + new data)
  Dec 2025: 10.981% ← RECOMPUTED (training + new data)
  Jan 2026: 27.500% ← NEW
```

### Console Output:
```
📂 RESUME MODE
  • Last prediction date: 2025-12-31
  • Overwrite buffer: 3 months
  • Resume from: 2025-09-30
  • Will re-predict from 2025-10-01 onwards

  [193/196] 2025-10: Training + Prediction
  [194/196] 2025-11: Training + Prediction
  [195/196] 2025-12: Training + Prediction
  [196/196] 2026-01: Training + Prediction
```

### Use Case Examples:

#### 1. Data Provider Revision
```bash
# Provider announced Q3 2025 earnings were restated
python forecast_returns_ml_walk_forward.py \
    --input-file data_corrected.csv \
    --output predictions_corrected.parquet \
    --resume-file predictions_old.parquet \
    --overwrite-months 3  # Recompute Oct, Nov, Dec 2025
```

#### 2. New Features Added (e.g., Moving Averages)
```bash
# You added 4 new MA features, want to recompute full year
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_jan.csv \
    --output predictions_with_ma.parquet \
    --resume-file predictions_2025.parquet \
    --overwrite-months 12  # Recompute all of 2025 with new features
    --num-leaves 127 \
    --n-estimators 1000
```

#### 3. Bug Fix in Data Pipeline
```bash
# Fixed a bug that affected last 6 months of LSEG data
python forecast_returns_ml_walk_forward.py \
    --input-file data_fixed.csv \
    --output predictions_fixed.parquet \
    --resume-file predictions_buggy.parquet \
    --overwrite-months 6  # Recompute affected period
```

#### 4. Symbol Mapping Error
```bash
# Fixed ticker changes (FB→META) for past year
python forecast_returns_ml_walk_forward.py \
    --input-file data_remapped.csv \
    --output predictions_remapped.parquet \
    --resume-file predictions_old.parquet \
    --overwrite-months 12  # Recompute full year
```

---

## Why Predictions Change When Adding New Data

Understanding why historical predictions change helps you decide which flag to use.

### Three Sources of Variance:

#### 1. **Stock Universe Changes** (60-70% of variance)
```
First run (Dec 2025):
  Training: Dec 2024 - Nov 2025 (4,440 stocks)

Second run (Jan 2026):
  Training: Dec 2024 - Nov 2025 (4,445 stocks) ← 5 new stocks added!
```

**Why stocks change**:
- IPOs with backfilled fundamentals
- Data provider coverage expansion
- Delisted stocks removed/added

#### 2. **Cross-Sectional Rankings** (30-40% of variance)
```
Stock XYZ: $500B market cap
First run:  Rank 3,774/4,440 = 0.8500 (85.00th percentile)
Second run: Rank 3,768/4,445 = 0.8475 (84.75th percentile) ← Changed!
```

**Why rankings change**:
- Universe size changes (more/fewer stocks in denominator)
- New stocks alter percentile calculations
- Affects 15+ ranking features (CompanyMarketCap_rank, return_20d_rank, etc.)

#### 3. **Data Revisions** (10-20% of variance)
```
AAPL earnings on 2025-11-15:
First run:  $1.64 EPS
Second run: $1.67 EPS ← Provider revised!
```

**Why data changes**:
- Earnings restatements
- Balance sheet corrections
- Corporate action adjustments

### Combined Effect:

Typical prediction drift when adding new data: **1-7%**

**Example**:
```
Dec 2025 prediction:
Run 1 (Dec data):  9.772%
Run 2 (Jan data): 10.981% ← +1.2% drift (within expected range)
Run 3 (Feb data): 11.350% ← +1.6% drift (within expected range)
```

This is **normal behavior** without `--preserve-existing`!

---

## Typical Production Workflow

### Phase 1: Initial Training (One-time)
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data_2009_2025.csv \
    --output predictions_2025.parquet \
    --lookback-months 24 \
    --forecast-days 10 \
    --target-return-days 90 \
    --no-lag \
    --num-leaves 127 \
    --n-estimators 1000
```

**Result**: Predictions for all months from 2009-2025

---

### Phase 2: Weekly Updates (Ongoing - Use `--preserve-existing`)
```bash
# Week 1: Add new week of data (2026-01-07)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_01_week1.csv \
    --output predictions_2026_01_week1.parquet \
    --resume-file predictions_2025.parquet \
    --lookback-months 24 \
    --forecast-days 10 \
    --target-return-days 90 \
    --no-lag \
    --num-leaves 127 \
    --n-estimators 1000 \
    --preserve-existing  # 🔒 Freeze all historical forecasts

# Week 2: Add another week (2026-01-14)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_01_week2.csv \
    --output predictions_2026_01_week2.parquet \
    --resume-file predictions_2026_01_week1.parquet \
    --lookback-months 24 \
    --forecast-days 10 \
    --target-return-days 90 \
    --no-lag \
    --num-leaves 127 \
    --n-estimators 1000 \
    --preserve-existing  # 🔒 Freeze all historical forecasts

# Week 3: Add another week (2026-01-21)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_01_week3.csv \
    --output predictions_2026_01_week3.parquet \
    --resume-file predictions_2026_01_week2.parquet \
    --lookback-months 24 \
    --forecast-days 10 \
    --target-return-days 90 \
    --no-lag \
    --num-leaves 127 \
    --n-estimators 1000 \
    --preserve-existing  # 🔒 Freeze all historical forecasts

# Week 4: End of month (2026-01-31)
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_01_week4.csv \
    --output predictions_2026_01_final.parquet \
    --resume-file predictions_2026_01_week3.parquet \
    --lookback-months 24 \
    --forecast-days 10 \
    --target-return-days 90 \
    --no-lag \
    --num-leaves 127 \
    --n-estimators 1000 \
    --preserve-existing  # 🔒 Freeze all historical forecasts
```

**Key Points**:
- ✅ Historical predictions (2009-2025) **never change** across all 4 weeks
- ✅ Only new weeks get predictions
- ✅ Stable backtests throughout the month
- ✅ Performance metrics remain consistent

---

### Phase 3: Model Updates (Occasional - Use `--overwrite-months`)

#### Scenario A: Added New Features
```bash
# You added 4 new MA features, want to recompute 2025
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_02.csv \
    --output predictions_2026_02_with_ma.parquet \
    --resume-file predictions_2026_01_final.parquet \
    --lookback-months 24 \
    --forecast-days 10 \
    --target-return-days 90 \
    --no-lag \
    --num-leaves 127 \
    --n-estimators 1000 \
    --overwrite-months 12  # Recompute all of 2025 + Jan 2026
```

#### Scenario B: Data Provider Revision
```bash
# LSEG revised Q4 2025 fundamentals
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_03_corrected.csv \
    --output predictions_2026_03.parquet \
    --resume-file predictions_2026_02_with_ma.parquet \
    --lookback-months 24 \
    --forecast-days 10 \
    --target-return-days 90 \
    --no-lag \
    --num-leaves 127 \
    --n-estimators 1000 \
    --overwrite-months 4  # Recompute Q4 2025 (Oct, Nov, Dec) + Jan, Feb, Mar 2026
```

#### Scenario C: Bug Fix
```bash
# Fixed symbol mapping error affecting last 6 months
python forecast_returns_ml_walk_forward.py \
    --input-file data_2026_04_fixed.csv \
    --output predictions_2026_04.parquet \
    --resume-file predictions_2026_03.parquet \
    --lookback-months 24 \
    --forecast-days 10 \
    --target-return-days 90 \
    --no-lag \
    --num-leaves 127 \
    --n-estimators 1000 \
    --overwrite-months 6  # Recompute affected 6 months
```

---

## Decision Tree: Which Flag Should I Use?

```
Are you adding NEW data to your input file?
│
├─ YES → Is this a normal weekly/monthly update?
│   │
│   ├─ YES → Use --preserve-existing ✅
│   │         (Keep historical forecasts stable)
│   │
│   └─ NO → Did something change?
│       │
│       ├─ Added new features → Use --overwrite-months N
│       │                       (Recompute with new features)
│       │
│       ├─ Data provider revised → Use --overwrite-months N
│       │                          (Correct affected period)
│       │
│       ├─ Bug fix in pipeline → Use --overwrite-months N
│       │                        (Fix affected period)
│       │
│       └─ Changed model params → Use --overwrite-months N
│                                 (See impact on recent predictions)
│
└─ NO → Are you just re-running the same data?
    │
    ├─ Testing/debugging → No resume flags needed
    │                      (Fresh run)
    │
    └─ Validating results → Use --preserve-existing
                           (Ensure reproducibility)
```

---

## Common Mistakes to Avoid

### Mistake 1: Using `--overwrite-months` for Weekly Updates
```bash
# ❌ WRONG: This will cause prediction drift every week
python forecast_returns_ml_walk_forward.py \
    --input-file data_week2.csv \
    --resume-file predictions_week1.parquet \
    --overwrite-months 1  # Last month keeps changing!

# ✅ CORRECT: Use --preserve-existing for stability
python forecast_returns_ml_walk_forward.py \
    --input-file data_week2.csv \
    --resume-file predictions_week1.parquet \
    --preserve-existing  # Historical forecasts frozen
```

### Mistake 2: Forgetting to Add `--overwrite-months` After Model Changes
```bash
# You added 4 new MA features...

# ❌ WRONG: New features only affect future predictions
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --resume-file old_predictions.parquet \
    --preserve-existing  # Historical predictions use OLD model without MA features!

# ✅ CORRECT: Recompute to include new features
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --resume-file old_predictions.parquet \
    --overwrite-months 12  # Recompute last year with NEW MA features
```

### Mistake 3: Not Realizing Lookback Window Changes Predictions
```bash
# Original run used 12-month lookback
# File: predictions_rolling12.parquet

# ❌ WRONG: Changing lookback will cause ALL predictions to change
python forecast_returns_ml_walk_forward.py \
    --resume-file predictions_rolling12.parquet \
    --lookback-months 24  # Different training window!
    --preserve-existing   # Won't help - training window changed!

# ✅ CORRECT: Keep same lookback for comparable predictions
python forecast_returns_ml_walk_forward.py \
    --resume-file predictions_rolling12.parquet \
    --lookback-months 12  # Match original
    --preserve-existing
```

### Mistake 4: Using `--overwrite-months` Too Small After Bug Fix
```bash
# Bug affected 6 months of data (Jul-Dec 2025)

# ❌ WRONG: Only recomputes 3 months, leaves 3 months with buggy data
python forecast_returns_ml_walk_forward.py \
    --input-file data_fixed.csv \
    --resume-file predictions_buggy.parquet \
    --overwrite-months 3  # Oct, Nov, Dec only

# ✅ CORRECT: Recompute full affected period
python forecast_returns_ml_walk_forward.py \
    --input-file data_fixed.csv \
    --resume-file predictions_buggy.parquet \
    --overwrite-months 6  # Jul, Aug, Sep, Oct, Nov, Dec
```

---

## Performance Impact

### `--preserve-existing` (Fast ⚡)
```
Previous predictions: 195 months
New data: 1 month

With --preserve-existing:
  • Skip 195 months (already predicted)
  • Train only 1 new month
  • Total time: ~30 seconds - 2 minutes

Speed: 95%+ faster than full retraining
```

### `--overwrite-months 12` (Medium ⏱️)
```
Previous predictions: 195 months
Overwrite: 12 months

With --overwrite-months 12:
  • Skip 183 months (keep existing)
  • Retrain 12 months
  • Total time: ~15-30 minutes

Speed: 90%+ faster than full retraining
```

### No Resume (Slow 🐌)
```
Full training: 196 months

Without resume:
  • Train all 196 months from scratch
  • Total time: ~2-4 hours

Speed: Baseline (full retraining)
```

---

## Summary

### For Production (Most Common):
```bash
# Use this for 95% of your weekly/monthly updates
python forecast_returns_ml_walk_forward.py \
    --input-file data_latest.csv \
    --output predictions_latest.parquet \
    --resume-file predictions_previous.parquet \
    --preserve-existing  # 🔒 FREEZE historical forecasts
```

### For Model Updates (Occasional):
```bash
# Use this when you add features, fix bugs, or handle data revisions
python forecast_returns_ml_walk_forward.py \
    --input-file data_latest.csv \
    --output predictions_latest.parquet \
    --resume-file predictions_previous.parquet \
    --overwrite-months N  # Recompute N months (3, 6, 12, etc.)
```

---

## Related Documentation

- **[USAGE.md](USAGE.md)** - Complete command-line reference
- **[Docs/FORECAST_STABILITY.md](Docs/FORECAST_STABILITY.md)** - Technical deep dive on prediction variance
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and v3.3.14 --preserve-existing release
- **[README.md](README.md)** - Feature overview and production guide

---

**Document Version**: 1.0
**Last Updated**: 2026-01-26
**Related Version**: v3.3.14+ (--preserve-existing flag introduced)
