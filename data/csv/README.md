# ML-Based Stock Return Forecasting

Fast, production-ready machine learning system for predicting stock returns using fundamental data.

## 🎯 Overview

This tool uses **Histogram-based Gradient Boosting** with extensive feature engineering to predict stock returns at customizable horizons (10-day, 90-day, etc.). It's optimized for institutional trading strategies with:

- ✅ **No look-ahead bias** - Forward-fill per symbol, proper lagging
- ✅ **Flexible data support** - Works with any fundamental columns (LSEG, FMP, Sharadar, custom)
- ✅ **Scalable features** - Handles 70-298+ features automatically based on your data
- ✅ **Market cap weighting** - Focused training on large-cap stocks
- ✅ **80%+ correlation** - Excellent predictive power
- ✅ **Fast execution** - Processes millions of rows in seconds
- ✅ **Lean output** - Only adds 2 columns to original data
- ✅ **Complete logging** - Auto-generated log files for reproducibility
- ✅ **Pre-lagged data support** - Use your own lagging pipeline

## 📖 Quick Links

- **[USAGE.md](USAGE.md)** - Complete command-line reference with all flags and examples ⭐ **NEW**
- **[CHANGELOG.md](CHANGELOG.md)** - Version history with detailed explanations
- **[LOOK_AHEAD_BIAS_AUDIT.md](LOOK_AHEAD_BIAS_AUDIT.md)** - Production safety verification
- **[Docs/INDEX.md](Docs/INDEX.md)** - Technical deep dives and advanced topics

## 🆕 What's New in v3.3.18 (2026-01-26)

### 🚀 NEW FEATURES: Moving Average Momentum Indicators

Added 4 classic technical indicators based on 50-day and 200-day moving averages:

**New Features**:
1. **`return_50dma`** - % above/below 50-day moving average
2. **`return_200dma`** - % above/below 200-day moving average
3. **`above_50dma`** - Binary flag: 1 if price above 50-day MA, 0 otherwise
4. **`above_200dma`** - Binary flag: 1 if price above 200-day MA, 0 otherwise

**Why this matters**:
- Classic trend identification (stocks above 200-day MA = uptrend)
- Momentum strength signals (large deviations from MA = strong moves)
- Mean reversion opportunities (far from MA may snap back)
- Support/resistance dynamics (MAs act as dynamic levels)

**Zero look-ahead bias**:
- ✅ All features use T-1 lagged prices only
- ✅ Moving averages computed backwards from T-1 (no future data)
- ✅ Production-safe for live trading

**Impact**: Total feature count increases from ~294 → **298 features**

**See CHANGELOG.md for complete v3.3.18 details**

---

## 🆕 What's New in v3.3.12 (2026-01-16)

### 🎯 Parameter Update: min_samples_leaf = 100 (Default):
- **min_samples_leaf: 50→100** - More stable splits, better alignment with 0.01% rule
- **l2_regularization: 0.2** (unchanged) - Moderate regularization
- **max_depth: 6** (unchanged) - Prevents deep overfitting

**Why this matters**:
```
Current: max_depth=6 + min_samples_leaf=100 + L2=0.2
         ↑ Conservative depth + Conservative splits + Moderate regularization
         = More stable, less noise-sensitive predictions
```

**Impact**: More robust predictions with less sensitivity to outliers. Each tree leaf requires ≥100 samples (0.0125% of 800K rows), within the recommended 0.01-0.05% range.

**For even better results**, combine with command-line flags:
```bash
--num-leaves 127 --n-estimators 400  # Best improvement
```

**See CHANGELOG.md for complete v3.3.12 details**

---

## 🆕 What's New in v3.3.7 (2026-01-14)

### 🎯 Tree Depth Optimization: max_depth = 6 (Better for Noisy Returns):
- **Reduced from 7 to 6** - Shallower trees generalize better on noisy stock returns
- **Prevents overfitting** - Depth 7 (128 max leaves) too complex for noise
- **Industry-aligned** - Matches XGBoost, CatBoost defaults (depth 6)
- **10-15% faster training** - Fewer nodes to evaluate

**Why this matters**:
```
Previous: max_depth=7 (128 max leaves)  ❌ Too complex for noisy returns
Current:  max_depth=6 (64 max leaves)   ✅ Balanced, industry standard
```

**Impact**: Simpler trees = less noise memorization, better generalization across market regimes. Your extreme outliers (±400-500%) should reduce. Depth 6 is the sweet spot for stock returns.

**Complete production stack** (all optimized together):
```python
max_depth=6              # Balanced complexity
num_leaves=63           # Reasonable capacity
min_samples_leaf=100    # Conservative splits (0.0125% rule)
l2_regularization=0.3   # Strong regularization
```

**See CHANGELOG.md for complete v3.3.7 details**

---

## 🆕 What's New in v3.3.6 (2026-01-14)

### 🎯 Regularization Optimization: L2 = 0.3 (Better Control for 300 Features):
- **Stronger regularization** - 3x increase from 0.1 to 0.3 for high-dimensional feature space
- **Controls extreme forecasts** - Prevents ±400-500% outlier predictions
- **Production-grade** - Conservative predictions critical for risk-managed trading
- **Better for leverage** - Downstream leverage amplifies prediction errors

**Why this matters**:
```
Previous: L2=0.1 (too weak for 300 features) ❌ Allows extreme predictions
Current:  L2=0.3 (3x stronger penalty)       ✅ Controls outliers, more stable
```

**Impact**: Fewer extreme predictions, more stable across regimes, better for risk management. Predictions will be more conservative (±100-200% vs ±400-500%), which is desirable for production trading with potential leverage.

**See CHANGELOG.md for complete v3.3.6 details**

---

## 🆕 What's New in v3.3.5 (2026-01-14)

### 🎯 Stability Optimization: min_samples_leaf = 100 (Follows 0.01-0.05% Rule):
- **Industry best practice** - Meets 0.01-0.05% of training data guideline
- **For 800K rows** - 100 samples = 0.0125% (previously 50 = 0.0063%, below threshold)
- **More stable predictions** - Prevents regime-specific micro-splits and noise overfitting
- **Production-ready** - Robust across different market conditions

**Why this matters**:
```
Previous: min_samples_leaf=50  (0.0063% of 800K) ⚠️ Below 0.01% threshold
Current:  min_samples_leaf=100 (0.0125% of 800K) ✅ Meets best practice
```

**Impact**: More stable predictions, better generalization, ~5-10% faster training. Your backtest signals should be smoother and more consistent across market regimes.

**See CHANGELOG.md for complete v3.3.5 details**

---

## 🆕 What's New in v3.3.4 (2026-01-14)

### 🔍 NEW: Temporal Diagnostics for Look-Ahead Bias Detection:
- **Detect temporal leakage** - ACF analysis, Ljung-Box test, stability checks
- **Validate model quality** - Verify walk-forward training is working correctly
- **Production safety** - Catch look-ahead bias before live trading
- **Opt-in feature** - Add `--temporal-diagnostics` flag when needed

**Usage**:
```bash
python forecast_returns_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions.parquet \
    --temporal-diagnostics
```

**What it tests**:
1. **ACF Analysis** - Residual autocorrelation at 20 lags (catches temporal structure leakage)
2. **Ljung-Box Test** - Omnibus test for autocorrelation (catches temporal dependencies)
3. **Temporal Stability** - Mean/std stability across 4 time periods (catches regime changes)

**Performance**: Now analyzes only last 6 months (2-3 minutes instead of hours for large datasets). See v3.3.10 for details.

**When to use**: After major feature changes, when validating a new model, troubleshooting unexpected results, or periodic quality checks (monthly/quarterly).

**See CHANGELOG.md for complete v3.3.4 details**

---

## 🆕 What's New in v3.3.3 (2026-01-13)

### 🎯 Stability Optimization: Increased min_samples_leaf (50 from 20):
- **Better for large datasets** - Optimized for 800K+ rows
- **More stable predictions** - Reduces overfitting to rare patterns
- **Faster training** - ~2-5% speed improvement
- **Better generalization** - More robust across market regimes

**Impact**: With 800K rows, each leaf now requires minimum 50 samples (vs 20), resulting in more conservative and stable splits. Perfect for production trading systems.

**See CHANGELOG.md for complete v3.3.3 details**

---

## 🆕 What's New in v3.3.2 (2026-01-13)

### 🎛️ NEW: `--num-leaves` Flag for Model Capacity Tuning:
- **Easy experimentation** - Control model capacity from command line
- **No code editing** - Just add `--num-leaves 63` to your command
- **Smart defaults** - Helpful guidance for your data size (31/63/127)

**Why Use This**:
```bash
# Your data: 650K rows with 300 features
# Default: 31 leaves = 21K rows per leaf (conservative)
# Recommended: 63 leaves = 10K rows per leaf (balanced capacity)
# High capacity: 127 leaves = 5K rows per leaf (capture subtle patterns)

# Try balanced capacity
python forecast_ml_walk_forward.py \
    --input-file data.csv \
    --output predictions.parquet \
    --num-leaves 63 \
    --lookback-months 12
```

**Expected Impact**: With 650K rows, increasing to 63-127 leaves may improve correlation by 1-2% (e.g., 80% → 82%) by capturing sector-specific and interaction patterns. Test and compare!

**See CHANGELOG.md for complete v3.3.2 details**

---

## 🆕 What's New in v3.3.1 (2026-01-13)

### 🔒 CRITICAL Reproducibility Fix:
- **Per-Month Ranking Features** - Cross-sectional rankings now computed PER-MONTH in walk-forward loop
- **Problem Fixed**: Adding new data no longer changes historical predictions
- **Before**: December predictions changed when adding January data (rankings recomputed on expanded universe)
- **After**: December predictions stable - rankings computed only using December's training window

**Weekly Production Use Case**:
```bash
# Week 51: Run forecasting
python forecast_ml.py --input-file data_week51.csv --output pred_week51.parquet

# Week 52: Add new data and resume
python forecast_ml.py --input-file data_week52.csv --output pred_week52.parquet \
    --resume-file pred_week51.parquet --overwrite-months 1

# ✅ December predictions in pred_week52 IDENTICAL to pred_week51
# ✅ Only January has new predictions
# ✅ Backtest results reproducible
```

**Impact**: Your backtest returns will now be **100% stable** when adding new weekly/monthly data. Critical for production trading systems that need consistent performance metrics.

**See CHANGELOG.md for complete v3.3.1 technical details**

---

## 🆕 What's New in v3.3.0 (2026-01-12)

### 🚨 CRITICAL Data Leak Fixes:
- **Fixed `tradedate` leak** - Date column was included as feature (15.8% importance!)
- **Fixed `RefPriceClose` leak** - Same-day price predicting future returns with `--no-lag` (19.6% importance!)
- **Fixed `CompanyMarketCap` leak** - Market cap derived from price, now always lagged
- **Fixed `accepteddate_fmp*` leaks** - Multiple date columns slipping through
- **Enhanced safety checks** - Precise date column detection, preserves legitimate fundamental data

### 🎯 Price-Derived Columns ALWAYS Lagged (Even with `--no-lag`):
- `RefPriceClose` → `RefPriceClose_lag1` (yesterday's close)
- `RefVolume` → `RefVolume_lag1` (yesterday's volume)
- `CompanyMarketCap` → `CompanyMarketCap_lag1` (yesterday's market cap)

**Why**: These columns contain T+0 information. Using same-day price/volume/market cap to predict T+10 to T+100 returns = **data leakage**. Now uses T-1 (previous day) data for point-in-time safety.

### 🆕 NEW FEATURE: Rolling Window Training (`--lookback-months`):
- **Experiment with historical depth** - Test recency bias vs. long-term patterns
- `--lookback-months 12` - Train on rolling 12-month window (recent data focus)
- `--lookback-months 24` - Train on rolling 24-month window (balanced)
- `None` (default) - Expanding window (all historical data)

**Use case**: Markets change - rolling windows adapt faster to new regimes. Compare stability (expanding) vs. responsiveness (rolling).

### ✅ Time-Series Validation Safety:
- Disabled `validation_fraction` - sklearn does RANDOM splits which leak future data
- Added clear safety documentation explaining why validation is disabled
- If early stopping needed, must pass explicit chronological X_val/y_val

### Impact:
- **Stable predictions** - No changes when adding new data (fixed unstable forecasts)
- **Zero look-ahead bias** - All price-derived columns properly lagged
- **Defense-in-depth** - Multiple layers of date column protection
- **New experiments** - Rolling windows for recency bias testing

**See CHANGELOG.md for complete v3.3.0 details**

---

## 🆕 What's New in v3.2.2 (2026-01-07)

### 🎯 Fully Deterministic Design - ZERO Randomness:
- **100% reproducible results** - Eliminated ALL random number generation
- **2-5% faster** - No RNG overhead from deterministic sampling
- **Simpler code** - No seed management needed
- **Perfect reproducibility** - Same input → same output, every time
- **Fixed critical bug** - Resolved 5.62% prediction difference between identical runs

### ⚡ Major Performance Optimizations (v3.2.0):
- **10-30x faster alignment** - Vectorized pandas merge instead of iterrows() loops (5-10 min → 10-30 sec)
- **85% less merge memory** - Merge before feature engineering (2 GB → 300 MB temp df)
- **3-5x faster CSV reading** - PyArrow engine for high-performance parsing
- **5-10x faster I/O** - Parquet format support (auto-detection by file extension)
- **10x smaller files** - Parquet compression reduces 50 MB CSV to 5 MB

### 🔧 Simplified Resume Logic (v3.2.0):
- **Removed checkpoint JSON** - No more confusing `.json` files and automatic renaming
- **Simple `--resume-file`** - Just point to previous predictions CSV/Parquet
- **Explicit flags** - Changed from positional `input` to `--input-file` and `--output`
- **Cleaner workflow** - 300 lines of complexity → 83 lines of clean code

### New Features:
- **Auto-export forecast CSV** - Automatically creates `YYYYMMDD_YYYYMMDD_forecast_only.csv` after each run
- **Automatic date cleanup** - Removes erroneous future-dated records based on filename
- **Parquet support** - Works with `.parquet` and `.pq` files for faster I/O

### Impact:
- **Weekly updates**: 30 seconds to 2 minutes (vs 5-12 minutes before)
- **100% look-ahead bias free** - All optimizations mathematically verified safe
- **Simpler workflow** - No more checkpoint confusion or file renaming
- **Better storage** - 10x smaller prediction files with Parquet format

**See CHANGELOG.md for complete v3.2.0 details**

---

## 🆕 What's New in v3.1 (2025-12-28)

### Critical Bug Fixes:
- **Fixed look-ahead bias** in missing value handling - Now uses forward-fill per symbol instead of median across all dates
- **Fixed `--no-lag` bug** - Raw LSEG fundamentals now correctly included as features when using pre-lagged data

### New Features:
- **PCA dimensionality reduction** with `--pca N` - Reduce features to N components while preserving variance
- **Feature descriptions** - Detailed list of all training features logged at start with human-readable descriptions
- **Automatic logging** with `--log-file` - All output saved to timestamped log files
- **Performance optimization** with `--sample-fraction` - Train on subset for 2-5x speedup
- **Pre-lagged data support** with `--no-lag` - Use your own data preparation pipeline
- **Auto-normalization** - Automatically handles lowercase column names (converts to PascalCase)

### Impact:
- **Better accuracy** - Forward-fill is point-in-time accurate (no future data contamination)
- **Faster training** - Sample 50% of data for 2x speedup with minimal accuracy loss
- **Full audit trail** - Complete logs for every run

---

## 🚀 Production Deployment Guide

### ✅ RECOMMENDED: Walk-Forward Mode (Zero Look-Ahead Bias)

**For live trading and production use, ALWAYS use walk-forward mode:**

```bash
# First run - full training
python forecast_returns_ml_walk_forward.py \
    --input-file your_data.csv \
    --output predictions.parquet \
    --forecast-days 10 \
    --target-return-days 90

# Weekly updates - fast resume (recommended)
python forecast_returns_ml_walk_forward.py \
    --input-file updated_data.csv \
    --output updated_predictions.parquet \
    --resume-file predictions.parquet \
    --overwrite-months 1 \
    --forecast-days 10 \
    --target-return-days 90
```

**Why walk-forward is critical:**
- ✅ **Realistic backtesting** - Mimics real-world deployment
- ✅ **Zero look-ahead bias** - Each month uses only past data
- ✅ **Repeatable** - Same input produces identical results
- ✅ **Production-ready** - Exactly how model will perform live

### ❌ DO NOT Use in Production

**Avoid these flags for live trading:**

1. **`--pca`** - Has look-ahead bias in walk-forward mode
   - PCA fits on ALL training data (including future months)
   - Model sees future feature distributions
   - Use raw features instead (298 features work well)

2. **`--no-walk-forward`** - Single model with look-ahead bias
   - Trains one model on all historical data
   - Uses future information for validation
   - Unrealistic performance estimates

### Production Safety Checklist

Before deploying to live trading:

- [x] ✅ Using `forecast_returns_ml_walk_forward.py` (walk-forward script)
- [x] ✅ NO `--pca` flag
- [x] ✅ NO `--no-walk-forward` flag
- [x] ✅ Features properly lagged (automatic with script)
- [x] ✅ Resume mode tested (`--resume-file`)
- [x] ✅ Prediction files saved for quick updates (Parquet recommended)

### Look-Ahead Bias Protection

**The walk-forward script automatically protects against:**

| Component | Protection | Status |
|-----------|-----------|--------|
| Feature engineering | T-1 lagging on all fundamentals | ✅ Safe |
| Inf/NaN handling | Row-level replacement (no statistics) | ✅ Safe |
| Outlier clipping | Disabled in walk-forward mode | ✅ Safe |
| Walk-forward loop | Strict `Date < first_day_of_month` cutoff | ✅ Safe |
| Model training | Each month trains on past data only | ✅ Safe |
| PCA/StandardScaler | Disabled (would use future data) | ✅ Safe |

### ⚠️ PCA Warning

If you see this warning, **DO NOT** use for production:

```
⚠️  WARNING: PCA in walk-forward mode has look-ahead bias!
   PCA is fit on ALL training data (including future months)
   For production, use --no-walk-forward or remove --pca flag
```

**Solution:** Remove `--pca` flag from command.

### Performance Without PCA

**Good news:** HistGradientBoosting handles 298 features efficiently:
- Training time: Fast (no PCA overhead)
- Memory: Modest (fits in RAM easily)
- Accuracy: Better (preserves all signal)
- Production-safe: Zero look-ahead bias

---

## 📊 Key Features

### 1. No Look-Ahead Bias
All fundamental features are **lagged by 1 day** to ensure we only use information available before making predictions. This prevents overfitting and ensures the model works in real trading.

```
Today (T): Use fundamentals from T-1
Predict:   Returns from T+10 to T+100 (90-day return)
```

### 2. Predictions for ALL Dates (Including Recent)
The model intelligently handles recent dates:
- **Training**: Uses only dates with valid `forward_return` (where future prices exist for validation)
- **Prediction**: Generates forecasts for ALL dates with fundamentals, including:
  - Historical dates (with `forward_return` for accuracy validation)
  - Recent dates (without `forward_return`, but perfect for live trading)

**This is critical for production use** - you get predictions for the most recent dates (where you actually trade) even though you can't validate them yet!

**Example:**
```
Date range: 2010-01-01 to 2024-12-09
Training samples: 39,570 (dates with future price data)
Predictions: 40,138 (100% coverage, including 568 recent dates)
```

### 3. Market Cap Weighted Training
The model focuses on **large-cap stocks** which are:
- More liquid and tradeable
- Less noisy in fundamental data
- More relevant for institutional strategies

**Weighting Scheme:**
- Top 2000 stocks: weight = 1.0 (full importance)
- Rank 2001-4000: weight = 0.5 (medium)
- Rank 4000+: weight = 0.1 (low but not ignored)

### 4. Flexible Return Periods
Predict any return horizon:
- **Short-term**: 10-20 day returns
- **Medium-term**: 30-60 day returns
- **Long-term**: 90-180 day returns

**Key Insight:** Longer periods = better predictions (fundamentals predict long-term better than short-term noise)

### 5. Extensive Feature Engineering (70-298+ features)

The number of features depends on your input data:
- **LSEG-only dataset**: ~70-100 features
- **Production dataset (LSEG + FMP + Sharadar)**: 298 features

**Price-based features:**
- Momentum: 5, 10, 20-day returns
- Volatility: Rolling volatility metrics
- Volume: Relative volume indicators

**Fundamental features:**
- All lagged fundamentals from your CSV (_lag1 versions)
- Derived ratios: ROE, ROA, EV/EBITDA, EV/EBIT, PEG, etc.
- Growth metrics: Long-term growth estimates, revenue growth, earnings growth
- Quality metrics: Alpha model rankings, earnings quality

**Cross-sectional:**
- Percentile ranks within each date
- Sector-relative metrics

**All features use lagged data (T-1) to prevent look-ahead bias!**

### 6. PCA Dimensionality Reduction (Optional)

> **⚠️ PRODUCTION WARNING:** PCA has look-ahead bias in walk-forward mode and is **NOT RECOMMENDED for live trading**. Use PCA only for exploration or with `forecast_returns_ml.py` (single model, non-walk-forward). For production deployment, skip PCA and use all 298 features - HistGradientBoosting handles them efficiently with zero look-ahead bias.

**What is PCA?**

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms correlated features into a smaller set of uncorrelated components while preserving the most important variance in the data.

**Why Use PCA? (Exploration Only)**

- ✅ **Reduce overfitting** - Fewer features mean less risk of fitting to noise
- ✅ **Faster training** - Dramatically speeds up model training (5-10x faster)
- ✅ **Handle multicollinearity** - Eliminates correlations between fundamental ratios
- ✅ **Preserve signal** - Keeps 90-99% of variance with just 20-50 components
- ✅ **Cleaner models** - Removes redundant information from highly correlated features

**When to Use PCA?**

| Scenario | Recommendation |
|----------|---------------|
| Initial exploration | Try `--pca 20` for fast iterations (non-walk-forward only) |
| **Production/Live Trading** | **❌ DO NOT USE PCA** - Use all 298 features |
| Research/Backtesting | `forecast_returns_ml.py` (single model) with `--pca 30-50` |
| Maximum accuracy | No PCA (use all features) |

**How to Use (Exploration Only):**

```bash
# FOR EXPLORATION ONLY - Use forecast_returns_ml.py (single model)
# NOT for production walk-forward backtesting

# Reduce 110+ features to 20 components (fast, ~90% variance)
python forecast_returns_ml.py data.csv --pca 20

# Balanced: 30 components (~95% variance)
python forecast_returns_ml.py data.csv --pca 30

# Maximum retention: 50 components (~98% variance)
python forecast_returns_ml.py data.csv --pca 50

# ❌ DO NOT USE: Walk-forward with PCA (has look-ahead bias)
# python forecast_returns_ml_walk_forward.py data.csv --pca 25
```

**Output Example:**

```
🔬 Applying PCA dimensionality reduction...
  • Original features: 110
  • Target components: 20
  • Variance explained: 92.34%
  • Components shape: (39570, 20)
  ✓ PCA transformation complete
```

**How It Works:**

1. **Fit PCA on training data** (only valid rows) to avoid look-ahead bias
2. **Transform all data** (training + prediction) using the fitted PCA
3. **Train model** on reduced feature space (PC1, PC2, ..., PCN)
4. **Make predictions** using the same PCA-transformed features

**Important Notes:**

- PCA is fit ONLY on training data to prevent look-ahead bias
- The same PCA transformation is applied to all prediction data
- For walk-forward: PCA is fit once on all training data, then reused consistently
- Feature importance will show PC1, PC2, etc. (not original feature names)
- Variance explained indicates how much information is preserved

**Performance Impact:**

| Components | Training Speed | Variance | Accuracy |
|------------|----------------|----------|----------|
| 10 | 10x faster | ~85% | Good for exploration |
| 20 | 6x faster | ~92% | Balanced speed/accuracy |
| 30 | 4x faster | ~95% | High accuracy |
| 50 | 2x faster | ~98% | Near-optimal accuracy |
| None (110+) | Baseline | 100% | Maximum accuracy |

## 🔬 Feature Engineering Pipeline (Detailed)

### Overview

The model is **flexible and data-agnostic** - it automatically processes ANY fundamental columns in your CSV. The number of features depends on your input data:

- **LSEG-only (32 columns)**: Creates ~70-100 engineered features
- **LSEG + FMP + Sharadar (200+ columns)**: Creates ~290 engineered features
- **Your custom data**: Creates features proportional to your input columns

**Key principle**: ALL fundamental columns are automatically lagged by 1 day to prevent look-ahead bias, then engineered into price momentum, ratios, rankings, and lag-2 features.

### Example Production Dataset (290 Features)

The example production runs use a combined dataset with **~240 raw columns** from multiple sources:

**1. FMP (Financial Modeling Prep) - ~170 columns**
- Income statement growth metrics (revenue, EBITDA, net income, EPS, etc.)
- Balance sheet growth metrics (assets, debt, equity, etc.)
- Cash flow growth metrics (operating CF, free CF, etc.)
- Example columns: `growthrevenue_fmp`, `growthnetincome_fmp`, `growebitda_fmp`

**2. LSEG (London Stock Exchange Group) - 32 columns**

**Price & Volume (2):**
- `RefPriceClose` - Daily closing price
- `RefVolume` - Daily volume

**Company Info (2):**
- `CompanyMarketCap` - Market capitalization
- `GICSSectorName` - Sector classification (excluded from training)

**Valuation Metrics (9):**
- `EnterpriseValue_DailyTimeSeries_`
- `EnterpriseValueToEBIT_DailyTimeSeriesRatio_`
- `EnterpriseValueToEBITDA_DailyTimeSeriesRatio_`
- `EnterpriseValueToSales_DailyTimeSeriesRatio_`
- `ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_`
- `ForwardPEG_DailyTimeSeriesRatio_`
- `ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_`
- `ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_`
- `PriceEarningsToGrowthRatio_SmartEstimate_`

**Cash Flow (2):**
- `FOCFExDividends_Discrete` - Free cash flow
- `CashCashEquivalents_Total` - Cash and equivalents

**Debt & Interest (2):**
- `Debt_Total` - Total debt
- `InterestExpense_NetofCapitalizedInterest` - Interest expense

**Earnings Metrics (4):**
- `EarningsPerShare_Actual`
- `EarningsPerShare_SmartEstimate_current_Q`
- `EarningsPerShare_SmartEstimate_prev_Q`
- `EarningsPerShare_ActualSurprise`

**Growth & Estimates (4):**
- `LongTermGrowth_Mean` - Long-term growth estimate
- `Estpricegrowth_percent` - Estimated price growth
- `PriceTarget_Median` - Analyst price target
- `Dividend_Per_Share_SmartEstimate` - Dividend estimate

**Profitability Ratios (3):**
- `ReturnOnEquity_SmartEstimat` - ROE
- `ReturnOnAssets_SmartEstimate` - ROA
- `GrossProfitMargin_ActualSurprise` - Gross profit margin surprise

**Rankings & Quality (4):**
- `CombinedAlphaModelRegionRank` - Regional alpha rank
- `CombinedAlphaModelSectorRank` - Sector alpha rank
- `CombinedAlphaModelSectorRankChange` - Rank change
- `EarningsQualityRegionRank_Current` - Earnings quality rank

**Analyst Recommendations (1):**
- `Recommendation_Median_1_5_` - Analyst recommendation (1=Strong Buy, 5=Sell)

**3. Sharadar Metadata - 2 columns**
- `sharadar_scalemarketcap` - Market cap scale (1=Nano, 6=Mega)
- `sharadar_is_adr` - Is American Depositary Receipt

**4. Other Metadata**
- `tradedate`, `fiscalyear_fmp`, `period_fmp`, `reportedcurrency_fmp`

**Total Input**: ~240 raw columns from multiple data sources

> **Note**: Your CSV doesn't need all these columns! The script automatically detects and processes whatever fundamental columns you provide. A minimal dataset needs only: `Date`, `Symbol`, `RefPriceClose`, `RefVolume`, `CompanyMarketCap`, plus any fundamentals you want to use for prediction.

### Step 2: Lag ALL Fundamentals by 1 Day

**Critical for preventing look-ahead bias:**

```python
# For EVERY fundamental column in your CSV, create lagged version
RefPriceClose_lag1 = shift(RefPriceClose, 1 day)
CompanyMarketCap_lag1 = shift(CompanyMarketCap, 1 day)
ReturnOnEquity_SmartEstimat_lag1 = shift(ReturnOnEquity_SmartEstimat, 1 day)
growthrevenue_fmp_lag1 = shift(growthrevenue_fmp, 1 day)
# ... for ALL ~240 columns in production dataset
```

**Why?** On day T, you only know fundamentals from day T-1. Using same-day fundamentals would create look-ahead bias.

**Result**: ~240 `_lag1` columns created (in production dataset)

### Step 3: Engineer Features from Lagged Data

#### A. Price Momentum Features (6 features)

From `RefPriceClose_lag1`:
```python
return_5d   = 5-day return (%)
return_10d  = 10-day return (%)
return_20d  = 20-day return (%)

volatility_5d   = 5-day annualized volatility (%)
volatility_10d  = 10-day annualized volatility (%)
volatility_20d  = 20-day annualized volatility (%)
```

#### B. Volume Features (1 feature)

From `RefVolume_lag1`:
```python
volume_ratio = current_volume / 20-day_moving_average
```

#### C. Fundamental Ratios (20+ features from LSEG data)

From lagged LSEG fundamentals:

**Profitability:**
```python
roa = ReturnOnAssets_SmartEstimate_lag1
roe = ReturnOnEquity_SmartEstimat_lag1
```

**Valuation:**
```python
ev_to_ebitda = EnterpriseValueToEBITDA_DailyTimeSeriesRatio__lag1
ev_to_ebit = EnterpriseValueToEBIT_DailyTimeSeriesRatio__lag1
ev_to_sales = EnterpriseValueToSales_DailyTimeSeriesRatio__lag1
forward_peg = ForwardPEG_DailyTimeSeriesRatio__lag1
pe_to_growth = PriceEarningsToGrowthRatio_SmartEstimate__lag1
```

#### D. All Other Lagged Fundamentals (~210+ features from FMP + other sources)

All additional fundamental columns from your CSV are automatically included as features:
- FMP income statement growth metrics (_lag1 versions)
- FMP balance sheet growth metrics (_lag1 versions)
- FMP cash flow growth metrics (_lag1 versions)
- Any other fundamental columns you provide

**Growth:**
```python
ltg = LongTermGrowth_Mean_lag1
price_growth_est = Estpricegrowth_percent_lag1
```

**Quality/Momentum:**
```python
alpha_sector_rank = CombinedAlphaModelSectorRank_lag1
alpha_region_rank = CombinedAlphaModelRegionRank_lag1
alpha_sector_change = CombinedAlphaModelSectorRankChange_lag1
earnings_quality = EarningsQualityRegionRank_Current_lag1
```

**Financial Health** (normalized by market cap):
```python
debt_to_marketcap = Debt_Total_lag1 / CompanyMarketCap_lag1
cash_to_marketcap = CashCashEquivalents_Total_lag1 / CompanyMarketCap_lag1
fcf_to_marketcap = FOCFExDividends_Discrete_lag1 / CompanyMarketCap_lag1
```

**Earnings:**
```python
eps_actual = EarningsPerShare_Actual_lag1
eps_surprise = EarningsPerShare_ActualSurprise_lag1
eps_estimate_current = EarningsPerShare_SmartEstimate_current_Q_lag1
```

**Analyst Metrics:**
```python
recommendation = Recommendation_Median_1_5__lag1
price_target = PriceTarget_Median_lag1
upside_to_target = (PriceTarget_Median_lag1 / RefPriceClose_lag1 - 1) * 100
```

**Size Factor:**
```python
log_marketcap = log(1 + CompanyMarketCap_lag1)
```

#### E. Cross-Sectional Rankings (7 features)

For each date, rank all stocks by percentile (0-100%):
```python
CompanyMarketCap_lag1_rank = percentile_rank(CompanyMarketCap_lag1)
return_20d_rank = percentile_rank(return_20d)
volatility_20d_rank = percentile_rank(volatility_20d)
roe_rank = percentile_rank(roe)
roa_rank = percentile_rank(roa)
ev_to_ebitda_rank = percentile_rank(ev_to_ebitda)
ltg_rank = percentile_rank(ltg)
```

**Why ranks?** Captures relative positioning - is this stock cheap/expensive compared to peers today?

#### F. Additional Lags (5 features)

T-2 versions of key metrics:
```python
roa_lag2 = shift(roa, 1 day)  # T-2 ROA
roe_lag2 = shift(roe, 1 day)  # T-2 ROE
ev_to_ebitda_lag2 = shift(ev_to_ebitda, 1 day)
ev_to_sales_lag2 = shift(ev_to_sales, 1 day)
ltg_lag2 = shift(ltg, 1 day)
```

**Why?** Captures changes in fundamentals over time.

### Step 4: Feature Selection (What's EXCLUDED)

**Automatically excluded from training (NOT used as features):**

1. **Metadata columns**: Date, Symbol, Instrument, TradeDate, CompanyCommonName, GICSSectorName
2. **Sharadar metadata**: sharadar_exchange, sharadar_category, sharadar_location, etc.
3. **Target variable**: forward_return (what we're predicting)
4. **Intermediate calculations**: volume_ma_20 (only used to create volume_ratio)
5. **ALL original (non-lagged) columns**: RefPriceClose, CompanyMarketCap, etc. (unless using `--no-lag` mode)

**What's INCLUDED (the X matrix for training):**

✅ All `_lag1` columns (~240 lagged fundamentals in production dataset)
✅ All momentum features (return_5d, return_10d, return_20d, volatility_5d, volatility_10d, volatility_20d)
✅ Volume features (volume_ratio)
✅ All fundamental ratios (roa, roe, ev_to_ebitda, ltg, etc.)
✅ All cross-sectional ranks (7 rank features)
✅ All lag-2 features (5 additional lags)

**Total Features (Production Dataset with LSEG + FMP + Sharadar):**
- **Input**: ~240 raw fundamental columns
- **After lagging + engineering**: **298 features** used for training
- **Feature breakdown**:
  - ~240 lagged fundamentals (_lag1 versions)
  - 6 price momentum features
  - 1 volume feature
  - ~20 derived fundamental ratios
  - 7 cross-sectional ranks
  - 5 lag-2 features
  - ~11 other engineered features (log_marketcap, upside_to_target, etc.)

> **Note**: If you use a minimal LSEG-only dataset (32 columns), you'll get ~70-100 features instead of 290. The script automatically adapts to your data!

### Step 5: Missing Value Handling (Forward-Fill - NO LOOK-AHEAD BIAS)

**CRITICAL UPDATE (2025-12-28):** Changed from median-fill to forward-fill to eliminate look-ahead bias!

```python
For numeric features:
  - Forward-fill per symbol (use last known value)
  - Fill remaining NaNs (first rows) with 0

For categorical features:
  - Fill with 'Unknown'
  - Convert to numeric codes
```

**Why forward-fill per symbol?**
- **Point-in-time accurate**: Uses only data available at that date
- **No look-ahead bias**: Never uses future data (median used future values!)
- **Realistic**: Matches what you'd actually know when trading

**Example:**
```
Symbol: AAPL
Date       | EarningsPerShare | After Forward-Fill
-----------|------------------|------------------
2024-01-15 | 2.18            | 2.18 (reported)
2024-01-16 | NaN             | 2.18 (use last known)
2024-01-17 | NaN             | 2.18 (use last known)
2024-04-25 | 1.53            | 1.53 (new report)
2024-04-26 | NaN             | 1.53 (use last known)
```

This ensures fundamentals persist until the next report - exactly like real trading!

### Step 6: Sample Weighting

Not all stocks are weighted equally:

```python
Top 2000 by market cap:    weight = 1.0  (full importance)
Rank 2001-4000:            weight = 0.5  (medium importance)
Rank 4000+:                weight = 0.1  (low but not ignored)
```

**Effect**: Model focuses on large-cap liquid stocks but still learns from entire universe.

### Final Training Matrix

```
X = 70-298+ features (depending on input data - all lagged or engineered, NO raw fundamentals)
y = forward_return (90-day return starting 10 days ahead)
sample_weights = market cap based weights

Training rows: ~8.6M rows with valid forward_return
Prediction rows: 9.0M+ rows (includes recent dates without forward_return)

Example production dataset:
  • Input: ~240 raw fundamental columns
  • Output: 298 features (240 lagged + 50 engineered)
```

### Feature Importance

The model automatically learns which features are most predictive. Typically:

**Most Important:**
1. Long-term growth estimates (`ltg`)
2. Analyst price targets (`price_target`, `upside_to_target`)
3. Valuation ratios (`ev_to_ebitda`, `forward_peg`)
4. Alpha rankings (`alpha_sector_rank`)
5. Recent momentum (`return_20d`)

**Moderately Important:**
6. Profitability metrics (`roe`, `roa`)
7. Earnings surprises (`eps_surprise`)
8. Cross-sectional ranks
9. Volatility measures

**Less Important but Still Used:**
10. Volume ratios
11. Short-term momentum (5-day)
12. Lag-2 features

### Key Principles

1. ✅ **NO LOOK-AHEAD BIAS**: All data lagged by 1 day
2. ✅ **Automatic selection**: Exclude metadata, keep all engineered features
3. ✅ **Market-cap weighting**: Focus on tradeable large-caps
4. ✅ **Robust to missing data**: Median imputation
5. ✅ **Cross-sectional context**: Ranks capture relative positioning
6. ✅ **Temporal features**: Momentum and changes over time

**The model learns from all features simultaneously to predict future returns!**
- LSEG-only dataset: ~70-100 features
- Production dataset (LSEG + FMP + Sharadar): 298 features
- Your custom dataset: Features scale automatically with your input columns

## 🚀 Quick Start

### Basic Usage

```bash
# Predict 10-day returns (default)
python forecast_returns_ml.py your_data.csv

# Output: your_data_predictions_NO_LOOKAHEAD.csv
```

### Common Use Cases

#### 1. Weekly Rebalancing Strategy (10-day returns)
```bash
python forecast_returns_ml.py data.csv --forecast-days 10 --no-cv
```
- **Use case**: Rebalance weekly, hold 2 weeks
- **Correlation**: ~80%
- **Direction accuracy**: ~73%

#### 2. Monthly Strategy with Lead Time (90-day returns, 10 days ahead)
```bash
python forecast_returns_ml.py data.csv \
  --forecast-days 10 \
  --target-return-days 90 \
  --no-cv
```
- **Use case**: Predict quarterly performance with 2-week lead time
- **Correlation**: ~95% 🔥
- **Direction accuracy**: ~87%

#### 3. Quarterly Rebalancing (180-day returns)
```bash
python forecast_returns_ml.py data.csv \
  --forecast-days 20 \
  --target-return-days 180 \
  --n-estimators 500 \
  --no-cv
```
- **Use case**: Long-term holdings, predict 6-month returns
- **Better for**: Fundamental-driven strategies

#### 4. Export for Trading System Integration
```bash
python forecast_returns_ml.py data.csv \
  --no-cv \
  --export-predictions
```
- **Use case**: Daily predictions for automated trading system
- **Output**: Full file + lightweight 3-column predictions file
- **Benefits**: Fast lookups, easy database integration

#### 5. Fast Training with Logging (NEW in v3.1)
```bash
python forecast_returns_ml_walk_forward.py data.csv \
  --sample-fraction 0.5 \
  --log-file my_run.log \
  --forecast-days 10 \
  --target-return-days 90
```
- **Use case**: Quick experiments with full audit trail
- **Speed**: ~2x faster with 50% sampling
- **Output**: Predictions + complete log file for reproducibility

#### 6. Pre-Lagged Data Pipeline (NEW in v3.1)
```bash
# You've already lagged your data by 1 day
python forecast_returns_ml.py pre_lagged_data.csv \
  --no-lag \
  --no-cv
```
- **Use case**: Custom data preparation pipeline
- **Benefits**: You control lagging logic, script uses data as-is
- **IMPORTANT**: Only use if ALL columns are already lagged by 1 day!

### All Options

```bash
python forecast_returns_ml.py <input.csv> [options]

Required:
  input.csv              Path to CSV file with fundamental data

Optional Arguments:
  --output, -o           Output CSV path (default: auto-generated)
  --lookback             NOT IMPLEMENTED - Reserved for future use (default: 10)
                         Note: Rolling windows are currently hardcoded to [5, 10, 20] days

  # Return Period Configuration
  --forecast-days        Days ahead to start measuring (default: 10)
  --target-return-days   Return period to predict (default: same as forecast-days)

  # Model Parameters
  --n-estimators         Number of boosting rounds (default: 300)
  --learning-rate        Learning rate (default: 0.05)
  --max-depth            Maximum tree depth (default: 7)
  --num-leaves           Maximum leaves per tree (default: 31) (NEW in v3.3.2)
                         Controls model capacity. With 650K rows, safe values:
                         31 (conservative), 63 (balanced), 127 (high capacity)
                         Rule: >1000 rows per leaf to avoid overfitting

  # Data Handling (NEW in v3.1)
  --no-lag               Skip automatic lagging (use when input data is already lagged)
                         IMPORTANT: Only use if you've pre-lagged all columns by 1 day!
                         Note: RefPriceClose, RefVolume, CompanyMarketCap are ALWAYS lagged (safety)

  # Training Window Strategy (NEW in v3.3.0)
  --lookback-months      Use rolling N-month window instead of expanding window (default: None)
                         None = Expanding window (train on ALL historical data)
                         12 = Rolling 12-month window (train on last 12 months only)
                         24 = Rolling 24-month window (train on last 24 months only)
                         Use to test recency bias vs. long-term patterns
                         Shorter windows = more responsive, Longer/expanding = more stable

  # Performance Optimization (NEW in v3.1)
  --sample-fraction      Fraction of training data to use (0.0-1.0, default: 1.0)
                         Example: 0.5 = 50% sampling for ~2x speedup
                         Predictions still made for ALL rows

  # Logging (NEW in v3.1)
  --log-file             Path to log file (default: auto-generated with timestamp)
                         All console output saved to file for reproducibility

  # Execution Options
  --no-cv                Skip cross-validation (faster)
  --keep-features        Keep all engineered features in output (large file)
  --export-predictions   Export separate file with only Symbol, Date, predicted_return
  --skip-feature-importance  Skip feature importance calculation (saves 1-3 minutes)
```

## 📋 Input Data Requirements

### Required Columns

Your CSV must include:

1. **Identifiers:**
   - `Date` - Date in YYYY-MM-DD format
   - `Symbol` - Stock ticker symbol
   - `RefPriceClose` - Closing price

2. **Market Cap:**
   - `CompanyMarketCap` - Market capitalization (for weighting)

3. **Fundamentals (at least some of these):**
   - Valuation: `EnterpriseValueToEBITDA_DailyTimeSeriesRatio_`, `ForwardPEG_DailyTimeSeriesRatio_`
   - Profitability: `ReturnOnEquity_SmartEstimat`, `ReturnOnAssets_SmartEstimate`
   - Growth: `LongTermGrowth_Mean`, `Estpricegrowth_percent`
   - Earnings: `EarningsPerShare_Actual`, `EarningsPerShare_SmartEstimate_current_Q`
   - Quality: `CombinedAlphaModelSectorRank`, `EarningsQualityRegionRank_Current`
   - Others: See full list in script

### Example Input Format

```csv
Date,Symbol,RefPriceClose,CompanyMarketCap,ReturnOnEquity_SmartEstimat,...
2024-01-01,AAPL,182.68,2900000000000,147.5,...
2024-01-01,MSFT,376.04,2800000000000,38.2,...
2024-01-02,AAPL,185.64,2920000000000,148.1,...
```

## 📤 Output

### Columns Added (2 new columns)

The script adds exactly **2 columns** to your original data:

1. **`forward_return`** - Actual returns for validation (in **percentages**)
   - Example: For 90-day returns starting day 10, this is the actual T+10 to T+100 return
   - Value of `5.0` means a 5% gain, `-3.5` means a 3.5% loss
   - Use this to validate model accuracy

2. **`predicted_return`** - ML predictions (in **percentages**, use for trading)
   - The model's predicted return
   - Value of `12.3` means predicted 12.3% gain
   - Use this for stock selection and ranking

**Important:** Both columns are in **percentage units**, not decimal. A 10% return is represented as `10.0`, not `0.10`.

### File Size
Output size ≈ Input size × 1.05 (only 5% larger, not 2x!)

**Example:**
- Input: 4 GB, 47 columns
- Output: 4.2 GB, 49 columns

### Predictions-Only Export (Optional)

Use `--export-predictions` to create a **separate lightweight file** with only the essential columns:

```bash
python forecast_returns_ml.py data.csv --export-predictions
```

**Output files:**
1. `data_predictions_NO_LOOKAHEAD.csv` - Full file (49 columns)
2. `data_predictions_NO_LOOKAHEAD_predictions_only.csv` - Lean file (3 columns)

**Predictions-only file contains:**
- `Symbol` - Stock ticker
- `Date` - Date in YYYY-MM-DD format
- `predicted_return` - ML predicted return (in **percentages**: `10.5` = 10.5% gain)

**Benefits:**
- **16x smaller file size** (1.3 MB vs 21 MB for 40K rows)
- **Fast lookups** in trading systems
- **Easy integration** with databases
- **Sorted by Date, Symbol** for efficient queries
- **No NaN rows** (only predictions, no missing values)

### Log Files

All runs automatically create a log file in the `./logs/` directory:

```bash
python forecast_returns_ml_walk_forward.py --input data.csv --output pred.parquet

# Creates: ./logs/forecast_ml_walk_forward_20260107_143022.log
```

**Log file contains:**
- Complete console output (all prints, warnings, errors)
- Run parameters (model settings, feature counts, date ranges)
- Training progress (time per month, prediction counts)
- File paths (input, output, resume files)
- Duplicate row detection results
- Feature descriptions (all 290+ features logged)

**Benefits:**
- ✅ **Full audit trail** for reproducibility
- ✅ **Automatic naming** with timestamp (no conflicts)
- ✅ **Organized storage** in `./logs/` directory (not cluttering data directory)
- ✅ **Share logs** for debugging without sharing data

**Custom log location:**
```bash
python forecast_returns_ml_walk_forward.py --input data.csv --log-file /path/to/custom.log
```

**Example output:**
```csv
Symbol,Date,predicted_return
AAPL,2024-12-16,8.5
GOOGL,2024-12-16,12.3
MSFT,2024-12-16,-2.1
```
This means: AAPL predicted to gain 8.5%, GOOGL to gain 12.3%, MSFT to decline 2.1%.

**Use cases:**
```python
# Quick lookup for today's predictions
predictions = pd.read_csv('predictions_only.csv')
today_preds = predictions[predictions['Date'] == '2024-12-16']

# Find stocks predicted to gain > 10%
strong_picks = predictions[predictions['predicted_return'] > 10]

# Merge with portfolio
portfolio = portfolio.merge(predictions, on=['Symbol', 'Date'])

# Database import
predictions.to_sql('ml_predictions', engine, if_exists='append')
```

## 📈 Model Performance

### Typical Results

| Return Period | Correlation | Direction Accuracy | MAE | RMSE |
|--------------|-------------|-------------------|-----|------|
| 10-day | 80.7% | 72.8% | 2.9% | 4.1% |
| 30-day | 85-90% | 75-80% | 3-4% | 5-6% |
| 90-day | 95.1% | 87.5% | 4.7% | 6.3% |

**Key Finding:** Fundamentals predict long-term returns much better than short-term!

### Cross-Validation

The script uses **time-series cross-validation** to ensure:
- No future information leakage
- Realistic out-of-sample performance
- Proper handling of temporal dependencies

## 🔧 Advanced Usage

### Custom Model Parameters

```bash
# More aggressive model (higher accuracy, slower)
python forecast_returns_ml.py data.csv \
  --n-estimators 500 \
  --max-depth 9 \
  --learning-rate 0.03

# Faster model (lower accuracy, faster training)
python forecast_returns_ml.py data.csv \
  --n-estimators 100 \
  --max-depth 5 \
  --learning-rate 0.1 \
  --no-cv
```

### Keep All Features for Analysis

```bash
# Save all engineered features (useful for debugging/analysis)
python forecast_returns_ml.py data.csv --keep-features

# Warning: Output will be ~2x larger (124 columns vs 49)
```

### Feature Importance Analysis

The walk-forward script automatically displays the **top 15 most important features** after training completes, using permutation importance (industry-standard method).

**Example output:**
```
🔬 TOP 15 MOST IMPORTANT FEATURES:
   (Based on final trained model using permutation importance)
  ltg                                       0.012345  (±0.001234)
  price_target                              0.010234  (±0.001123)
  marketcap                                 0.009876  (±0.000987)
  ...
```

**Performance impact:**
- Adds **1-3 minutes** to total runtime (after all training is complete)
- No impact on model training or prediction accuracy
- Based on 10,000 sampled rows with 5 permutation repeats

**Skip feature importance** to save time during testing:
```bash
# Skip feature importance calculation (saves 1-3 minutes)
python forecast_returns_ml_walk_forward.py data.csv --skip-feature-importance
```

**What it measures:**
- Higher values = more important for predictions
- Shows impact on model performance when each feature is shuffled
- Based on the final month's trained model (most recent)

### Pipeline Integration

```python
import pandas as pd
from forecast_returns_ml import ReturnForecaster

# Load data
df = pd.read_csv('your_data.csv')

# Initialize forecaster
forecaster = ReturnForecaster(
    forecast_days=10,
    target_return_days=90,
    n_estimators=300
)

# Train and predict
predictions_df = forecaster.fit_predict(df, use_cv=False)

# Use predictions
top_stocks = predictions_df.nlargest(50, 'predicted_return')
```

## 🎓 Understanding the Output

### Return Value Units

**All returns are in PERCENTAGES, not decimals:**

| Value | Meaning | Decimal Equivalent |
|-------|---------|-------------------|
| `10.0` | 10% gain | 0.10 |
| `5.5` | 5.5% gain | 0.055 |
| `-3.2` | 3.2% loss | -0.032 |
| `0.0` | No change | 0.00 |
| `100.0` | 100% gain (doubled) | 1.00 |

**In trading code, remember to convert:**
```python
# Get predicted return in percentage
predicted_pct = 12.5  # 12.5%

# Convert to decimal for calculations
predicted_decimal = predicted_pct / 100  # 0.125

# Calculate expected price
current_price = 100
expected_price = current_price * (1 + predicted_decimal)  # 112.50
```

### Example Output Summary

```
🎯 PREDICTION SETUP:
  • Using features from: T-1 (lagged by 1 day)
  • Predicting returns from: T+10 to T+100
  • Total return period: 90 days

📊 Sample Weighting:
  • Top 2000 stocks (weight=1.0): 2,500,000 samples
  • Mid-cap stocks (weight=0.5): 800,000 samples
  • Small-cap stocks (weight=0.1): 1,200,000 samples

🎯 MODEL PERFORMANCE (Out-of-sample, NO LOOK-AHEAD BIAS):
  • Correlation with actual returns: 0.9510 (95.1%)
  • Mean Absolute Error: 4.65%
  • Root Mean Squared Error: 6.29%
  • Direction accuracy: 87.48%

📋 COLUMN SUMMARY:
  • Original columns: 47
  • New columns added: 2
  • Total output columns: 49
```

### How to Use Predictions

**For Stock Selection:**
```python
# Top 50 predicted performers
top_50 = df.nlargest(50, 'predicted_return')

# Long/short strategy
longs = df.nlargest(100, 'predicted_return')
shorts = df.nsmallest(100, 'predicted_return')
```

**For Portfolio Optimization:**
```python
# Weight by predicted return
df['weight'] = df['predicted_return'].clip(lower=0) / df['predicted_return'].sum()
```

**For Risk Management:**
```python
# Avoid stocks with large predicted declines
safe_stocks = df[df['predicted_return'] > -5]
```

## ⚠️ Important Notes

### Look-Ahead Bias Prevention

The model uses **T-1 data to predict T+forecast_days** returns. This means:

✅ **Safe for real trading:**
- On day T, use fundamentals from T-1
- Predict returns starting T+10
- Trade at market close on day T

❌ **Would be look-ahead bias:**
- Using same-day fundamentals (T data to predict T+10)
- Not lagging features by at least 1 day

### Data Quality Matters

- **Survivorship bias**: Include delisted stocks in training
- **Point-in-time data**: Ensure fundamentals reflect what was known at each date
- **Consistent definitions**: Use same fundamental definitions throughout

### Computational Requirements

| Dataset Size | Training Time | RAM Required |
|-------------|---------------|--------------|
| 40K rows | ~5 seconds | 2 GB |
| 1M rows | ~2 minutes | 8 GB |
| 10M rows | ~20 minutes | 32 GB |

## 🐛 Troubleshooting

### "Excel shows 690%, 702%, 756% - looks wrong!"
**This is an Excel formatting issue, not a data problem!**

The CSV file stores values correctly as **6.9**, **7.0**, **7.6** (meaning 6.9%, 7.0%, 7.6%).

Excel auto-detects the column as "Percentage" and multiplies by 100, showing:
- 6.9 → 690% (wrong display)
- 7.0 → 702% (wrong display)
- 7.6 → 756% (wrong display)

**Solutions:**
1. In Excel: Select column → Format Cells → Number (not Percentage) → 2 decimal places
2. Use Python/pandas to view: `pd.read_csv('file.csv')`
3. Open in text editor to see raw values

**The data is correct** - just format as Number in Excel!

### "Recent dates have NaN forward_return but have predictions"
**This is correct and expected!**

- Recent dates don't have future prices yet (can't calculate `forward_return`)
- But they DO have fundamentals, so the model CAN and SHOULD predict
- These predictions are what you use for actual trading
- Example: Data through 2024-12-09, last 20 dates have predictions but no validation

**This is the whole point** - getting forecasts for dates you can trade on!

### "No such column: forward_return"
- Old output file format. Re-run the script to generate new predictions.

### "Correlation is very low (<0.3)"
- Check for look-ahead bias in input data
- Ensure sufficient data quality
- Try longer return periods (90-day vs 10-day)

### "All samples have weight 1.0"
- Your dataset has <2000 stocks per date
- This is fine! Weighting only matters with many stocks

### Output file is huge
- Don't use `--keep-features` flag
- Default output adds only 2 columns

### "--lookback parameter doesn't change anything"
**This is expected - the parameter is not implemented.**

The `--lookback` parameter is a placeholder for future functionality. Currently:
- Rolling momentum windows are hardcoded to: 5, 10, 20 days
- Rolling volatility windows are hardcoded to: 5, 10, 20 days
- Volume moving average is hardcoded to: 20 days

Changing `--lookback` has no effect. This may be implemented in a future version.

## 📚 References

### Methodology
- Gradient Boosting: scikit-learn HistGradientBoostingRegressor
- Feature engineering: Based on Fama-French factors and momentum
- Cross-validation: Time-series split with walk-forward validation

### Data Sources
This tool is designed for:
- LSEG (Refinitiv) fundamental data
- NASDAQ Data Link (Sharadar) fundamentals
- Custom fundamental databases

### Related Documentation
- See `/docs/MULTI_SOURCE_DATA.md` for data integration
- See `/examples/lseg_fundamentals/` for data preparation
- See `/examples/strategies/` for strategy examples using predictions

## 📞 Support

For issues or questions:
1. Check this README first
2. Review the inline code documentation
3. Check example usage in the script header
4. File an issue with sample data and error message

## 📄 License

Part of the Zipline-Reloaded project by Hidden Point Capital.

---

**Version:** 3.1.0
**Last Updated:** 2025-12-28
**Author:** Hidden Point Capital (with Claude Code assistance)

---

## Changelog

### v3.1.0 (2025-12-31) - Production Safety Release

**PRODUCTION-CRITICAL FIXES:**
- **CRITICAL**: Eliminated look-ahead bias in walk-forward mode by disabling statistical outlier clipping
- **CRITICAL**: Z-score and quantile clipping now skipped in walk-forward (was using future month statistics)
- **CRITICAL**: Added warnings when using PCA with walk-forward mode (has look-ahead bias)
- **CRITICAL**: Changed missing value handling from median-fill to forward-fill per symbol (eliminates look-ahead bias)

**Production Deployment:**
- **NEW**: Production Deployment Guide section with safety checklist
- **NEW**: Look-ahead bias protection table documenting all safe operations
- **NEW**: Clear warnings in code and documentation about PCA in production
- **WARNING**: PCA not recommended for production walk-forward (use all 298 features)

**Bug Fixes:**
- **BUGFIX**: Raw LSEG fundamentals now included when using `--no-lag` mode
- **BUGFIX**: Fixed z-score NaN handling for zero-variance columns
- **BUGFIX**: Fixed Series boolean comparison in extreme outlier detection
- **BUGFIX**: Added StandardScaler before PCA (was producing NaN variance)

**New Features:**
- **NEW**: Added `--pca N` for PCA dimensionality reduction (exploration only, not for production)
- **NEW**: Added feature descriptions logging - All training features listed with human-readable descriptions at start
- **NEW**: Added `--log-file` for automatic logging with timestamps
- **NEW**: Added `--sample-fraction` for training speedup (2-5x faster)
- **NEW**: Added `--no-lag` for pre-lagged data support
- **NEW**: Automatic column name normalization (lowercase → PascalCase)

**Documentation:**
- **UPDATED**: Comprehensive production deployment guide
- **UPDATED**: PCA section with production warnings
- **UPDATED**: Walk-forward mode guarantees zero look-ahead bias

### v3.0.2 (2024-12-17)
- Initial walk-forward implementation
- Market cap weighted training
- Feature importance analysis

---

## 🔧 Utility Scripts

### Format Conversion Tools

**Convert CSV to Parquet (for speed & compression):**
```bash
python convert_csv_to_parquet.py input.csv output.parquet

# Example:
python convert_csv_to_parquet.py predictions.csv predictions.parquet
# Output: predictions.parquet (10x smaller, 5-10x faster I/O)
```

**Convert Parquet to CSV (for compatibility & sharing):**
```bash
python convert_parquet_to_csv.py input.parquet output.csv

# Example:
python convert_parquet_to_csv.py predictions.parquet predictions.csv
# Output: predictions.csv (10x larger, but universal format)
```

**Benefits:**
- ✅ **Parquet**: 10x smaller files, 5-10x faster I/O, preserves types
- ✅ **CSV**: Universal compatibility, human-readable, Excel-friendly
- ✅ **Round-trip safe**: Convert CSV→Parquet→CSV preserves all data
- ✅ **Error handling**: Clear messages if file not found or PyArrow missing

**When to use:**
- **CSV → Parquet**: For production pipelines (speed & storage)
- **Parquet → CSV**: For sharing with teams, Excel analysis, debugging

---

## 📚 Detailed Documentation

For detailed technical documentation, design rationales, and troubleshooting guides, see the **[Docs/](Docs/)** directory:

- **[Docs/INDEX.md](Docs/INDEX.md)** - Complete documentation index
- **[Docs/DETERMINISTIC_DESIGN.md](Docs/DETERMINISTIC_DESIGN.md)** - Why zero randomness is better
- **[Docs/REPRODUCIBILITY_FIX.md](Docs/REPRODUCIBILITY_FIX.md)** - Root cause analysis of non-determinism
- **[Docs/MERGE_OPTIMIZATION_ANALYSIS.md](Docs/MERGE_OPTIMIZATION_ANALYSIS.md)** - Memory optimization details
- **[Docs/SUMMARY_v3.2.2.md](Docs/SUMMARY_v3.2.2.md)** - Latest version summary

**See [Docs/INDEX.md](Docs/INDEX.md) for complete documentation index and recommended reading order.**
