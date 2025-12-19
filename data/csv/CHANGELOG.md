# Changelog - ML Return Forecasting

## [3.0.2] - 2024-12-17

### New Features

#### Feature Importance Analysis

**Added**: Automatic feature importance display after walk-forward training completes.

**What it does**:
- Computes **permutation importance** for all features using the final trained model
- Displays **top 15 most important features** with importance scores and standard deviations
- Helps understand which features drive predictions

**Example output**:
```
🔬 TOP 15 MOST IMPORTANT FEATURES:
   (Based on final trained model using permutation importance)
  ltg                                       0.012345  (±0.001234)
  price_target                              0.010234  (±0.001123)
  marketcap                                 0.009876  (±0.000987)
  ...
```

**Performance**:
- Adds **1-3 minutes** to total runtime (after all training is complete)
- No impact on model training or prediction accuracy
- Based on 10,000 sampled rows with 5 permutation repeats
- Uses all CPU cores for parallel computation

**New flag**: `--skip-feature-importance`
- Skip feature importance calculation to save 1-3 minutes during testing
- Usage: `python forecast_returns_ml_walk_forward.py data.csv --skip-feature-importance`

**Code changes**:
- Added `permutation_importance` import from sklearn.inspection (line 55)
- Added `get_feature_importances()` method to ReturnForecaster class (lines 697-747)
- Store X and y in `fit_predict()` for later use (lines 954-956)
- Display feature importances in final output (lines 1348-1375)
- Added `--skip-feature-importance` command-line flag (lines 1084-1085)

**Documentation**:
- Added "Feature Importance Analysis" section to README.md
- Updated command-line flags table

---

## [3.0.1] - 2024-12-17

### Critical Bug Fixes

#### 1. Fixed Resume Prediction Alignment Bug

**Problem**: Resume mode was producing zero correlation (0%) instead of matching full training (37%).

**Root Cause**:
- Previous predictions were aligned with the unsorted dataframe
- `create_target()` sorts dataframe by ['Symbol', 'Date'] at line 267
- After sorting, row indices changed (e.g., [5, 2, 8, 1, ...] instead of [0, 1, 2, 3, ...])
- Prediction alignment broke: `previous_predictions[idx]` no longer matched `df.iloc[idx]`
- Walk-forward retrained ALL months with scrambled predictions, destroying model performance

**Example Impact**:
```
Stock A on 2020-06-01:
  Full training: pred=8.74%
  Resume (broken): pred=2.66%  ❌ WRONG prediction used!

Result: Correlation dropped from 37% to 0%
```

**Fix** (lines 865-895):
1. Pass previous predictions as **DataFrame** (not array) to `fit_predict()`
2. Align predictions **AFTER** dataframe is sorted
3. **Reset index** after sorting: `df.reset_index(drop=True)`
4. Use **position-based iteration**: `for position in range(len(df))` with `df.iloc[position]`

**Code Changes**:
```python
# BEFORE (broken):
# Aligned before sorting → indices mismatched after sort
previous_predictions = align_with_unsorted_df(df)
df = create_target(df)  # Sorts df, breaks alignment

# AFTER (fixed):
df = create_target(df)  # Sort first
df = df.reset_index(drop=True)  # Reset to [0,1,2,3,...]
previous_predictions = align_with_sorted_df(df)  # Align after sort
```

**Testing**: Resume mode now produces identical correlation to full training (37%).

#### 2. Fixed JSON Serialization Error

**Problem**: Checkpoint save failed with `TypeError: Object of type int64 is not JSON serializable`.

**Root Cause**: `valid_preds.sum()` returns numpy `int64`, which JSON can't serialize.

**Fix** (lines 1303-1304):
```python
# BEFORE:
'total_rows': len(df_predictions),
'rows_with_predictions': valid_preds.sum(),

# AFTER:
'total_rows': int(len(df_predictions)),
'rows_with_predictions': int(valid_preds.sum()),
```

#### 3. Improved Checkpoint Naming

**Change**: Checkpoint filename now based on **output file** instead of input file.

**Before**:
- Input: `20091231_20251216_with_metadata.csv`
- Output: `20091231_20251216_with_metadata_with_perdictions-180-90-10-wf.csv`
- Checkpoint: `20091231_20251216_with_metadata_checkpoint.json` ❌ No date range info!

**After**:
- Input: `20091231_20251216_with_metadata.csv`
- Output: `20091231_20251216_with_metadata_with_perdictions-180-90-10-wf.csv`
- Checkpoint: `20091231_20251216_with_metadata_with_perdictions-180-90-10-wf_checkpoint.json` ✓ Clear date range!

**Benefits**:
- ✅ Date range always visible in checkpoint filename
- ✅ Easy to identify which checkpoint goes with which predictions
- ✅ Multiple checkpoints with different date ranges clearly distinguished

**Implementation**: Moved output path generation to line 1068 (before checkpoint logic).

### Performance Impact

**Resume mode now works correctly**:
- Correlation: 0.37 (37%) - matches full training ✓
- Direction accuracy: ~59% - matches full training ✓
- Time: 2-3 minutes for weekly updates (95% faster than full training)

---

## [3.0.0] - 2024-12-16

### Major Feature: Walk-Forward Training with Checkpoint/Resume

**NEW SCRIPT**: `forecast_returns_ml_walk_forward.py` - Production-grade walk-forward training that eliminates ALL look-ahead bias.

#### Walk-Forward Training (Monthly Expanding Window)

**Problem**: The original `forecast_returns_ml.py` trains ONE model on ALL data (2009-2025), then uses it to predict historical dates. This means predictions for 2010 use knowledge of data from 2011-2025 (look-ahead bias).

**Solution**: Walk-forward training with expanding window:
- For Jan 2010: Train on 2009 data only → Predict Jan 2010
- For Jun 2015: Train on 2009-May 2015 data → Predict Jun 2015
- For Dec 2025: Train on 2009-Nov 2025 data → Predict Dec 2025

**Impact**:
- ✅ **NO look-ahead bias** - Each prediction uses only data available at that time
- ✅ **Realistic backtesting** - Results reflect what you could have achieved
- ✅ **Production ready** - Predictions are valid for strategy validation
- ⚠️ **Slower** - Takes 30 minutes vs 5 seconds (but see checkpoint/resume below)

**Implementation**:
- `_walk_forward_predict()` method (lines 696-835)
- Groups data by month
- For each month, trains on all data BEFORE that month
- Predicts for that month only
- Moves to next month with expanding window

#### Checkpoint/Resume System (95% Time Savings)

**Problem**: Walk-forward takes 30 minutes to train 197 models. When adding new data weekly, you don't want to retrain everything.

**Solution**: Checkpoint/resume system that saves progress and only trains new months.

**Features**:
1. **Automatic Checkpoints**: Saves checkpoint JSON after every run
2. **Smart Resume**: Loads previous predictions, only trains new months
3. **Overwrite Buffer**: Re-predicts last N months (default 3) to handle data revisions
4. **Parameter Validation**: Ensures checkpoint matches current model parameters
5. **LATEST Mode**: Auto-detects newest checkpoint by modification time

**Performance**:
```
First run:           197 models trained → 30 minutes
Update (--resume):   4 models trained   → 2-3 minutes (93% faster!)
```

**Implementation**:
- Checkpoint functions (lines 60-145):
  - `compute_data_hash()` - Data change detection
  - `save_checkpoint()` - Save checkpoint metadata
  - `load_checkpoint()` - Load checkpoint metadata
  - `validate_checkpoint()` - Parameter validation
- Resume logic in `_walk_forward_predict()` (lines 697-706, 750-757)
- Resume logic in `fit_predict()` (lines 837-874)
- Main resume workflow (lines 1032-1128)

**Command-Line Flags**:
```bash
--resume                    # Enable checkpoint resume
--overwrite-months N        # Re-predict last N months (default: 3)
--checkpoint-file PATH      # Checkpoint location (or "LATEST")
--force-full                # Force full retrain, ignore checkpoint
```

**Usage Examples**:
```bash
# First run (full training)
python forecast_returns_ml_walk_forward.py data.csv

# Weekly updates (fast resume)
python forecast_returns_ml_walk_forward.py new_data.csv --resume --checkpoint-file LATEST

# Monthly full retrain
python forecast_returns_ml_walk_forward.py data.csv --force-full
```

#### LATEST Checkpoint Auto-Detection

**Feature**: Use `--checkpoint-file LATEST` to automatically pick the newest checkpoint file in the directory.

**Implementation** (lines 1041-1063):
- Searches for all `*_checkpoint.json` files
- Sorts by modification time (newest first)
- Automatically selects newest
- Shows which checkpoint was selected

**Benefits**:
- ✅ No need to remember checkpoint filename
- ✅ Works with changing input filenames
- ✅ Always uses most recent checkpoint

**Example**:
```bash
python forecast_returns_ml_walk_forward.py data.csv --resume --checkpoint-file LATEST

# Output:
# 📂 LATEST checkpoint mode:
#   • Found 3 checkpoint file(s)
#   • Using newest: 20091231_20251209_with_metadata_checkpoint.json
#   • Modified: 2024-12-16 14:30:45
```

### New Feature: Window Strategy Testing Framework

**NEW SCRIPT**: `test_window_strategies.py` - Compare expanding vs rolling window approaches.

**Purpose**: Empirically test whether expanding window (uses all history) or rolling window (uses fixed lookback) produces better predictions for your data.

**Features**:
1. **Multiple Window Sizes**: Tests expanding + rolling 6, 12, 24, 36 months
2. **Comprehensive Metrics**:
   - Correlation (linear relationship)
   - IC (Information Coefficient - rank correlation)
   - RMSE, MAE (prediction errors)
   - Hit Rate (directional accuracy)
   - Top/Bottom Quintile Returns
   - Long/Short Spread (profitability)
3. **Visual Comparison**: Generates plots showing performance over time
4. **Fast Mode**: `--sample-every N` to test subset of months

**Usage**:
```bash
# Quick test (sample every 3 months)
python test_window_strategies.py data.csv --sample-every 3

# Full test with custom windows
python test_window_strategies.py data.csv --windows 12 24 36

# Custom forecast parameters
python test_window_strategies.py data.csv --forecast-days 10 --target-return-days 90
```

**Output Files**:
- `*_window_comparison.csv` - Detailed monthly metrics
- `*_window_comparison_summary.txt` - Summary statistics and recommendations
- `*_window_comparison_plots.png` - Visual comparison charts

### Bug Fixes

#### 1. Index Mismatch in Walk-Forward (lines 660-688)

**Problem**: Mixed pandas index values with numpy position indices, causing `ValueError: Input y contains NaN` on second month.

**Fix**: Use `np.where()` to get position indices consistently:
```python
# Before (wrong):
train_indices = df.index[train_mask].tolist()  # Pandas indices
y_train = y[train_indices]  # Numpy interprets as positions → WRONG

# After (correct):
train_positions = np.where(train_mask)[0]  # Position indices
y_train = y[train_positions]  # Numpy uses positions → CORRECT
```

#### 2. Checkpoint Not Saved with --force-full (line 1267)

**Problem**: `--force-full` flag prevented checkpoint from being saved, even though it should save for next run.

**Fix**: Changed condition from `if not args.force_full and not args.no_walk_forward:` to `if not args.no_walk_forward:`

**Reasoning**: `--force-full` means "ignore existing checkpoint" not "don't save checkpoint"

### Documentation Added

**New Files**:
1. **CHECKPOINT_RESUME_GUIDE.md** - Complete checkpoint/resume usage guide
   - How checkpoint/resume works
   - Command-line options
   - Example workflows (weekly updates, monthly retrain)
   - Safety features and validation
   - Troubleshooting guide
   - Performance comparison table
   - Production automation examples

2. **ML_FORECASTING_VERSIONS.md** - Comparison of all script versions
   - forecast_returns_ml.py (original - fast but has look-ahead bias)
   - forecast_returns_ml_walk_forward.py (new - slower but no look-ahead bias)
   - Side-by-side comparison table
   - When to use each version
   - Technical details of look-ahead bias

3. **TESTING_FRAMEWORK_README.md** - Window strategy testing guide
   - What the framework tests
   - How to run tests
   - Interpreting results
   - Metrics explanation
   - Advanced analysis techniques

### Migration Guide

**From forecast_returns_ml.py to forecast_returns_ml_walk_forward.py**:

1. **First run** (one-time, ~30 minutes):
   ```bash
   python forecast_returns_ml_walk_forward.py data.csv
   ```

2. **Future updates** (2-3 minutes):
   ```bash
   python forecast_returns_ml_walk_forward.py new_data.csv --resume --checkpoint-file LATEST
   ```

3. **Keep old script** for fast experimentation:
   ```bash
   # Quick testing (5 seconds, has look-ahead bias)
   python forecast_returns_ml.py data.csv

   # Final validation (30 minutes, no look-ahead bias)
   python forecast_returns_ml_walk_forward.py data.csv --force-full
   ```

### Performance Summary

| Script | Approach | Time (40K rows) | Look-Ahead Bias | Use Case |
|--------|----------|-----------------|-----------------|----------|
| forecast_returns_ml.py | Single model | 5 seconds | ⚠️ YES | Quick testing |
| forecast_ml_walk_forward.py (full) | 197 models | 30 minutes | ✅ NO | First run |
| forecast_ml_walk_forward.py (resume) | 4 models | 2-3 minutes | ✅ NO | Updates |

### Breaking Changes

**None** - All changes are in new script. Original `forecast_returns_ml.py` unchanged.

### Recommendations

**For production backtesting**: Always use `forecast_returns_ml_walk_forward.py` to ensure no look-ahead bias.

**For development iteration**: Use `forecast_returns_ml.py` for speed, then validate with walk-forward before deploying.

**For updates**: Use `--resume --checkpoint-file LATEST` for 95% time savings.

---

## [2.1.0] - 2024-12-16

### Fixed - Critical Bug: Predictions for Recent Dates

**Problem**: The script was only generating predictions for rows with valid `forward_return`. Recent dates (without future prices) got NaN predictions, making the model useless for live trading.

**Solution**: Modified the prediction pipeline to:
- Train ONLY on historical dates with valid `forward_return` (for validation)
- Predict for ALL dates including recent ones (for live trading)

**Changes**:
1. `prepare_features()` (line 396-409):
   - Now returns unfiltered feature matrix X
   - Identifies valid rows with `valid_idx` but doesn't filter X
   - Allows prediction on all rows while training on subset

2. `fit_predict()` (lines 600-621):
   - Filters data for training: `X_train = X[valid_idx]`
   - Predicts on all data: `predictions = model.predict(X)`
   - Assigns predictions to all rows, not just valid ones

**Impact**:
- ✅ Recent dates now get predictions even without `forward_return`
- ✅ 100% prediction coverage (all rows with fundamentals)
- ✅ Production-ready for live trading
- ✅ No change to training or validation logic

**Example**:
```
Before fix:
  Total rows: 40,138
  Rows with predictions: 39,570 (98.6%)
  Recent dates: 568 rows with NaN predictions ❌

After fix:
  Total rows: 40,138
  Rows with predictions: 40,138 (100%)
  Recent dates: 568 rows with valid predictions ✓
```

### Documentation - Clarified --lookback Parameter

**Status**: The `--lookback` parameter is **not implemented** and has no effect.

**Updated**:
- forecast_returns_ml.py: Marked as "NOT IMPLEMENTED" in help text and docstring
- README.md: Added to "All Options" with NOT IMPLEMENTED note
- README.md: Added troubleshooting entry explaining it's unused

**Current behavior**:
- Rolling windows are hardcoded to [5, 10, 20] days for momentum and volatility
- Volume moving average is hardcoded to 20 days
- Parameter is accepted but ignored

**Rationale**: The parameter was originally added for future feature engineering flexibility but was never implemented. Rather than remove it (breaking existing scripts), we've clearly documented it as unused.

### Documentation Updated

**README.md**:
- Added new section "Predictions for ALL Dates (Including Recent)"
- Added troubleshooting for "Recent dates have NaN forward_return but have predictions"
- Added troubleshooting for Excel formatting issue (690% display)
- Updated section numbering

**forecast_returns_ml.py**:
- Updated module docstring to highlight production-ready predictions
- Updated header comment block with PRODUCTION-READY PREDICTIONS section
- Updated class docstring with IMPORTANT note about prediction behavior
- Added example showing predictions for recent dates

### Notes

**This is the correct behavior for production ML forecasting**: You want predictions for the most recent dates (where you actually trade) even though you can't validate them yet. The model uses their fundamentals to generate forecasts.

**Excel Formatting Issue Documented**: CSV files store values correctly (6.9 = 6.9%), but Excel may auto-format as percentage and show 690%. This is a display issue, not a data problem. Solution: Format column as Number in Excel.

---

## [2.0.0] - 2024-12-15

### Initial Production Release

- No look-ahead bias (all features lagged by 1 day)
- Market cap weighted training (focus on top 2000 stocks)
- Customizable return periods
- 80-95% correlation with actual returns
- Fast execution (~5 seconds per 40K rows)
- Lean output (only adds 2 columns)

