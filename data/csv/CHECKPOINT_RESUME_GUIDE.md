# Checkpoint/Resume Feature - Usage Guide

## Overview

The checkpoint/resume feature allows you to **update predictions 95% faster** when adding new data to your CSV file. Instead of retraining 197 models (30 minutes), you only retrain the last few months (2-3 minutes).

## How It Works

### First Run (Full Training)
```bash
python forecast_returns_ml_walk_forward.py data.csv
```

**What happens:**
1. Trains 197 models (one per month from 2009-2024)
2. Saves predictions to `data_predictions_WALK_FORWARD.csv`
3. Creates checkpoint file: `data_checkpoint.json`
4. Time: ~30 minutes

**Checkpoint saved:**
```json
{
  "last_prediction_date": "2024-11-30",
  "output_file": "data_predictions_WALK_FORWARD.csv",
  "forecast_days": 10,
  "target_return_days": 10,
  "created": "2024-12-16 12:00:00"
}
```

### Subsequent Runs (Resume Mode)
```bash
# You've added December 2024 data
python forecast_returns_ml_walk_forward.py data_updated.csv --resume
```

**What happens:**
1. Loads checkpoint and previous predictions
2. Calculates resume date: Nov 30 - 3 months = Aug 31
3. Keeps predictions before Aug 31 (skips Jan 2010 - Aug 2024)
4. Trains only 4 models: Sep, Oct, Nov, Dec 2024
5. Overwrites Sep-Nov predictions (handles data revisions)
6. Adds Dec predictions (new month)
7. Time: ~2-3 minutes

**Time saved: 93%!**

## Command-Line Options

### Basic Usage

```bash
# Full training (default)
python forecast_returns_ml_walk_forward.py data.csv

# Resume from checkpoint (FAST)
python forecast_returns_ml_walk_forward.py data.csv --resume

# Force full retrain (ignore checkpoint)
python forecast_returns_ml_walk_forward.py data.csv --force-full
```

### Advanced Options

```bash
# Resume with custom overwrite buffer
python forecast_returns_ml_walk_forward.py data.csv --resume --overwrite-months 6

# Auto-detect newest checkpoint (RECOMMENDED)
python forecast_returns_ml_walk_forward.py data.csv --resume --checkpoint-file LATEST

# Custom checkpoint file location
python forecast_returns_ml_walk_forward.py data.csv --resume --checkpoint-file /path/to/checkpoint.json

# Resume with 1 month overwrite (faster but riskier if data revised)
python forecast_returns_ml_walk_forward.py data.csv --resume --overwrite-months 1
```

### 🆕 LATEST Checkpoint Auto-Detection

**The easiest way to resume** - no need to remember checkpoint filenames!

```bash
python forecast_returns_ml_walk_forward.py new_data.csv --resume --checkpoint-file LATEST
```

**What happens:**
1. Searches for all `*_checkpoint.json` files in the same directory
2. Sorts them by modification time (newest first)
3. Automatically picks the most recent checkpoint
4. Shows you which one it selected

**Output:**
```
📂 LATEST checkpoint mode:
  • Found 3 checkpoint file(s)
  • Using newest: 20091231_20251209_with_metadata_checkpoint.json
  • Modified: 2024-12-16 14:30:45
```

**Benefits:**
- ✅ **No filename hunting** - Always uses newest checkpoint
- ✅ **Works with changing input filenames** - Doesn't depend on input filename matching
- ✅ **Case-insensitive** - `LATEST`, `latest`, or `Latest` all work
- ✅ **Automatic fallback** - If no checkpoints found, does full training

**Recommended workflow:**
```bash
# First run
python forecast_returns_ml_walk_forward.py 20091231_20251209_data.csv

# All future updates - just use LATEST!
python forecast_returns_ml_walk_forward.py 20091231_20251215_data.csv --resume --checkpoint-file LATEST
python forecast_returns_ml_walk_forward.py 20091231_20251220_data.csv --resume --checkpoint-file LATEST
python forecast_returns_ml_walk_forward.py 20091231_20251231_data.csv --resume --checkpoint-file LATEST
```

No need to remember or specify the exact checkpoint filename - `LATEST` always finds it!

### All Resume Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--resume` | False | Enable checkpoint resume mode |
| `--overwrite-months N` | 3 | Re-predict last N months |
| `--checkpoint-file PATH` | auto | Checkpoint location: path, "LATEST", or auto-generated |
| `--force-full` | False | Ignore checkpoint, full retrain |

## Why Overwrite Last N Months?

Fundamental data often gets **revised**:
- Quarterly reports filed late
- Financial restatements
- Data provider corrections

By re-predicting the last 3 months, you ensure:
- ✅ Updated data incorporated
- ✅ Predictions stay accurate
- ✅ Safety buffer for corrections

## Example Workflows

### Weekly Updates (Fast)
```bash
# Monday morning: Download new data
download_data.sh  # Gets data through Friday

# Generate predictions (2 minutes)
python forecast_returns_ml_walk_forward.py latest_data.csv --resume

# Ready to trade!
```

### Monthly Full Retrain (Clean Slate)
```bash
# First Sunday of month: Full retrain
python forecast_returns_ml_walk_forward.py latest_data.csv --force-full

# Rest of month: Resume mode
python forecast_returns_ml_walk_forward.py latest_data.csv --resume
```

### After Major Data Correction
```bash
# Force full retrain to incorporate corrections
python forecast_returns_ml_walk_forward.py corrected_data.csv --force-full
```

## File Locations

### Auto-Generated Checkpoint (Based on Output Filename)

**Important**: As of v3.0.1, checkpoint filenames are based on the **output file** (not input file) for clarity.

```
Your working directory:
├── data.csv                                        # Input data
├── data_predictions_WALK_FORWARD.csv               # Predictions output
└── data_predictions_WALK_FORWARD_checkpoint.json   # Checkpoint metadata ✓

OR with custom output:
├── 20091231_20251216_with_metadata.csv                                # Input
├── 20091231_20251216_with_metadata_with_perdictions-180-90-10-wf.csv  # Output
└── 20091231_20251216_with_metadata_with_perdictions-180-90-10-wf_checkpoint.json  # Checkpoint ✓
```

**Benefits**:
- ✅ Date range always visible in checkpoint filename
- ✅ Easy to identify which checkpoint matches which predictions
- ✅ Multiple checkpoints with different parameters clearly distinguished

### Custom Checkpoint
```bash
python forecast_returns_ml_walk_forward.py data.csv \
  --resume \
  --checkpoint-file /backups/checkpoints/data_2024_checkpoint.json
```

## Safety Features

### 1. Parameter Validation
Checkpoint is **invalidated** if model parameters changed:
```bash
# First run
python forecast_ml_walk_forward.py data.csv --n-estimators 300

# Second run with different params
python forecast_ml_walk_forward.py data.csv --resume --n-estimators 500

# Output:
# ❌ Checkpoint invalid: Parameter mismatch: n_estimators (checkpoint=300, current=500)
# Running full training instead
```

### 2. Missing Files
```bash
python forecast_ml_walk_forward.py data.csv --resume

# If checkpoint.json exists but predictions.csv is deleted:
# ⚠️  Previous predictions file not found: data_predictions_WALK_FORWARD.csv
# Running full training
```

### 3. No New Data
```bash
# Checkpoint: through 2024-11-30
# New data:   through 2024-11-30 (same)

# Resume runs but only overwrites last 3 months
# (still useful if data was revised)
```

## Performance Comparison

For 40K rows, 16 years of data (197 months):

| Mode | Models Trained | Time | Use Case |
|------|---------------|------|----------|
| **Full run** | 197 | 30 min | First run, major changes |
| **Resume (overwrite 1M)** | 2 | 30 sec | Daily updates, no revisions |
| **Resume (overwrite 3M)** | 4 | 2 min | Weekly updates (recommended) |
| **Resume (overwrite 6M)** | 7 | 4 min | Monthly with safety buffer |
| **Resume (overwrite 12M)** | 13 | 7 min | Quarterly with large buffer |

**Recommendation:** Use `--overwrite-months 3` (default) as the sweet spot.

**Note on timing**: Times shown above include feature importance calculation (1-3 minutes). To skip it during testing:
```bash
python forecast_returns_ml_walk_forward.py data.csv --resume --skip-feature-importance
```

## Understanding Resume Output

```bash
python forecast_returns_ml_walk_forward.py data.csv --resume
```

**Output:**
```
======================================================================
  📂 CHECKPOINT RESUME MODE
======================================================================

✓ Checkpoint found: data_checkpoint.json
  • Last prediction date: 2024-11-30
  • Created: 2024-12-15 18:30:00

📂 Loading previous predictions from data_predictions_WALK_FORWARD.csv...
  ✓ Loaded 40,138 rows
  • Last prediction date: 2024-11-30
  • Overwrite buffer: 3 months
  • Resume from: 2024-08-31
  • Will re-predict from 2024-08-31 onwards
  ✓ Keeping 38,456 previous predictions

======================================================================
  🔄 RESUMING WALK-FORWARD PREDICTION
  (Starting from 2024-08-31)
======================================================================

📅 Date Range:
  • First month: 2010-01
  • Last month: 2024-12
  • Total months: 180
  • Total rows: 40,138
  • Resuming from: 2024-09
  • Months to process: 4 (skipping 176)

🔄 Training models month by month...

  [177/180] 2024-09: Trained on 38,123 rows → Predicted  4,235 rows (2.1s) ETA: 6s
  [178/180] 2024-10: Trained on 38,456 rows → Predicted  4,189 rows (2.2s) ETA: 4s
  [179/180] 2024-11: Trained on 38,892 rows → Predicted  4,156 rows (2.1s) ETA: 2s
  [180/180] 2024-12: Trained on 39,127 rows → Predicted  1,011 rows (2.0s) ETA: 0s

======================================================================
  ✅ Walk-Forward Complete!
======================================================================
  • Months processed: 4
  • Total predictions: 40,138 of 40,138
  • Total time: 0m 8s
  • Average time per month: 2.1s
```

## Edge Cases Handled

### 1. Different Input File
```bash
# First run
python forecast_ml_walk_forward.py data_v1.csv

# Second run with different file
python forecast_ml_walk_forward.py data_v2.csv --resume

# Works if data is compatible!
# Checkpoint stores filename but allows different input
```

### 2. Data Shape Changed
```
Checkpoint: 40,000 rows through Nov 2024
New data:   45,000 rows through Nov 2024

# More symbols added or data corrections
# Resume still works - aligns by (Date, Symbol) pairs
```

### 3. Missing Checkpoint
```bash
python forecast_ml_walk_forward.py data.csv --resume

# Output:
# ⚠️  No checkpoint found - running full training
# Looked for: data_checkpoint.json
```

## Troubleshooting

### "Parameter mismatch" Error
**Cause:** Model parameters changed between runs

**Solution:**
```bash
# Option 1: Use --force-full to retrain with new params
python forecast_ml_walk_forward.py data.csv --force-full --n-estimators 500

# Option 2: Revert to original parameters
python forecast_ml_walk_forward.py data.csv --resume --n-estimators 300
```

### "Previous predictions file not found"
**Cause:** Checkpoint references a file that was moved/deleted

**Solution:**
```bash
# Option 1: Full retrain
python forecast_ml_walk_forward.py data.csv --force-full

# Option 2: Specify correct predictions file location
# (Edit checkpoint.json to fix output_file path)
```

### Resume Seems Slower Than Expected
**Cause:** Large `--overwrite-months` value

**Check:**
```bash
# See how many months will be processed
python forecast_ml_walk_forward.py data.csv --resume --overwrite-months 12

# Output shows: "Months to process: 13 (skipping 184)"
# 13 months × 2 seconds = 26 seconds (still fast!)
```

**Solution:** Reduce overwrite buffer if data revisions are rare:
```bash
python forecast_ml_walk_forward.py data.csv --resume --overwrite-months 1
```

### Want to Start Fresh
**Cause:** Want to ignore all checkpoints

**Solution:**
```bash
# Option 1: Delete checkpoint file
rm data_checkpoint.json
python forecast_ml_walk_forward.py data.csv

# Option 2: Use --force-full flag
python forecast_ml_walk_forward.py data.csv --force-full
```

## Best Practices

### 1. Regular Full Retrains
```bash
# Monthly full retrain ensures clean slate
# First Sunday of each month:
python forecast_ml_walk_forward.py data.csv --force-full

# Rest of month: resume mode
python forecast_ml_walk_forward.py data.csv --resume
```

### 2. Backup Checkpoints
```bash
# Before major changes, backup checkpoint
cp data_checkpoint.json data_checkpoint_backup.json

# If something goes wrong, restore
mv data_checkpoint_backup.json data_checkpoint.json
```

### 3. Version Control
```bash
# Add checkpoint.json to .gitignore
echo "*_checkpoint.json" >> .gitignore

# Checkpoints are machine-specific, don't commit
```

### 4. Monitor Overwrite Buffer
```bash
# If your data provider often revises data, use larger buffer
python forecast_ml_walk_forward.py data.csv --resume --overwrite-months 6

# If data is stable, use smaller buffer for speed
python forecast_ml_walk_forward.py data.csv --resume --overwrite-months 1
```

## FAQ

### Q: Does resume mode affect prediction quality?
**A:** No! Each model still trains on ALL historical data. Resume only skips months you already predicted.

### Q: Can I resume with different model parameters?
**A:** No, the checkpoint will be invalidated. Use `--force-full` to retrain with new parameters.

### Q: What if I change the input CSV columns?
**A:** Resume will fail if required columns are missing. Add columns is OK, but changing fundamental columns requires `--force-full`.

### Q: Can I share checkpoints between machines?
**A:** Not recommended. Checkpoints reference absolute file paths. Use `--force-full` on new machine.

### Q: How much disk space do checkpoints use?
**A:** Checkpoint JSON is tiny (~1 KB). The predictions CSV is the same size with or without checkpoint.

### Q: Can I manually edit the checkpoint file?
**A:** Yes, it's JSON. But be careful - invalid checkpoint will trigger full retrain.

## Production Automation

### Cron Job Example
```bash
# Daily at 6 AM: Download data and update predictions
0 6 * * * /path/to/download_data.sh && \
          python /path/to/forecast_ml_walk_forward.py /path/to/data.csv --resume

# Monthly full retrain (1st of month at 2 AM)
0 2 1 * * python /path/to/forecast_ml_walk_forward.py /path/to/data.csv --force-full
```

### Error Handling Script
```bash
#!/bin/bash

# Try resume first
python forecast_ml_walk_forward.py data.csv --resume

# If failed, fall back to full retrain
if [ $? -ne 0 ]; then
    echo "Resume failed, running full retrain..."
    python forecast_ml_walk_forward.py data.csv --force-full
fi
```

---

## Version History

### v3.0.1 (2024-12-17) - Critical Bug Fixes

**Fixed Resume Prediction Alignment**:
- Resolved issue where resume mode produced 0% correlation instead of 37%
- Root cause: Previous predictions misaligned after dataframe sorting
- Fix: Reset index after sorting, align predictions AFTER sort operation
- Resume now produces identical results to full training ✓

**Improved Checkpoint Naming**:
- Checkpoint filenames now based on output file (includes date range)
- Example: `20091231_20251216_predictions_checkpoint.json` instead of `input_checkpoint.json`
- Makes it clear which checkpoint belongs to which predictions file

**Fixed JSON Serialization**:
- Convert numpy int64 to native Python int before saving checkpoint
- Prevents `TypeError: Object of type int64 is not JSON serializable`

### v3.0.0 (2024-12-16) - Initial Release

**Features**:
- Walk-forward training with checkpoint/resume
- LATEST auto-detection mode
- 95% time savings for updates

---

**Created:** 2024-12-16
**Last Updated:** 2024-12-17
**Version:** 3.0.1
**Feature:** Checkpoint/Resume for Walk-Forward Forecasting
