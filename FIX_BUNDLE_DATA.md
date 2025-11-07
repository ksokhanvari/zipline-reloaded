# Fix Bundle Data Issue

## Problem
Your backtest is failing because the partial bundle ingest didn't download complete price data. The strategy can't trade when `data.can_trade()` returns False due to missing prices.

## Diagnostic Steps

### 1. Check Bundle Data Availability

Run the diagnostic script:

```bash
python diagnose_bundle.py sharadar SPY
```

This will show you:
- What date range has data
- Which dates are missing
- Whether recent data (last 30 days) is available
- Recommendations for fixing

### 2. Understand the Output

The diagnostic will tell you:
- ✓ **Good**: All sessions have valid price data
- ⚠ **Warning**: Some historical data missing (may still work)
- 🔴 **Critical**: Recent data missing (backtests will fail)

## Solutions

### Option 1: Re-ingest the Complete Bundle (Recommended)

```bash
# Full re-ingest to get all data
zipline ingest -b sharadar
```

**When to use:**
- Missing recent data (last 30 days)
- Many gaps in historical data
- Want complete dataset

**Notes:**
- May take 10-30 minutes depending on data size
- Downloads all symbols and dates
- Most reliable solution

### Option 2: Adjust Backtest Dates

If re-ingesting is too slow, run the diagnostic script to find the valid date range, then adjust your backtest:

```python
# In your notebook, change these dates:
results = run_algorithm(
    start=pd.Timestamp('2020-01-01'),  # Change to first available date
    end=pd.Timestamp('2023-12-31'),    # Change to last available date
    ...
)
```

**When to use:**
- Historical data is complete, only recent data missing
- Quick testing without waiting for ingest
- Don't need to backtest recent dates

### Option 3: Incremental Ingest (For Future Updates)

After fixing with Option 1, use incremental updates:

```bash
# Regular updates (daily/weekly)
zipline ingest -b sharadar
```

The `sharadar_bundle` is configured with `incremental=True`, so it should only download new data after the initial ingest.

## Verify the Fix

After re-ingesting, run the diagnostic again:

```bash
python diagnose_bundle.py sharadar SPY
```

You should see:
```
✓ All sessions have valid price data!
✓ Bundle data looks good!
  You can safely backtest from YYYY-MM-DD to YYYY-MM-DD
```

Then re-run your backtest - it should work correctly.

## Common Issues

### Issue: "Bundle not found"
**Fix:** Check available bundles:
```bash
zipline bundles
```

### Issue: "Ingest takes too long"
**Fix:** The first ingest downloads all historical data and may take 20-30 minutes. Subsequent ingests are much faster (1-2 minutes).

### Issue: "Still getting 0% returns"
**Check:**
1. Did the diagnostic pass?
2. Are your backtest dates within the available data range?
3. Does the symbol exist in the bundle? Run: `python diagnose_bundle.py sharadar YOUR_SYMBOL`

## What the Partial Ingest Did Wrong

A "partial" ingest can happen when:
1. Network interruption during download
2. API rate limiting
3. Incomplete data from provider
4. Bundle cache corruption

The result is gaps in the price data, making `data.can_trade()` return False.

## Prevention

**Best Practices:**
1. Run full ingest initially: `zipline ingest -b sharadar`
2. Schedule regular updates (daily/weekly)
3. Check diagnostic before important backtests
4. Keep logs: `zipline ingest -b sharadar --log-file logs/ingest_$(date +%Y%m%d).log`

## Quick Reference

```bash
# Diagnose bundle issues
python diagnose_bundle.py sharadar SPY

# Fix with full re-ingest
zipline ingest -b sharadar

# Verify fix
python diagnose_bundle.py sharadar SPY

# Check what bundles you have
zipline bundles
```

## Expected Behavior After Fix

When you run your backtest, you should see:
```
INFO:root:Bought SPY at $XXX.XX  # This should appear early!
2025-11-07 22:25:12,804 INFO zipline.progress: [Buy-and-Hold] ----------      4%  2025-01-16              X.XX%  # Non-zero returns!
```

Not:
```
2025-11-07 22:25:12,804 INFO zipline.progress: [Buy-and-Hold] ----------      4%  2025-01-16              0%  # Zero = problem!
```
