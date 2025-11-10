# Critical Bug Fix: Incremental Bundle Ingest

## The Problem You Experienced

After running a "partial ingest" (incremental update) to get the latest daily data, **all your historical data disappeared**. Your bundle went from having years of data to only having 2025-11-07 (1 day).

This caused your backtests to fail with:
- 0% returns throughout the entire backtest
- Only buying on the last day
- 0 transactions recorded
- `data.can_trade()` returning False for all dates

## Root Cause

The incremental ingest logic had a critical bug:

1. ✅ **Price data was being merged** correctly (existing + new)
2. ❌ **Asset metadata was being OVERWRITTEN** instead of merged
3. Result: Assets showed `start_date = end_date = 2025-11-07` (only 1 day)

### Technical Details

In `src/zipline/data/bundles/sharadar_bundle.py`:

**Before the fix (lines 595-602):**
```python
# Created metadata from ONLY the current pricing data
metadata = prepare_asset_metadata(
    all_pricing_data,  # If this only has 1 day, metadata shows 1 day!
    actual_start,
    actual_end
)

# OVERWROTE the entire asset database
asset_db_writer.write(equities=metadata, exchanges=exchanges)
```

**After the fix (lines 595-691):**
```python
# For incremental updates: merge with existing metadata
if is_incremental_update:
    # Load existing asset metadata
    existing_metadata = pd.read_sql("SELECT * FROM equities", conn)

    # Create new metadata from current data
    new_metadata = prepare_asset_metadata(...)

    # Merge: keep EARLIEST start_date, update to LATEST end_date
    metadata = merge_metadata(existing_metadata, new_metadata)

    # Preserves historical date ranges!
```

## The Fix

I've implemented a complete fix that:

1. **Loads existing asset metadata** during incremental updates
2. **Merges** old and new metadata intelligently:
   - Keeps **earliest** start_date (preserves history)
   - Updates to **latest** end_date (new data)
   - Preserves **all** existing assets
   - Adds new assets that appear for first time
3. **Handles errors gracefully** (falls back safely if merge fails)

## What You Need To Do

### Step 1: Pull the Fix

```bash
cd /app
git pull
```

### Step 2: Clean Your Broken Bundle

Your current bundle only has 1 day of data and is unusable. Clean it:

```bash
zipline clean -b sharadar -a
```

**This removes ALL sharadar bundle data.** Don't worry - you'll re-ingest everything in the next step.

### Step 3: Re-ingest From Scratch

```bash
zipline ingest -b sharadar
```

**What to expect:**
- This will download ALL historical data (1998-present)
- Takes 15-30 minutes depending on your subscription and connection
- You'll see progress messages showing the download
- First time is slow, but future incremental updates will be fast (30-60 seconds)

### Step 4: Verify the Fix

After ingestion completes, run the diagnostic:

```bash
python diagnose_bundle.py sharadar SPY
```

You should now see:
```
✓ Found asset: Equity(8961 [SPY])
  - Start date: 1998-XX-XX  ← Multiple years, not just 1 day!
  - End date: 2025-11-07
  - Asset listed for 6000+ trading days  ← Not just 1!
```

### Step 5: Test Your Backtest

Re-run your buy-and-hold notebook. You should now see:
- ✅ "Bought SPY at $XXX.XX" on the FIRST day (not last!)
- ✅ Non-zero returns throughout the backtest
- ✅ Proper performance metrics
- ✅ Actual transactions

## How Incremental Updates Work Now

### Initial Ingest (Full Download)
```bash
zipline ingest -b sharadar
```
- Downloads all data from 1998-present
- Takes 15-30 minutes
- Creates complete historical database

### Daily Updates (Incremental)
```bash
zipline ingest -b sharadar
```
- Automatically detects existing data
- Downloads ONLY new data since last ingest
- **Merges** with existing data (price AND metadata)
- Takes 30-60 seconds
- **Preserves all historical data** ← THE FIX!

### Example Output (After Fix)

```
🔍 DEBUG: Checking for previous ingestions...
   Found 1 previous ingestion(s)

🔄 INCREMENTAL UPDATE DETECTED
Last ingestion ended: 2025-11-06
Downloading new data from: 2025-11-07 to 2025-11-07

Step 1/3: Downloading NEW Sharadar Equity Prices (incremental)...
Downloaded 8,245 equity price records

🔄 Merging with existing data...
   Newly downloaded: 8,245 records
   Loading from: 2025-11-06-223045
   Existing data: 15,234,567 records from 1998-01-05 to 2025-11-06
   Keeping 15,234,567 records before 2025-11-07
   ✓ Merged total: 15,242,812 records

Preparing asset metadata...
   Merging with existing asset metadata...
   Loaded 8,245 existing assets
   ✓ Merged metadata: 8,245 total assets
     New/updated: 8,245, Preserved: 0

✓ Sharadar bundle ingestion complete!
```

## Prevention

This bug is now fixed in the code, so:

1. **First ingest**: Always use full ingest after pulling the fix
2. **Daily updates**: Can safely use incremental updates
3. **No data loss**: Historical data will be preserved

## Troubleshooting

### If you still see 1-day assets after re-ingesting:

1. Make sure you pulled the latest code: `git pull`
2. Verify the fix is in `sharadar_bundle.py` lines 595-691
3. Check that you cleaned the old bundle: `zipline clean -b sharadar -a`
4. Try the diagnostic script: `python diagnose_bundle.py sharadar SPY`

### If the re-ingest fails:

1. Check your NASDAQ_DATA_LINK_API_KEY is set
2. Verify your subscription is active
3. Check network connectivity
4. Look at the error message - may indicate API limits

### If you want to avoid the long initial download:

The fix ensures incremental updates work correctly, but you **must** do at least one full ingest first to get the complete historical dataset.

## Summary

**Bug:** Incremental updates were deleting historical asset metadata
**Impact:** Backtests failed with 0% returns, only 1 day of data
**Fix:** Merge asset metadata during incremental updates (preserve history)
**Action Required:** Clean bundle + re-ingest once, then incremental updates will work correctly

The incremental update feature is now safe to use and will preserve your historical data! 🎉
