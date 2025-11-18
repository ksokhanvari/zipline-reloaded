# FB→META Symbol Mapping Fix

## Problem Discovery

When loading 8.9M rows of LSEG fundamental data, the temporal SID mapper showed:
```
Verifying FB→META continuity:
  FB (2020-01-01):   SID None      ❌
  META (2023-01-01): SID 194817    ✓
  ⚠ Different SIDs - this shouldn't happen
```

The mapper successfully processed 8.5M rows (95%) using parallel processing in seconds, but FB symbols weren't mapping correctly.

## Root Cause Analysis

### Investigation Steps

1. **Checked Sharadar Asset Database**
   ```bash
   /root/.zipline/data/sharadar/.../assets-7.sqlite
   ```

2. **Found SID 194817 exists as "META"**
   ```sql
   SELECT sid, asset_name FROM equities WHERE sid = 194817
   → SID 194817, asset_name = 'META'
   ```

3. **Checked for symbol history**
   ```sql
   SELECT * FROM equity_symbol_mappings
   WHERE share_class_symbol IN ('META', 'FB')
   → Empty result (no mappings)
   ```

### Key Finding

**The Sharadar bundle never contained "FB" as a separate symbol.**

- Sharadar only has META (SID 194817)
- No symbol mapping history for FB→META transition
- This is because data providers like LSEG/Sharadar ingest companies by their **current name**, not historical symbols

## Why This Happens

### Data Provider Behavior

LSEG and similar data vendors:
1. Export company data using the symbol **at the time of the data row**
2. So their CSV has:
   - `Symbol='FB'` for rows from 2012-2021
   - `Symbol='META'` for rows from 2022-2025

### Sharadar Bundle Behavior

Sharadar ingests companies by **current identity**:
1. Only knows the company as "META" (SID 194817)
2. Never ingested "FB" as a separate security
3. No historical symbol mapping in `equity_symbol_mappings`

### The Mismatch

```
LSEG CSV:          FB (2012-2021) → META (2022-2025)
Sharadar Bundle:   META only (SID 194817)
Result:            FB lookups fail → return None
```

## Solution Implemented

### Symbol Normalization

Added `KNOWN_SYMBOL_CHANGES` dictionary to `TemporalSIDMapper`:

```python
class TemporalSIDMapper:
    # Known symbol changes not in Sharadar bundle
    KNOWN_SYMBOL_CHANGES = {
        'FB': 'META',  # Facebook renamed to Meta in Oct 2021
    }

    def map_single_row(self, symbol, date):
        # Normalize symbol before lookup
        normalized_symbol = self.KNOWN_SYMBOL_CHANGES.get(symbol, symbol)

        # Now lookup META instead of FB
        asset = self.asset_finder.lookup_symbol(normalized_symbol, as_of_date=date)
        return asset.sid
```

### How It Works

1. **User's CSV has FB rows** (2012-2021) and META rows (2022-2025)
2. **Mapper normalizes** FB → META before lookup
3. **All rows map to SID 194817** (the current META SID)
4. **Continuous data** from 2012-2025 under one SID ✓

### Test Results

```
Symbol Mapping Results:
Symbol     Date            SID        Status
FB         2012-05-18      194817     ✓ Mapped
FB         2020-01-01      194817     ✓ Mapped
FB         2021-10-28      194817     ✓ Mapped
META       2022-01-01      194817     ✓ Mapped
META       2023-01-01      194817     ✓ Mapped

✓ SUCCESS: All FB and META rows map to same SID: 194817
  This ensures continuous data across the symbol change!
```

## Performance

The optimized mapper handles large datasets efficiently:

### Parallel Processing (>1M rows)
- **Dataset**: 8,995,098 rows
- **Strategy**: 16 parallel workers, 17 chunks
- **Mapped**: 8,546,736 rows (95.0%)
- **Time**: Seconds (not minutes)

### Unique Pairs Strategy (100k-1M rows)
For typical use cases:
```
Dataset: 600,000 rows
Unique (symbol, date) pairs: ~10,000
Reduction: 60x fewer lookups needed
```

## Files Modified

### `/examples/custom_data/temporal_sid_mapper.py`

**Changes**:
1. Added `KNOWN_SYMBOL_CHANGES` class variable
2. Modified `map_single_row()` to normalize symbols before lookup
3. Optimized `map_dataframe_grouped()` for better performance

**Commit**: `267ae758` - "fix: Add FB->META symbol normalization for temporal SID mapping"

## For Future Symbol Changes

### Adding New Symbol Changes

If you encounter other symbol changes not in Sharadar, add them to the dictionary:

```python
KNOWN_SYMBOL_CHANGES = {
    'FB': 'META',
    'GOOG': 'GOOGL',  # Example: if GOOG not in bundle
    # Add more as needed
}
```

### When to Use This

You need symbol normalization when:
1. Your CSV uses historical symbols (e.g., FB, GOOG)
2. The bundle only has current symbols (e.g., META, GOOGL)
3. No symbol mapping exists in `equity_symbol_mappings` table

### When You Don't Need This

If the bundle has proper symbol history (like some providers do), temporal lookups will work automatically without normalization.

## Summary

✅ **Root Cause**: Sharadar bundle doesn't contain FB symbol, only META
✅ **Solution**: Normalize FB→META before lookup
✅ **Result**: Continuous data 2012-2025 under SID 194817
✅ **Performance**: Parallel processing handles 8.9M rows in seconds
✅ **Testing**: Verified all FB/META rows map to same SID

The fix ensures your LSEG fundamental data loads correctly with continuous company history across symbol changes, regardless of whether the bundle provider tracks historical symbols.
