# Symbol Mapping Guide

This guide covers how to handle symbol changes and SID mapping when loading custom data into Zipline.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Temporal SID Mapping](#temporal-sid-mapping)
3. [Automatic Symbol Mapping](#automatic-symbol-mapping)
4. [Implementation](#implementation)
5. [Troubleshooting](#troubleshooting)

---

## The Problem

When companies change their ticker symbols (like FB → META), naive mapping creates **discontinuous data**:

```
BEFORE (Wrong approach):
  FB rows (2012-2021)   → Current 'FB' lookup → SID 644713
  META rows (2022-2025) → Current 'META' lookup → SID 194817

  Result: Two separate securities, broken continuity
```

Additionally, when loading LSEG CSV data, you encounter symbols that don't exist in the Sharadar bundle:

- **Your CSV**: `FB` (2012-2021), `META` (2022-2025)
- **Sharadar bundle**: Only `META` (current name)
- **Result**: FB rows fail to map → data loss

This happens because:
1. **LSEG exports use historical symbols** (symbol at time of data row)
2. **Bundle providers use current symbols** (company's current name)
3. No automatic translation between them

---

## Temporal SID Mapping

### The Solution: As-of-Date Lookups

Use **as-of-date lookups** to get the correct SID for each row's date:

```python
# For a row with Symbol='FB', Date='2020-01-01'
asset = asset_finder.lookup_symbol('FB', as_of_date='2020-01-01')
# Returns: SID 194817 (the Meta/Facebook company)

# For a row with Symbol='META', Date='2023-01-01'
asset = asset_finder.lookup_symbol('META', as_of_date='2023-01-01')
# Returns: SID 194817 (same company, different symbol)
```

**Result:** All rows for the same company get the same SID, maintaining continuity.

### How Zipline's Asset Database Works

Zipline's `asset_finder.lookup_symbol(symbol, as_of_date)` is intelligent:

1. **Knows about symbol changes**: The asset database tracks when companies changed tickers
2. **Returns the correct entity**: For FB on 2020-01-01, it knows this is the company now called META
3. **Consistent SIDs**: Always returns the same SID for the same company, regardless of symbol

### Using TemporalSIDMapper

The `temporal_sid_mapper.py` module provides a robust solution:

```python
from temporal_sid_mapper import TemporalSIDMapper

# Initialize with your asset finder
mapper = TemporalSIDMapper(asset_finder)

# Map your dataframe automatically
custom_data['Sid'] = mapper.map_dataframe_auto(
    custom_data,
    symbol_col='Symbol',
    date_col='Date',
    verbose=True
)
```

#### How It Works

1. **For each row** with (Symbol, Date)
2. **Looks up** `asset_finder.lookup_symbol(Symbol, as_of_date=Date)`
3. **Extracts SID** from the returned asset
4. **Caches results** for performance

#### Performance Modes

The mapper automatically chooses the best strategy:

- **< 100k rows**: Simple iteration
- **100k-1M rows**: Grouped lookups (same symbol processed together)
- **> 1M rows**: Parallel processing with threading

#### What This Handles Automatically

- **FB → META** (Facebook rebranded to Meta in 2021)
- **GOOG → GOOGL** (Google ticker changes)
- **Any company name change** in Zipline's asset database
- **Mergers and acquisitions** where ticker changed
- **Spin-offs** where new ticker was created

---

## Automatic Symbol Mapping

### AutoSymbolMapper

For symbols not in the bundle, use **intelligent fuzzy matching** based on company names.

```
┌─────────────────────┐
│  Your CSV           │
│  Symbol: FB         │
│  Name: Facebook Inc │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────────┐
    │ Auto Mapper      │
    │ 1. Detect: FB    │
    │    not in bundle │
    │ 2. Match by name │
    │ 3. Find: META    │
    │    (95% match)   │
    │ 4. Auto-apply    │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ Persistent File  │
    │ {"FB": "META"}   │
    └──────────────────┘
```

### Key Features

1. **Auto-Detection**: Finds unmapped symbols automatically
2. **Fuzzy Matching**: Uses company name similarity (not just symbols)
3. **Confidence Scoring**: Only auto-applies high-confidence matches (≥85%)
4. **Persistent Learning**: Saves mappings to JSON file
5. **Manual Review**: Flags uncertain matches for your approval

### Usage

```python
from symbol_mapper_auto import AutoSymbolMapper

# Create auto mapper
auto_mapper = AutoSymbolMapper(asset_finder)

# Auto-detect and apply mappings
custom_data = auto_mapper.auto_learn_and_map(
    custom_data,
    symbol_col='Symbol',
    name_col='CompanyCommonName',
    auto_threshold=0.85,  # 85% confidence for auto-apply
    verbose=True
)

# Now custom_data has normalized symbols!
# Proceed with temporal SID mapping...
```

### Example Output

```
================================================================================
DETECTING UNMAPPED SYMBOLS
================================================================================
Building bundle symbol cache...
Cached 30,701 bundle symbols

Found 2 unmapped symbols
  High confidence (>=0.85): 2
  Needs review (0.7-0.85): 0
  No match (<0.7): 0

================================================================================
APPLYING SYMBOL MAPPINGS
================================================================================
  ✓ FB → META (confidence: 0.95)
  ✓ GOOG → GOOGL (confidence: 0.92)

✓ Auto-applied 2 mappings
  Saved to: /root/.zipline/data/custom/symbol_mappings.json
```

### Mapping File Location

```
/root/.zipline/data/custom/symbol_mappings.json
```

- **Persistent**: Survives container restarts
- **Editable**: JSON format for manual adjustments
- **Portable**: Can copy between systems

### Confidence Scoring

The mapper scores matches based on:

1. **Company Name Similarity** (primary)
   - Uses SequenceMatcher for fuzzy matching

2. **Symbol Similarity** (bonus)
   - Similar symbols get +0.1 bonus

3. **Thresholds**:
   - **≥0.85**: Auto-apply (high confidence)
   - **0.70-0.85**: Manual review (uncertain)
   - **<0.70**: Skip (no good match)

---

## Implementation

### Complete Workflow in load_csv_fundamentals.ipynb

**Step 1: Auto Symbol Mapping** (before SID mapping):

```python
from symbol_mapper_auto import AutoSymbolMapper

print("="*80)
print("AUTO SYMBOL MAPPING")
print("="*80)

auto_mapper = AutoSymbolMapper(asset_finder)

custom_data = auto_mapper.auto_learn_and_map(
    custom_data,
    symbol_col='Symbol',
    name_col='CompanyCommonName',
    auto_threshold=0.85,
    verbose=True
)

print("="*80)
```

**Step 2: Temporal SID Mapping**:

```python
import sys
sys.path.insert(0, '/app/examples/shared_modules')
from temporal_sid_mapper import TemporalSIDMapper

print("="*80)
print("Mapping symbols to SIDs with temporal awareness")
print("="*80)

mapper = TemporalSIDMapper(asset_finder)

custom_data['Sid'] = mapper.map_dataframe_auto(
    custom_data,
    symbol_col='Symbol',
    date_col='Date',
    verbose=True
)

# Report results
mapped = custom_data['Sid'].notna().sum()
unmapped = custom_data['Sid'].isna().sum()

print(f"\n✓ Mapping results:")
print(f"  Mapped: {mapped:,} rows ({mapped/len(custom_data)*100:.1f}%)")
print(f"  Unmapped: {unmapped:,} rows")

# Remove unmapped rows
custom_data = custom_data[custom_data['Sid'].notna()].copy()
custom_data['Sid'] = custom_data['Sid'].astype(int)

print(f"✓ Final dataset: {len(custom_data):,} rows with valid SIDs")
```

### Verification

After mapping, verify continuity for companies that changed symbols:

```python
# Check META/FB data continuity
meta_sid = mapper.map_single_row('META', '2023-01-01')
fb_sid = mapper.map_single_row('FB', '2020-01-01')

print(f"META (2023): SID {meta_sid}")
print(f"FB (2020): SID {fb_sid}")
print(f"Same company: {meta_sid == fb_sid}")  # Should be True!

# Check data in database
conn = sqlite3.connect(db_path)
query = f"""
    SELECT
        MIN(Date) as first_date,
        MAX(Date) as last_date,
        COUNT(*) as rows,
        GROUP_CONCAT(DISTINCT Symbol) as symbol_list
    FROM Price
    WHERE Sid = {meta_sid}
"""
result = pd.read_sql(query, conn)
print("\nContinuity check:")
print(result)
# Should show: FB and META symbols, continuous date range
```

---

## Troubleshooting

### Q: Why are some symbols unmapped?

The symbol doesn't exist in your Sharadar bundle. Check:
- Symbol spelling (case-sensitive)
- Bundle coverage (some symbols may not be in Sharadar)
- Date range (symbol may not have existed on that date)

### Q: Mapping is slow. How to speed up?

1. The mapper uses caching - first run is slow, subsequent lookups are fast
2. For huge datasets (>1M rows), use `map_dataframe_parallel()`
3. Consider filtering data before mapping

### Q: How to verify mapping is correct?

```python
# Check a known symbol change
fb_2020 = mapper.map_single_row('FB', '2020-01-01')
meta_2023 = mapper.map_single_row('META', '2023-01-01')
assert fb_2020 == meta_2023, "FB and META should map to same SID!"
```

### Q: When fuzzy matching fails?

If company names are too different, add manually to mapping file:

```json
{
  "OLD_SYMBOL": "NEW_SYMBOL",
  "GOOG-A": "GOOGL",
  "GOOG-C": "GOOG"
}
```

### Q: Manual Review Required?

For uncertain matches (70-85% confidence), edit the mapping file:

```
/root/.zipline/data/custom/symbol_mappings.json
```

Format: `{"CSV_SYMBOL": "BUNDLE_SYMBOL"}`

---

## Best Practices

### When to Use Temporal Mapping

**Always use temporal mapping when:**
- Loading historical fundamental data
- Data spans multiple years
- Your data has company ticker symbols
- Companies in your data might have changed names

**You can skip temporal mapping when:**
- Loading only current data (last few months)
- You know for certain no symbols changed
- Data is already mapped to SIDs

### Performance Tips

1. **Enable caching**: The mapper caches (symbol, date) lookups
2. **Group by symbol first**: If manually processing, group data by symbol
3. **Use parallel mode**: For >1M rows, parallel processing is 4-8x faster
4. **Pre-filter dates**: Remove rows with dates outside your bundle's range

### File Locations

| File | Location |
|------|----------|
| TemporalSIDMapper | `examples/shared_modules/temporal_sid_mapper.py` |
| AutoSymbolMapper | `examples/shared_modules/symbol_mapper_auto.py` |
| Mapping File | `~/.zipline/data/custom/symbol_mappings.json` |

---

## Summary

| Aspect | Manual Approach | Auto Mapper + Temporal |
|--------|-----------------|------------------------|
| **Setup** | Edit Python code | Two function calls |
| **Learning** | None | Automatic from CSV |
| **Persistence** | Code changes | JSON file |
| **Maintenance** | Update for each change | Self-updating |
| **Symbol Changes** | Must know all | Handled automatically |

The combination of **AutoSymbolMapper** and **TemporalSIDMapper** provides a complete solution for mapping any custom data to Zipline's SID system, handling symbol changes automatically and maintaining data continuity.
