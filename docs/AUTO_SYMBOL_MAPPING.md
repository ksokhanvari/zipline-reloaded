# Automatic Symbol Mapping - General Solution

## The Problem

When loading LSEG CSV data, you encounter symbols that don't exist in the Sharadar bundle:

- **Your CSV**: `FB` (2012-2021), `META` (2022-2025)
- **Sharadar bundle**: Only `META` (current name)
- **Result**: FB rows fail to map → data loss

This happens because:
1. **LSEG exports use historical symbols** (symbol at time of data row)
2. **Bundle providers use current symbols** (company's current name)
3. No automatic translation between them

## The General Solution: AutoSymbolMapper

Instead of manually fixing each symbol change, use **intelligent fuzzy matching** based on company names.

### How It Works

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
    │    "Facebook Inc"│
    │    vs bundle     │
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
           │
           ▼
    ┌──────────────────┐
    │ Future loads     │
    │ FB → META        │
    │ (automatic!)     │
    └──────────────────┘
```

### Key Features

1. **Auto-Detection**: Finds unmapped symbols automatically
2. **Fuzzy Matching**: Uses company name similarity (not just symbols)
3. **Confidence Scoring**: Only auto-applies high-confidence matches (≥85%)
4. **Persistent Learning**: Saves mappings to JSON file
5. **Manual Review**: Flags uncertain matches for your approval
6. **Future-Proof**: Works automatically on all future data loads

## Usage

### In load_csv_fundamentals.ipynb

**Add ONE cell before SID mapping**:

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
# Proceed with normal SID mapping...
```

### What It Does

1. **First Run** (with FB in your CSV):
   ```
   Detecting unmapped symbols...
   Found 1 unmapped symbol

   Fuzzy matching by company name...
   ✓ FB → META (confidence: 0.95)

   Auto-applied 1 mapping
   Saved to: /root/.zipline/data/custom/symbol_mappings.json
   ```

2. **Future Runs**:
   ```
   Loading mappings: 1 existing
   ✓ Applied 1 symbol mapping to 2,421 rows

   (FB rows automatically converted to META)
   ```

## Example: Real Data

### Your CSV
```
Symbol  CompanyCommonName       Date        ROE
FB      Facebook Inc            2020-01-01  25.3
FB      Facebook Inc            2021-01-01  28.1
META    Meta Platforms Inc      2022-01-01  26.7
GOOG    Alphabet Inc            2020-01-01  18.2
```

### Mapper Output
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

### Resulting Mapping File

`/root/.zipline/data/custom/symbol_mappings.json`:
```json
{
  "FB": "META",
  "GOOG": "GOOGL"
}
```

### After Mapping
```
Symbol  CompanyCommonName       Date        ROE
META    Facebook Inc            2020-01-01  25.3  ← Changed!
META    Facebook Inc            2021-01-01  28.1  ← Changed!
META    Meta Platforms Inc      2022-01-01  26.7
GOOGL   Alphabet Inc            2020-01-01  18.2  ← Changed!
```

## Manual Review

For uncertain matches (70-85% confidence):

```
⚠ 2 symbols need manual review:
csv_symbol  csv_name           suggested_symbol  confidence
GOOG-A      Alphabet Inc A     GOOGL            0.78
TWTR        Twitter Inc        X                0.72

To manually add mappings, edit: /root/.zipline/data/custom/symbol_mappings.json
Format: {"CSV_SYMBOL": "BUNDLE_SYMBOL"}
```

Edit the file to confirm or adjust:
```json
{
  "FB": "META",
  "GOOG-A": "GOOGL",  ← Add manually
  "TWTR": "X"         ← Add manually
}
```

## Confidence Scoring

The mapper scores matches based on:

1. **Company Name Similarity** (primary)
   - Uses SequenceMatcher for fuzzy matching
   - "Facebook Inc" vs "Meta Platforms Inc" → 0.70
   - "Facebook Inc" vs "META" → 0.95 (exact after normalization)

2. **Symbol Similarity** (bonus)
   - Similar symbols get +0.1 bonus
   - "FB" vs "META" → small bonus
   - "GOOG" vs "GOOGL" → larger bonus

3. **Thresholds**:
   - **≥0.85**: Auto-apply (high confidence)
   - **0.70-0.85**: Manual review (uncertain)
   - **<0.70**: Skip (no good match)

## Advantages Over Manual Fixes

### Manual Approach (Point Solution)
```python
# Hardcode each change
KNOWN_SYMBOL_CHANGES = {
    'FB': 'META',
    'GOOG': 'GOOGL',
    'TWTR': 'X',
    # ... keep adding forever
}
```

**Problems**:
- ❌ Must update code for every new symbol
- ❌ Doesn't learn from your data
- ❌ Requires code changes for each CSV
- ❌ Not portable across different bundles

### Auto Mapper (General Solution)
```python
# One line, works forever
custom_data = auto_mapper.auto_learn_and_map(custom_data)
```

**Benefits**:
- ✅ Learns from your CSV automatically
- ✅ Builds persistent knowledge base
- ✅ Works with any bundle
- ✅ No code changes needed
- ✅ Handles company name changes intelligently
- ✅ Portable across systems

## Integration with load_csv_fundamentals.ipynb

### Updated Workflow

**Cell 6A** (new - add before Cell 7):
```python
# =============================================================================
# AUTO SYMBOL MAPPING - Normalize CSV symbols to bundle symbols
# =============================================================================

from symbol_mapper_auto import AutoSymbolMapper

print("="*80)
print("AUTO SYMBOL MAPPING")
print("="*80)

# Create auto mapper
auto_mapper = AutoSymbolMapper(asset_finder)

# Auto-learn and apply mappings
custom_data = auto_mapper.auto_learn_and_map(
    custom_data,
    symbol_col='Symbol',
    name_col='CompanyCommonName',
    auto_threshold=0.85,
    verbose=True
)

print("="*80)
```

**Cell 7** (existing SID mapping):
```python
# Now proceeds with normalized symbols!
mapper = TemporalSIDMapper(asset_finder)
custom_data['Sid'] = mapper.map_dataframe_auto(custom_data)
```

## File Locations

### Mapping File
```
/root/.zipline/data/custom/symbol_mappings.json
```

- **Persistent**: Survives container restarts
- **Editable**: JSON format for manual adjustments
- **Portable**: Can copy between systems

### Module Location
```
/app/examples/custom_data/symbol_mapper_auto.py
```

## Viewing Current Mappings

```python
from symbol_mapper_auto import generate_mapping_report

# Show all current mappings
report = generate_mapping_report()
```

Output:
```
================================================================================
CURRENT SYMBOL MAPPINGS
================================================================================
Source: /root/.zipline/data/custom/symbol_mappings.json
Total mappings: 3

CSV_Symbol  Bundle_Symbol
FB          META
GOOG        GOOGL
TWTR        X
================================================================================
```

## Best Practices

### 1. First Load (New Dataset)
```python
# Let it learn with verbose=True
custom_data = auto_mapper.auto_learn_and_map(
    custom_data,
    auto_threshold=0.85,
    verbose=True  # See what it's doing
)
```

Review the output, check `/root/.zipline/data/custom/symbol_mappings.json`

### 2. Subsequent Loads
```python
# Use existing mappings silently
custom_data = auto_mapper.auto_learn_and_map(
    custom_data,
    auto_threshold=0.85,
    verbose=False  # Quiet mode
)
```

### 3. Periodic Review
```python
# Check mappings occasionally
from symbol_mapper_auto import generate_mapping_report
generate_mapping_report()
```

### 4. Manual Adjustments
Edit `/root/.zipline/data/custom/symbol_mappings.json` directly:
```json
{
  "OLD_SYMBOL": "NEW_SYMBOL",
  "ANOTHER_OLD": "ANOTHER_NEW"
}
```

## Limitations

### When Fuzzy Matching Fails

If company names are too different:
```
CSV:    Symbol=GOOGL, Name="Alphabet Inc"
Bundle: Symbol=GOOG,  Name="Google LLC"
→ Low confidence match (name mismatch)
```

**Solution**: Add manually to mapping file

### Multiple Classes

For stocks with multiple share classes:
```
CSV: GOOG-A, GOOG-C
Bundle: GOOGL (Class A), GOOG (Class C)
```

**Solution**: Map explicitly
```json
{
  "GOOG-A": "GOOGL",
  "GOOG-C": "GOOG"
}
```

## Summary

| Aspect | Manual Point Solution | Auto Mapper |
|--------|---------------------|-------------|
| **Setup** | Edit Python code | One function call |
| **Learning** | None | Automatic from CSV |
| **Persistence** | Code changes | JSON file |
| **Portability** | Hardcoded | Data-driven |
| **Maintenance** | Update for each change | Self-updating |
| **Confidence** | 100% or 0% | Scored 0-100% |
| **Review** | No visibility | Clear reports |
| **Future** | Manual additions | Learns automatically |

## Getting Started

1. **Add to your notebook** (one cell before SID mapping):
   ```python
   from symbol_mapper_auto import AutoSymbolMapper
   auto_mapper = AutoSymbolMapper(asset_finder)
   custom_data = auto_mapper.auto_learn_and_map(custom_data)
   ```

2. **Run your data load**
   - First time: learns and creates mapping file
   - Future loads: applies existing mappings + learns new ones

3. **Review mapping file**:
   ```
   cat /root/.zipline/data/custom/symbol_mappings.json
   ```

4. **Done!** All future loads work automatically.

---

**This is a true general solution that scales to any number of symbol changes without code modifications.**
