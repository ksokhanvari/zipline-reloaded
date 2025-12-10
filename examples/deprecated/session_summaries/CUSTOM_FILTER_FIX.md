# CustomFilter Fix - Resolved TickerMetadata Conflict

## Problem Summary

The initial implementation of universe filtering using CustomFilter classes caused a `ValueError: 2` in the Pipeline loader's pivot operation.

**Root Cause**: The `TickerMetadata` DataSet class in `sharadar_filters.py` had `CODE = "fundamentals"`, which conflicted with the `CustomFundamentals` ms.Database class (also with `CODE = "fundamentals"`). When the CustomFilter classes referenced `TickerMetadata` columns, the Pipeline loader tried to process them as a separate DataSet, causing the pivot error.

## Solution

Changed the CustomFilter classes to accept columns as constructor parameters instead of hardcoding references to a conflicting DataSet.

### Changes Made

#### 1. `/Users/kamran/Documents/Code/Docker/zipline-reloaded/examples/strategies/sharadar_filters.py`

**Removed**:
- `TickerMetadata` DataSet class (lines 23-42)
- Hardcoded `inputs = [TickerMetadata.sharadar_exchange]` in filter classes

**Added**:
- `__init__()` methods to each CustomFilter class that accept column parameters
- Updated docstrings to show how to pass columns from your fundamentals dataset

**Example Changes**:

**Before**:
```python
class ExchangeFilter(CustomFilter):
    inputs = [TickerMetadata.sharadar_exchange]
    window_length = 1

    def compute(self, today, assets, out, exchange):
        # ... implementation
```

**After**:
```python
class ExchangeFilter(CustomFilter):
    window_length = 1

    def __init__(self, exchange_column):
        self.inputs = [exchange_column]
        super().__init__()

    def compute(self, today, assets, out, exchange):
        # ... implementation
```

#### 2. `/Users/kamran/Documents/Code/Docker/zipline-reloaded/examples/strategies/LS-ZR-ported.py`

**Removed**:
- `TickerMetadata` from imports (line 75)

**Changed**:
- Filter instantiation to pass `CustomFundamentals` columns as parameters

**Before**:
```python
exchange_filter = ExchangeFilter()  # NYSE, NASDAQ, NYSEMKT only
category_filter = CategoryFilter()  # Domestic Common Stock only
adr_filter = ADRFilter()  # Excludes ADRs (returns True for non-ADRs)
```

**After**:
```python
exchange_filter = ExchangeFilter(CustomFundamentals.sharadar_exchange)
category_filter = CategoryFilter(CustomFundamentals.sharadar_category)
adr_filter = ADRFilter(CustomFundamentals.sharadar_is_adr)
```

#### 3. Updated `create_sharadar_universe()` Convenience Function

**Before**:
```python
def create_sharadar_universe(
    exchanges=None,
    include_adrs=False,
    ...
):
    filters.append(ExchangeFilter())
    filters.append(CategoryFilter())
    # ...
```

**After**:
```python
def create_sharadar_universe(
    fundamentals_dataset,  # NEW: must pass dataset with metadata columns
    exchanges=None,
    include_adrs=False,
    ...
):
    filters.append(ExchangeFilter(fundamentals_dataset.sharadar_exchange))
    filters.append(CategoryFilter(fundamentals_dataset.sharadar_category))
    # ...
```

## How It Works Now

1. **CustomFundamentals** ms.Database defines metadata columns:
   - `sharadar_exchange = ms.Column(object)`
   - `sharadar_category = ms.Column(object)`
   - `sharadar_is_adr = ms.Column(float)`

2. **CustomFilter classes** accept these columns as parameters in their `__init__()` method

3. **No DataSet conflict** - Only one class (`CustomFundamentals`) uses `CODE = "fundamentals"`

4. **Pipeline loader** processes columns correctly without pivot errors

## Benefits

✅ **Eliminates DataSet conflict** - No more `TickerMetadata` competing with `CustomFundamentals`

✅ **More flexible** - Filter classes can be used with any fundamentals dataset that has the required columns

✅ **Cleaner architecture** - Filters don't hardcode their data source

✅ **No code duplication** - Metadata columns defined once in `CustomFundamentals`

## Usage Example

```python
# In your strategy file (LS-ZR-ported.py):

# Step 1: Define your fundamentals database with metadata columns
class CustomFundamentals(ms.Database):
    CODE = "fundamentals"

    # ... other columns ...

    # Sharadar metadata columns for universe filtering
    sharadar_exchange = ms.Column(object)
    sharadar_category = ms.Column(object)
    sharadar_is_adr = ms.Column(float)

# Step 2: In make_pipeline(), pass columns to filters
from sharadar_filters import ExchangeFilter, CategoryFilter, ADRFilter

exchange_filter = ExchangeFilter(CustomFundamentals.sharadar_exchange)
category_filter = CategoryFilter(CustomFundamentals.sharadar_category)
adr_filter = ADRFilter(CustomFundamentals.sharadar_is_adr)

# Step 3: Combine filters
us_equities_universe = (
    market_cap_filter &
    exchange_filter &
    category_filter &
    adr_filter
)
```

## Testing

After applying this fix:
1. The `ValueError: 2` in the pivot operation should no longer occur
2. Universe filtering will happen correctly in the Pipeline
3. The metadata columns will load as TEXT from the database (after database reload)
4. The strategy should run successfully with proper universe filtering

## Files Updated

1. `examples/strategies/sharadar_filters.py` - Removed TickerMetadata, updated all filter classes
2. `examples/strategies/LS-ZR-ported.py` - Updated filter instantiation, removed TickerMetadata import
3. Both files copied to Docker container

## Next Steps

The strategy is now ready to run. The CustomFilter-based universe filtering should work correctly without the pivot error.
