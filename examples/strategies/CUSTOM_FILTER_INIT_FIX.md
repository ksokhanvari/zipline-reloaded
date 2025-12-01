# CustomFilter Initialization Fix

## Problem

After fixing the TickerMetadata conflict, the strategy failed with:
```
Error: Term.__getitem__() expected a value of type Asset for argument 'key', but got int instead.
```

## Root Cause

The issue was with how CustomFilter classes were being initialized. When passing columns as parameters to `__init__()` and calling `super().__init__()`, the Pipeline framework wasn't properly setting up the filter instances. This caused the filter to try using integer indexing instead of Asset objects.

## Solution

Use `__new__()` instead of `__init__()` to set the `inputs` and `params` attributes **before** the CustomFilter base class initializes the instance.

### Changes Made

Updated all CustomFilter classes in `sharadar_filters.py` to use this pattern:

**Before**:
```python
class ExchangeFilter(CustomFilter):
    window_length = 1

    def __init__(self, exchange_column):
        self.inputs = [exchange_column]
        super().__init__()

    def compute(self, today, assets, out, exchange):
        # ... implementation
```

**After**:
```python
class ExchangeFilter(CustomFilter):
    window_length = 1

    def __new__(cls, exchange_column):
        # Create instance with inputs set BEFORE initialization
        instance = super().__new__(cls)
        instance.inputs = [exchange_column]
        return instance

    def __init__(self, exchange_column):
        # Don't call super().__init__() - __new__ handles initialization
        pass

    def compute(self, today, assets, out, exchange):
        # ... implementation
```

### Why This Works

1. **`__new__()` runs before `__init__()`** - This is Python's instance creation method
2. **Sets `inputs` before Pipeline initialization** - The CustomFilter base class needs `inputs` to be set when it initializes
3. **Avoids calling `super().__init__()`** - This would reset the inputs we just set

### Updated Classes

All five CustomFilter classes were updated with this pattern:
1. `ExchangeFilter`
2. `CategoryFilter`
3. `ADRFilter`
4. `SectorFilter` (also sets `params`)
5. `ScaleMarketCapFilter` (also sets `params`)

### Additional Improvements

Also optimized the `compute()` methods to use list comprehensions instead of explicit loops:

**Before**:
```python
result = np.zeros(len(exchange_vals), dtype=bool)
for i, val in enumerate(exchange_vals):
    val_str = str(val) if val is not None else ''
    result[i] = val_str in valid_exchanges
out[:] = result
```

**After**:
```python
out[:] = np.array([str(val) in valid_exchanges if val is not None else False
                   for val in exchange_vals], dtype=bool)
```

## Files Updated

- `examples/strategies/sharadar_filters.py` - All five CustomFilter classes updated
- Copied to Docker container

## Result

The CustomFilter classes should now properly initialize and work with the Pipeline framework. The `Term.__getitem__()` error should be resolved.
