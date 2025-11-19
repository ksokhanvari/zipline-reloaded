# Custom Fundamentals Examples - Consistency Guide

This document ensures all custom fundamentals examples use consistent patterns for database access, dynamic dates, and loader configuration.

---

## Two Separate Example Sets

### 1. Notebook Example (User's REFE Data)

**Purpose**: Load user's custom REFE fundamental data from CSV files

**Files**:
- `notebooks/load_csv_fundamentals.ipynb`

**Database**:
- Code: `"refe-fundamentals"`
- File: `~/.zipline/data/custom/refe-fundamentals.sqlite`

**Consistency Checklist**:
- ✅ Database code: `DATABASE_NAME = "refe-fundamentals"` (Cell 4)
- ✅ Database class: `CODE = DATABASE_NAME` (Cell 20)
- ✅ Explicit db_dir: `Path.home() / '.zipline' / 'data' / 'custom'` (Cell 29)
- ✅ Dynamic dates: Last 3 months via `pd.Timestamp.now().normalize()` (Cell 31)
- ✅ Loader caching: `_loader_cache` dictionary pattern (Cell 29)
- ✅ Domain-bound datasets: Checks `dataset.__name__` instead of CODE attribute (Cell 29)
- ✅ Returns same loader instance for all columns from same dataset

---

### 2. Python Script Examples (Standalone Sample Data)

**Purpose**: Standalone examples with sample fundamental data

**Database**:
- Code: `"fundamentals"`
- File: `~/.zipline/data/custom/fundamentals.sqlite`

#### File-by-File Consistency:

##### `create_fundamentals_db.py`
**Purpose**: Creates sample fundamentals database from CSV

**Consistency**:
- ✅ Database code: `DB_CODE = "fundamentals"`
- ✅ Uses `create_custom_db()` - defaults to correct directory
- ✅ Uses `load_csv_to_db()` - defaults to correct directory
- ✅ No loader configuration needed (database creation only)

##### `check_fundamentals_db.py`
**Purpose**: Diagnostic script to verify database contents

**Consistency**:
- ✅ Database code: `DB_CODE = "fundamentals"`
- ✅ Uses `connect_db(DB_CODE)` - defaults to correct directory
- ✅ Uses `describe_custom_db(DB_CODE)` - defaults to correct directory
- ✅ No loader configuration needed (read-only queries)

##### `test_fundamentals_only.py`
**Purpose**: Test fundamentals loading without pricing data

**Consistency**:
- ✅ Database code: `CODE = "fundamentals"`
- ✅ Explicit db_dir: `db_dir = Path.home() / '.zipline' / 'data' / 'custom'`
- ✅ Passes db_dir to loader: `CustomSQLiteLoader("fundamentals", db_dir=db_dir)`
- ✅ Dynamic dates: `end_date = pd.Timestamp.now(tz='UTC').normalize()`
- ✅ Date range: Last 3 months via `pd.DateOffset(months=3)`
- ✅ LoaderDict pattern: Handles domain-bound datasets by matching dataset name + column name
- ✅ Returns same loader instance for all registered columns

##### `backtest_with_fundamentals.py`
**Purpose**: Full backtest with custom fundamentals

**Consistency**:
- ✅ Database code: `CODE = "fundamentals"`
- ✅ Explicit db_dir: `db_dir = Path.home() / '.zipline' / 'data' / 'custom'`
- ✅ Passes db_dir to loader: `CustomSQLiteLoader("fundamentals", db_dir=db_dir)`
- ✅ Dynamic dates: `end_date = pd.Timestamp.now(tz='UTC').normalize()`
- ✅ Date range: Last 3 months via `pd.DateOffset(months=3)`
- ✅ LoaderDict pattern: Full `setup_custom_loader()` function
- ✅ Domain-bound dataset support: Matches by `str(dataset).split('<')[0]`
- ✅ Registers all columns automatically via `dir(CustomFundamentals)`

##### `database_class_approach.py`
**Purpose**: Documentation showing Database class pattern

**Consistency**:
- ✅ Database code: `CODE = "fundamentals"`
- ✅ Documentation only - no runtime code
- ✅ Shows proper Database class definition pattern
- ✅ Examples use correct attribute access patterns

---

## Key Consistency Patterns

### 1. Database Directory
**Always use**: `Path.home() / '.zipline' / 'data' / 'custom'`

When to specify:
- ✅ CustomSQLiteLoader instantiation in backtests
- ✅ Pipeline engine loader factory functions
- ❌ NOT needed for `create_custom_db()` (uses default)
- ❌ NOT needed for `connect_db()` (uses default)

### 2. Dynamic Dates
**Pattern**:
```python
end_date = pd.Timestamp.now(tz='UTC').normalize()
start_date = (end_date - pd.DateOffset(months=3)).normalize()
```

**Why**:
- Ensures examples work with recently loaded data
- Avoids hardcoded dates that become stale
- `.normalize()` sets time to midnight (required for trading calendar)

### 3. Loader Caching
**Pattern**:
```python
_loader_cache = {}

if cache_key not in _loader_cache:
    _loader_cache[cache_key] = CustomSQLiteLoader(db_code, db_dir=db_dir)
return _loader_cache[cache_key]
```

**Why**:
- Pipeline engine uses object identity for grouping columns
- Same loader instance must be returned for all columns from a dataset
- Different instances cause `KeyError` in Pipeline execution

### 4. Domain-Bound Dataset Handling
**Problem**: Pipeline binds datasets to domains (e.g., `US_EQUITIES`), creating:
- `CustomFundamentalsDataSet<US>` instead of `CustomFundamentals`
- Domain-bound datasets don't have the `CODE` attribute

**Solution**: Check dataset name instead of CODE:
```python
dataset_name = getattr(dataset, '__name__', '')
if 'CustomFundamentals' in dataset_name or 'CustomFundamentals' in str(dataset):
    return loader
```

### 5. LoaderDict Pattern
**Pattern**:
```python
class LoaderDict(dict):
    def get(self, key, default=None):
        # First try exact match
        if key in self:
            return self[key]

        # Match by dataset name + column name (ignoring domain)
        if hasattr(key, 'dataset') and hasattr(key, 'name'):
            key_dataset_name = str(key.dataset).split('<')[0]
            key_col_name = key.name

            for registered_col, loader in self.items():
                if hasattr(registered_col, 'dataset') and hasattr(registered_col, 'name'):
                    reg_dataset_name = str(registered_col.dataset).split('<')[0]
                    reg_col_name = registered_col.name

                    if key_dataset_name == reg_dataset_name and key_col_name == reg_col_name:
                        return loader

        raise KeyError(key)
```

**Why**:
- Handles domain-bound datasets automatically
- Works with `custom_loader` parameter in `run_algorithm()`
- Matches columns even when domains don't match exactly

---

## Core Fix Applied

### Removed Hardcoded 'quant_' Prefix

**File**: `src/zipline/data/custom/config.py`

**Before**:
```python
def get_db_filename(db_code: str) -> str:
    return f"quant_{db_code}.sqlite"
```

**After**:
```python
def get_db_filename(db_code: str) -> str:
    return f"{db_code}.sqlite"
```

**Impact**:
- Database codes now directly map to filenames
- `"refe-fundamentals"` → `refe-fundamentals.sqlite`
- `"fundamentals"` → `fundamentals.sqlite`
- No more unexpected `quant_` prefix

---

## Testing

To verify consistency:

1. **Create sample database**:
   ```bash
   cd examples/custom_data
   python create_fundamentals_db.py
   ```

2. **Check database**:
   ```bash
   python check_fundamentals_db.py
   ```

3. **Test fundamentals loading**:
   ```bash
   python test_fundamentals_only.py
   ```

4. **Run full backtest**:
   ```bash
   python backtest_with_fundamentals.py
   ```

5. **Test notebook**:
   - Open `notebooks/load_csv_fundamentals.ipynb`
   - Run All Cells
   - Verify Pipeline examples work (Cells 31-37)

---

## Common Issues

### Issue: `FileNotFoundError: Database not found`
**Cause**: Wrong database directory or filename
**Solution**: Ensure `db_dir` parameter is set correctly

### Issue: `KeyError: (<CustomSQLiteLoader object>, 0)`
**Cause**: Creating new loader instance each time instead of caching
**Solution**: Use loader caching pattern

### Issue: `ValueError: No loader for <column>`
**Cause**: Domain-bound datasets not matching due to CODE attribute check
**Solution**: Check dataset `__name__` instead of CODE attribute

### Issue: `ValueError: Pipeline start date is not a trading session`
**Cause**: Timezone-aware dates or time not set to midnight
**Solution**: Use `.normalize()` and avoid `tz='UTC'` in `sessions_in_range()`

---

## Summary

All custom fundamentals examples now follow consistent patterns:

1. ✅ Explicit database directory when needed
2. ✅ Dynamic date selection (last 3 months)
3. ✅ Loader caching for same instance reuse
4. ✅ Domain-bound dataset support
5. ✅ Correct database filename format (no `quant_` prefix)

These patterns ensure examples work reliably across different environments and with recently loaded data.
