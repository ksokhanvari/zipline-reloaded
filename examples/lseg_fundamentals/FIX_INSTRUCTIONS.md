# Fix for Sharadar Metadata Columns in load_csv_fundamentals.ipynb

## Problem
The Sharadar metadata columns (`sharadar_exchange`, `sharadar_category`, `sharadar_is_adr`, etc.) are currently created as **REAL** type in SQLite, but they contain **TEXT** values. This causes the Pipeline loader to read them as float64 with NaN values instead of as strings.

## Solution
Modify **Cell 15** in `load_csv_fundamentals.ipynb` to explicitly define these columns as TEXT type when creating the database table.

## Location
Find this code block around line 240 in Cell 15 (in the "Create table on first chunk" section):

### BEFORE (Current Code - BROKEN):
```python
    # Create table on first chunk
    if not table_created:
        cursor.execute("DROP TABLE IF EXISTS Price")

        # Build schema from chunk columns
        columns = []
        for col in chunk.columns:
            if col == 'Sid':
                columns.append(f'"{col}" INTEGER')
            elif col in ['Date', 'Symbol', 'CompanyCommonName', 'GICSSectorName', 'TradeDate']:
                columns.append(f'"{col}" TEXT')
            else:
                columns.append(f'"{col}" REAL')
```

### AFTER (Fixed Code):
```python
    # Create table on first chunk
    if not table_created:
        cursor.execute("DROP TABLE IF EXISTS Price")

        # Define which columns should be TEXT (not REAL)
        TEXT_COLUMNS = {
            'Date', 'Symbol', 'CompanyCommonName', 'GICSSectorName', 'TradeDate',
            # Sharadar metadata columns - MUST be TEXT for proper Pipeline filtering
            'sharadar_exchange', 'sharadar_category', 'sharadar_location',
            'sharadar_sector', 'sharadar_industry', 'sharadar_sicsector',
            'sharadar_sicindustry', 'sharadar_scalemarketcap'
        }

        # Build schema from chunk columns
        columns = []
        for col in chunk.columns:
            if col == 'Sid':
                columns.append(f'"{col}" INTEGER')
            elif col in TEXT_COLUMNS:
                columns.append(f'"{col}" TEXT')
            else:
                columns.append(f'"{col}" REAL')
```

## What Changed
Added a `TEXT_COLUMNS` set that includes all Sharadar metadata columns. Now when the table is created, these columns will be defined as TEXT type instead of REAL type.

## After Making This Change
1. Open `load_csv_fundamentals.ipynb` in Jupyter
2. Find Cell 15 (the large "MEMORY-EFFICIENT CHUNK PROCESSING" cell)
3. Replace the "Create table on first chunk" section with the fixed code above
4. Run the notebook to reload your fundamentals database
5. The Sharadar metadata columns will now be TEXT type and work properly with Pipeline filters

## Verification
After reloading, you can verify the fix worked:

```python
import sqlite3
conn = sqlite3.connect('/data/custom_databases/fundamentals.sqlite')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(Price)")
columns = cursor.fetchall()

# Check Sharadar columns
for col in columns:
    if 'sharadar' in col[1].lower():
        print(f"{col[1]:40} {col[2]}")  # Should show TEXT, not REAL
conn.close()
```

You should see:
```
sharadar_exchange                        TEXT
sharadar_category                        TEXT
sharadar_location                        TEXT
sharadar_sector                          TEXT
sharadar_industry                        TEXT
sharadar_sicsector                       TEXT
sharadar_sicindustry                     TEXT
sharadar_scalemarketcap                  TEXT
sharadar_is_adr                          REAL
```

Note: `sharadar_is_adr` remains REAL (float) because it's 0.0 or 1.0, not text.

## Impact
After this fix, you can:
- ✅ Use `CustomFundamentals.sharadar_exchange.latest` in Pipeline and get actual exchange names (not NaN)
- ✅ Use `CustomFundamentals.sharadar_category.latest` in Pipeline and get actual categories (not NaN)
- ✅ Filter directly in Pipeline using CustomFilter classes without the lazy-loading workaround
- ✅ Eliminate the need for the metadata cache in your strategy files
