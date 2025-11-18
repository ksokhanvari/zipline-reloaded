# FB/META SID Issue Analysis

## The Question

Why does META show NaN values before 2022-02 but has data after? The LSEG database should have continuous data for the same company, even after a name change.

## Investigation Results

### Zipline Bundle (Sharadar) Structure

**SID 194817 (META)**:
- Symbol: META
- Start date: 2012-05-18
- End date: 2025-11-17
- This is the PRIMARY equity for Facebook/Meta

**SID 644713 (FB)**:
- Symbol: FB
- Start date: 2025-06-26 (Future date!)
- End date: 2025-11-17
- This appears to be a DIFFERENT/FUTURE security, NOT the historical FB

### LSEG Database Structure

**Symbol 'FB'**:
- Rows: 2,422
- SID: 644713
- Date range: 2012-05-18 to 2021-12-31

**Symbol 'META'**:
- Rows: 970
- SID: 194817
- Date range: 2021-12-31 to 2025-11-11

## The Root Cause

Your LSEG data provider split the data into TWO different records when Facebook renamed to Meta:

1. **Historical FB data (2012-2021)**: Stored under SID 644713, Symbol "FB"
2. **Current META data (2022-2025)**: Stored under SID 194817, Symbol "META"

However, **Zipline treats SID 194817 as the continuous equity** from 2012 to 2025. This creates a mismatch:

- Before 2022: Zipline requests SID 194817 data → LSEG database has NO data for SID 194817 → Returns NaN
- After 2022: Zipline requests SID 194817 data → LSEG database HAS data for SID 194817 → Returns values

## The Solution

You need to **consolidate the historical FB data under SID 194817** so Zipline can access it. There are two approaches:

### Option 1: Update LSEG Database (Recommended)

Update all FB records (SID 644713) to use SID 194817:

```sql
-- Backup first!
CREATE TABLE Price_backup AS SELECT * FROM Price;

-- Update FB data to use META's SID
UPDATE Price
SET SID = 194817
WHERE SID = 644713 AND Symbol = 'FB';

-- Verify
SELECT Symbol, SID, COUNT(*), MIN(Date), MAX(Date)
FROM Price
WHERE SID = 194817
GROUP BY Symbol, SID;
```

After this update, SID 194817 should have:
- FB data from 2012-05-18 to 2021-12-31
- META data from 2021-12-31 to 2025-11-11
- Total: Continuous data for the entire company history

### Option 2: Custom Loader with Symbol Mapping (More Complex)

Modify `CustomSQLiteLoader` to map symbol changes:

```python
class SymbolMappingLoader(CustomSQLiteLoader):
    """
    Custom loader that handles symbol renames.
    Maps historical symbols to current SIDs.
    """

    SYMBOL_MAPPINGS = {
        'FB': 194817,  # Map FB to META's SID
    }

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        # When requesting SID 194817, also fetch data from FB (SID 644713)
        # Merge the results chronologically
        pass
```

## Why This Happened

When LSEG (or your data provider) exports fundamental data, they likely:

1. Export each ticker symbol as a separate record
2. When FB renamed to META, they created a NEW record with the new symbol
3. The SIDs in the export matched the tickers at export time

This is common with financial data providers - they track symbol changes as separate entities rather than maintaining continuity.

## Recommendation

**Use Option 1** - update the database. It's cleaner, simpler, and matches how Zipline expects data to be structured. Zipline treats SID 194817 as the continuous entity representing Facebook/Meta from 2012 onwards, so your database should too.

## Verification After Fix

After applying Option 1, run this query to verify:

```python
cursor.execute('''
    SELECT Symbol, COUNT(*), MIN(Date), MAX(Date)
    FROM Price
    WHERE SID = 194817
    GROUP BY Symbol
    ORDER BY MIN(Date)
''')

# Expected output:
# FB    | ~2,400 rows | 2012-05-18 | 2021-12-31
# META  | ~970 rows   | 2021-12-31 | 2025-11-11
# Total: ~3,370 rows of continuous data
```

Then re-run your backtest - META/FB should have data throughout the entire backtest period!
