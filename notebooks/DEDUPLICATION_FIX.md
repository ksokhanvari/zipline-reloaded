# Deduplication Fix for load_csv_fundamentals.ipynb

## Problem
IntegrityError: UNIQUE constraint failed: Price.Sid, Price.Date

## Solution
Add this code cell RIGHT AFTER the Date formatting cell and BEFORE the insertion loop.

## Find This Cell:
```python
# Convert Date to string format for SQLite
insert_data['Date'] = insert_data['Date'].dt.strftime('%Y-%m-%d')
```

## Add This NEW Cell RIGHT AFTER:

```python
# =============================================================================
# DEDUPLICATE DATA - Fix for UNIQUE constraint failures
# =============================================================================

print("\n" + "=" * 60)
print("DEDUPLICATING DATA")
print("=" * 60)

# Count before
rows_before = len(insert_data)
print(f"Rows before deduplication: {rows_before:,}")

# Check for duplicates
duplicates = insert_data[insert_data.duplicated(subset=['Sid', 'Date'], keep=False)]
if len(duplicates) > 0:
    print(f"⚠️  Found {len(duplicates):,} rows with duplicate (Sid, Date) pairs")

    # Show sample
    dup_counts = duplicates.groupby(['Sid', 'Date']).size().reset_index(name='count')
    dup_counts = dup_counts.sort_values('count', ascending=False).head(5)
    print("\nTop 5 most duplicated (Sid, Date) pairs:")
    print(dup_counts.to_string(index=False))

# Deduplicate - keep last occurrence (most recent data)
insert_data = insert_data.drop_duplicates(subset=['Sid', 'Date'], keep='last')

# Count after
rows_after = len(insert_data)
duplicates_removed = rows_before - rows_after

print(f"\nRows after deduplication: {rows_after:,}")
print(f"Duplicates removed: {duplicates_removed:,}")

if duplicates_removed == 0:
    print("✓ No duplicates found - data is clean!")
else:
    print(f"✓ Removed {duplicates_removed:,} duplicate records")
    print("  Strategy: Kept 'last' occurrence for each (Sid, Date) pair")

print("=" * 60)
```

## Then Continue With Existing Cells
The rest of your notebook should work as-is after adding this deduplication step.

## Why This Works
1. Checks for duplicates on the SAME columns that have the UNIQUE constraint: (Sid, Date)
2. Removes duplicates BEFORE attempting to insert into database
3. Keeps the 'last' occurrence, which is typically the most recent/updated data
4. Shows you exactly what was deduplicated

## Alternative: Quick One-Liner
If you just want a quick fix without the detailed reporting, use this instead:

```python
# Quick deduplication fix
print(f"Deduplicating... Before: {len(insert_data):,} rows")
insert_data = insert_data.drop_duplicates(subset=['Sid', 'Date'], keep='last')
print(f"After: {len(insert_data):,} rows (removed {len(insert_data) - len(insert_data):,} duplicates)")
```
