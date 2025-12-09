# Deprecated LSEG Fundamentals Files

This directory contains deprecated notebooks that have been superseded by the current recommended workflow.

## Deprecated Files

### `add_sharadar_metadata_to_fundamentals_full.ipynb`
- **Status**: Deprecated (Nov 2023)
- **Reason**: Replaced by single-source-of-truth pattern
- **Replaced By**: `../add_sharadar_metadata_to_fundamentals.ipynb` + `../add_sharadar_metadata_to_fundamentals.py`
- **Issue**: Had embedded functions instead of importing from `.py` script, leading to code duplication

### `add_sharadar_tickers.ipynb`
- **Status**: Deprecated (Nov 2023)
- **Reason**: Less efficient approach
- **Replaced By**: CSV enrichment workflow (`../add_sharadar_metadata_to_fundamentals.ipynb`)
- **Issue**: Creates massive `SharadarTickersDaily` table (~241M rows) for joins, uses more disk space and slower queries

### `add_sharadar_tickers_fast.ipynb`
- **Status**: Deprecated (Nov 2023)
- **Reason**: Same as `add_sharadar_tickers.ipynb` but with optimized chunking
- **Replaced By**: CSV enrichment workflow (`../add_sharadar_metadata_to_fundamentals.ipynb`)
- **Issue**: Still creates large normalized table when enriched CSV approach is simpler and faster

## Current Recommended Workflow

Use the **CSV Enrichment Workflow** instead:

1. **Enrich CSV**: Run `../add_sharadar_metadata_to_fundamentals.ipynb`
   - Adds 9 Sharadar metadata columns to LSEG CSV
   - 95.6% match rate
   - Interactive visualizations

2. **Load to Database**: Run `../load_csv_fundamentals.ipynb`
   - Loads enriched CSV into `fundamentals.sqlite`
   - Metadata columns available directly in Pipeline queries
   - No joins needed

## Why These Files Were Kept

These files are preserved for:
- Reference for alternative approaches
- Understanding historical implementation decisions
- Potential future use cases where normalized table approach is preferred

If you need to use these files for a specific use case, you can copy them back to the parent directory. However, the CSV enrichment workflow is recommended for most use cases.
