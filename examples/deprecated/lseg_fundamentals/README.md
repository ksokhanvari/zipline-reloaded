# Deprecated LSEG Fundamentals Files

This directory contains LSEG fundamentals-related files that have been superseded or are no longer needed.

## Deprecated Files

### FIX_INSTRUCTIONS.md
- **Status**: Deprecated (Dec 2025)
- **Reason**: Temporary fix instructions that have been implemented
- **Original Purpose**: Instructions for fixing forward fill issues in LSEG fundamentals loading
- **Resolution**: Fix has been incorporated into `load_csv_fundamentals.ipynb` (Cell 12)
- **Date**: November 30, 2025

## Current LSEG Fundamentals Workflow

For current LSEG fundamentals workflow, see:

### Active Files
- `examples/lseg_fundamentals/README.md` - Main documentation
- `examples/lseg_fundamentals/add_sharadar_metadata_to_fundamentals.py` - CLI enrichment script
- `examples/lseg_fundamentals/add_sharadar_metadata_to_fundamentals.ipynb` - Interactive enrichment
- `examples/lseg_fundamentals/load_csv_fundamentals.ipynb` - Load enriched CSV to database

### Deprecated But Preserved for Reference
- `examples/lseg_fundamentals/deprecated/` - Contains old workflow implementations:
  - `add_sharadar_metadata_to_fundamentals_full.ipynb` - Old embedded-functions version
  - `add_sharadar_tickers.ipynb` - Old table-based approach
  - `add_sharadar_tickers_fast.ipynb` - Old optimized table-based approach

## Current Recommended Workflow

1. **Enrich CSV**: Run `add_sharadar_metadata_to_fundamentals.ipynb` or `.py`
   - Auto-detects newest CSV file by date
   - Downloads Sharadar metadata from NASDAQ Data Link API
   - Adds 9 metadata columns for universe filtering

2. **Load to Database**: Run `load_csv_fundamentals.ipynb`
   - Auto-detects newest enriched CSV
   - Loads with proper forward fill
   - Creates fundamentals.sqlite with all columns

## Key Improvements Since Deprecation

- ✅ Auto-detection of newest CSV files by date pattern
- ✅ Direct API download of Sharadar TICKERS (no tickers.h5 dependency)
- ✅ Enhanced deduplication (60K+ → 30K unique tickers)
- ✅ Fixed forward fill to preserve metadata columns
- ✅ Proper CHUNK_SIZE and CSV_PATH configuration (Cell 12 fix)

---

**Last Updated**: 2025-12-10
**Deprecation Date**: 2025-12-10
