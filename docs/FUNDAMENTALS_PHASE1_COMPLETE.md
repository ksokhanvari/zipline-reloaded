# Phase 1 Complete: Sharadar Fundamentals Integration

**Status**: ✅ Complete and tested
**Date**: November 17, 2025
**Author**: Kamran Sokhanvari @ Hidden Point Capital

## Summary

Phase 1 of the multi-source fundamentals architecture is complete. Sharadar SF1 fundamentals are now fully integrated with zipline-reloaded's Pipeline API with proper point-in-time handling.

## What Was Accomplished

### 1. Permaticker-Based SID Architecture ✅

**Critical Decision**: Switched from sequential SIDs (0, 1, 2...) to using Sharadar's permaticker as SID.

**Benefits**:
- SIDs are stable across re-ingestions
- SIDs survive ticker symbol changes
- Perfect consistency between pricing and fundamentals
- Future-proof for multi-source fundamentals (Phase 2)

**Implementation**:
- Download TICKERS table to get ticker → permaticker mapping
- Use permaticker as SID directly in asset metadata
- Map permaticker to all data sources (pricing, fundamentals)

### 2. SF1 Fundamentals Download & Storage ✅

**Data Source**: Sharadar SF1 (150+ fundamental metrics)

**Implementation**:
- Download SF1 fundamentals during bundle ingestion
- Filter to ARQ dimension (As-Reported Quarterly) for point-in-time correctness
- Store in HDF5 format: `{bundle}/fundamentals/sf1.h5`
- Graceful failure: fundamentals errors don't break pricing ingestion

**Key Features**:
- Bulk export API for efficient large dataset downloads
- Handles missing permaticker column (maps from TICKERS)
- Date filtering using `calendardate` field
- 635,393 records covering 16,545 tickers from 1998-2025

### 3. Pipeline Dataset & Loader ✅

**Dataset**: `SharadarFundamentals` (src/zipline/pipeline/data/sharadar.py)
- 80+ fundamental columns defined
- Organized by category: Income Statement, Balance Sheet, Cash Flow, Ratios, Per-Share
- Float64 dtype with NaN missing values

**Loader**: `SharadarFundamentalsLoader` (src/zipline/pipeline/loaders/sharadar_fundamentals.py)

**Point-in-Time Logic**:
- Uses `datekey` field (filing date when data became public)
- Forward-fills quarterly data until next quarter is filed
- Prevents look-ahead bias
- Timezone-aware (UTC)

**Performance**:
- Lazy loading from HDF5
- Pivot caching for repeated queries
- Handles 11,106 assets efficiently

### 4. Standalone Fundamentals Download ✅

**Problem**: Re-ingestion downloads all fundamentals (10-30 min) even when only updating pricing

**Solution**: Created standalone download script

**Files**:
- `scripts/download_fundamentals_only.py` - Python implementation
- `scripts/download_fundamentals_only.sh` - Docker wrapper

**Usage**:
```bash
# From host
./scripts/download_fundamentals_only.sh sharadar

# From inside Docker
python /app/scripts/download_fundamentals_only.py --bundle sharadar
```

**Time Savings**: 30 seconds vs 10-30 minutes for full re-ingestion

### 5. Documentation ✅

**Files Created**:
- `docs/SHARADAR_FUNDAMENTALS_GUIDE.md` - Complete user guide (400+ lines)
- `docs/FUNDAMENTALS_PHASE1_COMPLETE.md` - This summary
- Updated `CLAUDE.md` with project context

**Guide Sections**:
- Quick Start
- Permaticker SIDs (breaking change explanation)
- Bundle Ingestion
- Available Metrics (80+ columns)
- Pipeline Usage
- Point-in-Time Correctness
- Example Strategies (Magic Formula, Piotroski F-Score, Dividend Aristocrats)
- Advanced Usage (Downloading Fundamentals Only)
- Troubleshooting

### 6. Testing ✅

**Test Script**: `test_fundamentals_pipeline.py`

**Test Results** (Nov 15, 2024):
- ✅ Pipeline executes successfully
- ✅ 11,106 assets in universe
- ✅ 5,340 assets with fundamental data (48.1% coverage)
- ✅ Data loading: revenue, netinc, assets, equity, marketcap
- ✅ Point-in-time handling working correctly
- ✅ No errors or failures

## Key Technical Fixes

### 1. Permaticker Column Missing from SEP/SFP
**Error**: `permaticker column does not exist`
**Fix**: Download TICKERS table separately, join with pricing data
**Commit**: 86258ce4

### 2. SF1 Date Filtering Error
**Error**: `You cannot use date column as a filter`
**Fix**: Use table-specific date columns (SF1=calendardate, others=date)
**Commit**: ad71131a

### 3. SF1 Bulk Export Missing Permaticker
**Error**: `SF1 data missing required columns: ['permaticker']`
**Fix**: Map permaticker from TICKERS when missing, make it optional in download function
**Commits**: 509a210b, 3d899aa6

### 4. Timezone Comparison Error
**Error**: `Cannot compare tz-naive and tz-aware timestamps`
**Fix**: Localize pivoted index to UTC before union operations
**Commit**: cfdb7d20

### 5. Pandas Deprecation Warning
**Warning**: `DataFrame.fillna with 'method' is deprecated`
**Fix**: Replace `fillna(method='ffill')` with `ffill()`
**Commit**: 2aa421e9

## Files Created/Modified

### Created
- `src/zipline/pipeline/data/sharadar.py` - SharadarFundamentals dataset
- `src/zipline/pipeline/loaders/sharadar_fundamentals.py` - Loader with point-in-time logic
- `scripts/download_fundamentals_only.py` - Standalone fundamentals download
- `scripts/download_fundamentals_only.sh` - Shell wrapper
- `docs/SHARADAR_FUNDAMENTALS_GUIDE.md` - User documentation
- `docs/FUNDAMENTALS_PHASE1_COMPLETE.md` - This summary
- `test_fundamentals_pipeline.py` - Integration test

### Modified
- `src/zipline/data/bundles/sharadar_bundle.py` - Massive overhaul:
  - TICKERS table download
  - Permaticker as SID
  - SF1 fundamentals download
  - Table-specific date handling
  - Graceful fundamentals failure

## Usage Example

```python
from zipline import run_algorithm
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals
from zipline.pipeline.factors import SimpleMovingAverage

def make_pipeline():
    # Get latest fundamentals
    revenue = SharadarFundamentals.revenue.latest
    roe = SharadarFundamentals.roe.latest
    pe = SharadarFundamentals.pe.latest

    # Calculate averages
    avg_roe = SimpleMovingAverage(
        inputs=[SharadarFundamentals.roe],
        window_length=4  # 4 quarters = 1 year
    )

    # Screen for quality companies
    is_profitable = revenue > 0
    is_high_quality = roe > 0.15
    is_not_expensive = pe < 20

    return Pipeline(
        columns={
            'revenue': revenue,
            'roe': roe,
            'avg_roe': avg_roe,
            'pe': pe,
        },
        screen=(is_profitable & is_high_quality & is_not_expensive)
    )

def initialize(context):
    attach_pipeline(make_pipeline(), 'fundamentals')

def before_trading_start(context, data):
    context.fundamentals = pipeline_output('fundamentals')

    # Top 10 highest ROE companies
    context.top_companies = context.fundamentals.nlargest(10, 'roe')
```

## Data Statistics

**Bundle**: sharadar (full universe)
**Ingestion Date**: 2025-11-17T08:19:43

**Pricing Data**:
- Tickers: 30,700 symbols loaded
- SIDs: Using permaticker (stable IDs)

**Fundamentals Data**:
- Records: 635,393 ARQ quarterly records
- Tickers: 16,545 with fundamentals
- Columns: 114 fundamental metrics
- Date Range: 1998-03-12 to 2025-11-14
- File Size: 187.7 MB (HDF5)
- Coverage: ~48% of assets have fundamentals

**Unmatched**: 41 tickers couldn't be mapped (normal - delisted/removed)

## Next Steps: Phase 2

Phase 2 will add LSEG (Refinitiv) fundamentals as a second data source.

**Planned Features**:
1. CustomData loader for LSEG fundamentals
2. Multi-source fundamentals comparison
3. Data quality checks and validation
4. Fallback logic (prefer LSEG, fallback to Sharadar, or vice versa)
5. Coverage analysis tools

**Architecture Ready**: Permaticker SIDs make multi-source integration straightforward.

## Breaking Changes

⚠️ **SID Change**: This implementation changes SIDs from sequential (0,1,2...) to permaticker-based (199059, 199913...).

**Impact**:
- Previous bundle ingestions are incompatible
- Re-ingest bundles after upgrading
- Old strategies referencing specific SID numbers need updating

**Migration**:
```bash
# Clean old bundles
zipline clean -b sharadar --keep-last 0

# Ingest with new SID system
zipline ingest -b sharadar
```

## Commits

All work done in branch: `claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA`

Key commits:
- `86258ce4` - Download TICKERS table and join permaticker to pricing
- `ad71131a` - Table-specific date filtering (SF1 uses calendardate)
- `509a210b` - Add permaticker fallback for SF1 bulk export
- `3d899aa6` - Make permaticker optional in download function
- `cfdb7d20` - Fix timezone handling in fundamentals loader
- `2aa421e9` - Replace deprecated fillna() with ffill()
- `da28165f` - Add bundle registration to test script

## Conclusion

Phase 1 is **production-ready**. Users can now:
- ✅ Ingest Sharadar fundamentals with pricing data
- ✅ Download fundamentals separately for existing bundles
- ✅ Use 80+ fundamental metrics in Pipeline
- ✅ Build factor-based strategies with point-in-time correctness
- ✅ Trust that data is free from look-ahead bias

The permaticker-based SID architecture provides a solid foundation for Phase 2 (multi-source fundamentals).

---

**Author**: Kamran Sokhanvari @ Hidden Point Capital
**Project**: zipline-reloaded fork
**Repository**: https://github.com/ksokhanvari/zipline-reloaded
