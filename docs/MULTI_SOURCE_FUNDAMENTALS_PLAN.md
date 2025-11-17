# Multi-Source Fundamentals Integration - Implementation Plan

**Author**: Kamran Sokhanvari - Hidden Point Capital
**Created**: 2025-11-16
**Status**: Planning Phase
**Goal**: Enable simultaneous use of Sharadar SF1 and LSEG fundamentals in Zipline strategies

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Design Decisions](#design-decisions)
4. [Implementation Phases](#implementation-phases)
5. [Technical Specifications](#technical-specifications)
6. [File Structure](#file-structure)
7. [Testing Strategy](#testing-strategy)
8. [Migration Path](#migration-path)
9. [Success Criteria](#success-criteria)
10. [Risk Mitigation](#risk-mitigation)

---

## Executive Summary

### Business Requirements

**Problem**: Need to use multiple fundamental data sources simultaneously:
- Sharadar SF1 (premium institutional data)
- LSEG Fundamentals (alternative data for alpha generation)

**Current State**:
- ✅ Sharadar pricing integrated via bundle
- ❌ Sharadar fundamentals NOT integrated (requires API calls)
- ❌ LSEG fundamentals NOT integrated
- ✅ CustomData system exists but has SID mapping challenges

**Target State**:
- ✅ Sharadar pricing + fundamentals in unified bundle
- ✅ LSEG fundamentals via enhanced CustomData
- ✅ Both sources use consistent Sharadar SIDs
- ✅ Seamless multi-source strategies in Pipeline
- ✅ Automatic updates for all sources

### Value Proposition

1. **Alpha Generation**: Cross-validate metrics across sources, find divergences
2. **Risk Management**: Detect data quality issues through comparison
3. **Professional Grade**: Institutional-quality multi-source data architecture
4. **Scalability**: Easy to add more sources (CapitalIQ, Bloomberg, etc.)

### Timeline

- **Phase 1** (Sharadar SF1): 2-3 days
- **Phase 2** (LSEG CustomData): 1-2 days
- **Phase 3** (Integration & Examples): 1 day
- **Total**: 4-6 days

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZIPLINE PIPELINE ENGINE                       │
│                  (Unified SID Space from Sharadar)              │
└────────┬─────────────────────┬──────────────────────────┬───────┘
         │                     │                          │
         ▼                     ▼                          ▼
┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│ Sharadar Bundle │  │ Sharadar Fundamentals│  │ LSEG Fundamentals│
│   (Pricing)     │  │       (SF1)          │  │   (CustomData)   │
├─────────────────┤  ├──────────────────────┤  ├──────────────────┤
│ • SEP table     │  │ • SF1 table          │  │ • CSV/API import │
│ • Daily OHLCV   │  │ • 150+ metrics       │  │ • Maps to SIDs   │
│ • Adjustments   │  │ • Quarterly data     │  │ • SQLite storage │
│ • Defines SIDs  │  │ • Same SID space     │  │ • Pipeline access│
│                 │  │ • Integrated updates │  │ • Independent    │
└─────────────────┘  └──────────────────────┘  └──────────────────┘
         │                     │                          │
         └─────────────────────┴──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Strategy Code     │
                    │  (Multi-Source)     │
                    └─────────────────────┘

Example Strategy:
  roe_sharadar = SharadarFundamentals.roe.latest
  roe_lseg = LSEGFundamentals.roe.latest
  roe_consensus = (roe_sharadar + roe_lseg) / 2
  screen = (roe_consensus > 15) & (roe_divergence < 2)
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Bundle Ingestion (Sharadar Pricing + Fundamentals) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
            zipline ingest -b sharadar
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    Download SEP (Pricing)            Download SF1 (Fundamentals)
            │                                   │
            ▼                                   ▼
    Store in bcolz files              Store in fundamentals.sqlite
            │                                   │
            └─────────────────┬─────────────────┘
                              │
                              ▼
                   Assign SIDs (permaticker-based)
                              │
                              ▼
              Store metadata in assets-7.sqlite

┌─────────────────────────────────────────────────────────────┐
│ STEP 2: LSEG Import (CustomData - Independent)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         Load LSEG data (CSV/API)
                              │
                              ▼
         Map tickers → Sharadar SIDs
         (using asset_finder)
                              │
                              ▼
    Store in /root/.zipline/data/custom/lseg_fundamentals.sqlite

┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Strategy Execution (Multi-Source Pipeline)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              run_algorithm(custom_loader=...)
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    Load Sharadar Data                Load LSEG Data
    (pricing + fundamentals)          (fundamentals only)
            │                                   │
            └─────────────────┬─────────────────┘
                              │
                              ▼
                  All data aligned by SID
                              │
                              ▼
                  Execute Pipeline logic
                  (cross-validate, combine, screen)
```

---

## Design Decisions

### Decision 1: SID Assignment Strategy

**Choice**: Use Sharadar bundle as **master SID source**

**Rationale**:
- Sharadar bundle already assigns SIDs during ingestion
- Sharadar has permaticker (unique, stable identifier)
- All other sources (LSEG, future sources) map TO Sharadar SIDs
- Guarantees consistency across all data sources

**Alternatives Considered**:
- ❌ Use permaticker directly as SID (sparse, Sharadar-specific)
- ❌ Independent SID spaces per source (mapping nightmare)
- ❌ Custom SID registry (unnecessary complexity)

### Decision 2: Sharadar Fundamentals Integration Method

**Choice**: Integrate SF1 into Sharadar bundle (native integration)

**Rationale**:
- SF1 and SEP use same permaticker → guaranteed SID consistency
- Automatic updates with `zipline ingest`
- Point-in-time accuracy guaranteed by Sharadar
- Native Pipeline integration (no custom loaders needed)
- Professional/institutional standard

**Alternatives Considered**:
- ❌ SF1 as CustomData (SID mapping issues, manual updates)
- ❌ Direct API calls in strategies (slow, not point-in-time accurate)

### Decision 3: LSEG Integration Method

**Choice**: Use enhanced CustomData system

**Rationale**:
- LSEG data comes from different source (not Sharadar)
- Flexible import (CSV, API, manual processing)
- Independent update schedule from Sharadar
- Maps to Sharadar SIDs for consistency
- Already have working CustomData infrastructure

**Alternatives Considered**:
- ❌ Separate LSEG bundle (SID mismatch issues)
- ❌ Merge into Sharadar bundle (inappropriate, different data source)

### Decision 4: Data Storage Format

**Sharadar SF1**:
```
/root/.zipline/data/sharadar/
└── 2025-11-15T07;54;38.129303/
    ├── assets-7.sqlite              # Existing: SID definitions
    ├── adjustments.sqlite            # Existing: Corporate actions
    ├── fundamentals.sqlite           # NEW: SF1 quarterly data
    └── [pricing bcolz files]         # Existing: Daily pricing
```

**LSEG**:
```
/root/.zipline/data/custom/
└── lseg_fundamentals.sqlite
    └── Price table (Sid, Date, metrics...)
```

**Rationale**:
- SF1: Keep with bundle for integrated updates
- LSEG: Separate for independent management
- Both use SQLite for consistency and performance

### Decision 5: Point-in-Time Handling

**SF1 (Quarterly)**: Store with `reportperiod` and `datekey`
```sql
CREATE TABLE fundamentals (
    sid INTEGER,
    reportperiod DATE,    -- e.g., 2023-09-30 (quarter end)
    datekey DATE,         -- e.g., 2023-11-01 (when filed/available)
    revenue REAL,
    netinc REAL,
    -- ... other metrics
    PRIMARY KEY (sid, reportperiod, datekey)
);
```

**Why**: Sharadar provides both dates, ensuring point-in-time accuracy

**LSEG**: Map to closest prior date
```python
# When importing LSEG data
for date in trading_dates:
    # Use most recent fundamental data available as of this date
    latest_fund = lseg_data[lseg_data['date'] <= date].iloc[-1]
```

---

## Implementation Phases

### Phase 1: Sharadar SF1 Integration

**Duration**: 2-3 days

**Objectives**:
1. Download SF1 table during bundle ingestion
2. Store SF1 data in fundamentals.sqlite
3. Create SharadarFundamentals Dataset class
4. Implement SharadarFundamentalsLoader
5. Test with pricing data in Pipeline

**Deliverables**:
- [ ] Modified `src/zipline/data/bundles/sharadar_bundle.py`
- [ ] New file: `src/zipline/pipeline/data/sharadar.py`
- [ ] New file: `src/zipline/data/loaders/sharadar_fundamentals.py`
- [ ] Test file: `tests/test_sharadar_fundamentals.py`
- [ ] Documentation: `docs/SHARADAR_FUNDAMENTALS_GUIDE.md`

**Tasks**:

**1.1 Download SF1 Data** (0.5 days)
```python
# In sharadar_bundle.py
def download_sharadar_fundamentals(api_key, tickers, start_date, end_date):
    """Download SF1 (fundamentals) table from Sharadar."""
    table = 'SF1'
    dimension = 'ARQ'  # As-Reported Quarterly

    # Download via API
    url = f'https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json'
    params = {
        'api_key': api_key,
        'dimension': dimension,
        'qopts.columns': 'ticker,dimension,datekey,reportperiod,revenue,netinc,equity,...',
        'datekey.gte': start_date,
        'datekey.lte': end_date,
    }

    if tickers:
        params['ticker'] = ','.join(tickers)

    # Download with pagination
    # Return DataFrame with all SF1 metrics
```

**1.2 Process and Store SF1 Data** (0.5 days)
```python
# In sharadar_bundle.py
def process_fundamentals(sf1_data, metadata):
    """Process SF1 data and map to SIDs."""

    # Map tickers to SIDs using metadata
    ticker_to_sid = dict(zip(metadata['symbol'], metadata['sid']))
    sf1_data['sid'] = sf1_data['ticker'].map(ticker_to_sid)

    # Remove unmapped tickers
    sf1_data = sf1_data.dropna(subset=['sid'])

    # Convert to proper types
    sf1_data['sid'] = sf1_data['sid'].astype(int)
    sf1_data['reportperiod'] = pd.to_datetime(sf1_data['reportperiod'])
    sf1_data['datekey'] = pd.to_datetime(sf1_data['datekey'])

    return sf1_data

def store_fundamentals(sf1_data, output_dir):
    """Store fundamentals in SQLite database."""
    import sqlite3

    db_path = os.path.join(output_dir, 'fundamentals.sqlite')
    conn = sqlite3.connect(db_path)

    # Create table with proper schema
    sf1_data.to_sql('fundamentals', conn, if_exists='replace', index=False)

    # Create indices for fast lookups
    conn.execute('CREATE INDEX idx_sid_date ON fundamentals(sid, datekey)')
    conn.execute('CREATE INDEX idx_reportperiod ON fundamentals(reportperiod)')

    conn.close()
```

**1.3 Create Dataset Class** (0.5 days)
```python
# New file: src/zipline/pipeline/data/sharadar.py
from zipline.pipeline.data import Column, DataSet

class SharadarFundamentals(DataSet):
    """
    Sharadar SF1 Fundamentals - Quarterly financial metrics.

    Data Source: NASDAQ Data Link Sharadar SF1 table
    Update Frequency: Quarterly (with filings)
    Point-in-Time: Yes (via datekey)

    Dimensions available:
    - ARQ: As-Reported Quarterly (raw GAAP data)
    - ARY: As-Reported Yearly
    - MRQ: Most Recent Quarterly (with restatements)
    - MRY: Most Recent Yearly

    This dataset uses ARQ (As-Reported Quarterly) for point-in-time accuracy.
    """

    # Income Statement
    revenue = Column(float, description="Total revenue")
    cor = Column(float, description="Cost of revenue")
    grossprofit = Column(float, description="Gross profit")
    netinc = Column(float, description="Net income")
    eps = Column(float, description="Earnings per share (basic)")

    # Balance Sheet
    assets = Column(float, description="Total assets")
    liabilities = Column(float, description="Total liabilities")
    equity = Column(float, description="Shareholders equity")
    debt = Column(float, description="Total debt")
    cashneq = Column(float, description="Cash and equivalents")

    # Cash Flow
    ncfo = Column(float, description="Net cash from operations")
    ncfi = Column(float, description="Net cash from investing")
    ncff = Column(float, description="Net cash from financing")

    # Ratios
    roe = Column(float, description="Return on equity")
    roa = Column(float, description="Return on assets")
    roic = Column(float, description="Return on invested capital")
    de = Column(float, description="Debt to equity ratio")
    currentratio = Column(float, description="Current ratio")

    # Valuation
    marketcap = Column(float, description="Market capitalization")
    ev = Column(float, description="Enterprise value")
    pe = Column(float, description="Price to earnings ratio")
    pb = Column(float, description="Price to book ratio")
    ps = Column(float, description="Price to sales ratio")

    # Metadata
    reportperiod = Column('datetime64[ns]', description="Fiscal period end date")
    datekey = Column('datetime64[ns]', description="Date data became available")

    # Add ~150 more SF1 columns...
```

**1.4 Implement Loader** (0.5 days)
```python
# New file: src/zipline/data/loaders/sharadar_fundamentals.py
from zipline.pipeline.loaders import PipelineLoader
import sqlite3
import pandas as pd
import numpy as np

class SharadarFundamentalsLoader(PipelineLoader):
    """
    Loads Sharadar SF1 fundamentals data for Pipeline.

    Key Features:
    - Point-in-time accuracy using datekey
    - Quarterly data forward-filled until next report
    - Handles restatements properly
    """

    def __init__(self, bundle_data):
        self.bundle_data = bundle_data
        self.db_path = os.path.join(
            os.path.dirname(bundle_data.equity_daily_bar_reader._rootdir),
            'fundamentals.sqlite'
        )

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        """
        Load fundamental data as AdjustedArrays.

        Logic:
        1. For each (date, sid), find most recent fundamental data
           where datekey <= date (point-in-time correct)
        2. Forward-fill quarterly data across trading days
        3. Return as 2D arrays: rows=dates, cols=sids
        """

        # Connect to fundamentals database
        conn = sqlite3.connect(self.db_path)

        # Extract column names
        col_names = [col.name for col in columns]

        # Build query for date/sid range with point-in-time logic
        query = f"""
        SELECT
            sid,
            datekey,
            {', '.join(col_names)}
        FROM fundamentals
        WHERE sid IN ({','.join(map(str, sids))})
          AND datekey <= ?
        ORDER BY sid, datekey
        """

        # For each date, get most recent data
        result_dict = {}

        for date in dates:
            df = pd.read_sql(query, conn, params=[date.strftime('%Y-%m-%d')])

            # For each sid, get the most recent row (point-in-time)
            latest = df.groupby('sid').last()

            # Store in dict
            for col_name in col_names:
                if col_name not in result_dict:
                    result_dict[col_name] = []

                # Create row for this date across all sids
                row = []
                for sid in sids:
                    if sid in latest.index:
                        value = latest.loc[sid, col_name]
                    else:
                        value = np.nan
                    row.append(value)

                result_dict[col_name].append(row)

        conn.close()

        # Convert to AdjustedArrays
        from zipline.lib.adjusted_array import AdjustedArray

        arrays = {}
        for col, col_name in zip(columns, col_names):
            data = np.array(result_dict[col_name], dtype=float)

            arrays[col] = AdjustedArray(
                data=data,
                adjustments={},
                missing_value=col.missing_value,
            )

        return arrays
```

**1.5 Testing** (0.5 days)
```python
# Test file: tests/test_sharadar_fundamentals.py

def test_sf1_download():
    """Test SF1 data download."""
    # Test API connection
    # Verify data format
    # Check column completeness

def test_fundamentals_loader():
    """Test fundamentals loader."""
    # Create test database
    # Load data for date range
    # Verify point-in-time accuracy
    # Check forward-fill logic

def test_pipeline_integration():
    """Test SF1 in Pipeline."""
    # Create simple pipeline with SF1
    # Run for date range
    # Verify output format
    # Check SID alignment with pricing
```

---

### Phase 2: LSEG CustomData Integration

**Duration**: 1-2 days

**Objectives**:
1. Create LSEG data import notebook
2. Implement SID mapping from LSEG → Sharadar
3. Define LSEGFundamentals Dataset class
4. Test multi-source Pipeline access
5. Document workflow

**Deliverables**:
- [ ] New notebook: `notebooks/load_lseg_fundamentals.ipynb`
- [ ] New file: `src/zipline/pipeline/data/lseg.py`
- [ ] Updated: `notebooks/STRATEGY_README.md` with LSEG examples
- [ ] Documentation: `docs/LSEG_INTEGRATION_GUIDE.md`

**Tasks**:

**2.1 LSEG Import Notebook** (0.5 days)
```python
# notebooks/load_lseg_fundamentals.ipynb

# Cell 1: Setup
import pandas as pd
import sqlite3
from pathlib import Path
from zipline.data.bundles import register, load
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

# Register and load Sharadar bundle (for SID mapping)
register('sharadar', sharadar_bundle())
bundle = load('sharadar')
asset_finder = bundle.asset_finder

# Cell 2: Load LSEG Data
lseg_data = pd.read_csv('lseg_fundamentals.csv')

# Expected columns:
# - ticker: Stock symbol
# - date: Data date
# - roe: Return on equity
# - ebitda: EBITDA
# - consensus_revenue: Consensus revenue estimate
# - analyst_rating: Analyst rating (1-5)
# - ... other LSEG metrics

# Cell 3: Map Tickers to Sharadar SIDs
def map_ticker_to_sid(ticker, as_of_date):
    """
    Map LSEG ticker to Sharadar SID.

    Critical: Use as_of_date to handle ticker changes.
    """
    try:
        asset = asset_finder.lookup_symbol(ticker, as_of_date=as_of_date)
        return asset.sid
    except Exception as e:
        print(f"Warning: Could not map {ticker} as of {as_of_date}: {e}")
        return None

# Apply mapping
lseg_data['Sid'] = lseg_data.apply(
    lambda row: map_ticker_to_sid(row['ticker'], row['date']),
    axis=1
)

# Cell 4: Data Quality Checks
unmapped = lseg_data[lseg_data['Sid'].isna()]
print(f"Unmapped tickers: {len(unmapped)} / {len(lseg_data)}")
print(f"Unmapped ticker list: {unmapped['ticker'].unique()}")

# Remove unmapped
lseg_data = lseg_data.dropna(subset=['Sid'])
lseg_data['Sid'] = lseg_data['Sid'].astype(int)

# Check for duplicates
duplicates = lseg_data.groupby(['Sid', 'date']).size()
duplicates = duplicates[duplicates > 1]
if len(duplicates) > 0:
    print(f"Warning: Found {len(duplicates)} duplicate (Sid, date) pairs")
    # Keep most recent or best quality
    lseg_data = lseg_data.drop_duplicates(subset=['Sid', 'date'], keep='last')

# Cell 5: Store in SQLite
db_dir = Path('/root/.zipline/data/custom')
db_dir.mkdir(parents=True, exist_ok=True)

db_path = db_dir / 'lseg_fundamentals.sqlite'
conn = sqlite3.connect(str(db_path))

# Define column types
text_columns = []  # Add any text columns
numeric_columns = [col for col in lseg_data.columns
                   if col not in ['Sid', 'Date'] + text_columns]

# Fill NaN for text columns
for col in text_columns:
    lseg_data[col].fillna('', inplace=True)

# Store
lseg_data.to_sql('Price', conn, if_exists='replace', index=False)

# Create index
conn.execute('CREATE INDEX idx_sid_date ON Price(Sid, Date)')
conn.close()

print(f"✓ Loaded {len(lseg_data):,} LSEG fundamental records")
print(f"✓ Date range: {lseg_data['date'].min()} to {lseg_data['date'].max()}")
print(f"✓ Unique SIDs: {lseg_data['Sid'].nunique():,}")
```

**2.2 LSEG Dataset Class** (0.25 days)
```python
# New file: src/zipline/pipeline/data/lseg.py
from zipline.pipeline.data import Column, DataSet

class LSEGFundamentals(DataSet):
    """
    LSEG Refinitiv Fundamentals - Alternative fundamental data.

    Data Source: LSEG Refinitiv
    Update Frequency: Variable (typically quarterly)
    Point-in-Time: Ensured via import process

    This dataset provides alternative fundamental estimates and
    metrics for cross-validation with Sharadar data.
    """

    CODE = "lseg_fundamentals"
    LOOKBACK_WINDOW = 252

    # Core Metrics
    roe = Column(float, description="Return on equity (LSEG calculation)")
    roa = Column(float, description="Return on assets")
    ebitda = Column(float, description="EBITDA")

    # Consensus Estimates
    consensus_revenue = Column(float, description="Consensus revenue estimate")
    consensus_eps = Column(float, description="Consensus EPS estimate")
    consensus_ebitda = Column(float, description="Consensus EBITDA estimate")

    # Analyst Data
    analyst_rating = Column(float, description="Average analyst rating (1-5)")
    analyst_count = Column(float, description="Number of analysts covering")
    target_price = Column(float, description="Average analyst target price")

    # LSEG-Specific Metrics
    starmine_score = Column(float, description="StarMine analyst revisions score")
    esg_score = Column(float, description="ESG combined score")

    # Add more LSEG-specific columns as needed...
```

**2.3 Test Multi-Source Pipeline** (0.5 days)
```python
# Test combining Sharadar + LSEG
def test_multi_source_pipeline():
    from zipline.pipeline import Pipeline
    from zipline.pipeline.data.sharadar import SharadarFundamentals
    from zipline.pipeline.data.lseg import LSEGFundamentals

    # Build pipeline using both sources
    pipe = Pipeline(
        columns={
            'roe_sharadar': SharadarFundamentals.roe.latest,
            'roe_lseg': LSEGFundamentals.roe.latest,
            'roe_diff': SharadarFundamentals.roe.latest - LSEGFundamentals.roe.latest,
        }
    )

    # Run and verify both sources work
    # Check SID alignment
    # Verify data quality
```

---

### Phase 3: Integration & Examples

**Duration**: 1 day

**Objectives**:
1. Create unified build_pipeline_loaders() function
2. Implement multi-source strategy examples
3. Add data quality monitoring utilities
4. Complete documentation
5. Create migration guide

**Deliverables**:
- [ ] Example: `examples/multi_source_fundamentals/strategy_consensus.py`
- [ ] Example: `examples/multi_source_fundamentals/strategy_divergence.py`
- [ ] Utility: `src/zipline/utils/data_quality.py`
- [ ] Documentation: `docs/MULTI_SOURCE_STRATEGIES.md`
- [ ] Migration: `docs/MIGRATION_CUSTOMDATA_TO_MULTI.md`

**Tasks**:

**3.1 Unified Loader Builder** (0.25 days)
```python
# Add to src/zipline/pipeline/loaders/__init__.py
def build_multi_source_loaders(bundle_name='sharadar'):
    """
    Build unified loader map for all data sources.

    Returns LoaderDict with:
    - USEquityPricing (Sharadar SEP)
    - SharadarFundamentals (Sharadar SF1)
    - LSEGFundamentals (LSEG CustomData)
    - ... extensible to more sources
    """
    from zipline.data.bundles import load
    from zipline.data.loaders import USEquityPricingLoader
    from zipline.data.loaders.sharadar_fundamentals import SharadarFundamentalsLoader
    from zipline.data.custom import CustomSQLiteLoader
    from pathlib import Path

    # Load bundle
    bundle_data = load(bundle_name)

    # Create loaders
    pricing_loader = USEquityPricingLoader.without_fx(
        bundle_data.equity_daily_bar_reader,
        bundle_data.adjustment_reader
    )

    sharadar_fund_loader = SharadarFundamentalsLoader(bundle_data)

    lseg_loader = CustomSQLiteLoader(
        db_code='lseg_fundamentals',
        db_dir=Path('/root/.zipline/data/custom')
    )

    # Build loader dict (with domain-aware matching)
    class LoaderDict(dict):
        # ... implementation from earlier

    loaders = LoaderDict()

    # Auto-register all columns
    # ... implementation

    return loaders
```

**3.2 Example Strategies** (0.5 days)

**Strategy 1: Consensus**
```python
# examples/multi_source_fundamentals/strategy_consensus.py
"""
Multi-Source Consensus Strategy

Logic:
1. Average ROE across Sharadar and LSEG
2. Only trade when both sources agree (low divergence)
3. Weight by data quality score
"""

def make_pipeline():
    # Get ROE from both sources
    roe_sharadar = SharadarFundamentals.roe.latest
    roe_lseg = LSEGFundamentals.roe.latest

    # Calculate consensus and divergence
    roe_consensus = (roe_sharadar + roe_lseg) / 2
    roe_divergence = (roe_sharadar - roe_lseg).abs()

    # Quality score: lower divergence = higher quality
    quality_score = 1 / (1 + roe_divergence)

    # Screen: High consensus ROE + low divergence
    screen = (
        (roe_consensus > 15) &
        (roe_divergence < 2) &
        (quality_score > 0.8)
    )

    return Pipeline(
        columns={
            'roe_consensus': roe_consensus,
            'quality_score': quality_score,
        },
        screen=screen.top(20)
    )
```

**Strategy 2: Divergence Alpha**
```python
# examples/multi_source_fundamentals/strategy_divergence.py
"""
Divergence Alpha Strategy

Hypothesis: Large divergences between sources indicate
data quality issues OR genuine alpha opportunities.

Logic:
1. Find stocks with large ROE divergence
2. If LSEG > Sharadar, potentially undervalued
3. If Sharadar > LSEG, potentially overvalued
4. Go long positive divergence, short negative
"""

def make_pipeline():
    roe_s = SharadarFundamentals.roe.latest
    roe_l = LSEGFundamentals.roe.latest
    revenue_s = SharadarFundamentals.revenue.latest
    revenue_consensus = LSEGFundamentals.consensus_revenue.latest

    # Metrics
    roe_divergence = roe_l - roe_s
    revenue_surprise = revenue_s - revenue_consensus

    # Combined signal
    alpha_signal = roe_divergence.zscore() + revenue_surprise.zscore()

    # Long top quintile, short bottom quintile
    longs = alpha_signal.top(40)
    shorts = alpha_signal.bottom(40)

    return Pipeline(
        columns={
            'alpha_signal': alpha_signal,
            'position': longs.astype(float) - shorts.astype(float),
        },
        screen=longs | shorts
    )
```

**3.3 Data Quality Utilities** (0.25 days)
```python
# src/zipline/utils/data_quality.py
"""
Data quality monitoring for multi-source fundamentals.
"""

def check_data_alignment(pipeline_data):
    """Check alignment between Sharadar and LSEG data."""

    # Find stocks with data from both sources
    has_sharadar = ~pipeline_data['roe_sharadar'].isna()
    has_lseg = ~pipeline_data['roe_lseg'].isna()
    both = has_sharadar & has_lseg

    coverage = {
        'sharadar_only': has_sharadar.sum() - both.sum(),
        'lseg_only': has_lseg.sum() - both.sum(),
        'both': both.sum(),
        'coverage_rate': both.sum() / len(pipeline_data),
    }

    return coverage

def detect_outliers(pipeline_data, metric='roe', threshold=3):
    """Detect outlier divergences between sources."""

    col_s = f'{metric}_sharadar'
    col_l = f'{metric}_lseg'

    # Calculate divergence
    divergence = (pipeline_data[col_s] - pipeline_data[col_l]).abs()

    # Find outliers (> threshold standard deviations)
    mean_div = divergence.mean()
    std_div = divergence.std()

    outliers = pipeline_data[divergence > (mean_div + threshold * std_div)]

    return outliers

def generate_quality_report(pipeline_data):
    """Generate comprehensive data quality report."""

    report = {
        'alignment': check_data_alignment(pipeline_data),
        'outliers': {
            'roe': len(detect_outliers(pipeline_data, 'roe')),
            'revenue': len(detect_outliers(pipeline_data, 'revenue')),
        },
        'summary': {
            'total_stocks': len(pipeline_data),
            'avg_roe_divergence': (pipeline_data['roe_sharadar'] - pipeline_data['roe_lseg']).abs().mean(),
        }
    }

    return report
```

---

## Technical Specifications

### Sharadar SF1 Table Structure

**Source**: `https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1`

**Key Columns**:
```
ticker          - Stock symbol
dimension       - ARQ, ARY, MRQ, MRY
datekey         - Date data became available (for point-in-time)
reportperiod    - Fiscal period end date
revenue         - Total revenue
netinc          - Net income
equity          - Shareholders equity
... (150+ more financial metrics)
```

**Dimensions**:
- **ARQ**: As-Reported Quarterly (we use this - point-in-time accurate)
- **ARY**: As-Reported Yearly
- **MRQ**: Most Recent Quarterly (with restatements)
- **MRY**: Most Recent Yearly

**Point-in-Time Logic**:
```python
# For a backtest date of 2023-11-15
# Get most recent fundamental where datekey <= 2023-11-15
# This ensures we only use data available at that time

SELECT * FROM fundamentals
WHERE sid = ? AND datekey <= '2023-11-15'
ORDER BY datekey DESC
LIMIT 1
```

### LSEG Data Format

**Expected CSV Format**:
```csv
ticker,date,roe,roa,ebitda,consensus_revenue,consensus_eps,analyst_rating,target_price
AAPL,2023-09-30,15.5,12.3,125000000000,100000000000,6.25,4.5,185.00
MSFT,2023-09-30,18.2,14.1,98000000000,55000000000,10.50,4.8,380.00
...
```

**After SID Mapping**:
```
Sid,Date,roe,roa,ebitda,consensus_revenue,consensus_eps,analyst_rating,target_price
68,2023-09-30,15.5,12.3,125000000000,100000000000,6.25,4.5,185.00
142,2023-09-30,18.2,14.1,98000000000,55000000000,10.50,4.8,380.00
...
```

### Database Schemas

**Sharadar fundamentals.sqlite**:
```sql
CREATE TABLE fundamentals (
    sid INTEGER NOT NULL,
    dimension TEXT NOT NULL,
    datekey DATE NOT NULL,
    reportperiod DATE NOT NULL,

    -- Income Statement
    revenue REAL,
    cor REAL,
    grossprofit REAL,
    netinc REAL,
    eps REAL,

    -- Balance Sheet
    assets REAL,
    liabilities REAL,
    equity REAL,
    debt REAL,
    cashneq REAL,

    -- Cash Flow
    ncfo REAL,
    ncfi REAL,
    ncff REAL,

    -- Ratios (150+ more columns)
    roe REAL,
    roa REAL,
    roic REAL,
    de REAL,
    currentratio REAL,

    -- Valuation
    marketcap REAL,
    ev REAL,
    pe REAL,
    pb REAL,
    ps REAL,

    PRIMARY KEY (sid, dimension, reportperiod),
    INDEX idx_sid_datekey (sid, datekey),
    INDEX idx_reportperiod (reportperiod)
);
```

**LSEG lseg_fundamentals.sqlite**:
```sql
CREATE TABLE Price (
    Sid INTEGER NOT NULL,
    Date TEXT NOT NULL,

    -- Core metrics
    roe REAL,
    roa REAL,
    ebitda REAL,

    -- Consensus estimates
    consensus_revenue REAL,
    consensus_eps REAL,
    consensus_ebitda REAL,

    -- Analyst data
    analyst_rating REAL,
    analyst_count INTEGER,
    target_price REAL,

    -- LSEG-specific
    starmine_score REAL,
    esg_score REAL,

    PRIMARY KEY (Sid, Date),
    INDEX idx_sid_date (Sid, Date)
);
```

---

## File Structure

### New Files to Create

```
zipline-reloaded/
├── src/zipline/
│   ├── pipeline/
│   │   └── data/
│   │       ├── sharadar.py                      # NEW: SharadarFundamentals dataset
│   │       └── lseg.py                          # NEW: LSEGFundamentals dataset
│   ├── data/
│   │   ├── bundles/
│   │   │   └── sharadar_bundle.py               # MODIFY: Add SF1 download
│   │   └── loaders/
│   │       ├── __init__.py                      # MODIFY: Add build_multi_source_loaders
│   │       └── sharadar_fundamentals.py         # NEW: SF1 loader
│   └── utils/
│       └── data_quality.py                      # NEW: Quality monitoring
│
├── notebooks/
│   ├── load_lseg_fundamentals.ipynb             # NEW: LSEG import workflow
│   └── STRATEGY_README.md                       # UPDATE: Add multi-source examples
│
├── examples/
│   └── multi_source_fundamentals/               # NEW: Example strategies
│       ├── strategy_consensus.py
│       ├── strategy_divergence.py
│       └── README.md
│
├── tests/
│   ├── test_sharadar_fundamentals.py            # NEW: SF1 tests
│   └── test_multi_source_pipeline.py            # NEW: Integration tests
│
└── docs/
    ├── MULTI_SOURCE_FUNDAMENTALS_PLAN.md        # THIS FILE
    ├── SHARADAR_FUNDAMENTALS_GUIDE.md           # NEW: SF1 usage guide
    ├── LSEG_INTEGRATION_GUIDE.md                # NEW: LSEG workflow
    ├── MULTI_SOURCE_STRATEGIES.md               # NEW: Strategy patterns
    └── MIGRATION_CUSTOMDATA_TO_MULTI.md         # NEW: Migration guide
```

### Modified Files

```
src/zipline/data/bundles/sharadar_bundle.py
- Add download_sharadar_fundamentals()
- Add process_fundamentals()
- Add store_fundamentals()
- Modify ingest() to download SF1
- Update incremental logic for SF1

src/zipline/data/loaders/__init__.py
- Add build_multi_source_loaders()
- Export new loaders

notebooks/STRATEGY_README.md
- Add multi-source examples
- Document LSEG workflow
- Update best practices
```

---

## Testing Strategy

### Unit Tests

**Test Coverage Requirements**: 80%+

**Test Categories**:

1. **SF1 Download Tests**
   ```python
   def test_sf1_api_connection()
   def test_sf1_column_completeness()
   def test_sf1_ticker_filtering()
   def test_sf1_date_range_filtering()
   def test_sf1_dimension_selection()
   ```

2. **SF1 Processing Tests**
   ```python
   def test_ticker_to_sid_mapping()
   def test_point_in_time_logic()
   def test_duplicate_handling()
   def test_missing_data_handling()
   ```

3. **SF1 Loader Tests**
   ```python
   def test_load_adjusted_array()
   def test_forward_fill_logic()
   def test_sid_alignment_with_pricing()
   def test_date_range_edge_cases()
   ```

4. **LSEG Import Tests**
   ```python
   def test_lseg_csv_parsing()
   def test_lseg_sid_mapping()
   def test_lseg_duplicate_detection()
   def test_lseg_data_quality_checks()
   ```

5. **Integration Tests**
   ```python
   def test_multi_source_pipeline()
   def test_loader_dict_routing()
   def test_sid_consistency_across_sources()
   def test_data_alignment()
   ```

### Manual Testing Checklist

- [ ] Run full Sharadar ingestion with SF1
- [ ] Verify fundamentals.sqlite created
- [ ] Check SF1 data completeness
- [ ] Import sample LSEG data
- [ ] Verify LSEG SID mapping
- [ ] Run multi-source pipeline
- [ ] Compare results with single-source
- [ ] Test incremental updates
- [ ] Verify point-in-time accuracy
- [ ] Check performance (speed, memory)

### Performance Benchmarks

**Target Performance**:
- SF1 download: < 5 minutes for full history
- SF1 incremental: < 30 seconds
- LSEG import (10k rows): < 2 minutes
- Pipeline execution (multi-source): < 2x single-source
- Memory usage: < 2GB for typical strategies

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `zipline ingest -b sharadar` downloads both SEP and SF1
- [ ] fundamentals.sqlite created with proper schema
- [ ] SharadarFundamentals dataset accessible in Pipeline
- [ ] Point-in-time accuracy verified with test cases
- [ ] Documentation complete

### Phase 2 Complete When:
- [ ] LSEG import notebook successfully maps tickers to SIDs
- [ ] lseg_fundamentals.sqlite created
- [ ] LSEGFundamentals accessible in Pipeline
- [ ] SID consistency verified between sources
- [ ] Documentation complete

### Phase 3 Complete When:
- [ ] build_multi_source_loaders() works for all sources
- [ ] Example strategies run successfully
- [ ] Data quality utilities functional
- [ ] All documentation complete
- [ ] Migration guide available

### Overall Project Success:
- [ ] All unit tests passing (80%+ coverage)
- [ ] All integration tests passing
- [ ] Performance benchmarks met
- [ ] Example strategies demonstrating value
- [ ] Documentation comprehensive
- [ ] Code reviewed and committed

---

## Risk Mitigation

### Risk 1: SF1 Download Timeouts

**Risk**: Downloading full SF1 history (1998-present) times out

**Mitigation**:
- Implement chunking (download by year)
- Add retry logic with exponential backoff
- Show progress indicators
- Cache intermediate results

**Fallback**:
- Download in batches
- Manual CSV import option

### Risk 2: SID Mapping Failures

**Risk**: LSEG tickers don't map to Sharadar SIDs

**Mitigation**:
- Use as_of_date in lookups (handles ticker changes)
- Log all unmapped tickers for review
- Provide manual mapping override file
- Generate mapping quality report

**Fallback**:
- Skip unmapped tickers with warning
- Manual review and correction

### Risk 3: Point-in-Time Accuracy Issues

**Risk**: Forward-fill logic creates look-ahead bias

**Mitigation**:
- Extensive testing with known filing dates
- Compare with third-party point-in-time data
- Document exact logic clearly
- Add validation utilities

**Fallback**:
- Conservative approach (lag fundamentals by 1 quarter)

### Risk 4: Performance Degradation

**Risk**: Multi-source Pipeline too slow

**Mitigation**:
- Profile and optimize hotspots
- Implement caching where appropriate
- Use efficient SQL queries
- Batch operations

**Fallback**:
- Load data separately and join in Python
- Pre-compute derived metrics

### Risk 5: Data Quality Issues

**Risk**: Divergences between sources indicate data problems

**Mitigation**:
- Implement comprehensive data quality checks
- Generate quality reports on every run
- Alert on large divergences
- Manual review process

**Fallback**:
- Use single source for critical decisions
- Weight by data quality score

---

## Migration Path

### For Existing CustomData Users

**Current State**: Using load_csv_fundamentals.ipynb with CustomFundamentals

**Migration Steps**:

1. **Keep Existing CustomData** (backward compatible)
   - Old code continues to work
   - No breaking changes

2. **Add Sharadar SF1** (new capability)
   - Run ingestion: `zipline ingest -b sharadar`
   - SF1 automatically included
   - Start using SharadarFundamentals in new strategies

3. **Migrate CustomData to LSEG** (optional)
   - Rename CustomFundamentals → LSEGFundamentals
   - Re-map using proper SID mapping
   - Ensures consistency with Sharadar

4. **Update Strategies** (gradual)
   - Modify strategies one at a time
   - Test thoroughly
   - Compare results

### Breaking Changes

**None!** This is additive only:
- ✅ Existing bundles work unchanged
- ✅ Existing CustomData works unchanged
- ✅ Existing strategies work unchanged
- ✅ New features are opt-in

---

## Next Steps

### Immediate Actions

1. **Review this plan** - Get feedback and approval
2. **Set up development branch** - `feature/multi-source-fundamentals`
3. **Create tracking issues** - GitHub issues for each phase
4. **Begin Phase 1** - Start with SF1 download implementation

### Development Process

**Workflow**:
1. Create feature branch
2. Implement phase
3. Write tests
4. Run tests
5. Review code
6. Merge to main
7. Update documentation
8. Move to next phase

**Git Strategy**:
```bash
# Create feature branch
git checkout -b feature/multi-source-fundamentals

# Work in sub-branches for each phase
git checkout -b feature/sharadar-sf1
# ... implement Phase 1
git merge feature/sharadar-sf1

git checkout -b feature/lseg-integration
# ... implement Phase 2
git merge feature/lseg-integration

git checkout -b feature/multi-source-examples
# ... implement Phase 3
git merge feature/multi-source-examples

# Final merge to main
git checkout main
git merge feature/multi-source-fundamentals
```

---

## Appendix

### References

- [Sharadar SF1 Documentation](https://data.nasdaq.com/databases/SFA/documentation)
- [Zipline Pipeline Documentation](https://zipline.ml4trading.io/bundles.html)
- [Current CustomData Implementation](../notebooks/load_csv_fundamentals.ipynb)
- [Current Sharadar Bundle](../src/zipline/data/bundles/sharadar_bundle.py)

### Glossary

- **ARQ**: As-Reported Quarterly (Sharadar dimension)
- **Point-in-Time**: Data as it was known at a specific historical date
- **SID**: Security IDentifier (Zipline's internal asset ID)
- **Permaticker**: Sharadar's permanent ticker identifier
- **Forward-fill**: Extending quarterly data across daily dates
- **CustomData**: Zipline's system for loading custom datasets
- **Pipeline**: Zipline's factor-based stock screening system
- **LSEG**: London Stock Exchange Group (Refinitiv data provider)

### Contact

For questions or issues during implementation:
- **Primary**: Kamran Sokhanvari - Hidden Point Capital
- **Documentation**: See docs/ folder
- **Code**: See src/zipline/

---

**Document Status**: ✅ Ready for Review
**Next Action**: Review and approve plan, then begin Phase 1
**Estimated Completion**: 4-6 days after approval
