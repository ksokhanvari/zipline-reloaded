# Custom Data Tutorial - Methodology & API Usage

## Overview

This document explains the methodology, API usage, and fixes for the custom fundamental data tutorial.

## Root Cause of NaN Issues

### Problem 1: Date Mismatch (CRITICAL)

**Issue**: Different companies have different fiscal year calendars:
- Most companies (AAPL, MSFT, GOOGL, etc.): Q4 2023 ends on 2023-12-31
- NVDA, WMT: Q4 2023 ends on 2024-01-31 (different fiscal calendar)

**Original Code**:
```python
test_date = pd.Timestamp('2023-12-31')
```

**Problem**: When querying with `as_of_date='2023-12-31'`:
- 9 companies return Q4 data (dated 2023-12-31) ✓
- NVDA and WMT return Q3 data (dated 2023-10-31) ✗

This creates **inconsistent data** - comparing Q3 vs Q4 quarters!

**Fix**:
```python
test_date = pd.Timestamp('2024-01-31')  # Latest date that includes all Q4 data
```

Now all companies return their Q4 2023 data.

### Problem 2: Division by Zero in Normalization

**Issue**: When all stocks have the same value for a metric, normalization causes division by zero → NaN

**Original Code**:
```python
roe_score = (roe_latest - np.nanmin(roe_latest)) / (np.nanmax(roe_latest) - np.nanmin(roe_latest))
# If max == min, we get: 0 / 0 = NaN
```

**Fix**:
```python
roe_min, roe_max = np.nanmin(roe_latest), np.nanmax(roe_latest)
if roe_max > roe_min:
    roe_score = (roe_latest - roe_min) / (roe_max - roe_min)
else:
    roe_score = np.full_like(roe_latest, 0.5)  # Neutral score if all same
```

Applied to:
- QualityScore CustomFactor
- Manual ranking calculations
- Heatmap normalization

## API Usage Reference

### 1. `get_latest_values()` - Point-in-Time Data Retrieval

**Purpose**: Get the most recent data for each asset as of a specific date.

**Signature**:
```python
get_latest_values(
    db_code: str,              # Database identifier
    as_of_date: str,           # ISO date 'YYYY-MM-DD'
    sids: List[int],           # List of asset IDs
    fields: List[str] = None,  # Specific columns (None = all)
    db_dir: Path = None        # Database directory (None = default)
) -> pd.DataFrame
```

**How it works**:
```sql
-- For each Sid, get the latest record where Date <= as_of_date
SELECT * FROM Price p1
WHERE Date = (
    SELECT MAX(Date)
    FROM Price p2
    WHERE p2.Sid = p1.Sid
    AND p2.Date <= '2024-01-31'  -- as_of_date parameter
)
```

**Example**:
```python
data = get_latest_values(
    db_code='fundamentals',
    as_of_date='2024-01-31',
    sids=[24, 5061, 26890],
)
# Returns: One row per Sid with latest data as of 2024-01-31
```

**Point-in-Time Correctness**: This ensures no look-ahead bias. If you backtest on 2024-01-15, you only get data that existed before that date.

### 2. `get_prices()` - Time Range Queries

**Purpose**: Get all data within a date range.

**Signature**:
```python
get_prices(
    db_code: str,
    start_date: str = None,
    end_date: str = None,
    sids: List[int] = None,
    fields: List[str] = None,
) -> pd.DataFrame
```

**Example**:
```python
# Get all quarterly data for AAPL (Sid=24) in 2023
history = get_prices(
    db_code='fundamentals',
    start_date='2023-01-01',
    end_date='2023-12-31',
    sids=[24],
    fields=['Revenue', 'EPS', 'ROE']
)
```

### 3. `make_custom_dataset_class()` - Pipeline Integration

**Purpose**: Create a Pipeline DataSet from your custom database.

**Signature**:
```python
make_custom_dataset_class(
    db_code: str,              # Database identifier
    columns: dict,             # Schema: {col_name: type}
    base_name: str = 'Custom'  # Class name prefix
) -> DataSet
```

**Example**:
```python
Fundamentals = make_custom_dataset_class(
    db_code='fundamentals',
    columns={
        'Revenue': 'int',
        'ROE': 'float',
        'Sector': 'text',
    },
    base_name='Fundamentals'
)

# Now you can use in Pipeline:
roe = Fundamentals.ROE.latest
high_roe_filter = (roe > 15.0)
```

**How `.latest` works**:
- Creates a `BoundColumn` that references the most recent value
- In backtests, automatically retrieves the latest value as of each trading day
- Respects point-in-time correctness

### 4. CustomFactor - Computed Metrics

**Purpose**: Create derived metrics from custom data columns.

**Example - Profit Margin**:
```python
class ProfitMargin(CustomFactor):
    inputs = [
        Fundamentals.NetIncome,
        Fundamentals.Revenue,
    ]
    window_length = 1  # How many periods to look back

    def compute(self, today, assets, out, net_income, revenue):
        # Get latest values (index 0 for window_length=1)
        latest_income = net_income[0]  # 2D array: [time, assets]
        latest_revenue = revenue[0]

        # Calculate with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            margin = (latest_income / latest_revenue) * 100.0
            margin = np.where(latest_revenue == 0, np.nan, margin)

        out[:] = margin

# Use in Pipeline
profit_margin = ProfitMargin()
pipe = Pipeline(columns={'margin': profit_margin})
```

**Key Points**:
- `window_length=1` means only current period (no historical lookback)
- `window_length=4` for quarterly data = 1 year lookback
- Input arrays are 2D: `[time_periods, num_assets]`
- Must handle NaN and division by zero

## Data Loading Best Practices

### 1. CSV Format Requirements

**Fundamentals CSV**:
```csv
Ticker,Date,Revenue,EPS,ROE,...
AAPL,2023-03-31,94836000000,1.52,47.7,...
AAPL,2023-06-30,81797000000,1.26,32.0,...
```

**Securities Mapping CSV**:
```csv
Ticker,Sid,Name,Exchange,Sector
AAPL,24,Apple Inc.,NASDAQ,Technology
MSFT,5061,Microsoft Corporation,NASDAQ,Technology
```

**Critical**: The identifier column name must match in both files (both use "Ticker").

### 2. Loading Data

```python
result = load_csv_to_db(
    csv_path='fundamentals.csv',
    db_code='fundamentals',
    sid_map=securities_df,          # DataFrame with Ticker and Sid columns
    id_col='Ticker',                # Column name in fundamentals CSV
    date_col='Date',                # Date column name
    on_duplicate='replace',         # replace | ignore | fail
    fail_on_unmapped=False,         # Skip unknown tickers
)

print(f"Loaded: {result['rows_inserted']} rows")
print(f"Skipped: {result['rows_skipped']} rows")
print(f"Unmapped: {result['unmapped_ids']}")
```

### 3. Point-in-Time Correctness

**Real-world consideration**: Fundamental data has reporting lag!

```python
# WRONG: Using report period date
# Q4 2023 earnings (Dec 31) are usually reported in Feb 2024
as_of_date = '2023-12-31'  # Data wouldn't exist yet!

# RIGHT: Add reporting lag
as_of_date = '2024-02-28'  # After typical earnings release
```

For backtesting, add ~45 days lag:
```python
def add_reporting_lag(report_date, days=45):
    return report_date + pd.Timedelta(days=days)

# Store with lag-adjusted dates for point-in-time correctness
df['Date'] = df['Report_Date'].apply(lambda d: add_reporting_lag(d))
```

## Normalization Methodology

### Min-Max Normalization

**Formula**:
```
normalized = (value - min) / (max - min)
```

**Range**: 0.0 (worst) to 1.0 (best)

**Direction**:
- ROE: Higher is better → use as-is
- P/E: Lower is better → invert: `1 - normalized`
- Debt/Equity: Lower is better → invert: `1 - normalized`

**Edge Cases**:
```python
if max > min:
    normalized = (value - min) / (max - min)
else:
    # All values identical
    normalized = 0.5  # Neutral score
```

### Composite Scores

```python
quality_score = (roe_score + pe_score + debt_score) / 3.0
```

**Weighting**: Equal weights (1/3 each). Can be customized:
```python
quality_score = (
    roe_score * 0.5 +      # 50% weight
    pe_score * 0.3 +       # 30% weight
    debt_score * 0.2       # 20% weight
)
```

## Testing Your Implementation

### 1. Verify Data Loading

```python
db_info = describe_custom_db('fundamentals')
print(f"Rows: {db_info['row_count']}")
print(f"Sids: {db_info['num_sids']}")
print(f"Date range: {db_info['date_range']}")
print(f"Columns: {list(db_info['columns'].keys())}")
```

### 2. Check Point-in-Time Retrieval

```python
# Test different dates
for date in ['2023-03-31', '2023-06-30', '2023-12-31', '2024-01-31']:
    data = get_latest_values('fundamentals', as_of_date=date, sids=[24])
    print(f"{date}: Returns data dated {data['Date'].iloc[0]}")
```

Expected output:
```
2023-03-31: Returns data dated 2023-03-31
2023-06-30: Returns data dated 2023-06-30
2023-12-31: Returns data dated 2023-12-31  (for AAPL)
2023-12-31: Returns data dated 2023-10-31  (for NVDA - missing Q4!)
2024-01-31: Returns data dated 2024-01-31  (for NVDA - has Q4)
```

### 3. Validate Normalization

```python
# Should not have NaNs
assert not ranking_data['Quality_Score'].isna().any()

# Should be in range [0, 1]
assert (ranking_data['Quality_Score'] >= 0).all()
assert (ranking_data['Quality_Score'] <= 1).all()
```

## Common Pitfalls

### ❌ Don't Do This

```python
# 1. Arithmetic on .latest objects
profit_margin = Fundamentals.NetIncome.latest / Fundamentals.Revenue.latest
# TypeError: unsupported operand type(s) for /: 'Latest' and 'Latest'

# 2. Using wrong identifier column names
securities_df = pd.read_csv('securities.csv')  # Has 'Symbol' column
load_csv_to_db(..., id_col='Ticker')           # Fundamentals has 'Ticker'
# KeyError: 'Ticker' not in fundamentals OR 'Symbol' not in securities

# 3. Inconsistent date queries
test_date = '2023-12-31'  # Some stocks return Q3, some Q4

# 4. Division without checks
score = (value - min_val) / (max_val - min_val)  # NaN if max == min
```

### ✓ Do This

```python
# 1. Use CustomFactor for calculations
class ProfitMargin(CustomFactor):
    inputs = [Fundamentals.NetIncome, Fundamentals.Revenue]
    def compute(self, today, assets, out, income, revenue):
        out[:] = (income[0] / revenue[0]) * 100

# 2. Match identifier column names
# Both CSVs use 'Ticker' or both use 'Symbol'

# 3. Use date that includes all data
test_date = '2024-01-31'  # All stocks return Q4

# 4. Check for division by zero
if max_val > min_val:
    score = (value - min_val) / (max_val - min_val)
else:
    score = 0.5
```

## Summary

**Key Fixes Applied**:
1. ✅ Changed test date from 2023-12-31 → 2024-01-31 (consistent Q4 data)
2. ✅ Added division-by-zero checks to all normalization code
3. ✅ Updated QualityScore CustomFactor with edge case handling
4. ✅ Improved chart layouts (no tight_layout warnings)
5. ✅ Fixed all Symbol → Ticker references

**Methodology Verified**:
- Point-in-time data retrieval using `get_latest_values()` ✓
- CustomFactor for derived metrics ✓
- Min-max normalization with proper bounds checking ✓
- Composite scoring with equal weights ✓

**API Usage Confirmed**:
- All function signatures correct ✓
- SQL queries use proper point-in-time logic ✓
- Error handling for edge cases ✓
