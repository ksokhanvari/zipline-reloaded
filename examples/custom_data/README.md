# Custom Data Examples

This directory contains comprehensive examples for using Zipline's custom data functionality to load and analyze fundamental data in quantitative research.

## Files in This Directory

### Jupyter Notebooks

- **`research_with_fundamentals.ipynb`** - Complete guide to loading fundamental data from CSV and using it in Pipeline research
  - Database creation and data loading
  - Pipeline factor creation
  - Stock screening and ranking
  - Sector analysis
  - Visualizations
  - Integration with backtesting
  - Troubleshooting guide

### Sample Data Files

- **`sample_fundamentals.csv`** - Example quarterly fundamental data
  - 10 stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM, WMT, XOM, V)
  - Q1-Q4 2023 data
  - 13 fundamental metrics (Revenue, NetIncome, EPS, ROE, P/E, etc.)
  - Realistic values for demonstration

- **`sample_securities.csv`** - Symbol-to-Sid mapping file
  - Maps ticker symbols to Zipline asset IDs (Sids)
  - Required for data loading
  - Includes company names, exchanges, and sectors

## Quick Start

### 1. Install Requirements

```bash
# From the zipline-reloaded root directory
pip install -e .

# Install additional dependencies for the notebook
pip install jupyter matplotlib seaborn
```

### 2. Launch Jupyter

```bash
cd examples/custom_data
jupyter notebook research_with_fundamentals.ipynb
```

### 3. Run the Notebook

Execute cells sequentially to:
1. Create a database for fundamental data
2. Load the sample CSV data
3. Create Pipeline factors
4. Screen and rank stocks
5. Visualize results

## Database Storage Locations

### Where Your Custom Data is Stored

**If running in Docker** (using docker-compose):
- **Container path**: `/data/custom_databases/`
- **Host path**: `./data/custom_databases/` (from zipline-reloaded project root)
- **Persistent**: Yes - data survives container restarts
- **Configured by**: `ZIPLINE_CUSTOM_DATA_DIR` environment variable in docker-compose.yml

**If running locally** (not in Docker):
- **Default path**: `~/.zipline/custom_data/`
- **Persistent**: Yes - stored alongside bundle data
- **Customizable**: Set `ZIPLINE_CUSTOM_DATA_DIR` environment variable

**Database filename format**: `quant_{db_code}.sqlite`
- Example: `quant_fundamentals.sqlite` for db_code='fundamentals'

### Checking Your Storage Location

```python
from zipline.data.custom.config import get_custom_data_dir
storage_dir = get_custom_data_dir()
print(f"Custom data directory: {storage_dir}")
```

### Docker Volume Mapping

The docker-compose.yml file includes this configuration:

```yaml
environment:
  - ZIPLINE_CUSTOM_DATA_DIR=/data/custom_databases
volumes:
  - ./data:/data  # Host ./data maps to container /data
```

This means:
- Custom databases are saved to `./data/custom_databases/` on your host machine
- The `./data` directory is in your zipline-reloaded project root
- You can access the SQLite files directly from your host filesystem
- Data persists even if you remove and recreate containers

## Database Management

### Listing Databases

```python
from zipline.data.custom import list_custom_dbs

# List all custom databases
dbs = list_custom_dbs()
print(f"Available databases: {dbs}")
```

### Getting Database Information

```python
from zipline.data.custom import describe_custom_db

# Get detailed info about a database
info = describe_custom_db('fundamentals')
print(f"Path: {info['db_path']}")
print(f"Rows: {info['row_count']}")
print(f"Columns: {info['columns']}")
print(f"Date range: {info['date_range']}")
print(f"Assets: {info['sids']}")
```

### Direct SQL Queries

```python
from zipline.data.custom import connect_db
import pandas as pd

# Connect and query
conn = connect_db('fundamentals')
df = pd.read_sql("SELECT * FROM Price WHERE Sid='24' ORDER BY Date", conn)
conn.close()
```

### Backing Up Your Data

```bash
# If running in Docker
cp ./data/custom_databases/quant_fundamentals.sqlite ./data/custom_databases/quant_fundamentals_backup.sqlite

# If running locally
cp ~/.zipline/custom_data/quant_fundamentals.sqlite ~/.zipline/custom_data/quant_fundamentals_backup.sqlite
```

### Deleting a Database

```python
import os
from zipline.data.custom import get_db_path

# CAUTION: This permanently deletes the database!
db_path = get_db_path('fundamentals')
os.remove(db_path)
```

## What You'll Learn

### Core Concepts

1. **Custom Data Architecture**
   - SQLite databases for flexible data storage
   - Schema definition and validation
   - Data loading with sid mapping

2. **Pipeline Integration**
   - Creating DataSets from custom data
   - Building custom factors
   - Screening and filtering stocks
   - Ranking and scoring

3. **Research Workflows**
   - Factor analysis and testing
   - Stock screening strategies
   - Sector comparison
   - Time series analysis

4. **Visualization**
   - Scatter plots (ROE vs P/E)
   - Bar charts (quality scores)
   - Heatmaps (metric comparison)
   - Time series (fundamental trends)

### Practical Applications

- **Value Investing**: Find undervalued stocks using P/E, P/B ratios
- **Quality Screening**: Identify companies with strong fundamentals
- **Sector Rotation**: Compare metrics across sectors
- **Factor Research**: Test custom factors based on fundamentals
- **Portfolio Construction**: Build portfolios using fundamental signals

## Sample Data Format

### Fundamentals CSV Format

```csv
Ticker,Date,Revenue,NetIncome,TotalAssets,TotalEquity,SharesOutstanding,EPS,BookValuePerShare,ROE,DebtToEquity,CurrentRatio,PERatio,Sector
AAPL,2023-03-31,94836000000,24160000000,346747000000,50672000000,15728700000,1.52,3.22,47.7,2.45,0.99,26.8,Technology
AAPL,2023-06-30,81797000000,19881000000,352755000000,62146000000,15697400000,1.26,3.96,32.0,2.23,1.01,28.5,Technology
...
```

**Required Columns:**
- `Ticker`: Stock symbol (must match securities.csv)
- `Date`: Date in YYYY-MM-DD format

**Data Columns:**
- Any fundamental metrics you want
- Column names must match schema definition
- Supported types: int, float, text, date, datetime

### Securities CSV Format

```csv
Ticker,Sid,Name,Exchange,Sector
AAPL,24,Apple Inc.,NASDAQ,Technology
MSFT,5061,Microsoft Corporation,NASDAQ,Technology
...
```

**Required Columns:**
- **Identifier column** (e.g., `Ticker`, `Symbol`): **Must match the identifier column name in your fundamentals CSV**
- `Sid`: Zipline asset ID (get from your bundle's asset database)

**IMPORTANT:** The identifier column name (e.g., `Ticker`) must be **identical** in both:
1. Your fundamentals CSV
2. Your securities CSV

If your fundamentals CSV has `Ticker`, then securities CSV must also have `Ticker`.
If your fundamentals CSV has `Symbol`, then securities CSV must also have `Symbol`.

**Optional Columns:**
- `Name`: Company name
- `Exchange`: Stock exchange
- `Sector`: Industry sector
- Any other metadata

## Getting Symbol-to-Sid Mappings

### From Your Bundle

```python
from zipline.data.bundles import load

# Load your bundle
bundle_data = load('sharadar')  # or 'quandl', etc.

# Get all assets
assets = bundle_data.asset_finder.retrieve_all(
    bundle_data.asset_finder.sids
)

# Create securities DataFrame
import pandas as pd

securities_df = pd.DataFrame({
    'Ticker': [a.symbol for a in assets],  # Use 'Ticker' to match fundamentals CSV
    'Sid': [a.sid for a in assets],
    'Name': [a.asset_name for a in assets],
    'Exchange': [a.exchange for a in assets],
})

# Save to CSV
securities_df.to_csv('my_securities.csv', index=False)

# NOTE: If your fundamentals CSV uses 'Symbol' instead of 'Ticker',
# change 'Ticker' above to 'Symbol' to match
```

### Manual Lookup

```python
# Look up a specific stock
asset = bundle_data.asset_finder.lookup_symbol('AAPL', as_of_date=None)
print(f"AAPL Sid: {asset.sid}")
```

## Customizing for Your Data

### 1. Prepare Your CSV Files

**Your fundamentals CSV should have:**
- Ticker column (any name, you'll specify it)
- Date column (any name, you'll specify it)
- Your fundamental metrics

**Your securities CSV should have:**
- Symbol column
- Sid column

### 2. Define Your Schema

In the notebook, modify the `fundamental_columns` dictionary:

```python
fundamental_columns = {
    'YourMetric1': 'float',
    'YourMetric2': 'int',
    'YourMetric3': 'text',
    # ... add all your columns
}
```

### 3. Update File Paths

```python
FUNDAMENTALS_CSV = 'path/to/your/fundamentals.csv'
SECURITIES_CSV = 'path/to/your/securities.csv'
```

### 4. Specify Column Names

```python
result = load_csv_to_db(
    csv_path=FUNDAMENTALS_CSV,
    db_code='your_db_name',
    sid_map=securities_df,
    id_col='YourTickerColumn',  # Name in your CSV
    date_col='YourDateColumn',   # Name in your CSV
)
```

### 5. Create Your Factors

```python
# Use your metrics in Pipeline
revenue = Fundamentals.YourMetric1.latest
ratio = Fundamentals.YourMetric2.latest

# Create custom factors
class MyCustomFactor(CustomFactor):
    inputs = [Fundamentals.YourMetric1, Fundamentals.YourMetric2]

    def compute(self, today, assets, out, metric1, metric2):
        out[:] = metric1[-1] / metric2[-1]  # Your calculation
```

## Advanced Usage

### Combining with Price Data

```python
from zipline.pipeline.data import EquityPricing

# In your pipeline
pipe = Pipeline(
    columns={
        'roe': Fundamentals.ROE.latest,
        'close': EquityPricing.close.latest,
        'volume': EquityPricing.volume.latest,
    }
)
```

### Point-in-Time Correctness

Fundamental data is reported quarterly with a lag (typically 45 days after quarter-end):

```python
# Example: Use Q1 2023 data (released ~May 15) from May 15 onwards
# The .latest operator handles this automatically based on dates in your CSV
```

### Multiple Databases

You can create separate databases for different data types:

```python
# Create separate databases
create_custom_db('fundamentals', '1 quarter', fundamental_columns)
create_custom_db('estimates', '1 day', estimate_columns)
create_custom_db('sentiment', '1 day', sentiment_columns)

# Create DataSets for each
Fundamentals = make_custom_dataset_class('fundamentals', ...)
Estimates = make_custom_dataset_class('estimates', ...)
Sentiment = make_custom_dataset_class('sentiment', ...)

# Use in Pipeline
pipe = Pipeline(
    columns={
        'roe': Fundamentals.ROE.latest,
        'eps_estimate': Estimates.EPSEstimate.latest,
        'sentiment_score': Sentiment.Score.latest,
    }
)
```

## Troubleshooting

### "Database already exists" Error

If you see this when creating a database:

**Option 1: Use existing database**
```python
# The notebook will automatically use the existing database
# Just skip to the data loading step
```

**Option 2: Delete and recreate**
```bash
# Delete the database
rm ~/.zipline/custom_data/quant_fundamentals.sqlite

# Re-run the database creation cell
```

### "sid_map DataFrame must have columns" Error

**Error Message:**
```
sid_map DataFrame must have columns 'Ticker' and 'Sid'
```

**Cause:** The identifier column name doesn't match between your files.

**Solution:** Ensure both CSVs use the **same identifier column name**:

```python
# ✗ WRONG - Column names don't match
# fundamentals.csv has: Ticker,Date,Revenue,...
# securities.csv has:   Symbol,Sid,Name,...     <- Symbol ≠ Ticker

# ✓ CORRECT - Column names match
# fundamentals.csv has: Ticker,Date,Revenue,...
# securities.csv has:   Ticker,Sid,Name,...     <- Both use 'Ticker'
```

**Quick Fix:**
```python
# If your securities CSV has 'Symbol' but you need 'Ticker'
securities_df = pd.read_csv('securities.csv')
securities_df = securities_df.rename(columns={'Symbol': 'Ticker'})

# Then load data
result = load_csv_to_db(..., sid_map=securities_df, id_col='Ticker')
```

### "Unmapped identifiers" Warning

If tickers in your fundamentals CSV aren't in securities.csv:

**Option 1: Add to securities.csv**
```python
# Look up missing tickers and add to securities.csv
```

**Option 2: Skip unmapped tickers**
```python
result = load_csv_to_db(
    ...,
    fail_on_unmapped=False,  # Skip instead of failing
)
```

### "No data returned" from Queries

Check that:
1. Data was loaded successfully (check `result['rows_inserted']`)
2. Your date range matches the data
3. Sids exist in the database (`describe_custom_db()`)

### Performance Issues

For large datasets:

```python
# Use chunking for large CSV files
result = load_csv_to_db(
    ...,
    chunk_size=50000,  # Process 50k rows at a time
)

# Filter queries by date range
df = get_prices(
    db_code='fundamentals',
    start_date='2023-01-01',
    end_date='2023-12-31',  # Limit date range
    sids=[24, 5061],        # Limit assets
)
```

## Resources

- **Zipline Documentation**: https://zipline.ml4trading.io/
- **Custom Data Module**: `src/zipline/data/custom/README.md`
- **Pipeline Tutorial**: https://zipline.ml4trading.io/pipeline.html
- **Example Strategies**: `notebooks/` directory

## Support

If you encounter issues:

1. Check the troubleshooting section in the notebook
2. Review `src/zipline/data/custom/README.md`
3. Run the diagnostic: `python diagnose_bundle.py`
4. Open an issue on GitHub with details

## Contributing

Have improvements or additional examples?

1. Add your example to this directory
2. Update this README
3. Submit a pull request

## License

Same as Zipline Reloaded (Apache 2.0)
