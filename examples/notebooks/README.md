# Zipline Reloaded - Jupyter Notebook Examples

This directory contains interactive Jupyter notebooks demonstrating various features of zipline-reloaded.

## Available Notebooks

### 1. `test_sharadar_fundamentals.ipynb` - Sharadar Fundamentals Stock Screening

**Purpose**: Comprehensive demonstration of using Sharadar SF1 fundamentals data in Pipeline API.

**What You'll Learn**:
- Access 80+ fundamental metrics (revenue, ROE, P/E, etc.)
- Point-in-time data handling (no look-ahead bias)
- 5 different stock screening strategies
- Data quality validation
- Visualization and analysis

**Strategies Demonstrated**:
1. **Magic Formula (Greenblatt)**: High ROE + High Earnings Yield
2. **High ROE**: Quality companies with exceptional profitability (>20% ROE)
3. **Profitable Growth**: Large revenue, strong margins, reasonable P/E
4. **Deep Value**: Low P/E (<10), Low P/B (<1.5), profitable
5. **Quality Score**: Multi-factor quality assessment (simplified Piotroski)

**Prerequisites**:
```bash
# Ingest Sharadar bundle with fundamentals
zipline ingest -b sharadar

# Or download fundamentals for existing bundle
./scripts/download_fundamentals_only.sh sharadar
```

**How to Run**:
```bash
# From inside Docker container
cd /notebooks
jupyter notebook test_sharadar_fundamentals.ipynb

# Or from host (if Jupyter is mounted)
cd examples/notebooks
jupyter notebook test_sharadar_fundamentals.ipynb
```

**Output Examples**:
- Data coverage report (% of stocks with each metric)
- Top 20 stocks from each strategy
- ROE vs P/E scatter plot with market cap coloring
- Multi-strategy picks (stocks appearing in multiple screens)
- Data quality validation checks

---

## Running Notebooks in Docker

### Option 1: Jupyter Server (Recommended)

If your Docker container has Jupyter server running:

```bash
# Access at http://localhost:8888
# Token is printed in Docker logs
docker logs zipline-reloaded-jupyter

# Navigate to /notebooks/test_sharadar_fundamentals.ipynb
```

### Option 2: JupyterLab

```bash
# Start JupyterLab in container
docker exec -it zipline-reloaded-jupyter jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access at http://localhost:8888
```

### Option 3: Convert to Python Script

```bash
# Convert notebook to Python script
docker exec zipline-reloaded-jupyter jupyter nbconvert \
    --to python /app/examples/notebooks/test_sharadar_fundamentals.ipynb \
    --output /tmp/test_fundamentals.py

# Run the script
docker exec zipline-reloaded-jupyter python /tmp/test_fundamentals.py
```

---

## Notebook Features

### Interactive Exploration
- Modify screening criteria in real-time
- Adjust thresholds (ROE > 20%, P/E < 10, etc.)
- Filter by market cap, sector, or other attributes
- Visualize different metric relationships

### Data Validation
- Coverage statistics for each metric
- Data quality checks (negative equity, extreme ratios, etc.)
- Summary statistics
- Outlier detection

### Export Results
- Save top picks to CSV
- Identify multi-strategy stocks (highest conviction)
- Generate watchlists for further research

---

## Common Modifications

### Change Test Date
```python
# In the "Load Bundle" cell, modify:
sessions = trading_calendar.sessions_in_range(
    pd.Timestamp('2024-01-01'),  # Start date
    pd.Timestamp('2024-12-31')   # End date
)
test_date = sessions[-1]  # Or choose specific session
```

### Adjust Screening Criteria
```python
# Example: More aggressive Magic Formula
magic_formula_data = explore_data[
    (explore_data['netinc'] > 0) &
    (explore_data['roe'] > 0.25) &      # Higher ROE threshold
    (explore_data['pe'] > 0) &
    (explore_data['pe'] < 15) &         # Lower P/E threshold
    (explore_data['marketcap'] > 5e9)   # Larger market cap
].copy()
```

### Add Custom Metrics
```python
# Add your own calculated metrics
explore_data['debt_to_assets'] = explore_data['debt'] / explore_data['assets']
explore_data['cash_ratio'] = explore_data['cashnequsd'] / explore_data['assets']
explore_data['enterprise_value'] = explore_data['ev']
```

---

## Troubleshooting

### Notebook Won't Load
**Problem**: Bundle not registered
```python
# Add at top of notebook:
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle
register('sharadar', sharadar_bundle())
```

### No Fundamental Data
**Problem**: Fundamentals not ingested
```bash
# Download fundamentals
docker exec zipline-reloaded-jupyter python /app/scripts/download_fundamentals_only.py --bundle sharadar
```

### Import Errors
**Problem**: Missing dependencies
```bash
# Install in container
docker exec zipline-reloaded-jupyter pip install matplotlib seaborn jupyter
```

### Memory Issues
**Problem**: Large dataset causes OOM
```python
# In notebook, filter to smaller universe first:
# Add to initial pipeline screen:
screen = (SharadarFundamentals.marketcap.latest > 1e9)  # Only large caps
```

---

## Creating Your Own Notebooks

### Template Structure
```python
# 1. Imports
import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.data.sharadar import SharadarFundamentals
# ... other imports

# 2. Load bundle and create engine
bundle_data = load('sharadar')
fundamentals_loader = make_sharadar_fundamentals_loader('sharadar')
engine = SimplePipelineEngine(...)

# 3. Define pipeline
pipeline = Pipeline(
    columns={
        'revenue': SharadarFundamentals.revenue.latest,
        # ... more columns
    },
    screen=(SharadarFundamentals.revenue.latest > 1e9)
)

# 4. Run pipeline
result = engine.run_pipeline(pipeline, start_date, end_date)

# 5. Analyze results
# Your custom analysis here
```

### Best Practices
1. **Clear Documentation**: Explain each strategy and metric
2. **Data Validation**: Always check coverage and quality
3. **Visualization**: Use plots to understand data distributions
4. **Reproducibility**: Set random seeds, document dates used
5. **Performance**: Filter early to reduce dataset size

---

## Additional Resources

### Documentation
- **Fundamentals Guide**: `docs/SHARADAR_FUNDAMENTALS_GUIDE.md`
- **Phase 1 Summary**: `docs/FUNDAMENTALS_PHASE1_COMPLETE.md`
- **Available Metrics**: `src/zipline/pipeline/data/sharadar.py`

### Example Strategies
See `docs/SHARADAR_FUNDAMENTALS_GUIDE.md` for complete backtest examples:
- Magic Formula (full implementation)
- Piotroski F-Score
- Dividend Aristocrats

### Support
- GitHub Issues: https://github.com/ksokhanvari/zipline-reloaded/issues
- Community: https://exchange.ml4trading.io

---

**Author**: Kamran Sokhanvari @ Hidden Point Capital
**Repository**: https://github.com/ksokhanvari/zipline-reloaded
