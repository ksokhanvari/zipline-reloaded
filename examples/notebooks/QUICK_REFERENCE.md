# Sharadar Fundamentals - Quick Reference Card

## Running the Test Notebook

### From Docker
```bash
# Option 1: Jupyter Notebook
docker exec -it zipline-reloaded-jupyter bash
cd /notebooks
jupyter notebook test_sharadar_fundamentals.ipynb

# Option 2: JupyterLab
docker exec -it zipline-reloaded-jupyter jupyter lab \
    --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access at: http://localhost:8888
```

### Convert to Python Script
```bash
# Convert and run
docker exec zipline-reloaded-jupyter jupyter nbconvert \
    --to python /app/examples/notebooks/test_sharadar_fundamentals.ipynb \
    --output /tmp/test_fundamentals.py

docker exec zipline-reloaded-jupyter python /tmp/test_fundamentals.py
```

---

## Key Fundamental Metrics

### Income Statement
| Metric | Description | Usage |
|--------|-------------|-------|
| `revenue` | Total revenue | Size, growth |
| `netinc` | Net income | Profitability |
| `ebitda` | Earnings before interest, tax, depreciation | Cash generation |
| `eps` | Earnings per share | Per-share profitability |

### Balance Sheet
| Metric | Description | Usage |
|--------|-------------|-------|
| `assets` | Total assets | Size |
| `equity` | Shareholders equity | Book value |
| `debt` | Total debt | Leverage |
| `cashnequsd` | Cash and equivalents (USD) | Liquidity |

### Ratios
| Metric | Description | Good Value |
|--------|-------------|------------|
| `roe` | Return on equity | > 15% |
| `roa` | Return on assets | > 5% |
| `pe` | Price to earnings | 10-20 |
| `pb` | Price to book | < 3 |
| `de` | Debt to equity | < 1 |

### Market Data
| Metric | Description | Usage |
|--------|-------------|-------|
| `marketcap` | Market capitalization | Company size |
| `ev` | Enterprise value | Total valuation |

---

## Common Screening Patterns

### Magic Formula
```python
# High quality + Reasonable price
screen = (
    (SharadarFundamentals.roe.latest > 0.15) &
    (SharadarFundamentals.pe.latest > 0) &
    (SharadarFundamentals.pe.latest < 20)
)
```

### High ROE
```python
# Exceptional profitability, low debt
screen = (
    (SharadarFundamentals.roe.latest > 0.20) &
    (SharadarFundamentals.de.latest < 2) &
    (SharadarFundamentals.netinc.latest > 0)
)
```

### Deep Value
```python
# Low valuation, profitable
screen = (
    (SharadarFundamentals.pe.latest < 10) &
    (SharadarFundamentals.pb.latest < 1.5) &
    (SharadarFundamentals.netinc.latest > 0)
)
```

### Profitable Growth
```python
# Large revenue, strong margins
revenue = SharadarFundamentals.revenue.latest
netinc = SharadarFundamentals.netinc.latest
margin = netinc / revenue

screen = (
    (revenue > 1e9) &  # $1B+ revenue
    (margin > 0.10) &  # 10%+ margin
    (SharadarFundamentals.pe.latest < 30)
)
```

---

## Notebook Sections

### 1. Setup
- Import libraries
- Load bundle
- Create Pipeline engine

### 2. Data Exploration
- Check data coverage (% of stocks with each metric)
- View sample companies
- Validate data quality

### 3. Strategy Screens (5 strategies)
- **Magic Formula**: High ROE + Earnings Yield
- **High ROE**: ROE > 20%, low debt
- **Profitable Growth**: Large revenue, strong margins
- **Deep Value**: Low P/E, low P/B
- **Quality Score**: Multi-factor assessment

### 4. Visualizations
- ROE vs P/E scatter plot
- Market cap coloring
- Identify sweet spots

### 5. Data Validation
- Negative equity check
- Extreme ratio detection
- Missing data analysis
- Summary statistics

### 6. Export Results
- Combine top picks
- Identify multi-strategy stocks
- Prepare for further research

---

## Customization Examples

### Change Universe Size
```python
# Only large caps
screen = (SharadarFundamentals.marketcap.latest > 10e9)

# Only mid caps
screen = (
    (SharadarFundamentals.marketcap.latest > 2e9) &
    (SharadarFundamentals.marketcap.latest < 10e9)
)
```

### Calculate Custom Metrics
```python
# Profit margin
profit_margin = (
    SharadarFundamentals.netinc.latest /
    SharadarFundamentals.revenue.latest
)

# Debt to assets
debt_ratio = (
    SharadarFundamentals.debt.latest /
    SharadarFundamentals.assets.latest
)

# Cash to debt
cash_coverage = (
    SharadarFundamentals.cashnequsd.latest /
    SharadarFundamentals.debt.latest
)
```

### Time Series Analysis
```python
from zipline.pipeline.factors import SimpleMovingAverage

# Average ROE over 4 quarters (1 year)
avg_roe = SimpleMovingAverage(
    inputs=[SharadarFundamentals.roe],
    window_length=4
)

# Trend: Current vs Average
improving_roe = (
    SharadarFundamentals.roe.latest > avg_roe
)
```

---

## Troubleshooting

### No Fundamentals Data
```bash
# Download fundamentals for existing bundle
docker exec zipline-reloaded-jupyter python \
    /app/scripts/download_fundamentals_only.py --bundle sharadar
```

### Import Errors
```bash
# Install visualization libraries
docker exec zipline-reloaded-jupyter pip install \
    matplotlib seaborn jupyter
```

### Memory Issues
```python
# Filter to smaller universe first
pipeline = Pipeline(
    columns={...},
    screen=(SharadarFundamentals.marketcap.latest > 1e9)  # Large caps only
)
```

### Timezone Warnings
- Already fixed in loader (uses UTC throughout)
- If you see warnings, ensure you've pulled latest code

---

## Output Expectations

### Data Coverage (Typical)
- **Total assets**: ~11,000
- **With fundamentals**: ~5,000-6,000 (45-50%)
- **Revenue coverage**: ~45-50%
- **Ratio coverage**: ~40-45% (varies by metric)

### Strategy Results (Typical)
- **Magic Formula**: 100-300 qualifying stocks
- **High ROE**: 50-150 stocks
- **Deep Value**: 30-100 stocks
- **Quality Score**: 200-500 stocks (score >= 4)

### Runtime
- **Load bundle**: 5-10 seconds
- **Run pipeline**: 10-30 seconds (depends on metrics)
- **Full notebook**: 1-2 minutes

---

## Next Steps After Testing

### 1. Build Full Backtest
```python
from zipline import run_algorithm

def initialize(context):
    # Attach your pipeline
    attach_pipeline(make_pipeline(), 'screener')

def make_pipeline():
    # Use your tested screen
    screen = (
        (SharadarFundamentals.roe.latest > 0.15) &
        (SharadarFundamentals.pe.latest < 20)
    )
    return Pipeline(columns={...}, screen=screen)

# Run backtest
results = run_algorithm(
    start=pd.Timestamp('2020-01-01', tz='UTC'),
    end=pd.Timestamp('2024-01-01', tz='UTC'),
    initialize=initialize,
    bundle='sharadar',
)
```

### 2. Add Technical Factors
```python
from zipline.pipeline.factors import Returns, SimpleMovingAverage

# Combine fundamentals + momentum
momentum = Returns(window_length=126)  # 6-month returns
sma = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=50)

combined_screen = (
    (SharadarFundamentals.roe.latest > 0.15) &  # Fundamental
    (momentum > 0) &                             # Technical
    (USEquityPricing.close.latest > sma)        # Technical
)
```

### 3. Portfolio Construction
```python
def rebalance(context, data):
    # Get fundamentals from pipeline
    fundamentals = context.pipeline_output

    # Rank by ROE
    ranked = fundamentals.sort_values('roe', ascending=False)

    # Top 20 stocks
    target_stocks = ranked.head(20)

    # Equal weight
    for asset in target_stocks.index:
        order_target_percent(asset, 1.0 / 20)
```

---

**Full Documentation**: `docs/SHARADAR_FUNDAMENTALS_GUIDE.md`
**Notebook README**: `examples/notebooks/README.md`
**Phase 1 Summary**: `docs/FUNDAMENTALS_PHASE1_COMPLETE.md`
