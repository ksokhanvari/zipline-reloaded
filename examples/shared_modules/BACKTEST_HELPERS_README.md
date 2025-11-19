# Backtest Helpers

Easy-to-use helper functions for running Zipline backtests and analyzing results with pyfolio.

## Quick Start

```python
from backtest_helpers import backtest, analyze_results

# Run backtest (matches your exact signature!)
backtest(
    "my_strategy.py",              # Algorithm file
    "my-strategy-v1",               # Name/identifier
    bundle="sharadar",
    data_frequency='daily',
    progress='D',
    start_date="2021-01-01",
    end_date="2025-01-01",
    capital_base=5000000,
    filepath_or_buffer="results.csv"  # Output filename
)

# Analyze results
analyze_results("./backtest_results/results.csv", benchmark_symbol="SPY")
```

## Installation

The helper functions are in `backtest_helpers.py` in this directory. Just import them:

```python
from backtest_helpers import backtest, analyze_results, quick_backtest
```

**Optional dependency for full analysis:**

```bash
pip install pyfolio-reloaded
```

If pyfolio is not installed, basic plots and metrics will still be generated.

## Functions

### `backtest()` - Run Backtest

Run a backtest and save results to disk.

**Signature:**

```python
backtest(
    algo_filename,           # Path to algorithm .py file
    name,                    # Backtest identifier/name
    bundle="sharadar",       # Data bundle name
    data_frequency='daily',  # 'daily' or 'minute'
    segment=None,            # Optional segment identifier
    progress='D',            # Progress bar: 'D', 'W', 'M', or None
    start_date="2020-01-01", # Start date (YYYY-MM-DD)
    end_date="2023-12-31",   # End date (YYYY-MM-DD)
    capital_base=100000,     # Starting capital
    filepath_or_buffer=None, # Output filename (defaults to {name}.csv)
    output_dir="./backtest_results",  # Output directory
    save_pickle=True,        # Also save as pickle
    **kwargs                 # Additional args for run_algorithm()
) -> pd.DataFrame
```

**What it does:**

1. Loads your algorithm from file
2. Runs the backtest
3. Saves results to CSV and pickle
4. Saves metadata JSON
5. Prints summary statistics
6. Returns performance DataFrame

**Output files:**

```
backtest_results/
├── my-strategy-v1.csv      # CSV format (easy to read)
├── my-strategy-v1.pkl      # Pickle format (preserves all data types)
└── my-strategy-v1.json     # Metadata (config, timestamps, etc.)
```

**Example:**

```python
perf = backtest(
    "backtest_with_fundamentals.py",
    "quality-strat-2023",
    bundle="sharadar",
    start_date="2023-01-01",
    end_date="2023-12-31",
    capital_base=1000000,
    filepath_or_buffer="quality-2023.csv"
)

# perf is a DataFrame with columns:
#  - portfolio_value
#  - returns
#  - positions
#  - transactions
#  - orders
#  - etc.
```

### `analyze_results()` - Analyze Backtest

Load saved backtest results and generate comprehensive analysis.

**Signature:**

```python
analyze_results(
    filepath_or_buffer,          # Path to CSV or pickle file
    benchmark_symbol='SPY',      # Benchmark ticker
    output_file=None,            # Save plots to file
    live_start_date=None,        # Separate in-sample vs out-of-sample
    show_plots=True,             # Display plots
    save_plots=True,             # Save plots to disk
    figsize=(14, 10),            # Figure size
    return_dict=False,           # Return detailed results dict
) -> dict (optional)
```

**What it does:**

1. Loads results from CSV or pickle
2. Extracts returns and metrics
3. Generates pyfolio tearsheet (if installed)
4. Creates performance plots:
   - Portfolio value over time
   - Cumulative returns
   - Drawdown
   - Monthly returns heatmap
5. Calculates metrics:
   - Total return
   - Annual return
   - Sharpe ratio
   - Max drawdown
   - Calmar ratio
   - Win rate

**Example:**

```python
# Basic analysis
analyze_results("./backtest_results/quality-2023.csv")

# Get detailed results
results = analyze_results(
    "./backtest_results/quality-2023.csv",
    benchmark_symbol="SPY",
    live_start_date="2024-01-01",  # Out-of-sample starts here
    return_dict=True
)

# Access components
returns = results['returns']
metrics = results['metrics']
perf = results['perf']
```

### `quick_backtest()` - Run and Analyze

Convenience function to run backtest and analyze in one step.

**Signature:**

```python
quick_backtest(
    algo_filename,
    name=None,                   # Defaults to algo filename
    start_date="2021-01-01",
    end_date="2023-12-31",
    capital_base=100000,
    analyze=True,                # Run analysis after backtest
    **kwargs
) -> pd.DataFrame
```

**Example:**

```python
# Run and analyze in one step
perf = quick_backtest(
    "my_strategy.py",
    start_date="2023-01-01",
    end_date="2023-12-31",
    capital_base=100000
)
```

## Algorithm File Requirements

Your algorithm file must define an `initialize` function:

```python
def initialize(context):
    """Called once at start of backtest"""
    context.my_var = 10
    # ... setup code ...

def handle_data(context, data):
    """Called every bar (optional)"""
    # ... trading logic ...

def before_trading_start(context, data):
    """Called before market open (optional)"""
    # ... pre-market logic ...
```

See `backtest_with_fundamentals.py` for a complete example.

## Examples

### Example 1: Basic Backtest

```python
from backtest_helpers import backtest, analyze_results

# Run backtest
perf = backtest(
    "my_strategy.py",
    "test-run-1",
    bundle="sharadar",
    start_date="2022-01-01",
    end_date="2023-12-31",
    capital_base=100000,
    filepath_or_buffer="test-run-1.csv"
)

# Analyze
analyze_results("./backtest_results/test-run-1.csv")
```

### Example 2: Multiple Scenarios

```python
scenarios = [
    {'name': 'bear-market', 'start': '2022-01-01', 'end': '2022-12-31'},
    {'name': 'bull-market', 'start': '2023-01-01', 'end': '2023-12-31'},
]

results = {}
for scenario in scenarios:
    perf = backtest(
        "my_strategy.py",
        scenario['name'],
        start_date=scenario['start'],
        end_date=scenario['end'],
        capital_base=100000,
        filepath_or_buffer=f"{scenario['name']}.csv"
    )
    results[scenario['name']] = perf

# Compare
for name, perf in results.items():
    total_return = (perf['portfolio_value'].iloc[-1] / 100000 - 1) * 100
    print(f"{name}: {total_return:.2f}%")
```

### Example 3: Reload and Re-analyze

```python
# Run backtest once
backtest(
    "my_strategy.py",
    "long-term-test",
    start_date="2020-01-01",
    end_date="2023-12-31",
    capital_base=1000000,
    filepath_or_buffer="long-term.csv"
)

# Later: Re-analyze without re-running backtest
analyze_results("./backtest_results/long-term.csv")

# Or: Load for custom analysis
import pandas as pd
perf = pd.read_csv("./backtest_results/long-term.csv", index_col=0, parse_dates=True)

# Custom metrics
monthly_returns = perf['portfolio_value'].resample('M').last().pct_change()
print(f"Best month: {monthly_returns.max():.2%}")
print(f"Worst month: {monthly_returns.min():.2%}")
```

### Example 4: Custom Analysis

```python
results = analyze_results(
    "./backtest_results/my-strategy.csv",
    show_plots=False,
    return_dict=True
)

# Access data
returns = results['returns']
perf = results['perf']
metrics = results['metrics']

# Custom plot: Underwater (drawdown) chart
import matplotlib.pyplot as plt
import numpy as np

cumulative = (returns + 1).cumprod()
running_max = cumulative.cummax()
underwater = (cumulative / running_max - 1) * 100

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(underwater.index, underwater, 0, alpha=0.3, color='red')
ax.plot(underwater.index, underwater, color='red', linewidth=2)
ax.set_title('Underwater Plot (Drawdown)', fontsize=14, fontweight='bold')
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.3)
plt.show()
```

## Metrics Calculated

The helper automatically calculates:

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative return over period |
| **Annual Return** | Annualized return (CAGR) |
| **Annual Volatility** | Annualized standard deviation |
| **Sharpe Ratio** | Risk-adjusted return (assumes 0% risk-free rate) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Calmar Ratio** | Annual return / max drawdown |
| **Win Rate** | Percentage of positive return days |
| **Final Portfolio Value** | Ending capital |

## Pyfolio Integration

If `pyfolio-reloaded` is installed, you get the full tearsheet with:

- **Returns analysis**: Daily, monthly, yearly returns
- **Risk metrics**: Sharpe, Sortino, Calmar ratios
- **Drawdown analysis**: Underwater plot, top drawdowns
- **Rolling metrics**: Rolling Sharpe, volatility
- **Monthly returns heatmap**
- **Distribution analysis**: Return distribution, Q-Q plot
- **Worst periods**: Worst 5 drawdowns with details

**Install pyfolio:**

```bash
pip install pyfolio-reloaded
```

## File Structure

```
backtest_results/
├── my-strategy.csv              # Performance data (CSV)
├── my-strategy.pkl              # Performance data (pickle)
└── my-strategy.json             # Metadata
```

**Metadata JSON contains:**

```json
{
  "backtest_name": "my-strategy",
  "algo_file": "my_strategy.py",
  "bundle": "sharadar",
  "data_frequency": "daily",
  "start_date": "2022-01-01T00:00:00+00:00",
  "end_date": "2023-12-31T00:00:00+00:00",
  "capital_base": 100000,
  "run_timestamp": "2025-01-09T10:30:45.123456",
  "num_rows": 252,
  "csv_path": "./backtest_results/my-strategy.csv",
  "pickle_path": "./backtest_results/my-strategy.pkl"
}
```

## Tips

### 1. Always Save Results

Running backtests is time-consuming. Always save results so you can re-analyze later:

```python
# Good: Results saved, can re-analyze anytime
backtest(..., filepath_or_buffer="my-results.csv")

# Bad: Results lost after this session
perf = backtest(..., filepath_or_buffer=None)  # Not saved!
```

### 2. Use Descriptive Names

```python
# Good: Clear what this is
backtest(..., name="momentum-top20-monthly", filepath_or_buffer="momentum-top20-monthly.csv")

# Bad: Unclear
backtest(..., name="test1", filepath_or_buffer="test1.csv")
```

### 3. Save Pickle for Complex Data

CSV files lose some data types. Pickle preserves everything:

```python
backtest(..., save_pickle=True)  # Saves both CSV and pickle

# Later: Load from pickle for complete data
import pandas as pd
perf = pd.read_pickle("./backtest_results/my-strategy.pkl")
```

### 4. Compare Multiple Periods

```python
periods = [
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
]

for name, start, end in periods:
    backtest(
        "my_strategy.py",
        f"test-{name}",
        start_date=start,
        end_date=end,
        capital_base=100000,
        filepath_or_buffer=f"test-{name}.csv"
    )
```

### 5. Separate In-Sample and Out-of-Sample

```python
# Run full backtest
backtest(
    "my_strategy.py",
    "full-period",
    start_date="2020-01-01",  # In-sample: 2020-2023
    end_date="2024-12-31",     # Out-of-sample: 2024
    filepath_or_buffer="full-period.csv"
)

# Analyze with live start date
analyze_results(
    "./backtest_results/full-period.csv",
    live_start_date="2024-01-01"  # Separates in-sample vs out-of-sample
)
```

## Troubleshooting

### Algorithm File Not Found

```
FileNotFoundError: Algorithm file not found: my_strategy.py
```

**Solution:** Use absolute path or check current directory:

```python
import os
print(os.getcwd())  # Check current directory

# Use absolute path
backtest(
    "/full/path/to/my_strategy.py",
    ...
)
```

### No 'initialize' Function

```
ValueError: Algorithm file must define an 'initialize' function
```

**Solution:** Ensure your algorithm file has:

```python
def initialize(context):
    # ... your code ...
    pass
```

### Results File Not Found

```
FileNotFoundError: File not found: ./backtest_results/my-strategy.csv
```

**Solution:** Check the filename matches exactly:

```python
# When saving
backtest(..., filepath_or_buffer="my-strategy.csv")

# When loading (must match!)
analyze_results("./backtest_results/my-strategy.csv")
```

### Pyfolio Errors

If you get errors with pyfolio, the helpers will fall back to basic plots:

```
⚠ Warning: Could not create full tearsheet: ...
  Generating basic plots instead...
```

Basic plots are still generated even without pyfolio.

## See Also

- **[backtest_helpers_example.ipynb](backtest_helpers_example.ipynb)** - Complete examples
- **[backtest_with_fundamentals.py](backtest_with_fundamentals.py)** - Example algorithm
- **[BACKTEST_README.md](BACKTEST_README.md)** - General backtest guide
- **[DATABASE_CLASS_GUIDE.md](DATABASE_CLASS_GUIDE.md)** - Custom data integration
