# Zipline Example Notebooks

Comprehensive examples demonstrating backtesting strategies with Zipline and pyfolio analysis.

---

## 📚 Notebook Overview

| Notebook | Topic | Difficulty | Runtime |
|----------|-------|------------|---------|
| [01_quickstart_sharadar_flightlog.ipynb](./01_quickstart_sharadar_flightlog.ipynb) | Quick start with FlightLog | ⭐ Beginner | 5 min |
| [02_buy_and_hold_strategy.ipynb](./02_buy_and_hold_strategy.ipynb) | Buy and hold SPY | ⭐ Beginner | 3 min |
| [03_moving_average_crossover.ipynb](./03_moving_average_crossover.ipynb) | MA crossover strategy | ⭐⭐ Intermediate | 5 min |
| [04_pyfolio_analysis.ipynb](./04_pyfolio_analysis.ipynb) | Complete tearsheet analysis | ⭐⭐ Intermediate | 8 min |
| [05_pipeline_factors.ipynb](./05_pipeline_factors.ipynb) | Factor-based stock screening | ⭐⭐⭐ Advanced | 10 min |
| [06_strategy_comparison.ipynb](./06_strategy_comparison.ipynb) | Compare multiple strategies | ⭐⭐⭐ Advanced | 15 min |

---

## 🚀 Getting Started

### Prerequisites

1. **Data ingested:**
   ```bash
   zipline ingest -b sharadar
   ```

2. **Jupyter Lab running:**
   ```bash
   # Already running in Docker
   ```

3. **FlightLog (optional):**
   ```bash
   docker compose run --rm flightlog
   ```

### Recommended Learning Path

**Beginners** (New to Zipline):
1. Start with `01_quickstart_sharadar_flightlog.ipynb`
2. Then try `02_buy_and_hold_strategy.ipynb`
3. Move to `03_moving_average_crossover.ipynb`

**Intermediate** (Familiar with basics):
1. Skip to `04_pyfolio_analysis.ipynb` for performance metrics
2. Try `06_strategy_comparison.ipynb` to compare approaches

**Advanced** (Ready for factor investing):
1. Dive into `05_pipeline_factors.ipynb`
2. Build your own multi-factor strategies

---

## 📖 Detailed Descriptions

### 01 - Quick Start with FlightLog
**What you'll learn:**
- Running your first backtest
- Connecting to FlightLog for real-time monitoring
- Basic strategy structure
- Viewing results

**Key concepts:**
- `initialize()`, `handle_data()`
- `order_target_percent()`
- FlightLog integration

---

### 02 - Buy and Hold Strategy
**What you'll learn:**
- Simplest possible strategy
- One-time purchase and hold
- Performance visualization
- Transaction logging

**Use case:** Passive investing baseline

**Key code:**
```python
def handle_data(context, data):
    if not context.has_bought:
        order_target_percent(context.spy, 1.0)
        context.has_bought = True
```

---

### 03 - Moving Average Crossover
**What you'll learn:**
- Technical analysis strategy
- Buy/sell signal generation
- Position tracking
- Trade log analysis

**Strategy:**
- **Buy:** Fast MA (50-day) crosses above slow MA (200-day)
- **Sell:** Fast MA crosses below slow MA

**Key features:**
- Signal visualization
- Entry/exit points marked on chart
- Drawdown analysis
- Win rate calculation

---

### 04 - Pyfolio Analysis
**What you'll learn:**
- Generate professional tearsheets
- Comprehensive performance metrics
- Risk analysis (Sharpe, Sortino, Calmar)
- Drawdown periods
- Monthly returns heatmap
- Rolling statistics

**Pyfolio features demonstrated:**
- `create_full_tear_sheet()`
- Drawdown plots
- Return distributions
- Transaction analysis
- Round-trip trades

**Metrics covered:**
- Annual return & volatility
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Calmar ratio
- Win rate

---

### 05 - Pipeline Factors
**What you'll learn:**
- Zipline's Pipeline API
- Custom factor creation
- Multi-factor stock screening
- Quantitative stock selection
- Factor combination strategies

**Pipeline concepts:**
- Custom factors (Momentum, Volatility)
- Universe definition
- Factor normalization (z-scores)
- Stock ranking
- Top-N selection

**Example strategy:**
- Universe: Top 500 liquid stocks
- Factors: Momentum (60-day) + Low Volatility (20-day)
- Selection: Top 5 by combined score
- Rebalance: Monthly

**Advanced topics:**
- Adding value/quality factors
- Sector filters
- Correlation screens
- Multi-factor rankings

---

### 06 - Strategy Comparison
**What you'll learn:**
- Running multiple backtests
- Side-by-side performance comparison
- Risk-adjusted returns
- Metric visualization
- Strategy selection criteria

**Strategies compared:**
1. **Buy & Hold** - Passive SPY
2. **MA Crossover** - Trend following
3. **Momentum Rotation** - Active factor

**Comparisons shown:**
- Total return
- Sharpe ratio
- Max drawdown
- Annual volatility
- Win rate
- Risk-return scatter plot
- Cumulative returns

**Decision framework:**
- When to use each strategy
- Risk tolerance considerations
- Market regime awareness

---

## 💡 Common Patterns

### Basic Strategy Structure

```python
def initialize(context):
    # Setup: runs once at start
    context.stock = symbol('SPY')

def handle_data(context, data):
    # Trading logic: runs every bar
    price = data.current(context.stock, 'price')
    order_target_percent(context.stock, 1.0)
```

### Scheduled Rebalancing

```python
def initialize(context):
    schedule_function(
        rebalance,
        date_rules.month_start(),  # When: start of month
        time_rules.market_open(hours=1)  # Time: 1 hour after open
    )

def rebalance(context, data):
    # Rebalancing logic
    pass
```

### Recording Metrics

```python
def handle_data(context, data):
    record(
        portfolio_value=context.portfolio.portfolio_value,
        cash=context.portfolio.cash,
        leverage=context.account.leverage,
    )
```

### Using Pipeline

```python
from zipline.pipeline import Pipeline, CustomFactor

def initialize(context):
    pipe = make_pipeline()
    attach_pipeline(pipe, 'my_pipeline')

def before_trading_start(context, data):
    context.output = pipeline_output('my_pipeline')
```

---

## 📊 Available Data

**With Sharadar bundle you have access to:**
- **8,000+ US equities** (1998-present)
- **ETFs and funds**
- **Daily OHLCV data**
- **Corporate actions** (splits, dividends)

**Data frequency:**
- Daily bars (default)
- Minute bars (if ingested)

**Universe filtering:**
- Dollar volume
- Market cap
- Sector/industry
- Custom criteria

---

## 🛠️ Customization Tips

### Change Stock/ETF

```python
# Instead of SPY, try:
context.stock = symbol('QQQ')   # Nasdaq 100
context.stock = symbol('AAPL')  # Individual stock
context.stock = symbol('GLD')   # Gold ETF
```

### Adjust Time Period

```python
results = run_algorithm(
    start=pd.Timestamp('2015-01-01', tz='UTC'),
    end=pd.Timestamp('2020-12-31', tz='UTC'),
    # ... other params
)
```

### Change Parameters

```python
# MA Crossover
FAST_MA = 20   # Faster signals
SLOW_MA = 50   # Faster signals

# Momentum
TOP_N = 5      # Hold more stocks
MOMENTUM_WINDOW = 120  # Longer lookback
```

### Add Stop Loss

```python
def handle_data(context, data):
    for stock in context.portfolio.positions:
        cost_basis = context.portfolio.positions[stock].cost_basis
        current_price = data.current(stock, 'price')

        # 10% stop loss
        if current_price < cost_basis * 0.90:
            order_target_percent(stock, 0.0)
```

---

## 🐛 Troubleshooting

### "Bundle not found"

```bash
# Make sure sharadar bundle is ingested
zipline bundles
zipline ingest -b sharadar
```

### "Symbol not found"

Check ticker availability:
```python
from zipline.data import bundles
bundle = bundles.load('sharadar')
finder = bundle.asset_finder
asset = finder.lookup_symbol('AAPL', as_of_date=None)
print(f"{asset.start_date} to {asset.end_date}")
```

### "Not enough data"

Ensure your date range is within bundle data:
```python
# Check bundle date range first
bundle = bundles.load('sharadar')
sessions = bundle.equity_daily_bar_reader.sessions
print(f"Data available: {sessions[0]} to {sessions[-1]}")
```

### Pyfolio import error

```bash
pip install pyfolio-reloaded
```

### FlightLog not receiving logs

```bash
# Make sure FlightLog is running
docker ps | grep flightlog

# Check connection in notebook
from zipline.utils.flightlog_client import enable_flightlog
enable_flightlog(host='flightlog', port=9020)
```

---

## 📈 Performance Tips

### Speed up development

1. **Use smaller date ranges** while testing:
   ```python
   start=pd.Timestamp('2023-01-01', tz='UTC')  # Just 1 year
   ```

2. **Reduce logging verbosity:**
   ```python
   logging.basicConfig(level=logging.WARNING, force=True)
   ```

3. **Test with fewer stocks:**
   ```python
   UNIVERSE = ['SPY', 'QQQ', 'IWM']  # Just 3 instead of 20
   ```

### Memory optimization

For large universes:
- Use Pipeline for stock screening (more efficient)
- Avoid loading unnecessary history
- Batch your analysis

---

## 🎓 Learning Resources

**Zipline Documentation:**
- Main docs: https://zipline.ml4trading.io
- API reference: https://zipline.ml4trading.io/api-reference.html
- Pipeline tutorial: https://zipline.ml4trading.io/pipeline.html

**Pyfolio:**
- GitHub: https://github.com/quantopian/pyfolio
- Example notebooks: In pyfolio repo

**Books:**
- "Machine Learning for Algorithmic Trading" by Stefan Jansen
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Quantitative Momentum" by Wesley Gray

**Papers:**
- AQR Capital: https://www.aqr.com/Insights/Research
- Research Affiliates: https://www.researchaffiliates.com/research

---

## 💬 Getting Help

**Issues with notebooks:**
1. Check this README first
2. Review notebook comments and markdown cells
3. See [../docs/README.md](../docs/README.md) for data management
4. Open issue on GitHub if still stuck

**Community:**
- Forum: https://exchange.ml4trading.io
- GitHub: https://github.com/stefan-jansen/zipline-reloaded/issues

---

## 🚦 Next Steps

After completing these notebooks:

1. **Modify existing strategies** - Change parameters, add filters
2. **Combine strategies** - Ensemble methods, strategy rotation
3. **Add risk management** - Stop losses, position sizing
4. **Test robustness** - Different time periods, market regimes
5. **Build your own** - Custom factors, unique trading ideas
6. **Paper trade** - Test with live data (if available)

---

## ⚠️ Important Notes

**These examples are for educational purposes:**
- Not financial advice
- Past performance doesn't guarantee future results
- Always consider transaction costs in real trading
- Test thoroughly before using real money

**Backtesting limitations:**
- Look-ahead bias
- Survivorship bias
- Market impact not modeled
- Slippage assumptions
- Transaction costs

**Best practices:**
- Out-of-sample testing
- Walk-forward analysis
- Monte Carlo simulation
- Stress testing
- Regular strategy review

---

## 📝 License

These notebooks are part of Zipline-Reloaded and are licensed under Apache 2.0.

---

**Happy backtesting!** 🚀📊💰
