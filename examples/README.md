# Zipline Examples

Production-ready examples demonstrating FlightLog and Progress Logging.

## Quick Start

All examples require FlightLog running in Terminal 1:

```bash
# Terminal 1: Start FlightLog
python scripts/flightlog.py --host 0.0.0.0 --level INFO
```

Then run any example in Terminal 2 and watch the colored logs appear in Terminal 1!

## Examples

### 1. Simple Demo (Start Here!)
**File:** `simple_flightlog_demo.py`

The simplest possible example - buy and hold AAPL with FlightLog monitoring.

```bash
python examples/simple_flightlog_demo.py
```

**What you'll see:**
- ✅ Real-time progress bars
- ✅ Strategy initialization logs
- ✅ Position entry logs
- ✅ Final performance summary

**Perfect for:** Learning the basics, quick testing

---

### 2. Momentum Strategy (Production-Ready)
**File:** `momentum_strategy_with_flightlog.py`

Complete momentum trading strategy with professional logging.

```bash
python examples/momentum_strategy_with_flightlog.py
```

**Features:**
- ✅ Multi-stock momentum ranking
- ✅ Weekly rebalancing with logs
- ✅ Error handling and recovery
- ✅ Comprehensive analytics

**What you'll learn:**
- Strategic logging patterns
- Error handling with FlightLog
- Performance monitoring
- Professional code structure

**Perfect for:** Real strategy development, production use

---

### 3. Elegant Jupyter Integration
**File:** `../notebooks/flightlog_elegant_example.ipynb`

Interactive Jupyter notebook showing clean FlightLog integration.

```bash
# Open in Jupyter
# Navigate to: notebooks/flightlog_elegant_example.ipynb
```

**Features:**
- ✅ Step-by-step setup
- ✅ Interactive cells
- ✅ Best practices guide
- ✅ Plotting and analysis

**Perfect for:** Jupyter users, interactive development

---

## Output Examples

### Terminal 1 (FlightLog) - Color-Coded Logs

```
2025-11-03 10:30:15 INFO     algorithm: Initializing Momentum Strategy (lookback=20d, top=5)
2025-11-03 10:30:15 INFO     algorithm: Strategy initialized with 10 stocks
2025-11-03 10:30:45 INFO     zipline.progress: [Momentum-Strategy]  Progress      Pct    Date             Cum Returns        Sharpe      Max DD             Cum PNL
2025-11-03 10:30:45 INFO     zipline.progress: [Momentum-Strategy]  ----------      2%  2023-01-10              1%          0.00          0%               $1.2K
2025-11-03 10:31:02 INFO     algorithm: Rebalance #1: Selected 5 stocks
2025-11-03 10:31:02 INFO     algorithm: Rebalance complete: 5 orders, 5 positions, Portfolio: $101,250, Cash: $1,250
2025-11-03 10:31:25 INFO     zipline.progress: [Momentum-Strategy]  ██--------     15%  2023-02-28              5%          8.43          1%               $5.2K
2025-11-03 10:32:45 INFO     zipline.progress: [Momentum-Strategy]  ██████████    100%  2023-12-31             28%          1.88         12%              $28.0K
2025-11-03 10:32:46 INFO     algorithm: ============================================================
2025-11-03 10:32:46 INFO     algorithm: BACKTEST SUMMARY
2025-11-03 10:32:46 INFO     algorithm: Total Return: +28.27%
2025-11-03 10:32:46 INFO     algorithm: Final Portfolio Value: $128,268
2025-11-03 10:32:46 INFO     algorithm: Sharpe Ratio: 1.31
```

**Colors:**
- 🟢 **INFO** = Green
- 🟡 **WARNING** = Yellow
- 🔴 **ERROR** = Red
- 🟣 **CRITICAL** = Magenta
- 🔵 **DEBUG** = Cyan

### Terminal 2 (Backtest) - Progress Summary

```
Simple FlightLog Demo
============================================================
Watch Terminal 1 for real-time colored logs!

[Simple-Demo] Backtest initialized: 2023-01-03 to 2023-12-29 (251 trading days)

============================================================
Total Return: +28.27%
Final Value: $128,268
============================================================
```

## Usage Patterns

### Pattern 1: Minimal Setup (3 lines)

```python
import logging
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog

logging.basicConfig(level=logging.INFO)
enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='My-Strategy', update_interval=5)
```

### Pattern 2: Strategic Logging

```python
from zipline.utils.flightlog_client import log_to_flightlog

def initialize(context):
    log_to_flightlog('Strategy initialized', level='INFO')

def rebalance(context, data):
    log_to_flightlog(f'Rebalanced: {positions} positions', level='INFO')

def analyze(context, perf):
    total_return = perf['algorithm_period_return'].iloc[-1] * 100
    log_to_flightlog(f'Final return: {total_return:+.2f}%', level='INFO')
```

### Pattern 3: Error Handling

```python
try:
    # Strategy logic
    order_target_percent(stock, weight)
except Exception as e:
    log_to_flightlog(f'Order failed: {str(e)}', level='ERROR')
```

## Best Practices

### ✅ DO

1. **Log important events only**
   ```python
   log_to_flightlog('Rebalancing portfolio', level='INFO')  # ✅
   ```

2. **Use appropriate log levels**
   ```python
   log_to_flightlog('Low liquidity warning', level='WARNING')  # ✅
   ```

3. **Include context in messages**
   ```python
   log_to_flightlog(f'Portfolio: ${value:,.0f}', level='INFO')  # ✅
   ```

### ❌ DON'T

1. **Don't log on every bar**
   ```python
   def handle_data(context, data):
       log_to_flightlog('Processing...', level='INFO')  # ❌ Too noisy
   ```

2. **Don't use wrong levels**
   ```python
   log_to_flightlog('Order placed', level='CRITICAL')  # ❌ Not critical
   ```

3. **Don't log sensitive data**
   ```python
   log_to_flightlog(f'API key: {key}', level='INFO')  # ❌ Security risk
   ```

## Configuration

### Update Intervals

Choose based on backtest length:

```python
# Short backtests (< 1 year)
enable_progress_logging(update_interval=1)  # Daily

# Medium backtests (1-5 years)
enable_progress_logging(update_interval=5)  # Weekly

# Long backtests (5+ years)
enable_progress_logging(update_interval=20)  # Monthly
```

### FlightLog Options

```bash
# Basic
python scripts/flightlog.py

# With file output
python scripts/flightlog.py --file logs/backtest.log

# Debug level
python scripts/flightlog.py --level DEBUG

# Custom port
python scripts/flightlog.py --port 9021
```

## Troubleshooting

### Connection Failed

**Problem:** `enable_flightlog()` returns `False`

**Solutions:**
1. Check FlightLog is running: `docker compose ps`
2. Use `localhost` when inside Docker
3. Verify port: Default is 9020
4. Test: `nc -zv localhost 9020`

### No Logs Appearing

**Problem:** Logs sent but not visible

**Solutions:**
1. Set logging level: `logging.basicConfig(level=logging.INFO)`
2. Check FlightLog level: Start with `--level DEBUG`
3. Verify connection: Check return value of `enable_flightlog()`

### Performance Issues

**Problem:** Backtest runs slowly

**Solutions:**
1. Reduce logging frequency
2. Increase `update_interval`
3. Use INFO level instead of DEBUG
4. Don't log in `handle_data()`

## Documentation

- **Best Practices:** `../docs/FLIGHTLOG_BEST_PRACTICES.md`
- **Full Guide:** `../docs/FLIGHTLOG_USAGE.md`
- **Getting Started:** `../GETTING_STARTED.md`

## Summary

| Example | Complexity | Use Case | Features |
|---------|-----------|----------|----------|
| `simple_flightlog_demo.py` | ⭐ Simple | Learning | Basic logging, buy & hold |
| `momentum_strategy_with_flightlog.py` | ⭐⭐⭐ Advanced | Production | Full strategy, error handling |
| `flightlog_elegant_example.ipynb` | ⭐⭐ Medium | Interactive | Jupyter, step-by-step |

Start with `simple_flightlog_demo.py` to learn the basics, then move to the momentum strategy for production patterns!

## Quick Test

```bash
# Terminal 1
python scripts/flightlog.py

# Terminal 2
python examples/simple_flightlog_demo.py

# Watch Terminal 1 for colored logs! 🎉
```
