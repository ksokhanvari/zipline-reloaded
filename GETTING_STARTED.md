# Getting Started with Your New Features

You now have progress logging and FlightLog merged into main! Here's how to use them.

## 🎯 Quick Start - Test Everything Works

### Test 1: Progress Logging (Simple)

```bash
# Inside your Docker container
python test_progress_quick.py
```

Expected output:
```
[Quick-Test]  Progress      Pct    Date          Cum Returns      Sharpe      Max DD      Cum PNL
[Quick-Test]  ----------     10%   2020-01-01              0%        0.00          0%           $0
[Quick-Test]  ██--------     20%   2020-01-02              1%        5.21          0%         $500
[Quick-Test]  ██████████    100%   2020-01-10             10%        1.45          5%      $10,000
```

### Test 2: FlightLog (Two Terminals)

**Terminal 1 - Start FlightLog Server:**
```bash
python scripts/flightlog.py --host 0.0.0.0 --level INFO
```

**Terminal 2 - Run Test Backtest:**
```bash
python notebooks/test_flightlog_backtest.py
```

Watch the logs stream to Terminal 1 with color coding!

## 📊 Use in Your Real Backtests

### Simple Integration (Just Progress)

```python
from zipline import run_algorithm
from zipline.api import *
from zipline.utils.progress import enable_progress_logging
import pandas as pd

# Enable progress logging (one line!)
enable_progress_logging(algo_name='My-Strategy', update_interval=5)

# Your strategy
def initialize(context):
    context.stock = symbol('AAPL')

def handle_data(context, data):
    order_target_percent(context.stock, 0.95)

# Run - progress displays automatically
result = run_algorithm(
    start=pd.Timestamp('2020-01-01'),
    end=pd.Timestamp('2020-12-31'),
    initialize=initialize,
    handle_data=handle_data,
    capital_base=100000,
    bundle='sharadar'
)
```

You'll see output like:
```
[My-Strategy] Backtest initialized: 2020-01-01 to 2020-12-31 (252 trading days)
[My-Strategy]  Progress      Pct    Date          Cum Returns      Sharpe      Max DD      Cum PNL
[My-Strategy]  █---------      7%   2020-01-20              3%        5.21          1%       $3,145
[My-Strategy]  ██████████    100%   2020-12-31             15%        1.89          8%      $15,000
```

### Full Integration (Progress + FlightLog)

```python
import logging
from zipline import run_algorithm
from zipline.api import *
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)

# Enable FlightLog (Terminal 1 must be running: python scripts/flightlog.py)
enable_flightlog(host='localhost', port=9020)

# Enable progress
enable_progress_logging(algo_name='My-Strategy', update_interval=5)

def initialize(context):
    log_to_flightlog("Strategy initialized!", level='INFO')
    context.stock = symbol('AAPL')

def handle_data(context, data):
    if not context.portfolio.positions:
        order_target_percent(context.stock, 0.95)
        log_to_flightlog(f"Opened position: {context.stock.symbol}", level='INFO')

# Run
result = run_algorithm(
    start=pd.Timestamp('2020-01-01'),
    end=pd.Timestamp('2020-12-31'),
    initialize=initialize,
    handle_data=handle_data,
    capital_base=100000,
    bundle='sharadar'
)
```

## 🎨 Configuration Options

### Progress Update Intervals

Choose based on backtest length:

```python
# Daily updates (< 1 year backtests)
enable_progress_logging(update_interval=1)

# Weekly updates (1-5 year backtests)
enable_progress_logging(update_interval=5)

# Monthly updates (5-10 year backtests)
enable_progress_logging(update_interval=20)

# Quarterly updates (10+ year backtests)
enable_progress_logging(update_interval=60)
```

### Custom Algorithm Names

```python
# Match QuantRocket style
enable_progress_logging(algo_name='LS-Prod-Algo')

# Or your own naming
enable_progress_logging(algo_name='MeanReversion-v2')
enable_progress_logging(algo_name='Momentum-2024')
```

### FlightLog Options

```bash
# Different log level
python scripts/flightlog.py --level DEBUG

# Save to custom file
python scripts/flightlog.py --file logs/my_backtest_$(date +%Y%m%d).log

# No colors (for piping to file)
python scripts/flightlog.py --no-color > output.log

# Different port
python scripts/flightlog.py --port 9021
```

## 📁 What's Included

### New Files
- `src/zipline/utils/progress.py` - Progress logging implementation
- `src/zipline/utils/flightlog_client.py` - FlightLog client library
- `scripts/flightlog.py` - FlightLog server
- `docs/FLIGHTLOG_USAGE.md` - Detailed documentation
- `notebooks/test_flightlog_backtest.py` - Working example
- `test_progress_quick.py` - Quick verification test

### Features
✅ Real-time progress bars
✅ Live portfolio metrics (Returns, Sharpe, Drawdown, PNL)
✅ QuantRocket-style output formatting
✅ Color-coded log levels in FlightLog
✅ Separate terminal for logs
✅ Automatic log file saving
✅ Multiple backtest support
✅ Zero impact on backtest performance

## 🔍 Troubleshooting

### Progress not showing
Make sure to call `enable_progress_logging()` before `run_algorithm()`:
```python
# ✅ CORRECT
enable_progress_logging(algo_name='MyStrat')
result = run_algorithm(...)

# ❌ WRONG
result = run_algorithm(...)
enable_progress_logging(algo_name='MyStrat')  # Too late!
```

### FlightLog not connecting
1. Make sure server is running in Terminal 1
2. Check connection:
```python
from zipline.utils.flightlog_client import get_flightlog_status
print(get_flightlog_status())
```

### Messages not appearing in FlightLog
Set logging level:
```python
import logging
logging.basicConfig(level=logging.INFO)  # ← Important!
```

### Port already in use
Kill existing FlightLog:
```bash
pkill -f flightlog.py
# Then restart
python scripts/flightlog.py
```

## 📚 Documentation

- **Full FlightLog Guide**: `docs/FLIGHTLOG_USAGE.md`
- **Example Backtest**: `notebooks/test_flightlog_backtest.py`
- **Quick Test**: `test_progress_quick.py`

## 🚀 Next Steps

1. **Test the features:**
   ```bash
   python test_progress_quick.py
   ```

2. **Try FlightLog:**
   - Terminal 1: `python scripts/flightlog.py`
   - Terminal 2: `python notebooks/test_flightlog_backtest.py`

3. **Use in your notebooks:**
   Add `enable_progress_logging()` to your existing algos

4. **Read the docs:**
   Check `docs/FLIGHTLOG_USAGE.md` for advanced usage

## 🎉 You're Ready!

Everything is set up and working. Start with the quick test, then integrate into your real strategies.

Happy backtesting! 📈
