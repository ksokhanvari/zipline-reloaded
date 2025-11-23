# FlightLog Guide

FlightLog is a real-time log streaming system that displays backtest logs in a separate terminal window, similar to QuantRocket's flightlog.

---

## Features

- Color-coded log levels (green INFO, yellow WARNING, red ERROR)
- Real-time streaming via TCP socket
- **Channel separation**: LOG channel (9020) for progress/algorithm logs, PRINT channel (9021) for print output
- Progress bar support with portfolio metrics
- Filtering options to show only what you need
- **IPython magic commands** for easy server management
- Automatic stdout redirection to keep notebooks clean

---

## Quick Start

### Step 1: Start FlightLog Servers

**Option A: Using IPython Magic (Recommended)**

In your Jupyter notebook:
```python
%load_ext flightlog_commands
%flightlog both
```

This starts both servers:
- Port 9020: LOG channel (progress bars, algorithm logs)
- Port 9021: PRINT channel (all print output)

**Option B: Using Docker Compose**

```bash
# Terminal 1: LOG channel
docker compose run --rm flightlog python /app/scripts/flightlog.py --port 9020

# Terminal 2: PRINT channel
docker compose run --rm flightlog python /app/scripts/flightlog.py --port 9021
```

### Step 2: Run Your Backtest with backtest_helpers

The `backtest()` function in `backtest_helpers.py` automatically handles FlightLog integration:

```python
from examples.utils.backtest_helpers import backtest

# Run backtest - FlightLog is automatically configured
results = backtest(
    algo_filename='/app/examples/strategies/LS-ZR-ported.py',
    start='2023-01-01',
    end='2024-01-01',
    capital_base=1000000,
    bundle='sharadar'
)
```

The backtest helper will:
- Connect to LOG channel (9020) for progress and `log_to_flightlog()` calls
- Connect to PRINT channel (9021) for all print output
- Suppress print output in notebook when FlightLog is connected
- Show daily progress updates with strategy name

### Step 3: Manual Setup (Alternative)

If not using backtest_helpers:

```python
import logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
from zipline.utils.progress import enable_progress_logging

# Setup logging
logging.basicConfig(level=logging.INFO, force=True)

# Connect to FlightLog
enable_flightlog(host='localhost', port=9020)

# Enable progress logging
enable_progress_logging(algo_name='My-Strategy', update_interval=5)

# Run your backtest
result = run_algorithm(...)
```

### Step 4: Log Important Events

```python
def initialize(context):
    log_to_flightlog('Strategy initialized', level='INFO')

def rebalance(context, data):
    log_to_flightlog('Portfolio rebalanced', level='INFO')
```

---

## IPython Magic Commands

The `flightlog_commands` extension provides convenient magic commands:

```python
# Load the extension (auto-loaded on startup)
%load_ext flightlog_commands

# Available commands
%flightlog log      # Start LOG server on port 9020
%flightlog print    # Start PRINT server on port 9021
%flightlog both     # Start both servers
%flightlog status   # Check server status
%flightlog stop     # Stop all servers
```

You can also import functions directly:
```python
from flightlog_commands import start_log_server, start_print_server, start_both, status, stop_servers
```

---

## Channel Separation

FlightLog uses two channels to separate different types of output:

| Channel | Port | Purpose |
|---------|------|---------|
| LOG | 9020 | Progress bars, `log_to_flightlog()` calls, algorithm logs |
| PRINT | 9021 | All print statements (redirected from notebook) |

**Benefits:**
- Keep your notebook clean - print output goes to FlightLog
- Separate important logs from debug clutter
- Run different filter options on each channel

**Terminal Setup:**
```bash
# Terminal 1: Important logs only
python scripts/flightlog.py --port 9020

# Terminal 2: All print output
python scripts/flightlog.py --port 9021
```

---

## Command-Line Options

```bash
python scripts/flightlog.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | `0.0.0.0` | Host to bind to |
| `--port PORT` | `9020` | Port to listen on |
| `--level LEVEL` | `INFO` | Min log level |
| `--file FILE` | None | Save logs to file |
| `--no-color` | False | Disable color output |
| `--filter-progress` | False | Hide progress logs |

### Examples

```bash
# Default: All logs with colors
python scripts/flightlog.py

# Algorithm logs only (hide progress bars)
python scripts/flightlog.py --filter-progress

# Debug level, save to file
python scripts/flightlog.py --level DEBUG --file backtest.log
```

---

## Usage Patterns

### Pattern 1: See Everything (Default)

```bash
docker compose run --rm flightlog
```

Shows progress bars, algorithm logs, system logs, and errors.

### Pattern 2: Algorithm Logs Only

```bash
docker compose run --rm flightlog python /app/scripts/flightlog.py --filter-progress --host 0.0.0.0
```

Hides progress bars, shows only your strategy logs in bold.

### Pattern 3: Save Logs to File

```bash
docker compose run --rm flightlog python /app/scripts/flightlog.py --file /logs/backtest.log
```

---

## Log Levels

| Level | Color | When to Use |
|-------|-------|-------------|
| `DEBUG` | Cyan | Detailed debugging |
| `INFO` | Green | Normal operations |
| `WARNING` | Yellow | Unusual conditions |
| `ERROR` | Red | Failures |
| `CRITICAL` | Magenta | Critical issues |

---

## Best Practices

### DO

**Log Strategic Events**
```python
log_to_flightlog('Entered position in AAPL', level='INFO')
log_to_flightlog('Risk limit reached', level='WARNING')
```

**Include Context**
```python
portfolio_value = context.portfolio.portfolio_value
positions = len(context.portfolio.positions)
log_to_flightlog(
    f'Rebalance complete: {positions} positions, ${portfolio_value:,.0f}',
    level='INFO'
)
```

**Log Summary Metrics**
```python
def analyze(context, perf):
    total_return = perf['algorithm_period_return'].iloc[-1] * 100
    log_to_flightlog(f'Total Return: {total_return:+.2f}%', level='INFO')
```

### DON'T

**Don't Log Too Frequently**
```python
# Bad: Logs on every bar
def handle_data(context, data):
    log_to_flightlog('Processing bar...', level='INFO')  # Too noisy!
```

**Don't Duplicate Progress Information**
```python
# Bad: Progress logging already shows this
log_to_flightlog(f'Portfolio value: ${context.portfolio.portfolio_value}')
```

**Don't Log Sensitive Information**
```python
# Bad: Security risk
log_to_flightlog(f'API Key: {api_key}', level='INFO')
```

---

## Common Patterns

### Initialization Logging

```python
def initialize(context):
    log_to_flightlog('Initializing strategy...', level='INFO')
    
    context.lookback = 20
    context.top_n = 5
    
    log_to_flightlog(
        f'Config: lookback={context.lookback}d, top_n={context.top_n}',
        level='INFO'
    )
```

### Error Handling

```python
def safe_calculate_momentum(stock, data):
    try:
        prices = data.history(stock, 'price', 20, '1d')
        return (prices[-1] / prices[0] - 1) * 100
    except Exception as e:
        log_to_flightlog(
            f'Error calculating momentum for {stock.symbol}: {e}',
            level='WARNING'
        )
        return None
```

### Risk Monitoring

```python
def check_risk_limits(context, data):
    leverage = context.account.leverage
    
    if leverage > 1.5:
        log_to_flightlog(
            f'High leverage detected: {leverage:.2f}x',
            level='WARNING'
        )
```

---

## Performance Tips

### Update Intervals

```python
# Short backtests (< 1 year)
enable_progress_logging(update_interval=1)  # Daily

# Medium backtests (1-5 years)
enable_progress_logging(update_interval=5)  # Weekly

# Long backtests (5-10 years)
enable_progress_logging(update_interval=20)  # Monthly
```

### Batch Logging

```python
# Instead of logging each order
for stock in stocks:
    log_to_flightlog(f'Ordering {stock}')  # Too many logs

# Log summary
log_to_flightlog(f'Placed {len(orders)} orders', level='INFO')  # Better
```

---

## Troubleshooting

### No Logs Appearing

1. **Is FlightLog running?**
   ```bash
   docker ps | grep flightlog
   ```

2. **Is `enable_flightlog()` called BEFORE backtest?**
   ```python
   enable_flightlog(host='localhost', port=9020)
   results = run_algorithm(...)  # Must come after
   ```

3. **Test connection:**
   ```python
   import socket
   sock = socket.socket()
   result = sock.connect_ex(('localhost', 9020))
   print("Open" if result == 0 else "Closed")
   ```

### Connection Refused

1. Check FlightLog is running: `docker compose ps flightlog`
2. Verify hostname: Use `localhost` inside Docker
3. Check port: Default is 9020

### No Colors

Run FlightLog in foreground instead of using `docker logs`:
```bash
docker compose run --rm flightlog
```

### Port Already in Use

```bash
lsof -i :9020
kill -9 <PID>
```

---

## Complete Example

**Terminal 1: Start FlightLog**
```bash
docker compose run --rm flightlog
```

**Terminal 2: Run strategy**
```python
import logging
import pandas as pd
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
from zipline.utils.progress import enable_progress_logging

# Setup
logging.basicConfig(level=logging.INFO, force=True)
enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='BuyAndHold', update_interval=5)

def initialize(context):
    log_to_flightlog('Strategy starting!', level='INFO')
    context.stock = symbol('AAPL')
    context.bought = False

def handle_data(context, data):
    if not context.bought and data.can_trade(context.stock):
        order_target_percent(context.stock, 0.95)
        context.bought = True
        log_to_flightlog('Bought AAPL at 95%', level='INFO')

def analyze(context, perf):
    log_to_flightlog(
        f'Backtest complete! Return: {perf["returns"].sum()*100:.2f}%',
        level='INFO'
    )

results = run_algorithm(
    start=pd.Timestamp('2023-01-01', tz='UTC'),
    end=pd.Timestamp('2024-01-01', tz='UTC'),
    initialize=initialize,
    handle_data=handle_data,
    analyze=analyze,
    capital_base=10000,
    bundle='sharadar',
)
```

---

## Summary

**FlightLog provides:**
- Real-time log streaming to separate terminal
- Color-coded log levels
- Progress bars and algorithm logs
- Filtering options
- File output for archiving
- Easy integration (3 lines of code)

**Perfect for:**
- Development and testing
- Production monitoring
- Debugging strategies
- Keeping audit trails
