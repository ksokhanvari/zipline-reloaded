# FlightLog Usage Guide

FlightLog is a real-time log streaming system that displays backtest logs in a separate terminal window, similar to QuantRocket's flightlog.

**Features:**
- 🎨 **Color-coded log levels** (green INFO, yellow WARNING, red ERROR)
- **Bold algorithm logs** for your strategy messages
- **Real-time streaming** via TCP socket (port 9020)
- **Progress bar support** with portfolio metrics
- **Filtering options** to show only what you need (--filter-progress)
- **No duplicate messages** even when running multiple backtests

---

## Quick Start

### Step 1: Start FlightLog Server

**Inside Docker container (recommended):**

```bash
# Start in foreground (shows colors)
docker compose run --rm flightlog

# Or in background
docker compose up -d flightlog
```

**Or with custom options:**

```bash
# Filter out progress bars (show only algorithm logs)
docker compose run --rm flightlog python /app/scripts/flightlog.py --filter-progress --host 0.0.0.0

# Debug level with file output
docker compose run --rm flightlog python /app/scripts/flightlog.py --level DEBUG --file /logs/backtest.log
```

You'll see:
```
======================================================================
FlightLog Server - Real-time Zipline Backtest Logging
======================================================================
Listening on: 0.0.0.0:9020
Log level: INFO

Waiting for backtest connections...
Press Ctrl+C to stop
======================================================================
```

### Step 2: Enable FlightLog in Your Strategy

**In your Jupyter notebook or Python script:**

```python
import logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
from zipline.utils.progress import enable_progress_logging

# Setup logging (force=True prevents duplicates on re-runs)
logging.basicConfig(level=logging.INFO, force=True)

# Enable FlightLog (connects to port 9020)
enable_flightlog(host='localhost', port=9020)

# Enable progress logging (optional)
enable_progress_logging(algo_name='My-Strategy', update_interval=5)

# Now run your backtest - logs appear in FlightLog terminal!
result = run_algorithm(...)
```

### Step 3: Run Your Backtest

All logs from your strategy will now appear in the FlightLog terminal in real-time!

---

## Viewing FlightLog Output

### ✅ Recommended: Run in Foreground

**Best way to see colors and real-time updates:**

```bash
docker compose run --rm flightlog
```

**Why:** You see ANSI colors, real-time updates, and can easily stop with Ctrl+C.

### ❌ Not Recommended: `docker logs`

**If you use:**
```bash
docker logs -f zipline-flightlog
```

**You'll see:** Plain text without colors (Docker strips ANSI codes).

**Alternative:** Attach to running container:
```bash
docker attach zipline-flightlog
```
Press Ctrl+P, Ctrl+Q to detach without stopping.

---

## Command-Line Options

```bash
python scripts/flightlog.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | `0.0.0.0` | Host to bind to |
| `--port PORT` | `9020` | Port to listen on |
| `--level LEVEL` | `INFO` | Min log level (DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL) |
| `--file FILE` | None | Save logs to file (optional) |
| `--no-color` | False | Disable color output |
| `--filter-progress` | False | **NEW!** Hide progress logs (show only algorithm logs) |

### Examples:

```bash
# Default: All logs with colors
python scripts/flightlog.py

# Algorithm logs only (hide progress bars) 🆕
python scripts/flightlog.py --filter-progress

# Debug level, save to file
python scripts/flightlog.py --level DEBUG --file backtest.log

# No colors (for log parsing)
python scripts/flightlog.py --no-color

# Custom port
python scripts/flightlog.py --port 9021
```

---

## Usage Patterns

### Pattern 1: See Everything (Default)

**Start FlightLog:**
```bash
docker compose run --rm flightlog
```

**What you'll see:**
- ✅ Progress bars with metrics (zipline.progress)
- ✅ Algorithm logs (your strategy)
- ✅ System logs (zipline.finance.metrics)
- ✅ All errors and warnings

**Use when:** You want complete visibility into your backtest.

**Example output:**
```
2025-11-04 12:30:45 INFO zipline.progress: [My-Strategy] ██--------  20%  2023-03-15  12%  2.5  -5%  $112K
2025-11-04 12:30:45 INFO algorithm: Strategy initializing...
2025-11-04 12:30:45 INFO algorithm: Rebalanced: 3 positions, Portfolio: $112,000
2025-11-04 12:30:46 INFO zipline.progress: [My-Strategy] ████------  40%  2023-06-30  18%  2.8  -8%  $118K
```

---

### Pattern 2: Algorithm Logs Only 🆕

**Start FlightLog with filtering:**
```bash
docker compose run --rm flightlog python /app/scripts/flightlog.py --filter-progress --host 0.0.0.0
```

**What you'll see:**
- ❌ Progress bars (hidden)
- ✅ Algorithm logs (your strategy) in **bold**
- ✅ System logs
- ✅ Errors and warnings

**Use when:** You only care about your strategy's logic and want cleaner output.

**Example output:**
```
2025-11-04 12:30:45 INFO algorithm: Strategy initializing...
2025-11-04 12:30:45 INFO algorithm: Initialized with 3 stocks
2025-11-04 12:30:46 INFO algorithm: Rebalanced: 3 positions, Portfolio: $100,661
2025-11-04 12:30:46 INFO algorithm: Rebalanced: 3 positions, Portfolio: $105,331
2025-11-04 12:31:15 INFO algorithm: Backtest complete: +59.77% return
```

---

### Pattern 3: Debug Mode

**Start FlightLog with debug level:**
```bash
docker compose run --rm flightlog python /app/scripts/flightlog.py --level DEBUG
```

**What you'll see:**
- ✅ Everything from default mode
- ✅ DEBUG level messages
- ✅ Detailed execution traces

**Use when:** Troubleshooting issues or developing new strategies.

---

### Pattern 4: Save Logs to File

**Start FlightLog with file output:**
```bash
docker compose run --rm flightlog python /app/scripts/flightlog.py --file /logs/backtest.log
```

**What you'll see:**
- ✅ Real-time output in terminal
- ✅ Logs also saved to file
- ✅ File persists after backtest completes

**Use when:** You need to review logs later or share results.

---

## From Jupyter Notebooks

In your notebook:

```python
import logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
from zipline.utils.progress import enable_progress_logging

# Setup logging (safe to run multiple times)
logging.basicConfig(level=logging.INFO, force=True)

# Connect to FlightLog
connected = enable_flightlog(host='localhost', port=9020)
if connected:
    print("✓ Connected to FlightLog!")
else:
    print("⚠ FlightLog not available - logs will show here instead")

# Enable progress logging
enable_progress_logging(algo_name='My-Strategy', update_interval=5)

# Now run your backtest
result = run_algorithm(...)

# Safe to re-run this cell multiple times - no duplicate messages!
```

---

## Custom Logging

Add custom log messages in your algorithm:

```python
from zipline.utils.flightlog_client import log_to_flightlog
from zipline.api import *

def initialize(context):
    log_to_flightlog("🚀 Strategy initialized!", level='INFO')
    context.stock = symbol('AAPL')

def handle_data(context, data):
    # Log when opening position
    if not context.portfolio.positions:
        order_target_percent(context.stock, 0.95)
        log_to_flightlog(
            f"Opened position in {context.stock.symbol}",
            level='INFO'
        )

    # Warning for high volatility
    if data.current(context.stock, 'volume') > 1000000:
        log_to_flightlog(
            f"⚠️ High volume detected: {data.current(context.stock, 'volume'):,}",
            level='WARNING'
        )
```

---

## Log Levels

FlightLog supports standard Python logging levels with colors:

| Level | Color | When to Use |
|-------|-------|-------------|
| `DEBUG` | Cyan | Detailed debugging information |
| `INFO` | **Green** | General informational messages |
| `WARNING` | **Yellow** | Warning messages |
| `ERROR` | **Red** | Error messages |
| `CRITICAL` | **Magenta** | Critical issues |

**Algorithm logs are shown in bold** to make them stand out!

---

## Troubleshooting

### Problem: "No logs appearing in FlightLog"

**Checklist:**

1. **Is FlightLog server running?**
   ```bash
   docker ps | grep flightlog
   # Should show a running container
   ```

2. **Is `enable_flightlog()` called BEFORE running backtest?**
   ```python
   # ✅ Correct order
   enable_flightlog(host='localhost', port=9020)
   results = run_algorithm(...)

   # ❌ Wrong order
   results = run_algorithm(...)
   enable_flightlog(host='localhost', port=9020)  # Too late!
   ```

3. **Check connection:**
   ```python
   from zipline.utils.flightlog_client import get_flightlog_status
   status = get_flightlog_status()
   print(status)  # Should show connected=True
   ```

4. **Verify port is correct:**
   - FlightLog server default: `9020`
   - Your code must match: `enable_flightlog(port=9020)`

### Problem: "Progress logs not appearing, only algorithm logs"

**Cause:** You're running an older version of the code.

**Fix:**
```bash
cd /home/user/zipline-reloaded
git pull origin claude/continue-work-011CUkRmXtm24f1Qc3mHbrGJ
```

**Why this happened:** Earlier versions had `propagate=False` on the progress logger which prevented logs from reaching FlightLog. The latest version explicitly copies SocketHandlers to the progress logger so FlightLog receives them.

### Problem: "Duplicate log messages (each message appears 2-3 times)"

**Cause:** Running multiple backtests without restarting Python kernel (already fixed!).

**Fix:** Just pull latest code:
```bash
git pull origin claude/continue-work-011CUkRmXtm24f1Qc3mHbrGJ
```

The code now:
- ✅ Prevents duplicate handlers
- ✅ Checks for existing SocketHandlers before adding
- ✅ Clears old progress logger handlers on each run
- ✅ Uses `logging.basicConfig(force=True)` in examples

### Problem: "ModuleNotFoundError: No module named 'zipline.data._adjustments'"

**Cause:** Docker image is out of date or corrupted.

**Fix:** Rebuild the Docker image:

```bash
cd /path/to/zipline-reloaded

# Pull latest code
git pull origin claude/continue-work-011CUkRmXtm24f1Qc3mHbrGJ

# Rebuild Docker image (this compiles Cython extensions inside container)
docker compose build zipline-jupyter

# Restart containers
docker compose up -d
```

**Note:** After any `git pull` that includes code changes, you should rebuild the Docker image to ensure all Cython extensions are properly compiled.

### Problem: "Connection refused"

**Possible causes:**

1. **FlightLog not running:**
   ```bash
   docker compose run --rm flightlog
   ```

2. **Wrong host in Docker:**
   ```python
   # ✅ Jupyter container to same container:
   enable_flightlog(host='localhost', port=9020)

   # ✅ Jupyter to separate FlightLog container:
   enable_flightlog(host='flightlog', port=9020)
   ```

3. **Firewall blocking port 9020:**
   Check your firewall settings.

### Problem: "No colors in output"

**Cause:** Viewing with `docker logs` which strips ANSI codes.

**Fix:** Run FlightLog in foreground:
```bash
docker compose run --rm flightlog
```

Or attach to running container:
```bash
docker attach zipline-flightlog
```

### Problem: "Port already in use"

```bash
# Find what's using port 9020
lsof -i :9020

# Kill it if needed
kill -9 <PID>

# Or use a different port
docker compose run --rm flightlog python /app/scripts/flightlog.py --port 9021
```

Then in your code:
```python
enable_flightlog(host='localhost', port=9021)
```

---

## Multiple Backtests

FlightLog can handle multiple backtests simultaneously!

**Terminal 1 - FlightLog:**
```bash
docker compose run --rm flightlog
```

**Terminal 2 - Strategy A:**
```python
enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='Strategy-A')
# ... run backtest
```

**Terminal 3 - Strategy B:**
```python
enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='Strategy-B')
# ... run backtest
```

Both strategies will stream to Terminal 1, differentiated by their algo names!

---

## Docker Compose Integration

### Background Mode (Auto-start)

**In `docker-compose.yml` (already configured):**

```yaml
services:
  flightlog:
    command: python /app/scripts/flightlog.py --host 0.0.0.0 --file /logs/flightlog.log
    ports:
      - "9020:9020"
```

**Start all services:**
```bash
docker compose up -d
```

**View FlightLog output:**
```bash
# View logs (no colors)
docker compose logs -f flightlog

# Or attach (with colors)
docker attach zipline-flightlog
```

**Connect from notebooks:**
```python
# When FlightLog is a separate container
enable_flightlog(host='flightlog', port=9020)

# When FlightLog is in same container
enable_flightlog(host='localhost', port=9020)
```

---

## Best Practices

### ✅ DO:

1. **Call `enable_flightlog()` once at the start**
   ```python
   logging.basicConfig(level=logging.INFO, force=True)
   enable_flightlog(host='localhost', port=9020)
   ```

2. **Use `force=True` with basicConfig**
   - Prevents duplicate handlers on re-runs
   ```python
   logging.basicConfig(level=logging.INFO, force=True)
   ```

3. **Run FlightLog in foreground for development**
   ```bash
   docker compose run --rm flightlog
   ```

4. **Use `--filter-progress` for production logs**
   - Focus on strategy logic only
   ```bash
   python scripts/flightlog.py --filter-progress
   ```

5. **Use meaningful algorithm names**
   ```python
   enable_progress_logging(algo_name='MomentumProd-v2.1')
   ```

6. **Save important runs to file**
   ```bash
   python scripts/flightlog.py --file production_run.log
   ```

### ❌ DON'T:

1. **Call `enable_flightlog()` multiple times in same session**
   - Already handled in latest code, but best to avoid

2. **Use `docker logs` to view FlightLog**
   - You won't see colors

3. **Forget to start FlightLog server first**
   - Your code will still run, logs just won't appear remotely

4. **Mix log levels carelessly**
   - Use INFO for normal events, WARNING for concerns, ERROR for problems

5. **Log too frequently**
   - Use appropriate `update_interval` for progress logging
   - Don't log on every bar unless debugging

---

## Advanced Usage

### Conditional Logging

```python
def rebalance(context, data):
    # Only log when something important happens
    if context.portfolio.positions_value < context.target_value * 0.9:
        log_to_flightlog(
            f'⚠️  Portfolio underweight: {context.portfolio.positions_value:.2f}',
            level='WARNING'
        )

    # Log significant trades
    if abs(context.target_positions - context.current_positions) > 5:
        log_to_flightlog(
            f'Major rebalance: {context.target_positions - context.current_positions} positions',
            level='INFO'
        )
```

### Multiple FlightLog Instances

You can run FlightLog on different ports for different strategies:

```bash
# Terminal 1: Strategy A logs
python scripts/flightlog.py --port 9020

# Terminal 2: Strategy B logs
python scripts/flightlog.py --port 9021
```

```python
# Strategy A
enable_flightlog(host='localhost', port=9020)

# Strategy B
enable_flightlog(host='localhost', port=9021)
```

### Log with Emoji for Visibility

```python
log_to_flightlog('🚀 Backtest starting', level='INFO')
log_to_flightlog('📈 Portfolio up 10%', level='INFO')
log_to_flightlog('⚠️  High volatility detected', level='WARNING')
log_to_flightlog('❌ Order rejected', level='ERROR')
log_to_flightlog('✅ Backtest complete', level='INFO')
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

# Setup (do once)
logging.basicConfig(level=logging.INFO, force=True)
enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='BuyAndHold', update_interval=5)

def initialize(context):
    log_to_flightlog('🚀 Strategy starting!', level='INFO')
    context.stock = symbol('AAPL')
    context.bought = False

def handle_data(context, data):
    if not context.bought and data.can_trade(context.stock):
        order_target_percent(context.stock, 0.95)
        context.bought = True
        log_to_flightlog('📈 Bought AAPL at 95%', level='INFO')

def analyze(context, perf):
    log_to_flightlog(
        f'✅ Backtest complete! Return: {perf["returns"].sum()*100:.2f}%',
        level='INFO'
    )

# Run
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

**FlightLog Terminal Output:**
```
2025-11-04 12:30:45 INFO zipline.progress: [BuyAndHold] Backtest initialized: 2023-01-01 to 2024-01-01 (252 trading days)
2025-11-04 12:30:45 INFO algorithm: 🚀 Strategy starting!
2025-11-04 12:30:45 INFO algorithm: 📈 Bought AAPL at 95%
2025-11-04 12:30:46 INFO zipline.progress: [BuyAndHold] ██--------  20%  2023-03-15  12%  2.5  -5%  $112K
2025-11-04 12:30:47 INFO zipline.progress: [BuyAndHold] ████------  40%  2023-06-30  18%  2.8  -8%  $118K
...
2025-11-04 12:30:50 INFO algorithm: ✅ Backtest complete! Return: 45.23%
```

---

## Tips & Tricks

1. **Check connection status before running:**
   ```python
   connected = enable_flightlog(host='localhost', port=9020)
   if not connected:
       print("⚠️ FlightLog not available - continuing without it")
   ```

2. **Use different update intervals for different backtest lengths:**
   ```python
   # Short backtest (< 1 year)
   enable_progress_logging(update_interval=1)  # Daily

   # Medium backtest (1-3 years)
   enable_progress_logging(update_interval=5)  # Weekly

   # Long backtest (5+ years)
   enable_progress_logging(update_interval=20)  # Monthly
   ```

3. **Filter progress in production, show all in development:**
   ```bash
   # Development
   python scripts/flightlog.py

   # Production
   python scripts/flightlog.py --filter-progress --file production.log
   ```

4. **Restart FlightLog without restarting backtests:**
   - Backtests will buffer logs and send when FlightLog reconnects

5. **Save different log levels to different files:**
   ```bash
   # Terminal 1: All logs
   python scripts/flightlog.py --level INFO --file all.log

   # Terminal 2: Errors only
   python scripts/flightlog.py --level ERROR --file errors.log --port 9021
   ```

---

## Examples

Complete working examples:
- `examples/simple_flightlog_demo.py` - Minimal example (50 lines)
- `examples/momentum_strategy_with_flightlog.py` - Production example (286 lines)
- `notebooks/01_quickstart_sharadar_flightlog.ipynb` - Interactive tutorial
- `notebooks/flightlog_elegant_example.ipynb` - Advanced patterns

---

## See Also

- [FlightLog Best Practices](FLIGHTLOG_BEST_PRACTICES.md) - DO/DON'T patterns
- [Progress Logging Documentation](PROGRESS_LOGGING.md) - In-terminal progress bars
- [FlightLog Architecture](FLIGHTLOG.md) - Technical details
- [Getting Started Guide](../GETTING_STARTED.md) - Quick start for new users

---

## Summary

**FlightLog provides:**
- ✅ Real-time log streaming to separate terminal
- ✅ Color-coded log levels (green/yellow/red)
- ✅ Progress bars and algorithm logs
- ✅ Filtering options (--filter-progress)
- ✅ File output for archiving
- ✅ No duplicate messages (fixed in latest version)
- ✅ Easy integration (3 lines of code)
- ✅ Bold algorithm logs for easy identification

**Perfect for:**
- 🔬 Development and testing
- 📊 Production monitoring
- 🐛 Debugging strategies
- 📝 Keeping audit trails
- 👥 Running multiple strategies simultaneously

Start using FlightLog today for better backtest visibility!
