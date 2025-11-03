# FlightLog Usage Guide

FlightLog is a real-time log streaming system that displays backtest logs in a separate terminal window, similar to QuantRocket's flightlog.

## Quick Start

### Two Simple Steps

**Terminal 1 - Start FlightLog Server:**
```bash
# Inside the Docker container
python scripts/flightlog.py --host 0.0.0.0 --level INFO
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

**Terminal 2 - Run Your Backtest:**
```python
import logging
from zipline.utils.flightlog_client import enable_flightlog
from zipline import run_algorithm

# Setup logging level
logging.basicConfig(level=logging.INFO)

# Enable FlightLog
enable_flightlog(host='localhost', port=9020)

# Run your backtest - logs will stream to Terminal 1!
result = run_algorithm(...)
```

## From Jupyter Notebooks

In your notebook:

```python
import logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
from zipline.utils.progress import enable_progress_logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Connect to FlightLog
connected = enable_flightlog(host='localhost', port=9020)
if connected:
    print("✓ Connected to FlightLog!")
else:
    print("⚠ FlightLog not available - logs will show here instead")

# Enable progress logging too
enable_progress_logging(algo_name='My-Strategy', update_interval=5)

# Now run your backtest
result = run_algorithm(...)
```

## Test It Works

**Terminal 1:**
```bash
python scripts/flightlog.py --host 0.0.0.0 --level DEBUG
```

**Terminal 2:**
```bash
python notebooks/test_flightlog_backtest.py
```

You should see logs streaming in Terminal 1 with color-coded output!

## Output Format

FlightLog displays logs with QuantRocket-style formatting:

```
[Test-Strategy] Backtest initialized: 2020-01-01 to 2020-06-30 (126 trading days)
[Test-Strategy]              Date    Cumulative Returns    Sharpe Ratio    Max Drawdown    Cumulative PNL
[Test-Strategy]  ----------     0%      2020-01-02                    0%              0%                $0
[Test-Strategy]  ██--------    15%      2020-02-15                    3%            5.21              1%          $32,145
[Test-Strategy]  ██████████   100%      2020-06-30                   12%            1.89              8%         $120,450
```

## Custom Logging

Add custom log messages in your algorithm:

```python
from zipline.utils.flightlog_client import log_to_flightlog
from zipline.api import *

def initialize(context):
    log_to_flightlog("Strategy initialized!", level='INFO')
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
            f"High volume detected: {data.current(context.stock, 'volume'):,}",
            level='WARNING'
        )
```

## Log Levels

FlightLog supports standard Python logging levels:

- `DEBUG`: Detailed debugging information (cyan)
- `INFO`: General informational messages (green)
- `WARNING`: Warning messages (yellow)
- `ERROR`: Error messages (red)
- `CRITICAL`: Critical issues (magenta)

Start FlightLog with different levels:
```bash
# Show everything
python scripts/flightlog.py --level DEBUG

# Info and above (default)
python scripts/flightlog.py --level INFO

# Only warnings and errors
python scripts/flightlog.py --level WARNING
```

## Persistent Logs

FlightLog automatically saves all logs to `/logs/flightlog.log`:

```bash
# View from inside container
tail -f /logs/flightlog.log

# View from host machine
tail -f logs/flightlog.log

# Search logs
grep "ERROR" logs/flightlog.log
```

## Troubleshooting

### Connection Issues

**Test connection:**
```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 9020))
if result == 0:
    print("✓ FlightLog is reachable")
else:
    print("✗ Cannot connect to FlightLog")
sock.close()
```

### Messages Not Appearing

Make sure logging level is set:
```python
import logging
logging.basicConfig(level=logging.INFO)  # ← Important!

# Also set algorithm logger
algo_logger = logging.getLogger('algorithm')
algo_logger.setLevel(logging.INFO)
```

### Port Already in Use

If port 9020 is busy:
```bash
# Kill existing FlightLog
pkill -f flightlog.py

# Use different port
python scripts/flightlog.py --host 0.0.0.0 --port 9021
```

Then in your code:
```python
enable_flightlog(host='localhost', port=9021)
```

## Multiple Backtests

FlightLog can handle multiple backtests simultaneously!

**Terminal 1 - FlightLog:**
```bash
python scripts/flightlog.py --host 0.0.0.0 --level INFO
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

## Docker Compose Integration

If you want FlightLog to run automatically:

```bash
# From host machine - start all services including FlightLog
docker compose up -d

# View FlightLog output
docker compose logs -f flightlog

# Then connect from notebooks using host='flightlog'
enable_flightlog(host='flightlog', port=9020)
```

## Tips

1. **Always set logging level:** Add `logging.basicConfig(level=logging.INFO)` at the start of your scripts

2. **Use meaningful algo names:** Makes it easy to track multiple backtests
   ```python
   enable_progress_logging(algo_name='LS-Prod-v2')
   ```

3. **Check connection status:**
   ```python
   connected = enable_flightlog(host='localhost', port=9020)
   if not connected:
       print("Warning: FlightLog not available")
   ```

4. **Save important logs:** FlightLog automatically saves to `/logs/flightlog.log`

5. **Use different update intervals for different backtest lengths:**
   ```python
   # Short backtest (< 1 year)
   enable_progress_logging(update_interval=1)  # Daily

   # Long backtest (5+ years)
   enable_progress_logging(update_interval=20)  # Monthly
   ```

## Examples

Complete working examples:
- `notebooks/test_flightlog_backtest.py` - Simple buy-and-hold with FlightLog
- `notebooks/flightlog_quickstart.ipynb` - Jupyter notebook walkthrough
- `examples/flightlog_demo.py` - Full-featured demo

## See Also

- [Progress Logging Documentation](PROGRESS_LOGGING.md) - In-terminal progress bars
- [FlightLog Architecture](FLIGHTLOG.md) - Technical details and architecture
