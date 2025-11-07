# FlightLog Multi-Channel Guide

FlightLog now supports routing different types of logs to different terminal windows. This is useful for:
- Separating algorithm logs from progress updates
- Isolating errors in a dedicated window
- Monitoring different subsystems simultaneously
- Reducing clutter in log output

## Quick Start

### Single Channel (Original Behavior)

**Terminal 1:** Start FlightLog server
```bash
python scripts/flightlog.py
```

**Your Script:** Connect to FlightLog
```python
from zipline.utils.flightlog_client import enable_flightlog

enable_flightlog(host='localhost')
# All logs go to this one terminal
```

### Multi-Channel Setup

**Terminal 1:** Algorithm logs only
```bash
python scripts/flightlog.py --port 9020 --channel algorithm --logger-filter algorithm
```

**Terminal 2:** Progress logs only
```bash
python scripts/flightlog.py --port 9021 --channel progress --logger-filter zipline.progress
```

**Terminal 3:** Errors only
```bash
python scripts/flightlog.py --port 9022 --channel errors --level ERROR
```

**Your Script:** Route logs to different channels
```python
from zipline.utils.flightlog_client import enable_multi_channel_flightlog

enable_multi_channel_flightlog({
    'algorithm': {'port': 9020, 'logger': 'algorithm'},
    'progress': {'port': 9021, 'logger': 'zipline.progress'},
    'errors': {'port': 9022, 'level': logging.ERROR},
})
```

## Server Options

### Basic Options
- `--host HOST` - Host to bind to (default: 0.0.0.0)
- `--port PORT` - Port to listen on (default: 9020)
- `--level LEVEL` - Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--channel NAME` - Channel name for identification (default: default)
- `--file PATH` - Save logs to file (optional)
- `--no-color` - Disable colored output

### Filtering Options
- `--filter-progress` - Hide progress logs (shortcut for excluding zipline.progress)
- `--logger-filter LOGGER` - Only show logs from specific logger (e.g., "algorithm")
- `--exclude-logger LOGGER` - Exclude logs from specific logger (e.g., "zipline.finance")

### Examples

**Algorithm logs only with color:**
```bash
python scripts/flightlog.py --port 9020 --channel algorithm --logger-filter algorithm
```

**Progress logs without color (for piping):**
```bash
python scripts/flightlog.py --port 9021 --channel progress --logger-filter zipline.progress --no-color
```

**Errors only, saved to file:**
```bash
python scripts/flightlog.py --port 9022 --channel errors --level ERROR --file errors.log
```

**Everything except progress:**
```bash
python scripts/flightlog.py --port 9020 --channel main --exclude-logger zipline.progress
```

## Client Configuration

### Simple Configuration (Port Numbers Only)
```python
enable_multi_channel_flightlog({
    'main': 9020,
    'errors': 9021,
})
```

### Detailed Configuration
```python
import logging
from zipline.utils.flightlog_client import enable_multi_channel_flightlog

enable_multi_channel_flightlog({
    'algorithm': {
        'port': 9020,
        'logger': 'algorithm',  # Only algorithm logs
        'level': logging.INFO,
    },
    'progress': {
        'port': 9021,
        'logger': 'zipline.progress',  # Only progress logs
        'level': logging.INFO,
    },
    'errors': {
        'port': 9022,
        'level': logging.ERROR,  # Only errors, from any logger
    },
}, host='localhost')
```

### Using Presets
```python
from zipline.utils.flightlog_client import enable_multi_channel_flightlog, FLIGHTLOG_PRESETS

# Preset: Split algorithm and progress
enable_multi_channel_flightlog(FLIGHTLOG_PRESETS['split'])

# Preset: Three-way split
enable_multi_channel_flightlog(FLIGHTLOG_PRESETS['three_way'])

# Preset: Separate errors
enable_multi_channel_flightlog(FLIGHTLOG_PRESETS['errors_separate'])

# Preset: Debug by subsystem (4 channels)
enable_multi_channel_flightlog(FLIGHTLOG_PRESETS['debug'])
```

## Common Patterns

### Pattern 1: Clean Separation for Presentations/Demos

When presenting your backtest results, you might want clean algorithm logs without progress noise:

**Terminal 1 (present this):**
```bash
python scripts/flightlog.py --port 9020 --channel "Algorithm" --logger-filter algorithm
```

**Terminal 2 (hide this):**
```bash
python scripts/flightlog.py --port 9021 --channel "Background" --exclude-logger algorithm
```

**Script:**
```python
enable_multi_channel_flightlog({
    'clean': {'port': 9020, 'logger': 'algorithm'},
    'background': {'port': 9021},
})
```

### Pattern 2: Debugging Multiple Subsystems

When debugging, monitor different parts of Zipline in separate windows:

```bash
# Terminal 1: Data loading
python scripts/flightlog.py --port 9020 --channel "DATA" --logger-filter zipline.data

# Terminal 2: Trade execution
python scripts/flightlog.py --port 9021 --channel "FINANCE" --logger-filter zipline.finance

# Terminal 3: Your algorithm
python scripts/flightlog.py --port 9022 --channel "ALGORITHM" --logger-filter algorithm

# Terminal 4: Everything else
python scripts/flightlog.py --port 9023 --channel "OTHER"
```

**Script:**
```python
enable_multi_channel_flightlog({
    'data': {'port': 9020, 'logger': 'zipline.data', 'level': logging.DEBUG},
    'finance': {'port': 9021, 'logger': 'zipline.finance', 'level': logging.DEBUG},
    'algorithm': {'port': 9022, 'logger': 'algorithm', 'level': logging.DEBUG},
    'other': {'port': 9023, 'level': logging.DEBUG},
})
```

### Pattern 3: Error Monitoring

Keep an error-only window open to quickly spot issues:

**Terminal 1 (errors):**
```bash
python scripts/flightlog.py --port 9022 --channel "🚨 ERRORS" --level ERROR --file errors.log
```

This terminal will normally be quiet, but flash red when errors occur, and save them to a file.

### Pattern 4: stdout vs stderr Separation

Route different streams to different terminals:

```python
import sys
import logging

# Terminal 1: Regular logs via FlightLog
enable_multi_channel_flightlog({'logs': 9020})

# Terminal 2: Redirect stdout (print statements)
# Use `tee` or pipe to another terminal
# In bash: your_script.py > >(some_command)

# Terminal 3: Redirect stderr
# In bash: your_script.py 2> >(some_command)
```

## Checking Connection Status

```python
from zipline.utils.flightlog_client import get_flightlog_status

status = get_flightlog_status()
print(f"Connected: {status['connected']}")
print(f"Active channels: {status['handlers']}")

for channel in status['channels']:
    print(f"  {channel['logger']}: {channel['host']}:{channel['port']} ({channel['level']})")
```

## Jupyter Notebook Usage

In Jupyter notebooks, you typically want to see algorithm logs inline and send progress to a FlightLog window:

```python
# Cell 1: Setup
import logging
from zipline.utils.flightlog_client import enable_multi_channel_flightlog

# Start FlightLog server in separate terminal:
# python scripts/flightlog.py --port 9021 --channel progress --logger-filter zipline.progress

# Send only progress logs to FlightLog, keep algorithm logs in notebook
enable_multi_channel_flightlog({
    'progress': {'port': 9021, 'logger': 'zipline.progress'},
}, host='localhost')

# Cell 2: Run backtest
# Your algorithm logs will appear in the notebook
# Progress logs will appear in the FlightLog terminal
```

## Docker Compose Setup

For Docker users, set up multiple FlightLog services:

```yaml
services:
  flightlog-algorithm:
    image: zipline
    command: python scripts/flightlog.py --port 9020 --channel algorithm --logger-filter algorithm
    ports:
      - "9020:9020"

  flightlog-progress:
    image: zipline
    command: python scripts/flightlog.py --port 9021 --channel progress --logger-filter zipline.progress
    ports:
      - "9021:9021"

  flightlog-errors:
    image: zipline
    command: python scripts/flightlog.py --port 9022 --channel errors --level ERROR
    ports:
      - "9022:9022"
```

Then in your script:
```python
enable_multi_channel_flightlog({
    'algorithm': {'port': 9020, 'logger': 'algorithm'},
    'progress': {'port': 9021, 'logger': 'zipline.progress'},
    'errors': {'port': 9022, 'level': logging.ERROR},
}, host='localhost')  # or Docker service names if within same network
```

## Tips and Tricks

1. **Use meaningful channel names** - They appear in the server banner and help identify windows
2. **Color coding** - Each log level has a different color (ERROR=red, WARNING=yellow, INFO=green)
3. **File logging** - Add `--file` to any server to keep a permanent record
4. **No progress** - Use `--filter-progress` as a quick way to hide progress logs
5. **Terminal multiplexers** - Use tmux or screen to manage multiple FlightLog windows
6. **Window titles** - Set your terminal window title to match the channel name for easy identification

## Troubleshooting

**"Channel not reachable" warning:**
- Make sure the FlightLog server is running on the specified port
- Check firewall settings if using Docker or remote hosts
- Try `localhost` instead of `127.0.0.1` or vice versa

**Logs appearing in wrong window:**
- Check logger hierarchy (e.g., 'algorithm' is different from 'algorithm.strategy')
- Remember that child loggers propagate to parent loggers
- Use `--exclude-logger` on the receiving servers to prevent duplicates

**Too much/too little logging:**
- Adjust `--level` on the server side
- Adjust `level` parameter in client configuration
- Use logger filters to be more specific

**Performance concerns:**
- FlightLog uses non-blocking sockets, minimal overhead
- Each channel is independent, no cross-talk
- Close unused handlers with `disable_flightlog()` when done
