# Zipline Scripts and Utilities

This directory contains standalone scripts and utilities for Zipline.

## FlightLog - Real-time Log Viewer

FlightLog is a real-time log viewer for Zipline backtests and data operations, similar to QuantRocket's flightlog.

### Quick Start

```bash
# Start FlightLog server
python scripts/flightlog.py

# In your backtest or script
from zipline.utils.flightlog_client import enable_flightlog
enable_flightlog(host='localhost')
```

### Features

- **Real-time log streaming** - See backtest logs as they happen in a separate terminal
- **Color-coded output** - Different colors for DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Multi-channel support** - Route different log types to different terminal windows
- **File logging** - Save logs to file while viewing in real-time
- **Filtering** - Show only specific loggers, exclude noisy loggers, or filter by level
- **Docker support** - Works seamlessly with Docker containers

### Documentation

See [FLIGHTLOG_MULTI_CHANNEL_GUIDE.md](FLIGHTLOG_MULTI_CHANNEL_GUIDE.md) for comprehensive documentation including:
- Multi-channel setup
- Common usage patterns
- Data ingestion monitoring
- Jupyter notebook integration
- Docker Compose configuration
- Troubleshooting guide

### Command-Line Options

```bash
python scripts/flightlog.py --help

Options:
  --host HOST              Host to bind to (default: 0.0.0.0)
  --port PORT              Port to listen on (default: 9020)
  --level LEVEL            Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  --channel NAME           Channel name for identification
  --file PATH              Save logs to file
  --no-color               Disable colored output
  --filter-progress        Hide progress logs
  --logger-filter LOGGER   Only show logs from specific logger
  --exclude-logger LOGGER  Exclude logs from specific logger
```

### Example: Multi-Channel Backtest Monitoring

**Terminal 1: Algorithm logs**
```bash
python scripts/flightlog.py --port 9020 --channel algorithm --logger-filter algorithm
```

**Terminal 2: Progress updates**
```bash
python scripts/flightlog.py --port 9021 --channel progress --logger-filter zipline.progress
```

**Terminal 3: Errors only**
```bash
python scripts/flightlog.py --port 9022 --channel errors --level ERROR --file logs/errors.log
```

**Your backtest script:**
```python
from zipline.utils.flightlog_client import enable_multi_channel_flightlog

enable_multi_channel_flightlog({
    'algorithm': {'port': 9020, 'logger': 'algorithm'},
    'progress': {'port': 9021, 'logger': 'zipline.progress'},
    'errors': {'port': 9022, 'level': logging.ERROR},
}, host='localhost')

# Run your backtest - logs will be routed to appropriate windows
```

## Data Ingestion Logging

Zipline now supports comprehensive logging for data bundle ingestion operations.

### CLI Usage

The `zipline ingest` command now supports the `--log-file` option:

```bash
# Save ingest logs to a file
zipline ingest -b sharadar --log-file logs/ingest.log

# Use timestamped log files
zipline ingest -b sharadar --log-file logs/ingest_$(date +%Y%m%d_%H%M%S).log

# Daily log files (appends if run multiple times same day)
zipline ingest -b sharadar --log-file logs/ingest_$(date +%Y%m%d).log
```

### What Gets Logged

- Bundle name and start time
- Data fetching progress
- Database operations (creating tables, writing data)
- Warnings and errors
- Completion status and timing
- Whether bundle was skipped (already up-to-date)

### Example Log Output

```
[2025-01-07T10:23:45-0500-INFO][root]
 Logging ingest to file: logs/ingest.log
[2025-01-07T10:23:45-0500-INFO][root]
 Starting ingest for bundle: sharadar
[2025-01-07T10:23:46-0500-INFO][zipline.data.bundles.core]
 Ingesting sharadar
[2025-01-07T10:24:15-0500-INFO][zipline.data.bundles.sharadar_bundle]
 Fetching price data from Sharadar...
[2025-01-07T10:26:42-0500-INFO][root]
 Bundle sharadar ingestion completed successfully
```

### Combining with FlightLog

Monitor ingestion in real-time while also saving to a log file:

**Terminal 1: FlightLog for real-time monitoring**
```bash
python scripts/flightlog.py --port 9030 --channel "INGEST" --file logs/ingest_history.log
```

**Terminal 2: Run ingest with both FlightLog and file logging**
```python
from zipline.utils.flightlog_client import enable_flightlog

# Connect to FlightLog
enable_flightlog(host='localhost', port=9030)

# Run ingest (can also use CLI with --log-file)
from zipline.data.bundles import ingest
ingest('sharadar', show_progress=True)
```

Or from command line:
```bash
# Terminal 2: Connect to FlightLog first, then run CLI ingest
python -c "
from zipline.utils.flightlog_client import enable_flightlog
enable_flightlog(host='localhost', port=9030)
import os
os.system('zipline ingest -b sharadar --log-file logs/ingest.log')
"
```

This gives you:
- **Real-time monitoring** in FlightLog terminal
- **Permanent record** in the log file
- **Console output** for immediate feedback

### Automated Ingestion with Logging

For scheduled/automated ingestion (cron jobs, etc.):

**Script: `scripts/ingest_daily.sh`**
```bash
#!/bin/bash
# Daily ingestion with logging

LOG_DIR="$HOME/.zipline/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/ingest_$(date +%Y%m%d).log"

echo "=== Starting daily ingest: $(date) ===" >> "$LOG_FILE"

zipline ingest -b sharadar --log-file "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "=== Ingest completed successfully: $(date) ===" >> "$LOG_FILE"
else
    echo "=== Ingest FAILED: $(date) ===" >> "$LOG_FILE"
    # Send alert email/notification here
fi
```

**Crontab entry:**
```bash
# Run daily at 7 AM
0 7 * * * /path/to/scripts/ingest_daily.sh
```

### Log Management

**View today's logs:**
```bash
tail -f ~/.zipline/logs/ingest_$(date +%Y%m%d).log
```

**Check for errors:**
```bash
grep -i error ~/.zipline/logs/ingest_*.log
```

**Rotate old logs:**
```bash
# Keep only last 30 days
find ~/.zipline/logs -name "ingest_*.log" -mtime +30 -delete
```

## Multi-Channel Ingest Monitoring

For detailed monitoring, route different ingest subsystems to different FlightLog windows:

**Terminal 1: Data fetching logs**
```bash
python scripts/flightlog.py --port 9030 --channel "FETCH" \
    --logger-filter zipline.data --level DEBUG
```

**Terminal 2: Database writes**
```bash
python scripts/flightlog.py --port 9031 --channel "WRITE" \
    --logger-filter zipline.data.bundles --level INFO
```

**Terminal 3: Errors**
```bash
python scripts/flightlog.py --port 9032 --channel "ERRORS" \
    --level ERROR --file logs/ingest_errors.log
```

**Ingest script:**
```python
from zipline.utils.flightlog_client import enable_multi_channel_flightlog
import logging

enable_multi_channel_flightlog({
    'fetch': {'port': 9030, 'logger': 'zipline.data', 'level': logging.DEBUG},
    'write': {'port': 9031, 'logger': 'zipline.data.bundles', 'level': logging.INFO},
    'errors': {'port': 9032, 'level': logging.ERROR},
}, host='localhost')

from zipline.data.bundles import ingest
ingest('sharadar', show_progress=True)
```

## Benefits

### For Development
- **Debugging** - See logs in real-time without cluttering your main terminal
- **Multi-tasking** - Keep one eye on logs while working in another window
- **Error isolation** - Dedicated error window catches problems immediately

### For Production
- **Monitoring** - Watch automated backtests or data updates
- **Audit trail** - Complete log history for compliance and troubleshooting
- **Performance tracking** - Compare timing across runs
- **Alerting** - Integrate with monitoring systems via log files

## See Also

- [FLIGHTLOG_MULTI_CHANNEL_GUIDE.md](FLIGHTLOG_MULTI_CHANNEL_GUIDE.md) - Comprehensive FlightLog documentation
- Main Zipline documentation: https://zipline.ml4trading.io
