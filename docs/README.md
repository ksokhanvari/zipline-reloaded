# Zipline-Reloaded Documentation

Complete guides for using Zipline-Reloaded with Sharadar data, FlightLog monitoring, and backtesting.

---

## Quick Navigation

### Getting Started

**New to Zipline?** Start here:

1. **[QUICK_START_DATA.md](./QUICK_START_DATA.md)** - Get data set up in 5 minutes
2. **[FLIGHTLOG.md](./FLIGHTLOG.md)** - Real-time backtest monitoring
3. **[DEVELOPMENT.md](./DEVELOPMENT.md)** - Development setup and workflow

### Data Management

**Managing Sharadar market data:**

- **[DATA_MANAGEMENT.md](./DATA_MANAGEMENT.md)** - Complete guide to bundles, updates, and cleanup
- **[SHARADAR_FUNDAMENTALS_GUIDE.md](./SHARADAR_FUNDAMENTALS_GUIDE.md)** - Using fundamentals in pipelines
- **[SYMBOL_MAPPING.md](./SYMBOL_MAPPING.md)** - Handling symbol changes (FB→META)

### Multi-Source Data

**Combining multiple data sources:**

- **[MULTI_SOURCE_DATA.md](./MULTI_SOURCE_DATA.md)** - Architecture for combining Sharadar + custom data
- **[MULTI_SOURCE_QUICKREF.md](./MULTI_SOURCE_QUICKREF.md)** - Quick reference for multi-source pipelines

### Monitoring & Development

**Tools and development:**

- **[FLIGHTLOG.md](./FLIGHTLOG.md)** - Real-time log streaming and progress monitoring
- **[DEVELOPMENT.md](./DEVELOPMENT.md)** - Docker build optimization and development workflow

---

## Documentation Summary

| Guide | Purpose | Time |
|-------|---------|------|
| [QUICK_START_DATA.md](./QUICK_START_DATA.md) | Get started fast | 5 min |
| [DATA_MANAGEMENT.md](./DATA_MANAGEMENT.md) | Bundle management, updates, cleanup | 15 min |
| [SHARADAR_FUNDAMENTALS_GUIDE.md](./SHARADAR_FUNDAMENTALS_GUIDE.md) | Fundamentals in pipelines | 20 min |
| [SYMBOL_MAPPING.md](./SYMBOL_MAPPING.md) | Handle symbol changes | 10 min |
| [MULTI_SOURCE_DATA.md](./MULTI_SOURCE_DATA.md) | Combine multiple data sources | 15 min |
| [FLIGHTLOG.md](./FLIGHTLOG.md) | Real-time monitoring | 10 min |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | Development workflow | 10 min |

---

## Common Tasks

### Daily Workflow

**After market close each trading day:**

```bash
# Update data (incremental, ~30 seconds)
docker compose exec zipline-jupyter zipline ingest -b sharadar

# Run backtest with monitoring
# Start FlightLog
docker compose run --rm flightlog

# In Jupyter notebook:
from zipline.utils.flightlog_client import enable_flightlog
from zipline.utils.progress import enable_progress_logging

enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='MyStrategy')

results = run_algorithm(...)
```

### First-Time Setup

```bash
# 1. Configure API key
echo 'NASDAQ_DATA_LINK_API_KEY=your-key' > .env

# 2. Build and start containers
docker compose build
docker compose up -d

# 3. Download all historical data (10-20 minutes)
docker compose exec zipline-jupyter zipline ingest -b sharadar

# 4. Verify data
docker compose exec zipline-jupyter zipline bundles

# 5. Test with FlightLog
docker compose run --rm flightlog
```

### Automated Daily Updates

```bash
# Add to crontab (runs Mon-Fri at 6 PM)
crontab -e

# Add this line:
0 18 * * 1-5 cd /path/to/zipline-reloaded && \
    docker compose exec -T zipline-jupyter zipline ingest -b sharadar >> \
    /path/to/logs/data_update.log 2>&1
```

---

## File Structure

```
docs/
├── README.md                      # This file - navigation guide
├── QUICK_START_DATA.md           # 5-minute quick start
├── DATA_MANAGEMENT.md            # Bundle management, updates, cleanup
├── SHARADAR_FUNDAMENTALS_GUIDE.md # Fundamentals in pipelines
├── SYMBOL_MAPPING.md             # Handle FB→META symbol changes
├── MULTI_SOURCE_DATA.md          # Combine Sharadar + custom data
├── MULTI_SOURCE_QUICKREF.md      # Quick reference for multi-source
├── FLIGHTLOG.md                  # Real-time log monitoring
└── DEVELOPMENT.md                # Docker build, development workflow
```

---

## Key Features

### Incremental Data Updates

**Problem:** Traditional data bundles download ALL data every time (10-20 minutes daily)

**Solution:** Sharadar bundle with incremental updates
- First download: 1998-present (~10-20 minutes)
- Daily updates: Yesterday only (~10-30 seconds)
- Automatic detection of last ingestion date
- Combines new data with existing history

**Impact:** 40x faster daily updates!

### Real-Time Monitoring

**Problem:** Long-running backtests with no visibility into progress

**Solution:** FlightLog TCP streaming
- Real-time log display in separate terminal
- Color-coded log levels (INFO=green, ERROR=red)
- Progress bars with portfolio metrics
- Optional filtering (algorithm-only, no progress bars)
- Multiple concurrent connections

**Impact:** See exactly what's happening during backtests

---

## Getting Help

### Documentation

- **Quick Start:** [QUICK_START_DATA.md](./QUICK_START_DATA.md)
- **Data Management:** [DATA_MANAGEMENT.md](./DATA_MANAGEMENT.md)
- **Fundamentals:** [SHARADAR_FUNDAMENTALS_GUIDE.md](./SHARADAR_FUNDAMENTALS_GUIDE.md)
- **Symbol Mapping:** [SYMBOL_MAPPING.md](./SYMBOL_MAPPING.md)
- **Multi-Source:** [MULTI_SOURCE_DATA.md](./MULTI_SOURCE_DATA.md)
- **Monitoring:** [FLIGHTLOG.md](./FLIGHTLOG.md)
- **Development:** [DEVELOPMENT.md](./DEVELOPMENT.md)

### External Resources

- **Zipline Docs:** https://zipline.ml4trading.io
- **Community Forum:** https://exchange.ml4trading.io
- **Sharadar Data:** https://data.nasdaq.com/databases/SFA
- **NASDAQ Support:** https://data.nasdaq.com/contact

### Troubleshooting

**Data Issues:**
- See [DATA_MANAGEMENT.md - Troubleshooting](./DATA_MANAGEMENT.md#troubleshooting)
- Check `~/.zipline/ingestions/sharadar_*.log`

**FlightLog Issues:**
- See [FLIGHTLOG.md - Troubleshooting](./FLIGHTLOG.md#troubleshooting)
- Verify connection: `docker ps | grep flightlog`

**General Issues:**
- GitHub Issues: https://github.com/stefan-jansen/zipline-reloaded/issues
- Check Docker logs: `docker compose logs zipline-jupyter`

---

## Contributing

Found an error or have a suggestion? Please:

1. Check existing documentation for similar issues
2. Create a GitHub issue with:
   - What you were trying to do
   - What documentation you were following
   - What went wrong or could be improved
3. Submit a pull request if you'd like to contribute improvements

---

## Quick Command Reference

### Data Management

```bash
# List bundles
zipline bundles

# Ingest/update data
zipline ingest -b sharadar

# Clean old ingestions
zipline clean -b sharadar --keep-last 7

# Verify in Python
from zipline.data import bundles
bundle = bundles.load('sharadar')
```

### FlightLog

```bash
# Start FlightLog (foreground, shows colors)
docker compose run --rm flightlog

# Start in background
docker compose up -d flightlog

# Filter progress bars
docker compose run --rm flightlog python /app/scripts/flightlog.py --filter-progress

# View logs
docker compose logs -f flightlog
```

### Docker

```bash
# Start services
docker compose up -d

# Rebuild image
docker compose build zipline-jupyter

# Restart container
docker compose restart zipline-jupyter

# Execute command in container
docker compose exec zipline-jupyter <command>

# View logs
docker compose logs zipline-jupyter
```

---

## License

These documentation files are part of Zipline-Reloaded and are licensed under Apache 2.0.

---

**Last Updated:** 2025-11-19

**Maintained by:** Zipline-Reloaded Community
