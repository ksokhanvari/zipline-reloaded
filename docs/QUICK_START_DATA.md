# Sharadar Data Quick Start

**TL;DR** - Get up and running with daily market data in 5 minutes.

---

## One-Time Setup (5 minutes)

### 1. Get API Key

1. Sign up: https://data.nasdaq.com/databases/SFA (requires paid subscription)
2. Copy your API key from: https://data.nasdaq.com/account/profile

### 2. Configure Key

```bash
# Create .env file
cd /path/to/zipline-reloaded
echo 'NASDAQ_DATA_LINK_API_KEY=your-key-here' > .env
```

### 3. Initial Download

```bash
# Start container
docker compose up -d

# Download all historical data (takes 10-20 minutes)
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

**Expected:**
- Time: 10-20 minutes
- Storage: ~10 GB
- Date range: 1998 to today
- Tickers: ~8,000 US equities + ETFs

---

## Daily Updates (30 seconds)

After the initial download, updates are **automatic and fast**:

```bash
# Run after market close (6 PM ET or later)
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

**Expected:**
- Time: 10-30 seconds (not minutes!)
- Storage: ~50 MB (just yesterday's data)
- Automatic: Detects last download, only gets new data

### When to Run

| Time (ET) | Status | Action |
|-----------|--------|--------|
| Before 5:00 PM | Data not ready | ❌ Don't run yet |
| 5:00-6:00 PM | Processing | ⏳ May work, may be incomplete |
| After 6:00 PM | Ready | ✅ Safe to run |
| Weekends | No trading | ❌ Skip (no new data) |

---

## Verify Your Data

### Check Ingestion Status

```bash
docker compose exec zipline-jupyter zipline bundles

# Shows:
# sharadar  2025-11-04 19:23:14.123456  ← Latest ingestion
```

### Check Date Range

```python
# In Jupyter notebook
from zipline.data import bundles

bundle = bundles.load('sharadar')
sessions = bundle.equity_daily_bar_reader.sessions

print(f"First date: {sessions[0].date()}")
print(f"Last date:  {sessions[-1].date()}")
print(f"Total days: {len(sessions):,}")

# Expected:
# First date: 1998-01-02
# Last date:  2025-11-04  ← Should be yesterday or today
# Total days: 6,891
```

### Check Specific Symbol

```python
from zipline.data import bundles

bundle = bundles.load('sharadar')
finder = bundle.asset_finder

aapl = finder.lookup_symbol('AAPL', as_of_date=None)
print(f"AAPL data: {aapl.start_date.date()} to {aapl.end_date.date()}")

# Expected:
# AAPL data: 1998-01-02 to 2025-11-04
```

---

## Automation

### Linux/Mac Cron Job

```bash
# Edit crontab
crontab -e

# Add this line (runs Mon-Fri at 6 PM)
0 18 * * 1-5 cd /path/to/zipline-reloaded && docker compose exec -T zipline-jupyter zipline ingest -b sharadar >> /path/to/logs/data_update.log 2>&1
```

### Manual Script

```bash
#!/bin/bash
# update-data.sh

cd /path/to/zipline-reloaded
docker compose exec -T zipline-jupyter zipline ingest -b sharadar

if [ $? -eq 0 ]; then
    echo "✓ Data updated successfully at $(date)"
else
    echo "✗ Data update failed at $(date)"
fi
```

Make executable and run:

```bash
chmod +x update-data.sh
./update-data.sh
```

---

## Troubleshooting

### "API key required"

```bash
# Check if .env file exists
cat .env

# Should show:
NASDAQ_DATA_LINK_API_KEY=abc123xyz

# If missing, recreate it:
echo 'NASDAQ_DATA_LINK_API_KEY=your-key' > .env

# Restart container
docker compose restart zipline-jupyter
```

### "Out of memory"

Increase Docker memory to **16 GB**:
- Docker Desktop → Settings → Resources → Memory → 16 GB

For incremental updates, **4 GB** is usually sufficient.

### "No data for today"

Data has a processing delay:
- Market closes: 4:00 PM ET
- Data available: ~5:00 PM ET
- Safe to run after: 6:00 PM ET

Wait 1-2 hours after market close, then re-run:

```bash
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

### Incremental update not working

If it downloads ALL data every time (taking 10+ minutes):

```bash
# Clean and reingest
docker compose exec zipline-jupyter zipline clean -b sharadar --keep-last 1
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

Next ingestion should be fast (incremental).

---

## Alternative Bundles

### Tech Stocks Only (Fast Testing)

```bash
# Takes 30 seconds instead of 20 minutes
docker compose exec zipline-jupyter zipline ingest -b sharadar-tech

# Uses these tickers:
# AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX, ADBE, CRM,
# INTC, AMD, ORCL, CSCO, AVGO
```

### Equities Only (No ETFs)

```bash
# Smaller dataset
docker compose exec zipline-jupyter zipline ingest -b sharadar-equities
```

### Custom Ticker List

Edit `~/.zipline/extension.py`:

```python
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

register('my-stocks', sharadar_bundle(
    tickers=['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA'],
    incremental=True,
))
```

Then ingest:

```bash
docker compose exec zipline-jupyter zipline ingest -b my-stocks
```

---

## Quick Command Reference

```bash
# List bundles and ingestion dates
zipline bundles

# Ingest/update data
zipline ingest -b sharadar

# Clean old ingestions (keep last 7)
zipline clean -b sharadar --keep-last 7

# Remove all ingestions
zipline clean -b sharadar

# Check bundle in Python
from zipline.data import bundles
bundle = bundles.load('sharadar')
```

---

## Resource Requirements

| Operation | Time | Storage | RAM |
|-----------|------|---------|-----|
| **Initial download** | 10-20 min | 10 GB | 16 GB |
| **Daily update** | 10-30 sec | +50 MB | 4 GB |
| **Tech bundle** | 30 sec | 50 MB | 2 GB |

---

## What's Next?

- **Full documentation:** See [DATA_MANAGEMENT.md](./DATA_MANAGEMENT.md)
- **Run backtests:** See [FLIGHTLOG_USAGE.md](./FLIGHTLOG_USAGE.md)
- **Monitor logs:** See `~/.zipline/ingestions/sharadar_*.log`

---

## Summary

**Initial setup:**
```bash
echo 'NASDAQ_DATA_LINK_API_KEY=your-key' > .env
docker compose up -d
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

**Daily updates (automated with cron):**
```bash
0 18 * * 1-5 cd /path/to/zipline-reloaded && \
    docker compose exec -T zipline-jupyter zipline ingest -b sharadar
```

**That's it!** Your data is now updated daily in 10-30 seconds instead of downloading everything each time.
