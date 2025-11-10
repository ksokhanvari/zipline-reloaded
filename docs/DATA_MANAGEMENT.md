# Sharadar Data Management Guide

Complete guide to downloading, updating, and managing Sharadar market data for Zipline backtesting.

---

## Table of Contents

1. [Overview](#overview)
2. [Initial Setup](#initial-setup)
3. [First-Time Data Download](#first-time-data-download)
4. [Daily Incremental Updates](#daily-incremental-updates)
5. [Monitoring Your Data](#monitoring-your-data)
6. [Automation](#automation)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)
9. [Command Reference](#command-reference)

---

## Overview

### What is Sharadar?

Sharadar is a **premium** dataset from NASDAQ Data Link that provides:
- **Daily OHLCV pricing** for ~8,000 US equities
- **ETF and fund pricing** data
- **Corporate actions** (splits, dividends, etc.)
- **Historical data** from 1998 to present
- **Daily updates** with T+0 availability

**Cost:** Requires a paid subscription from [NASDAQ Data Link](https://data.nasdaq.com/databases/SFA)

### Incremental Updates (Key Feature)

The Sharadar bundle supports **incremental updates**, which dramatically speeds up daily data refreshes:

| Update Type | Date Range | Time Required | Data Downloaded |
|-------------|------------|---------------|-----------------|
| **Initial download** | 1998 - Present | 10-20 minutes | ~10 GB (all history) |
| **Daily update** | Yesterday only | 10-30 seconds | ~50 MB (1 day) |

**How it works:**
1. First ingestion downloads all historical data since 1998
2. Subsequent ingestions **automatically detect** the last download date
3. Only new data since last ingestion is downloaded
4. Full history is preserved and combined with new data

---

## Initial Setup

### Step 1: Get Your API Key

1. Sign up for Sharadar at: https://data.nasdaq.com/databases/SFA
2. Get your API key from: https://data.nasdaq.com/account/profile
3. Copy the key (looks like: `aBcDeFgHiJkLmNoPqR12`)

### Step 2: Configure API Key

**On host machine (Mac/Linux):**

```bash
# Add to your shell profile (~/.zshrc, ~/.bashrc, etc.)
echo 'export NASDAQ_DATA_LINK_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc

# Verify it's set
echo $NASDAQ_DATA_LINK_API_KEY
```

**For Docker (recommended):**

Create a `.env` file in your zipline-reloaded directory:

```bash
cd /path/to/zipline-reloaded

# Create .env file
cat > .env << 'EOF'
NASDAQ_DATA_LINK_API_KEY=your-api-key-here
EOF

# Verify
cat .env
```

The docker-compose.yml automatically loads this file.

### Step 3: Verify Bundle Registration

```bash
# Inside Docker container
docker compose exec zipline-jupyter bash

# List available bundles
zipline bundles

# You should see:
# sharadar              <no ingestions>
# sharadar-equities     <no ingestions>
# sharadar-tech         <no ingestions>
```

If bundles are missing, check that `extension.py` exists at `~/.zipline/extension.py`.

---

## First-Time Data Download

### Option 1: Full Dataset (Recommended for Production)

Downloads **all** US equities and ETFs since 1998.

```bash
# Inside Docker container
docker compose exec zipline-jupyter bash

# Run ingestion
zipline ingest -b sharadar

# Expected output:
# ============================================================
# Sharadar Bundle Ingestion
# ============================================================
# Date range: 1998-01-01 to 2025-11-04
# Tickers: ALL (this may take a while and use significant storage)
# ============================================================
#
# Step 1/4: Downloading Sharadar Equity Prices (SEP table)...
# Downloaded 60,000,000 equity price records for 8,234 tickers
#
# Step 2/4: Downloading Sharadar Fund Prices (SFP table)...
# Downloaded 15,000,000 fund price records for 2,456 tickers
#
# Combined 75,000,000 total price records
#
# Step 3/4: Downloading corporate actions (ACTIONS table)...
# Downloaded 450,000 corporate action records
#
# Step 4/4: Processing data for zipline...
# Writing asset metadata...
# Writing daily bars...
# Writing adjustments...
#
# ============================================================
# ✓ Sharadar bundle ingestion complete!
# ============================================================
```

**Time required:** 10-20 minutes (depending on internet speed)
**Storage required:** ~10-15 GB
**Memory required:** 8-16 GB RAM

### Option 2: Tech Stocks Sample (Fast for Testing)

Downloads only major tech stocks for faster testing.

```bash
# Use the pre-configured tech bundle
zipline ingest -b sharadar-tech

# Expected output:
# Date range: 1998-01-01 to 2025-11-04
# Tickers: 15 symbols
# Downloaded 125,000 equity price records for 15 tickers
```

**Time required:** 30-60 seconds
**Storage required:** ~50 MB
**Memory required:** 2 GB RAM

### Option 3: Custom Ticker List

Edit `~/.zipline/extension.py` to create a custom bundle:

```python
from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

# Your custom watchlist
register(
    'my-watchlist',
    sharadar_bundle(
        tickers=[
            'SPY', 'QQQ', 'IWM',  # Index ETFs
            'AAPL', 'MSFT', 'GOOGL',  # Your stocks
        ],
        incremental=True,
        include_funds=True,
    ),
)
```

Then ingest:

```bash
zipline ingest -b my-watchlist
```

---

## Daily Incremental Updates

### How Incremental Updates Work

When you run `zipline ingest -b sharadar` the second time:

1. **Detection:** Checks the existing bundle database for the last ingestion date
2. **Smart Download:** Downloads only data from `last_date + 1` to today
3. **Merging:** Combines new data with existing historical data
4. **Speed:** ~10-30 seconds instead of 10-20 minutes

### Manual Daily Update

**Recommended workflow** (run once per day after market close):

```bash
# Inside Docker container
docker compose exec zipline-jupyter bash

# Run incremental update
zipline ingest -b sharadar

# Expected output:
# ============================================================
# 🔄 INCREMENTAL UPDATE DETECTED
# ============================================================
# Last ingestion ended: 2025-11-03
# Downloading new data from: 2025-11-04 to 2025-11-04
# This will be much faster than a full download!
# ============================================================
#
# Step 1/4: Downloading NEW Sharadar Equity Prices (incremental)...
# Downloaded 8,234 equity price records for 8,234 tickers
#
# Step 2/4: Downloading NEW Sharadar Fund Prices (incremental)...
# Downloaded 2,456 fund price records for 2,456 tickers
#
# Combined 10,690 total price records
#
# Step 3/4: Downloading NEW corporate actions (incremental)...
# Downloaded 15 corporate action records
#
# Step 4/4: Processing data for zipline...
# ✓ Sharadar bundle ingestion complete!
```

**Time required:** 10-30 seconds
**Data downloaded:** ~50 MB (1 day)
**When to run:** After 5:00 PM ET (market close + data processing)

### Verify Update

```bash
# Check ingestion history
zipline bundles

# Output shows:
# sharadar 2025-11-04 19:23:14.123456
# sharadar 2025-11-03 19:15:42.789012
# sharadar 2025-11-02 19:10:33.456789
```

The most recent timestamp is your latest ingestion.

---

## Monitoring Your Data

### Check Bundle Status

```bash
# List all bundles and ingestion dates
zipline bundles

# Output:
# sharadar              2025-11-04 19:23:14.123456
# sharadar-equities     <no ingestions>
# sharadar-tech         2025-11-03 18:45:22.987654
```

### Check Data Coverage

Create a script to verify your data range:

```python
# scripts/check_data_coverage.py
from zipline.data import bundles
import pandas as pd

bundle_name = 'sharadar'
bundle_data = bundles.load(bundle_name)

# Get date range
sessions = bundle_data.equity_daily_bar_reader.sessions
print(f"\n{'='*60}")
print(f"Bundle: {bundle_name}")
print(f"{'='*60}")
print(f"First date: {sessions[0].date()}")
print(f"Last date:  {sessions[-1].date()}")
print(f"Total trading days: {len(sessions):,}")

# Get asset count
asset_finder = bundle_data.asset_finder
all_sids = asset_finder.sids
print(f"Total assets: {len(all_sids):,}")

# Get last 10 dates
print(f"\nMost recent trading days:")
for session in sessions[-10:]:
    print(f"  {session.date()}")
print(f"{'='*60}\n")
```

Run it:

```bash
docker compose exec zipline-jupyter python /app/scripts/check_data_coverage.py
```

### Check Specific Symbol

```python
# In Jupyter notebook
from zipline.data import bundles

bundle = bundles.load('sharadar')
asset_finder = bundle.asset_finder

# Look up a symbol
aapl = asset_finder.lookup_symbol('AAPL', as_of_date=None)

print(f"Symbol: {aapl.symbol}")
print(f"Start date: {aapl.start_date.date()}")
print(f"End date: {aapl.end_date.date()}")
print(f"Exchange: {aapl.exchange}")
```

### Ingestion Logs

Zipline saves ingestion logs to:

```bash
# View latest ingestion log
docker compose exec zipline-jupyter bash
ls -lt ~/.zipline/ingestions/

# Read the log
cat ~/.zipline/ingestions/sharadar_YYYYMMDD_HHMMSS.log
```

---

## Automation

### Option 1: Cron Job (Linux/Mac)

**On your host machine** (not in Docker), set up a daily cron job:

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 6 PM)
0 18 * * 1-5 cd /path/to/zipline-reloaded && docker compose exec -T zipline-jupyter zipline ingest -b sharadar >> /path/to/logs/sharadar_ingest.log 2>&1
```

**Explanation:**
- `0 18 * * 1-5` = 6:00 PM, Monday-Friday only
- `-T` flag = disable pseudo-TTY for cron compatibility
- `>> ... 2>&1` = append logs to file

**View logs:**

```bash
tail -f /path/to/logs/sharadar_ingest.log
```

### Option 2: Shell Script

Create a dedicated update script:

```bash
#!/bin/bash
# scripts/update_sharadar_data.sh

# Configuration
BUNDLE_NAME="sharadar"
LOG_DIR="/path/to/zipline-reloaded/logs"
LOG_FILE="${LOG_DIR}/sharadar_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Run ingestion
echo "========================================" | tee -a "$LOG_FILE"
echo "Sharadar Data Update - $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd /path/to/zipline-reloaded

docker compose exec -T zipline-jupyter zipline ingest -b "$BUNDLE_NAME" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Update completed successfully" | tee -a "$LOG_FILE"
else
    echo "✗ Update failed with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
    # Send email alert (optional)
    # echo "Sharadar update failed" | mail -s "Zipline Alert" your@email.com
fi

echo "========================================" | tee -a "$LOG_FILE"
```

Make it executable:

```bash
chmod +x scripts/update_sharadar_data.sh

# Test it
./scripts/update_sharadar_data.sh
```

Then add to crontab:

```bash
0 18 * * 1-5 /path/to/zipline-reloaded/scripts/update_sharadar_data.sh
```

### Option 3: GitHub Actions (for remote updates)

Create `.github/workflows/update-data.yml`:

```yaml
name: Update Sharadar Data

on:
  schedule:
    # Run Monday-Friday at 6 PM ET (11 PM UTC)
    - cron: '0 23 * * 1-5'
  workflow_dispatch:  # Allow manual trigger

jobs:
  update-data:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Docker
        run: docker compose build

      - name: Run data ingestion
        env:
          NASDAQ_DATA_LINK_API_KEY: ${{ secrets.NASDAQ_API_KEY }}
        run: |
          docker compose up -d zipline-jupyter
          docker compose exec -T zipline-jupyter zipline ingest -b sharadar

      - name: Archive bundle data
        if: always()
        run: |
          # Optionally backup bundle data to S3, artifact, etc.
          tar -czf sharadar-bundle.tar.gz -C ~/.zipline/data .
```

**Note:** Store your API key in GitHub Secrets (`Settings > Secrets > NASDAQ_API_KEY`)

---

## Troubleshooting

### Problem: "No module named 'nasdaqdatalink'"

**Cause:** Missing package in Docker image.

**Fix:**

```bash
# Rebuild Docker image (package should be included)
docker compose build zipline-jupyter
docker compose up -d
```

Verify `nasdaq-data-link` is in `Dockerfile`:

```dockerfile
RUN pip install --no-cache-dir nasdaq-data-link
```

### Problem: "NASDAQ Data Link API key required"

**Cause:** API key not set or not accessible to container.

**Fix:**

```bash
# Check if .env file exists
cat .env

# Should contain:
NASDAQ_DATA_LINK_API_KEY=your-key-here

# Restart container to load .env
docker compose restart zipline-jupyter

# Verify inside container
docker compose exec zipline-jupyter bash
echo $NASDAQ_DATA_LINK_API_KEY
```

### Problem: "Out of memory" during ingestion

**Cause:** Insufficient RAM for processing 60M+ records.

**Solutions:**

**1. Increase Docker memory:**
- Docker Desktop → Settings → Resources → Memory
- Set to **16 GB** for full download
- Set to **8 GB** minimum for incremental updates

**2. Use smaller ticker list:**

```python
# Edit ~/.zipline/extension.py
register(
    'sharadar-small',
    sharadar_bundle(
        tickers=['SPY', 'QQQ', 'AAPL', 'MSFT'],  # Just a few tickers
        incremental=True,
    ),
)
```

**3. Use chunked download:**

```python
register(
    'sharadar',
    sharadar_bundle(
        use_chunks=True,  # Download in 5-year chunks
        incremental=True,
    ),
)
```

### Problem: Incremental update not working (full download every time)

**Cause:** Bundle database not found or corrupted.

**Fix:**

```bash
# Check existing ingestions
zipline bundles

# If shows ingestions but still downloads full data, clean and reingest
zipline clean -b sharadar --keep-last 1
zipline ingest -b sharadar
```

### Problem: "Bulk export not ready after X hours"

**Cause:** NASDAQ servers take very long to prepare full historical export (1998-present).

**Solutions:**

1. **Wait and retry** - File might be ready now
2. **Download smaller date range first:**

```python
register(
    'sharadar-recent',
    sharadar_bundle(
        start_date='2020-01-01',  # Last 5 years only
        incremental=True,
    ),
)
```

3. **Contact NASDAQ support** if persistent

### Problem: "Invalid subscription" or "No data returned"

**Cause:** Your NASDAQ account doesn't have Sharadar access.

**Fix:**

1. Verify subscription at: https://data.nasdaq.com/databases/SFA
2. Ensure subscription is **active** (not expired)
3. Test API key manually:

```python
import nasdaqdatalink
nasdaqdatalink.ApiConfig.api_key = 'your-key'
data = nasdaqdatalink.get_table('SHARADAR/SEP', ticker='AAPL', paginate=True)
print(data.head())
```

### Problem: Data seems stale or missing recent days

**Cause:** Sharadar data has a processing delay.

**Expected behavior:**
- Market closes: 4:00 PM ET
- Data available: ~5:00 PM ET (T+0 typically)
- Run ingestion after: 6:00 PM ET to be safe

**Fix:**

```bash
# Check what date range was actually downloaded
zipline bundles

# Re-run ingestion if needed
zipline ingest -b sharadar
```

---

## Advanced Usage

### Multiple Bundles for Different Strategies

You can maintain separate bundles for different use cases:

```python
# ~/.zipline/extension.py

# Full dataset for production
register('sharadar-prod', sharadar_bundle(
    incremental=True,
    include_funds=True,
))

# Equities only for stock strategies
register('sharadar-stocks', sharadar_bundle(
    incremental=True,
    include_funds=False,
))

# Tech sector for sector rotation
register('sharadar-tech', sharadar_bundle(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
    incremental=True,
))

# ETFs only for portfolio allocation
register('sharadar-etfs', sharadar_bundle(
    tickers=['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'USO'],
    incremental=True,
))
```

Update each one:

```bash
zipline ingest -b sharadar-prod
zipline ingest -b sharadar-stocks
zipline ingest -b sharadar-tech
zipline ingest -b sharadar-etfs
```

### Force Full Re-download

To force a complete re-download (not incremental):

**Option 1: Clean and reingest**

```bash
# Remove all previous ingestions
zipline clean -b sharadar

# Fresh download
zipline ingest -b sharadar
```

**Option 2: Create non-incremental bundle**

```python
# Temporary bundle for full refresh
register('sharadar-full-refresh', sharadar_bundle(
    incremental=False,  # Disable incremental
))
```

```bash
zipline ingest -b sharadar-full-refresh
```

### Keep Last N Ingestions

To save disk space, keep only recent ingestions:

```bash
# Keep last 7 days (delete older)
zipline clean -b sharadar --keep-last 7

# Keep only the latest
zipline clean -b sharadar --keep-last 1

# Remove specific ingestion by timestamp
zipline clean -b sharadar --before 2025-10-01
```

### Backup Your Bundle Data

Bundle data is stored at `~/.zipline/data/`:

```bash
# Backup to tar.gz
tar -czf sharadar-bundle-$(date +%Y%m%d).tar.gz \
    -C ~/.zipline/data .

# Restore from backup
mkdir -p ~/.zipline/data
tar -xzf sharadar-bundle-20251104.tar.gz -C ~/.zipline/data/
```

**For Docker:**

```bash
# Backup Docker volume
docker run --rm \
    -v zipline-reloaded_zipline-data:/data \
    -v $(pwd):/backup \
    ubuntu tar -czf /backup/zipline-data-backup.tar.gz -C /data .

# Restore Docker volume
docker run --rm \
    -v zipline-reloaded_zipline-data:/data \
    -v $(pwd):/backup \
    ubuntu tar -xzf /backup/zipline-data-backup.tar.gz -C /data
```

### Custom Date Ranges

Download specific date ranges:

```python
# Last year only
register('sharadar-2024', sharadar_bundle(
    start_date='2024-01-01',
    end_date='2024-12-31',
    incremental=False,
))

# Financial crisis period
register('sharadar-crisis', sharadar_bundle(
    start_date='2007-01-01',
    end_date='2009-12-31',
    incremental=False,
))
```

---

## Command Reference

### Bundle Management

```bash
# List all registered bundles
zipline bundles

# Ingest a bundle
zipline ingest -b <bundle-name>

# Ingest with verbose output
zipline ingest -b <bundle-name> --show-progress

# Clean old ingestions (keep last N)
zipline clean -b <bundle-name> --keep-last <N>

# Clean old ingestions (before date)
zipline clean -b <bundle-name> --before YYYY-MM-DD

# Remove all ingestions for a bundle
zipline clean -b <bundle-name>
```

### Docker Commands

```bash
# Enter container
docker compose exec zipline-jupyter bash

# Run command in container (from host)
docker compose exec -T zipline-jupyter <command>

# View container logs
docker compose logs zipline-jupyter

# Restart container
docker compose restart zipline-jupyter

# Rebuild image
docker compose build zipline-jupyter
```

### Data Inspection

```python
# In Python/Jupyter
from zipline.data import bundles

# Load bundle
bundle = bundles.load('sharadar')

# Get asset finder
finder = bundle.asset_finder

# Look up symbol
asset = finder.lookup_symbol('AAPL', as_of_date=None)
print(f"SID: {asset.sid}")
print(f"Range: {asset.start_date} to {asset.end_date}")

# Get daily bar reader
reader = bundle.equity_daily_bar_reader

# Get sessions
sessions = reader.sessions
print(f"First: {sessions[0]}")
print(f"Last: {sessions[-1]}")
```

---

## Best Practices

### 1. Daily Update Schedule

**Recommended timing:**
- ✅ After **6:00 PM ET** (market close + data processing)
- ✅ **Monday-Friday** only (skip weekends/holidays)
- ✅ Use **incremental mode** for speed

### 2. Monitoring

**Check weekly:**
- Last ingestion date matches recent trading day
- Data coverage is continuous (no gaps)
- Log files for any errors

**Alerts to set up:**
- Email notification if ingestion fails
- Slack/Discord webhook for completion status
- Disk space monitoring (keep 20+ GB free)

### 3. Backup Strategy

**Daily:** Keep last 7 ingestions
```bash
zipline clean -b sharadar --keep-last 7
```

**Weekly:** Backup to external storage
```bash
tar -czf sharadar-weekly-$(date +%Y%m%d).tar.gz ~/.zipline/data/
```

**Monthly:** Archive to S3/cloud storage

### 4. Resource Management

**Memory:**
- Full download: 16 GB RAM
- Incremental update: 4 GB RAM
- Adjust Docker Desktop settings accordingly

**Disk:**
- Reserve **20 GB** for bundle data
- Monitor with `df -h`
- Clean old ingestions regularly

### 5. Testing

Before relying on automation:

```bash
# Test manual update
zipline ingest -b sharadar

# Verify data
python scripts/check_data_coverage.py

# Test backtest with latest data
python examples/simple_flightlog_demo.py
```

---

## Summary

**For daily incremental updates:**

```bash
# 1. One-time setup
echo 'NASDAQ_DATA_LINK_API_KEY=your-key' > .env

# 2. Initial download (once)
docker compose exec zipline-jupyter zipline ingest -b sharadar

# 3. Daily update (automated)
0 18 * * 1-5 cd /path/to/zipline-reloaded && \
    docker compose exec -T zipline-jupyter zipline ingest -b sharadar
```

**Key points:**
- ✅ First download: 10-20 minutes, ~10 GB
- ✅ Daily updates: 10-30 seconds, ~50 MB
- ✅ Automatic incremental detection
- ✅ Run after market close (6 PM ET)
- ✅ Monitor logs for errors
- ✅ Backup weekly

---

## Getting Help

**Issues with Sharadar data:**
- NASDAQ Data Link Support: https://data.nasdaq.com/contact
- Sharadar Forums: https://data.nasdaq.com/databases/SFA/documentation

**Issues with Zipline bundle:**
- Zipline Docs: https://zipline.ml4trading.io
- Community: https://exchange.ml4trading.io
- GitHub Issues: https://github.com/stefan-jansen/zipline-reloaded/issues

**Check bundle logs:**
```bash
ls -lt ~/.zipline/ingestions/
cat ~/.zipline/ingestions/sharadar_*.log
```
