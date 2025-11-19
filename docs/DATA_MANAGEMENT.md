# Data Management Guide

Complete guide to managing Sharadar data bundles, including setup, daily updates, incremental updates, and cleanup.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Initial Setup](#initial-setup)
3. [Daily Updates](#daily-updates)
4. [Incremental Updates](#incremental-updates)
5. [Bundle Cleanup](#bundle-cleanup)
6. [Alternative Bundles](#alternative-bundles)
7. [Automation](#automation)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### One-Time Setup (5 minutes)

```bash
# 1. Get API key from https://data.nasdaq.com/databases/SFA
# 2. Configure key
echo 'NASDAQ_DATA_LINK_API_KEY=your-key-here' > .env

# 3. Start container and download data (10-20 minutes)
docker compose up -d
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

### Daily Updates (30 seconds)

```bash
# Run after market close (6 PM ET)
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

---

## Initial Setup

### What is Sharadar?

Sharadar is a **premium, institutional-grade dataset** from NASDAQ Data Link that provides:

- **SEP (Sharadar Equity Prices)**: Daily OHLCV pricing data
- **SF1 (Sharadar Fundamentals)**: Fundamental data (revenue, earnings, ratios)
- **ACTIONS**: Corporate actions (splits, dividends)
- **TICKERS**: Ticker metadata and classification

### Why Choose Sharadar?

| Feature | Sharadar | Yahoo Finance |
|---------|----------|---------------|
| **Data Quality** | Institutional-grade | Good |
| **Point-in-Time** | Yes | No |
| **Corporate Actions** | Comprehensive | Basic |
| **Historical Depth** | Complete history | ~20 years |
| **Cost** | Paid subscription | Free |

**Use Sharadar if:**
- Building professional trading strategies
- Need point-in-time accuracy for backtesting
- Require comprehensive fundamental data

**Use Yahoo Finance if:**
- Learning zipline
- Prototyping strategies
- Budget is limited

### Get Your API Key

1. Subscribe at: https://data.nasdaq.com/databases/SFA
2. Log in and go to Account Settings
3. Copy your API key (format: `abc123xyz...`)

### Configure API Key

**Option A: Via .env file (Recommended)**

```bash
cd /path/to/zipline-reloaded
echo 'NASDAQ_DATA_LINK_API_KEY=your_key_here' > .env
docker compose restart
```

**Option B: Environment variable**

```bash
export NASDAQ_DATA_LINK_API_KEY='your_key_here'
```

### Initial Download

```bash
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

**Expected:**
- Time: 60-90 minutes for full dataset
- Storage: ~10-15 GB
- Date range: 1998 to today
- Tickers: ~8,000 US equities + ETFs

---

## Daily Updates

After the initial download, updates are **fast and automatic**:

```bash
docker compose exec zipline-jupyter zipline ingest -b sharadar
```

**Expected:**
- Time: 2-5 minutes
- Storage: ~50-100 MB (just new data)

### When to Run

| Time (ET) | Status | Action |
|-----------|--------|--------|
| Before 5:00 PM | Data not ready | Don't run yet |
| 5:00-6:00 PM | Processing | May be incomplete |
| After 6:00 PM | Ready | Safe to run |
| Weekends | No trading | Skip |

### Verify Your Data

```bash
# Check ingestion status
docker compose exec zipline-jupyter zipline bundles
```

---

## Incremental Updates

### How It Works

**First Ingestion**: Downloads ALL historical data (60-90 minutes)

**Subsequent Ingestions**: Only downloads NEW data since last ingestion (2-5 minutes)

The bundle automatically:
1. Checks for existing data in `~/.zipline/data/sharadar/`
2. Finds the last date in the database
3. Requests data from `last_date + 1 day` onwards

### Performance Comparison

| Scenario | Duration | Data Downloaded |
|----------|----------|-----------------|
| **First Ingestion** | 60-90 min | 27 years (1998-2025) |
| **Daily Update** | 2-3 min | 1 day |
| **Weekly Update** | 3-4 min | 7 days |
| **Monthly Update** | 4-5 min | 30 days |

---

## Bundle Cleanup

### Why Cleanup is Important

Each ingestion creates a new timestamped directory. Cleanup saves disk space:

```bash
# Keep only the 2 most recent ingestions
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 2

# Keep only the most recent
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 1
```

### Recommended Keep Count

| Scenario | Keep Last | Reason |
|----------|-----------|--------|
| **Development** | 1-2 | Save disk space |
| **Production Daily** | 2-3 | Quick rollback |
| **Production Critical** | 5-7 | Multiple rollback points |

### Storage Calculation

**Sharadar bundle size**: ~600 MB per ingestion

| Keep Last | Disk Usage |
|-----------|------------|
| 1 | ~600 MB |
| 2 | ~1.2 GB |
| 3 | ~1.8 GB |
| 5 | ~3.0 GB |

---

## Alternative Bundles

### Tech Stocks Only (Fast Testing)

```bash
# Takes 30 seconds instead of 20 minutes
docker exec zipline-reloaded-jupyter zipline ingest -b sharadar-tech
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

---

## Automation

### Cron Job (Recommended)

```bash
# Edit crontab
crontab -e

# Add line (runs Mon-Fri at 6 PM ET)
0 18 * * 1-5 cd /path/to/zipline-reloaded && \
    docker compose exec -T zipline-jupyter zipline ingest -b sharadar \
    >> /var/log/sharadar-update.log 2>&1
```

### Update Script with Cleanup

```bash
#!/bin/bash
BUNDLE=${1:-sharadar}
KEEP_LAST=${2:-2}

docker exec zipline-reloaded-jupyter zipline ingest -b $BUNDLE
docker exec zipline-reloaded-jupyter zipline clean -b $BUNDLE --keep-last $KEEP_LAST
```

---

## Troubleshooting

### "API key required"

```bash
# Check .env file
cat .env

# If missing
echo 'NASDAQ_DATA_LINK_API_KEY=your-key' > .env
docker compose restart
```

### "Out of memory"

Increase Docker memory to **16 GB** for initial ingestion (Docker Desktop → Settings → Resources).

For incremental updates, **4 GB** is sufficient.

### "No data for today"

Wait 1-2 hours after market close (safe after 6 PM ET).

### Emergency: Disk Space Full

```bash
# Aggressive cleanup - keep only latest
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 1
```

---

## Resource Requirements

| Operation | Time | Storage | RAM |
|-----------|------|---------|-----|
| **Initial download** | 60-90 min | 10-15 GB | 16 GB |
| **Daily update** | 2-5 min | +50-100 MB | 4 GB |
| **Tech bundle** | 30 sec | 50 MB | 2 GB |

---

## Quick Command Reference

```bash
# List bundles and ingestion dates
zipline bundles

# Ingest/update data
zipline ingest -b sharadar

# Clean old ingestions (keep last 2)
zipline clean -b sharadar --keep-last 2

# Check disk usage
du -sh /root/.zipline/data/sharadar
```

---

## Best Practices

1. **Start small**: Test with tech bundle before full ingestion
2. **Run daily updates**: Keeps ingestion fast (2-5 minutes)
3. **Use Docker volumes**: Persist data between container restarts
4. **Monitor disk space**: Clean old ingestions regularly
5. **Backup bundles**: Before major changes
6. **Keep multiple ingestions**: At least 2 for rollback

---

## Related Documentation

- [QUICK_START_DATA.md](./QUICK_START_DATA.md) - 5-minute setup
- [SHARADAR_FUNDAMENTALS_GUIDE.md](./SHARADAR_FUNDAMENTALS_GUIDE.md) - Using fundamentals
- [MULTI_SOURCE_DATA.md](./MULTI_SOURCE_DATA.md) - Combining data sources
