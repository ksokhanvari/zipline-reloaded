# Bundle Cleanup Guide

## Why Cleanup is Important

Each time you ingest a Zipline bundle, it creates a new timestamped directory. This allows rollback if needed, but can quickly consume disk space with large datasets like Sharadar.

**Example**: With 3 Sharadar ingestions:
```
588 MB - 2025-11-08 (oldest)
588 MB - 2025-11-12
588 MB - 2025-11-15 (newest)
────────────────────
1.8 GB total
```

After cleanup (keeping last 2):
```
588 MB - 2025-11-12
588 MB - 2025-11-15 (newest)
────────────────────
1.2 GB total (600 MB saved!)
```

## Manual Cleanup

### Clean a Specific Bundle

Keep only the 2 most recent ingestions:

```bash
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 2
```

Keep only the most recent:

```bash
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 1
```

Keep last 5 (for safety/rollback):

```bash
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 5
```

### Clean All Bundles

```bash
# Keep last 2 for all bundles
for bundle in sharadar sharadar-tech sharadar-sp500; do
    docker exec zipline-reloaded-jupyter zipline clean -b "$bundle" --keep-last 2
done
```

## Automated Cleanup

### Option 1: Use the Update Script (Recommended)

The `update_and_clean_sharadar.sh` script does both ingest and cleanup:

```bash
# Update and clean (keeps last 2)
./scripts/update_and_clean_sharadar.sh

# Specify bundle name and keep count
./scripts/update_and_clean_sharadar.sh sharadar-tech 3
```

**What it does**:
1. Ingests new data
2. Cleans old ingestions
3. Shows disk usage and remaining ingestions

### Option 2: Add to Cron Job

For daily automated updates with cleanup:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 6 PM ET, Monday-Friday, keeps last 2)
0 18 * * 1-5 /path/to/zipline-reloaded/scripts/update_and_clean_sharadar.sh >> /var/log/sharadar-update.log 2>&1
```

### Option 3: Manual Commands

Run update and cleanup separately:

```bash
# Ingest
docker exec zipline-reloaded-jupyter zipline ingest -b sharadar

# Clean (keeps last 2)
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 2
```

## Best Practices

### Recommended Keep Count

| Scenario | Keep Last | Reason |
|----------|-----------|--------|
| **Development/Testing** | 1-2 | Save disk space, don't need history |
| **Production Daily** | 2-3 | Quick rollback if latest has issues |
| **Production Critical** | 5-7 | Multiple rollback points, weekly retention |
| **Long-term Backtesting** | 10+ | Test against different data snapshots |

### Storage Calculation

**Sharadar bundle size**: ~600 MB per ingestion

| Keep Last | Disk Usage |
|-----------|------------|
| 1 | ~600 MB |
| 2 | ~1.2 GB |
| 3 | ~1.8 GB |
| 5 | ~3.0 GB |
| 10 | ~6.0 GB |

**For large universes** (sharadar-all with 8,000+ tickers):
- Per ingestion: ~10-20 GB
- Keep 2: ~20-40 GB
- Keep 5: ~50-100 GB

### Disk Space Monitoring

Check bundle sizes:

```bash
# All bundles
docker exec zipline-reloaded-jupyter du -sh /root/.zipline/data/*

# Specific bundle
docker exec zipline-reloaded-jupyter du -sh /root/.zipline/data/sharadar

# Individual ingestions
docker exec zipline-reloaded-jupyter bash -c "
for dir in /root/.zipline/data/sharadar/*/; do
    du -sh \"\$dir\"
done
"
```

## Rollback to Previous Ingestion

If you need to rollback to an older ingestion:

### Option 1: List Available Ingestions

```bash
docker exec zipline-reloaded-jupyter zipline bundles
```

Output:
```
sharadar 2025-11-15 07:54:38.129303  ← Current
sharadar 2025-11-12 07:22:48.179626  ← Available rollback
```

### Option 2: Use Specific Timestamp

Zipline automatically uses the most recent ingestion. To use an older one, you need to specify the timestamp in your code:

```python
from zipline.data.bundles import load

# Load specific ingestion by timestamp
bundle = load('sharadar', timestamp=pd.Timestamp('2025-11-12 07:22:48.179626'))
```

Or delete the newer ingestion to make the older one active:

```bash
# Manually delete newest
docker exec zipline-reloaded-jupyter rm -rf /root/.zipline/data/sharadar/2025-11-15T07\;54\;38.129303
```

## Troubleshooting

### "No ingestions for bundle X"

If you accidentally delete all ingestions:

```bash
# Re-ingest from scratch
docker exec zipline-reloaded-jupyter zipline ingest -b sharadar
```

### Check What Will Be Deleted

Before cleaning, see what exists:

```bash
docker exec zipline-reloaded-jupyter bash -c "
echo 'Current ingestions:'
ls -lt /root/.zipline/data/sharadar/ | grep '^d'
"
```

The oldest directories (at the bottom) will be deleted first.

### Cleanup Didn't Free Space

Check if cleanup actually ran:

```bash
# Before
docker exec zipline-reloaded-jupyter du -sh /root/.zipline/data/sharadar

# Run cleanup
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 2

# After (should be smaller)
docker exec zipline-reloaded-jupyter du -sh /root/.zipline/data/sharadar
```

## Emergency: Disk Space Full

If you run out of disk space:

```bash
# 1. Keep only most recent ingestion (aggressive cleanup)
docker exec zipline-reloaded-jupyter zipline clean -b sharadar --keep-last 1

# 2. Remove all but newest for all bundles
for bundle in sharadar sharadar-tech sharadar-sp500 sharadar-all; do
    docker exec zipline-reloaded-jupyter zipline clean -b "$bundle" --keep-last 1 2>/dev/null || true
done

# 3. Check space saved
docker exec zipline-reloaded-jupyter df -h /root/.zipline
```

## Automation Script Reference

### update_and_clean_sharadar.sh

**Location**: `scripts/update_and_clean_sharadar.sh`

**Usage**:
```bash
# Default (sharadar bundle, keep last 2)
./scripts/update_and_clean_sharadar.sh

# Custom bundle
./scripts/update_and_clean_sharadar.sh sharadar-tech

# Custom bundle and keep count
./scripts/update_and_clean_sharadar.sh sharadar-tech 3
```

**Features**:
- Ingests new data
- Cleans old ingestions
- Shows disk usage
- Lists remaining ingestions
- Error handling

## Related Documentation

- [SHARADAR_GUIDE.md](SHARADAR_GUIDE.md) - Sharadar setup and usage
- [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md) - General data management
- [DOCKER_DISK_SPACE_FIX.md](../DOCKER_DISK_SPACE_FIX.md) - Docker disk space issues

## Quick Reference

```bash
# Check ingestions
zipline bundles

# Clean (keep last 2)
zipline clean -b sharadar --keep-last 2

# Update + Clean (automated script)
./scripts/update_and_clean_sharadar.sh

# Check disk usage
du -sh /root/.zipline/data/sharadar
```
