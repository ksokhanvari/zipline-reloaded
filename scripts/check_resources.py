#!/usr/bin/env python
"""
Check Docker container resource limits and usage.
Run this to diagnose why the ingestion is being killed.
"""

import os
import subprocess
import sys

print("=" * 70)
print("DOCKER RESOURCE DIAGNOSTICS")
print("=" * 70)

# Check memory info
print("\n1. Container Memory Info:")
try:
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
        limit_bytes = int(f.read().strip())
        limit_gb = limit_bytes / (1024**3)
        print(f"   Memory Limit: {limit_gb:.2f} GB")

        if limit_gb < 8:
            print(f"   ⚠️  WARNING: Memory limit is LOW ({limit_gb:.2f} GB)")
            print("   Sharadar full ingestion needs at least 8-16 GB RAM")
        elif limit_gb < 16:
            print(f"   ⚠️  CAUTION: Memory limit is {limit_gb:.2f} GB")
            print("   Sharadar ingestion may struggle. Recommended: 16+ GB")
        else:
            print(f"   ✅ Memory limit looks good ({limit_gb:.2f} GB)")
except Exception as e:
    print(f"   ⚠️  Could not read memory limit: {e}")

try:
    with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
        usage_bytes = int(f.read().strip())
        usage_gb = usage_bytes / (1024**3)
        print(f"   Current Usage: {usage_gb:.2f} GB")
except Exception as e:
    print(f"   ⚠️  Could not read memory usage: {e}")

# Check available memory
print("\n2. System Memory Info:")
try:
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()

    for line in meminfo.split('\n'):
        if 'MemTotal' in line or 'MemAvailable' in line or 'MemFree' in line:
            parts = line.split()
            if len(parts) >= 2:
                value_kb = int(parts[1])
                value_gb = value_kb / (1024**2)
                print(f"   {parts[0]:<20} {value_gb:.2f} GB")
except Exception as e:
    print(f"   ⚠️  Could not read /proc/meminfo: {e}")

# Check disk space
print("\n3. Disk Space:")
try:
    result = subprocess.run(['df', '-h', '/root/.zipline'],
                          capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"   ⚠️  Could not check disk space: {e}")

# Check Sharadar data size
print("\n4. Existing Sharadar Data:")
try:
    result = subprocess.run(['du', '-sh', '/root/.zipline/data/sharadar'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   Current size: {result.stdout.strip()}")
    else:
        print("   No existing data found")
except Exception as e:
    print(f"   Could not check: {e}")

# Check for OOM kills in dmesg
print("\n5. Recent OOM Kills:")
try:
    result = subprocess.run(['dmesg', '-T'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        oom_lines = [line for line in result.stdout.split('\n')
                     if 'Out of memory' in line or 'oom-kill' in line.lower()]
        if oom_lines:
            print("   ⚠️  OOM kills detected:")
            for line in oom_lines[-5:]:  # Last 5
                print(f"   {line}")
        else:
            print("   ✅ No OOM kills found in dmesg")
    else:
        print("   ⚠️  Cannot read dmesg (requires privileges)")
except Exception as e:
    print(f"   ⚠️  Could not check dmesg: {e}")

# Recommendations
print("\n" + "=" * 70)
print("DIAGNOSIS & RECOMMENDATIONS")
print("=" * 70)

print("""
The "Killed" message during Step 4 is caused by the OOM (Out of Memory) killer.

Sharadar full ingestion requires:
  - Download: ~3-5 GB (ZIP files)
  - Processing: ~8-16 GB RAM (in-memory pandas operations)
  - Final storage: ~10-15 GB (bcolz/SQLite)

SOLUTIONS:

1. INCREASE DOCKER DESKTOP MEMORY (Recommended)

   On your Mac:
   a. Open Docker Desktop
   b. Settings → Resources → Memory
   c. Increase from current to 16 GB minimum (32 GB ideal)
   d. Apply & Restart
   e. Retry: zipline ingest -b sharadar

2. USE SMALLER TICKER SET (Faster alternative)

   Instead of all tickers, use a subset:

   # Edit ~/.zipline/extension.py and change to:
   register(
       'sharadar',
       sharadar_bundle(
           tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                    'NVDA', 'TSLA', 'NFLX', 'SPY', 'QQQ'],
           incremental=True,
           include_funds=False,
       ),
   )

   Then: zipline ingest -b sharadar

   This will:
   - Complete in 5-10 minutes
   - Use only 1-2 GB RAM
   - Let you test strategies immediately

3. USE PRE-CONFIGURED SMALL BUNDLE

   Use the tech stocks bundle:
   zipline ingest -b sharadar-tech

   This includes 15 major tech stocks and completes quickly.

4. PROCESS IN BATCHES (Advanced)

   Modify the bundle to download in date ranges or ticker batches.
   This requires code changes to sharadar_bundle.py.

RECOMMENDATION:
Start with option #2 or #3 (smaller ticker set) to test your setup,
then increase Docker memory and do a full ingestion later.
""")

print("=" * 70)
