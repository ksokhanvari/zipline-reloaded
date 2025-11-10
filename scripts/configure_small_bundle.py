#!/usr/bin/env python
"""
Configure a smaller Sharadar bundle for testing.
Use this if full ingestion is getting killed due to memory issues.
"""

from pathlib import Path
import shutil

extension_path = Path.home() / '.zipline' / 'extension.py'
backup_path = Path.home() / '.zipline' / 'extension.py.backup'

# Backup current extension.py
if extension_path.exists():
    shutil.copy(extension_path, backup_path)
    print(f"✓ Backed up current extension.py to {backup_path}")

# Create new extension.py with small ticker set
small_bundle_config = '''"""
Zipline Extensions - Small ticker set for testing.

This configuration uses only 20 popular tickers to test the setup
without requiring huge amounts of RAM.

For full dataset, see extension.py.backup
"""

from zipline.data.bundles import register
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

# Popular tech stocks
TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
]

# Popular ETFs
ETFS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
    'VOO', 'VEA', 'VWO', 'AGG', 'BND',
]

# Register small test bundle (20 tickers only)
register(
    'sharadar',
    sharadar_bundle(
        tickers=TECH_STOCKS + ETFS,  # Only 20 tickers
        incremental=True,
        include_funds=False,  # Skip funds to save memory
    ),
)

# Still register the full bundles for when you have more RAM
register(
    'sharadar-all',
    sharadar_bundle(
        tickers=None,  # ALL tickers (requires 16+ GB RAM)
        incremental=True,
        include_funds=True,
    ),
)

register(
    'sharadar-tech',
    sharadar_bundle(
        tickers=TECH_STOCKS,
        incremental=True,
        include_funds=False,
    ),
)

print("✓ Small Sharadar bundle registered (20 tickers)")
print("  - Tickers:", ', '.join(TECH_STOCKS + ETFS))
print("  - Use 'sharadar-all' for full dataset (requires 16+ GB RAM)")
'''

# Write new configuration
extension_path.write_text(small_bundle_config)

print(f"\n✓ Created new extension.py with small ticker set")
print(f"\nConfiguration:")
print("  - Bundle: sharadar")
print("  - Tickers: 20 popular stocks + ETFs")
print("  - RAM needed: ~2 GB (vs 16 GB for full dataset)")
print("  - Ingestion time: 5-10 minutes (vs 60-90 minutes)")
print("\nTo ingest:")
print("  zipline ingest -b sharadar")
print("\nTo restore full configuration:")
print(f"  cp {backup_path} {extension_path}")
print("  Then increase Docker memory to 16+ GB")
print("  Then: zipline ingest -b sharadar-all")
