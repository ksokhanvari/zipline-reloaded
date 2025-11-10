#!/usr/bin/env python
"""
List all available bundles and their ingestions.
"""

import os
from zipline.data.bundles import register, bundles, ingestions_for_bundle
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

# Register sharadar bundle
if 'sharadar' not in bundles:
    register('sharadar', sharadar_bundle(tickers=None, incremental=True, include_funds=True))
    print("✓ Registered sharadar bundle\n")

print("="*70)
print("Available Bundles and Ingestions")
print("="*70)

for bundle_name in sorted(bundles.keys()):
    if bundle_name.startswith('.'):
        continue

    print(f"\n{bundle_name}:")

    try:
        bundle_ingestions = list(ingestions_for_bundle(bundle_name))
        if bundle_ingestions:
            for timestamp in bundle_ingestions:
                print(f"  - {timestamp}")
        else:
            print(f"  <no ingestions>")
    except Exception as e:
        print(f"  Error: {e}")

print(f"\n{'='*70}")
print("Note: If a bundle shows '<no ingestions>', run:")
print("  zipline ingest -b <bundle_name>")
print("="*70)
