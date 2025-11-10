#!/usr/bin/env python
"""
Debug why Sharadar bundle is not registering.
Run this inside the Docker container to diagnose the issue.
"""

import sys
import traceback
from pathlib import Path

print("=" * 70)
print("SHARADAR BUNDLE REGISTRATION DEBUGGER")
print("=" * 70)

# Step 1: Check extension.py exists
print("\n1. Checking extension.py file...")
extension_path = Path.home() / '.zipline' / 'extension.py'
if extension_path.exists():
    print(f"   ✅ Found: {extension_path}")
else:
    print(f"   ❌ NOT FOUND: {extension_path}")
    print("\n   SOLUTION: Copy extension.py to ~/.zipline/")
    print("   cp extension.py ~/.zipline/extension.py")
    sys.exit(1)

# Step 2: Check if sharadar_bundle.py exists
print("\n2. Checking sharadar_bundle.py module...")
try:
    import zipline
    zipline_path = Path(zipline.__file__).parent
    bundle_file = zipline_path / 'data' / 'bundles' / 'sharadar_bundle.py'

    if bundle_file.exists():
        print(f"   ✅ Found: {bundle_file}")
    else:
        print(f"   ❌ NOT FOUND: {bundle_file}")
        print("\n   SOLUTION: Reinstall zipline or check installation")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error checking: {e}")
    sys.exit(1)

# Step 3: Try to import sharadar_bundle
print("\n3. Testing import of sharadar_bundle module...")
try:
    from zipline.data.bundles.sharadar_bundle import sharadar_bundle
    print("   ✅ Import successful!")
except ImportError as e:
    print(f"   ❌ ImportError: {e}")
    print("\n   This usually means missing dependencies.")
    print("   Check if nasdaqdatalink is installed:")
    print("   pip install nasdaqdatalink")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Check what's registered before loading extension
print("\n4. Checking bundles BEFORE loading extension.py...")
from zipline.data.bundles import bundles
bundles_before = list(bundles.keys())
print(f"   Registered: {bundles_before}")

# Step 5: Manually execute extension.py
print("\n5. Manually executing extension.py...")
try:
    with open(extension_path, 'r') as f:
        extension_code = f.read()

    # Create a namespace for execution
    extension_namespace = {
        '__name__': 'extension',
        '__file__': str(extension_path),
    }

    # Execute the extension.py code
    exec(extension_code, extension_namespace)
    print("   ✅ Extension.py executed successfully!")

except Exception as e:
    print(f"   ❌ Error executing extension.py: {e}")
    print("\n   Full traceback:")
    traceback.print_exc()
    sys.exit(1)

# Step 6: Check what's registered AFTER loading extension
print("\n6. Checking bundles AFTER loading extension.py...")
bundles_after = list(bundles.keys())
print(f"   Registered: {bundles_after}")

new_bundles = set(bundles_after) - set(bundles_before)
if new_bundles:
    print(f"   ✅ New bundles registered: {list(new_bundles)}")
else:
    print("   ⚠️  No new bundles registered!")

# Step 7: Check specifically for sharadar
print("\n7. Checking for 'sharadar' bundle...")
if 'sharadar' in bundles:
    print("   ✅ Sharadar bundle IS registered!")
    print("\n   Available Sharadar bundles:")
    for name in bundles:
        if 'sharadar' in name.lower():
            print(f"     - {name}")
else:
    print("   ❌ Sharadar bundle NOT registered!")
    print("\n   This means the register() call in extension.py failed silently.")

# Step 8: Try registering manually
print("\n8. Attempting manual registration...")
try:
    from zipline.data.bundles import register
    from zipline.data.bundles.sharadar_bundle import sharadar_bundle

    # Try to register
    register(
        'sharadar-manual-test',
        sharadar_bundle(
            tickers=['AAPL'],  # Just one ticker for testing
            incremental=True,
            include_funds=False,
        ),
    )

    if 'sharadar-manual-test' in bundles:
        print("   ✅ Manual registration successful!")
        print("\n   This means the code works, but extension.py isn't being loaded by Zipline.")
    else:
        print("   ❌ Manual registration failed!")

except Exception as e:
    print(f"   ❌ Error during manual registration: {e}")
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

final_bundles = list(bundles.keys())
print(f"\nTotal registered bundles: {len(final_bundles)}")
print(f"Bundles: {final_bundles}")

if 'sharadar' in final_bundles or 'sharadar-manual-test' in final_bundles:
    print("\n✅ SUCCESS: Sharadar bundle can be registered!")
    print("\nTo use it:")
    print("  zipline ingest -b sharadar")
else:
    print("\n❌ FAILURE: Sharadar bundle registration not working!")
    print("\nPossible issues:")
    print("  1. Extension.py is not being loaded by Zipline automatically")
    print("  2. There's a hidden import error")
    print("  3. nasdaqdatalink is not installed")

print("\n" + "=" * 70)
