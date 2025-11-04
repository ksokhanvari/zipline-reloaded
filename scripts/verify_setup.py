#!/usr/bin/env python
"""
Verify Zipline setup is correct.

This script checks:
1. Zipline is installed
2. Extension.py exists and loads correctly
3. Sharadar bundle is registered
4. Data has been ingested
5. FlightLog server is reachable
"""

import sys
from pathlib import Path


def check_zipline_installed():
    """Check if Zipline is installed."""
    print("1. Checking Zipline installation...")
    try:
        import zipline
        print(f"   ✅ Zipline installed (version: {zipline.__version__})")
        return True
    except ImportError as e:
        print(f"   ❌ Zipline not installed: {e}")
        return False


def check_extension_file():
    """Check if extension.py exists."""
    print("\n2. Checking extension.py file...")
    extension_path = Path.home() / '.zipline' / 'extension.py'

    if extension_path.exists():
        print(f"   ✅ Extension file exists: {extension_path}")

        # Try to read and validate
        try:
            content = extension_path.read_text()
            if 'sharadar_bundle' in content:
                print("   ✅ Extension file contains sharadar_bundle")
                return True
            else:
                print("   ⚠️  Extension file exists but doesn't import sharadar_bundle")
                return False
        except Exception as e:
            print(f"   ⚠️  Could not read extension file: {e}")
            return False
    else:
        print(f"   ❌ Extension file not found at: {extension_path}")
        print("\n   To fix:")
        print("   1. Copy extension.py to ~/.zipline/")
        print("      cp extension.py ~/.zipline/extension.py")
        print("   2. Or rebuild Docker container")
        return False


def check_bundle_registration():
    """Check if Sharadar bundle is registered."""
    print("\n3. Checking bundle registration...")
    try:
        # First, manually load extension.py (like zipline does)
        import runpy
        extension_path = Path.home() / '.zipline' / 'extension.py'

        if extension_path.exists():
            try:
                print("   Loading extension.py...")
                runpy.run_path(str(extension_path))
            except Exception as e:
                print(f"   ⚠️  Error loading extension.py: {e}")

        from zipline.data.bundles import bundles

        registered = list(bundles.keys())
        print(f"   Registered bundles: {registered}")

        if 'sharadar' in registered:
            print("   ✅ Sharadar bundle is registered!")
            return True
        else:
            print("   ❌ Sharadar bundle NOT registered")
            print("\n   Troubleshooting:")
            print("   1. Check extension.py exists (see step 2)")
            print("   2. Check for import errors in extension.py")
            print("   3. Try: zipline bundles")
            return False
    except Exception as e:
        print(f"   ❌ Error checking bundles: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_sharadar_module():
    """Check if sharadar_bundle module can be imported."""
    print("\n4. Checking sharadar_bundle module...")
    try:
        from zipline.data.bundles.sharadar_bundle import sharadar_bundle
        print("   ✅ sharadar_bundle module loads successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Cannot import sharadar_bundle: {e}")
        print("\n   This means the sharadar_bundle.py file is missing or has errors.")
        print("   Check: src/zipline/data/bundles/sharadar_bundle.py")
        return False
    except Exception as e:
        print(f"   ❌ Error loading sharadar_bundle: {e}")
        return False


def check_data_ingested():
    """Check if Sharadar data has been ingested."""
    print("\n5. Checking ingested data...")
    data_path = Path.home() / '.zipline' / 'data' / 'sharadar'

    if data_path.exists():
        ingestions = list(data_path.glob('*'))
        if ingestions:
            print(f"   ✅ Found {len(ingestions)} ingestion(s)")
            latest = max(ingestions, key=lambda p: p.stat().st_mtime)
            print(f"   Latest: {latest.name}")

            # Check for key files
            assets_db = list(latest.glob('assets-*.db'))
            pricing = latest / 'daily_equity_pricing.bcolz'
            adjustments = latest / 'adjustments.sqlite'

            # Count what we have
            files_found = []
            if assets_db:
                files_found.append("assets DB")
            if pricing.exists():
                files_found.append("pricing")
            if adjustments.exists():
                files_found.append("adjustments")

            if len(files_found) >= 2:  # At least 2 of 3 key files
                print(f"   ✅ Data appears complete ({', '.join(files_found)})")
                return True
            elif len(files_found) > 0:
                print(f"   ⚠️  Partial data found ({', '.join(files_found)})")
                print("   Ingestion may have been interrupted")
                return False
            else:
                print("   ⚠️  No data files found in latest ingestion")
                return False
        else:
            print(f"   ⚠️  Data directory exists but no ingestions found")
            return False
    else:
        print("   ❌ No data directory found")
        print(f"   Expected: {data_path}")
        print("\n   To ingest data:")
        print("   zipline ingest -b sharadar")
        print("\n   Note: First ingestion takes 60-90 minutes")
        return False


def check_flightlog():
    """Check if FlightLog server is reachable."""
    print("\n6. Checking FlightLog server...")
    try:
        import socket

        # Try to connect to localhost:9020
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 9020))
        sock.close()

        if result == 0:
            print("   ✅ FlightLog server is reachable on localhost:9020")
            return True
        else:
            print("   ⚠️  FlightLog server not reachable on localhost:9020")
            print("\n   This is optional - backtests will work without it")
            print("   To start FlightLog:")
            print("   docker compose up -d flightlog")
            return False
    except Exception as e:
        print(f"   ⚠️  Could not check FlightLog: {e}")
        return False


def check_environment():
    """Check environment variables."""
    print("\n7. Checking environment variables...")
    import os

    api_key = os.getenv('NASDAQ_DATA_LINK_API_KEY')
    if api_key:
        masked = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
        print(f"   ✅ NASDAQ_DATA_LINK_API_KEY is set: {masked}")
        return True
    else:
        print("   ⚠️  NASDAQ_DATA_LINK_API_KEY not set")
        print("\n   Set in .env file or environment:")
        print("   export NASDAQ_DATA_LINK_API_KEY=your_key_here")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Zipline Setup Verification")
    print("=" * 60)

    results = []
    results.append(("Zipline installed", check_zipline_installed()))
    results.append(("Extension file", check_extension_file()))
    results.append(("Sharadar module", check_sharadar_module()))
    results.append(("Bundle registered", check_bundle_registration()))
    results.append(("Data ingested", check_data_ingested()))
    results.append(("FlightLog server", check_flightlog()))
    results.append(("Environment vars", check_environment()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nPassed: {passed_count}/{total_count}")

    if passed_count >= 4:  # At least core functionality working
        print("\n✅ Setup looks good! You can run backtests.")
    else:
        print("\n❌ Setup incomplete. Fix the issues above before running backtests.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
