#!/usr/bin/env python
"""
Remap SIDs in the custom fundamentals database to match Sharadar bundle SIDs.

This script updates the Sid column in the LSEG fundamentals database to use
the same SID values as the Sharadar bundle, ensuring both data sources can
be used together in Pipeline.
"""

import sqlite3
from pathlib import Path
from zipline.data.bundles import register, load as load_bundle
from zipline.data.bundles.sharadar_bundle import sharadar_bundle

# Register bundle
register('sharadar', sharadar_bundle())

def remap_sids(db_path: str, dry_run: bool = False):
    """
    Remap SIDs in the fundamentals database to match Sharadar bundle.

    Parameters
    ----------
    db_path : str
        Path to the fundamentals.sqlite database
    dry_run : bool
        If True, only show what would be changed without making changes
    """
    print("=" * 80)
    print("REMAPPING FUNDAMENTALS DATABASE SIDS")
    print("=" * 80)
    print()

    # Load Sharadar bundle
    print("Loading Sharadar bundle...")
    bundle_data = load_bundle('sharadar')
    asset_finder = bundle_data.asset_finder
    print("✓ Bundle loaded")
    print()

    # Connect to database
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print("✓ Connected")
    print()

    # Get all unique symbols and their current SIDs
    print("Finding symbols in database...")
    cursor.execute("SELECT DISTINCT Symbol, Sid FROM Price WHERE Symbol IS NOT NULL ORDER BY Symbol")
    symbols_with_sids = cursor.fetchall()
    print(f"✓ Found {len(symbols_with_sids)} unique symbols")
    print()

    # Build mapping: old_sid -> new_sid
    sid_mapping = {}
    unmapped_symbols = []

    print("Building SID mapping...")
    for symbol, old_sid in symbols_with_sids:
        try:
            # Look up the asset in Sharadar bundle
            asset = asset_finder.lookup_symbol(symbol, as_of_date=None)
            if asset is not None:
                new_sid = asset.sid
                sid_mapping[old_sid] = new_sid
                if old_sid != new_sid:
                    print(f"  {symbol}: {old_sid} -> {new_sid}")
            else:
                unmapped_symbols.append(symbol)
        except Exception as e:
            unmapped_symbols.append(symbol)
            print(f"  ⚠ {symbol}: Could not map (error: {e})")

    print()
    print(f"✓ Mapped {len(sid_mapping)} SIDs")
    if unmapped_symbols:
        print(f"⚠ Could not map {len(unmapped_symbols)} symbols: {', '.join(unmapped_symbols[:10])}")
        if len(unmapped_symbols) > 10:
            print(f"  ... and {len(unmapped_symbols) - 10} more")
    print()

    if not sid_mapping:
        print("No SIDs to remap!")
        conn.close()
        return

    # Perform the remapping
    if dry_run:
        print("DRY RUN - No changes will be made")
        print()
        print("Would update SIDs for:")
        for old_sid, new_sid in sorted(sid_mapping.items())[:20]:
            cursor.execute("SELECT COUNT(*) FROM Price WHERE Sid = ?", (old_sid,))
            count = cursor.fetchone()[0]
            print(f"  SID {old_sid} -> {new_sid} ({count} rows)")
        if len(sid_mapping) > 20:
            print(f"  ... and {len(sid_mapping) - 20} more SIDs")
    else:
        print("Updating database...")

        # Update SIDs one by one
        updated_count = 0
        for old_sid, new_sid in sid_mapping.items():
            cursor.execute("UPDATE Price SET Sid = ? WHERE Sid = ?", (new_sid, old_sid))
            updated_count += cursor.rowcount

        conn.commit()
        print(f"✓ Updated {updated_count} rows")
        print()

        # Verify the changes
        print("Verifying changes...")
        cursor.execute("SELECT DISTINCT Symbol, Sid FROM Price WHERE Symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA') ORDER BY Symbol")
        verification = cursor.fetchall()

        print("Sample SIDs after remapping:")
        for symbol, sid in verification:
            print(f"  {symbol}: {sid}")
        print()

    conn.close()

    print("=" * 80)
    if dry_run:
        print("DRY RUN COMPLETE - No changes made")
    else:
        print("REMAPPING COMPLETE")
    print("=" * 80)
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Remap SIDs in fundamentals database')
    parser.add_argument(
        '--db-path',
        default=str(Path.home() / '.zipline' / 'data' / 'custom' / 'fundamentals.sqlite'),
        help='Path to fundamentals.sqlite database'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making changes'
    )

    args = parser.parse_args()

    remap_sids(args.db_path, args.dry_run)
