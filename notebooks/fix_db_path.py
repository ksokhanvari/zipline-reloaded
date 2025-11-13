#!/usr/bin/env python3
"""
Rename the database file to match CustomSQLiteLoader's expected format.
"""
from pathlib import Path
import shutil

db_dir = Path.home() / '.zipline' / 'data' / 'custom'
old_path = db_dir / 'refe-fundamentals.sqlite'
new_path = db_dir / 'quant_refe-fundamentals.sqlite'

if old_path.exists():
    print(f"Renaming database file...")
    print(f"  From: {old_path}")
    print(f"  To: {new_path}")
    shutil.move(str(old_path), str(new_path))
    print(f"✓ Database renamed successfully!")
    print(f"\nNew database location: {new_path}")
    print(f"Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print(f"Database not found at {old_path}")
    if new_path.exists():
        print(f"Database already exists at correct location: {new_path}")
    else:
        print("No database found. Please run cells 1-18 to create the database.")
