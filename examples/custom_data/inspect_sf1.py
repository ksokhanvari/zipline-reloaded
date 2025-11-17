"""
Inspect the contents of the sf1.h5 file to see what data is available.
"""
import pandas as pd
import os

# Path to sf1.h5
bundle_path = os.path.expanduser('~/.zipline/data/sharadar/2025-11-17T08;19;43.398169')
sf1_path = os.path.join(bundle_path, 'fundamentals', 'sf1.h5')

print(f"Inspecting: {sf1_path}")
print(f"File exists: {os.path.exists(sf1_path)}")
print(f"File size: {os.path.getsize(sf1_path) / 1024 / 1024:.2f} MB\n")

# Open HDF5 file
with pd.HDFStore(sf1_path, 'r') as store:
    print("Keys in HDF5 file:")
    for key in store.keys():
        print(f"  {key}")

    # Read the main dataframe
    if '/sf1' in store.keys():
        df = store['sf1']
        print(f"\n✓ Found /sf1 table")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:20]}...")  # First 20 columns
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

        # Check if ROE column exists
        if 'roe' in df.columns:
            print(f"\n✓ 'roe' column exists")
            print(f"  Non-null values: {df['roe'].notna().sum()} / {len(df)} ({df['roe'].notna().sum()/len(df)*100:.1f}%)")

            # Filter for test date
            test_date = pd.Timestamp('2023-01-05')
            nearby = df[(df.index >= test_date - pd.Timedelta(days=90)) &
                       (df.index <= test_date + pd.Timedelta(days=1))]
            print(f"\n  Data near 2023-01-05 (±90 days): {len(nearby)} records")
            if len(nearby) > 0:
                print(f"  ROE non-null: {nearby['roe'].notna().sum()} / {len(nearby)}")
                print("\nSample records:")
                print(nearby[['roe']].head(10))
        else:
            print(f"\n⚠️  'roe' column NOT found")
            print(f"Available columns: {list(df.columns)}")
    else:
        print("\n⚠️  /sf1 table not found")
