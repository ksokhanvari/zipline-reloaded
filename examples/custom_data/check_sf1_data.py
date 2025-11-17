"""
Check which columns in sf1.h5 actually have data.
"""
import pandas as pd
import os

bundle_path = os.path.expanduser('~/.zipline/data/sharadar/2025-11-17T08;19;43.398169')
sf1_path = os.path.join(bundle_path, 'fundamentals', 'sf1.h5')

print(f"Checking data availability in sf1.h5\n")

with pd.HDFStore(sf1_path, 'r') as store:
    df = store['sf1']

    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}\n")

    # Check data availability for each column
    print("Data availability by column:")
    print("=" * 80)

    availability = []
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        availability.append((col, non_null, pct))

    # Sort by availability
    availability.sort(key=lambda x: x[2], reverse=True)

    print(f"{'Column':<25} {'Non-null':<15} {'Percentage':>10}")
    print("-" * 80)
    for col, non_null, pct in availability[:30]:  # Top 30
        print(f"{col:<25} {non_null:<15,} {pct:>9.1f}%")

    print("\n" + "=" * 80)
    print("Key fields we care about:")
    print("-" * 80)
    for col in ['roe', 'fcf', 'marketcap', 'pe', 'revenue', 'netinc', 'equity']:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"{col:<25} {non_null:<15,} {pct:>9.1f}%")

    # Show a sample record
    print("\n" + "=" * 80)
    print("Sample record:")
    print("-" * 80)
    print(df.iloc[0])
