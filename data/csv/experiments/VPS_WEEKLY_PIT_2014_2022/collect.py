"""Stitch the yearly chunks into one forecast-only file + a sanity report."""
import glob, pandas as pd
parts = []
for f in sorted(glob.glob('out/weekly_pit_*.parquet')):
    y = int(f.split('_')[-1].split('.')[0])
    x = pd.read_parquet(f, columns=['Symbol', 'Date', 'predicted_return'])
    x['Date'] = pd.to_datetime(x['Date'])
    x = x[x['predicted_return'].notna() & (x['Date'].dt.year == y)]   # each chunk contributes its own year only
    parts.append(x); print(f'{y}: {len(x):>9,} rows  {x.Date.nunique():>4} dates  {x.Symbol.nunique():,} symbols')
D = pd.concat(parts).sort_values(['Date', 'Symbol'])
assert not D.duplicated(['Date', 'Symbol']).any()
D.to_csv('WEEKLY_PIT_2014_2022_forecast_only.csv', index=False)
print(f'\nWEEKLY_PIT_2014_2022_forecast_only.csv: {len(D):,} rows, {D.Date.min():%Y-%m-%d} -> {D.Date.max():%Y-%m-%d}')
print('ship this file back; it joins the existing 2023+ weekly file (PIT_BASE_WEEKLY_90d_from2023_LSEG).')
