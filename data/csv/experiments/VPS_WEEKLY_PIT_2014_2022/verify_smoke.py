"""Compare the VPS smoke run to the Mac reference. Pass = identical predictions."""
import sys, numpy as np, pandas as pd
ref = pd.read_csv('reference/smoke_reference_forecast_only.csv', parse_dates=['Date'])
vps = pd.read_parquet('smoke/smoke_vps.parquet', columns=['Symbol', 'Date', 'predicted_return'])
vps = vps[vps['predicted_return'].notna()]; vps['Date'] = pd.to_datetime(vps['Date'])
m = ref.merge(vps, on=['Symbol', 'Date'], suffixes=('_ref', '_vps'))
d = (m['predicted_return_ref'] - m['predicted_return_vps']).abs()
print(f"reference rows {len(ref):,} | vps rows {len(vps):,} | matched {len(m):,}")
print(f"max abs diff {d.max():.3e} | mean abs diff {d.mean():.3e}")
ok = len(m) == len(ref) == len(vps) and d.max() < 1e-6
if not ok:
    rc = m.groupby('Date').apply(lambda t: t['predicted_return_ref'].corr(t['predicted_return_vps'], method='spearman')).min()
    print(f"worst per-day rank corr {rc:.6f}")
print('SMOKE TEST: PASS -- environment reproduces the Mac exactly' if ok else
      'SMOKE TEST: FAIL -- do NOT run the full job; send smoke/smoke_vps.log back')
sys.exit(0 if ok else 1)
