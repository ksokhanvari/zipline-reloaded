"""Pre-flight for a long run. Catches the class of error that wasted 6.5 hours.

Runs the REAL pipeline on a small slice and asserts:
  1. no feature is the target, or target bookkeeping, or derivable from it
  2. --technicals-only actually keeps technicals and drops fundamentals
  3. features are not mostly NaN / constant (nothing to learn from)
  4. one full train+predict period completes and produces sane predictions
Takes ~2 minutes instead of 6.5 hours.
"""
import importlib.util, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

SRC = 'experiments/20230101_20260915_SUBSTITUTED.parquet'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'technicals'

sp = importlib.util.spec_from_file_location("wk", "forecast_returns_ml_weekly_pit.py")
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)

print(f"loading a slice of {SRC} ...")
df = pd.read_parquet(SRC, engine='pyarrow')
df['date'] = pd.to_datetime(df['date'])
# keep ~9 months and a manageable symbol count, enough for one real period
keep = df['symbol'].drop_duplicates().head(400)
df = df[df['symbol'].isin(keep) & (df['date'] >= '2023-01-01') & (df['date'] <= '2023-09-30')]
# Use the SCRIPT's own PascalCase normalisation, not a hand-rolled copy --
# reimplementing it is how this preflight first failed.
src = open('forecast_returns_ml_weekly_pit.py').read()
blk = src[src.index('        known_columns = {'):]
blk = blk[:blk.index('        }') + 9]
ns = {}
exec('known_columns = ' + blk.split('known_columns = ', 1)[1].strip(), ns)
KNOWN = ns['known_columns']
df = df.rename(columns={c: KNOWN.get(c.lower(), c) for c in df.columns})
print(f"  normalised via the script's own map ({len(KNOWN)} known columns)")
print(f"  {len(df):,} rows x {df.shape[1]} cols, {df['Symbol'].nunique()} symbols\n")

f = m.ReturnForecaster(forecast_days=1, target_return_days=20, no_lag=True,
                       walk_frequency='weekly',
                       technicals_only=(MODE == 'technicals'),
                       fundamental_only=(MODE == 'fundamental'))
t = f.create_target(df.copy())
t = f.engineer_features(t)
X, y, names, w, vi = f.prepare_features(t)
print(f"\nfeatures: {len(names)}   valid target rows: {int(np.sum(vi)):,}")

fail = []

# 1. LEAK: no feature may be the label or its bookkeeping
bad_exact = {'forward_return', 'forward_return_raw', 'target_date'}
leak = [n for n in names if n in bad_exact or str(n).startswith('_cmp_')]
print(f"\n1. label/bookkeeping in features : {leak if leak else 'NONE'}")
if leak: fail.append("label columns leaked into features")

# 1b. LEAK: no feature may correlate ~perfectly with the target
yv = pd.Series(y)[vi]
sus = []
for n in names:
    col = pd.Series(X[n].values)[vi] if hasattr(X, 'columns') else None
    if col is None or col.notna().sum() < 500: continue
    c = abs(col.corr(yv))
    if c == c and c > 0.95: sus.append((n, c))
print(f"1b. |corr(feature, target)| > 0.95 : {sus if sus else 'NONE'}")
if sus: fail.append(f"suspiciously predictive features: {sus[:3]}")

# 2. MODE: the filter did what it claims
if MODE == 'technicals':
    tech_pref = ('t_', 'return_', 'volatility_', 'momentum_', 'volume_',
                 'RefPriceClose', 'RefVolume', 'CompanyMarketCap')
    sector = {'GICSSectorName', 'sharadar_sicsector', 'sharadar_sicindustry'}
    off = [n for n in names if not str(n).startswith(tech_pref) and n not in sector]
    print(f"2. non-technical features kept   : {len(off)} {off[:6] if off else ''}")
    if off: fail.append(f"{len(off)} fundamentals survived --technicals-only")
    if len(names) < 30: fail.append(f"only {len(names)} features left - too few to train")

# 3. QUALITY: features must carry information
Xv = X[names] if hasattr(X, 'columns') else pd.DataFrame(X, columns=names)
nan_frac = Xv.isna().mean()
const = [n for n in names if Xv[n].nunique(dropna=True) <= 1]
mostly_nan = nan_frac[nan_frac > 0.9].index.tolist()
print(f"3. constant features             : {len(const)}")
print(f"   >90% NaN features             : {len(mostly_nan)}")
if len(const) + len(mostly_nan) > len(names) * 0.5:
    fail.append("over half the features are constant or empty")

# 4. SMOKE: one real train+predict
try:
    idx = np.where(vi)[0]
    cut = int(len(idx) * 0.7)
    tr, pr = idx[:cut], idx[cut:]
    f.train(Xv.iloc[tr], y[tr], sample_weight=w[tr])
    p = f.model.predict(Xv.iloc[pr])
    print(f"4. train+predict smoke test      : OK  "
          f"({len(tr):,} train -> {len(pr):,} pred, "
          f"mean {np.nanmean(p):+.3f}, std {np.nanstd(p):.3f})")
    if not np.isfinite(p).any(): fail.append("predictions are all non-finite")
    if np.nanstd(p) < 1e-9: fail.append("predictions are constant")
except Exception as e:
    print(f"4. train+predict smoke test      : FAILED  {type(e).__name__}: {e}")
    fail.append(f"pipeline raised {type(e).__name__}")

print("\n" + "=" * 62)
if fail:
    print("PREFLIGHT FAILED - do not launch:")
    for x in fail: print(f"   - {x}")
    sys.exit(1)
print("PREFLIGHT PASSED - safe to launch the long run")
