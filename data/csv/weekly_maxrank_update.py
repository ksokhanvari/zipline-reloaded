#!/usr/bin/env python3
"""
Weekly MAX-RANK forecast update -- run after the weekly production forecast.

    python weekly_maxrank_update.py            # auto-detects the newest production files
    python weekly_maxrank_update.py --dry-run  # report only, write nothing

What it produces
----------------
MLData/<range>_maxrank_forecast_only.csv  (Symbol, Date, predicted_return) -- load it
in place of <range>_forecast_only.csv. Per trading day:

    score = max( pct-rank 19-factor GBM , pct-rank production mlf1 )

then quantile-mapped onto a FROZEN reference of production's mlf1 distribution, so
the algo's mlf1_crzsoft ** 1.2 sizing sees the units it always has.

Validated: 2026 live (Feb-Sep) Sharpe 3.45 vs 2.10 for production alone at the same
return (+65.3% vs +65.9%), max DD -6.1% vs -12.4%; also ahead over 2018-26 and 2023-26.

How it stays stable (same principle as --preserve-existing)
-----------------------------------------------------------
* 19-factor GBM: retrained once per COMPLETED month-end (60-month window, PIT guard:
  training snapshots >= 135 days old so every 90-day target had realised). Scores are
  stored in the state dir and never recomputed.
* Output rows already written are never rewritten; each week only new dates are added.
* The quantile reference is frozen on first run, so old values never drift.
* The factor list is frozen (selected on 2010-2017, experiments/FACTOR_STUDY/).

State lives in experiments/MAXRANK_LIVE/ (panel, 19-factor scores, reference, last output).
"""
import argparse, re, subprocess, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
STATE = HERE / 'experiments' / 'MAXRANK_LIVE'
FACTORS = HERE / 'experiments' / 'FACTOR_STUDY' / 'selected_factors_2010_2017.csv'
SEED_FROM = pd.Timestamp('2015-01-01')          # first month the 60-month GBM window allows
GBM = dict(max_depth=6, min_samples_leaf=100, l2_regularization=0.2,
           learning_rate=0.05, max_iter=300, random_state=0)


def newest(pattern, exclude=None):
    """Newest MLData file by the END date in its YYYYMMDD_YYYYMMDD prefix."""
    best = None
    for f in (HERE / 'MLData').glob(pattern):
        if exclude and exclude in f.name:
            continue
        m = re.match(r'(\d{8})_(\d{8})', f.name)
        if m and (best is None or m.group(2) > best[0]):
            best = (m.group(2), f)
    if best is None:
        sys.exit(f'ABORT: no MLData/{pattern}')
    return best[1]


def build_panel(prod_parquet, panel):
    """Rebuild the PIT factor panel from the newest production parquet (once per input)."""
    stamp = STATE / 'panel.source'
    if panel.exists() and stamp.exists() and stamp.read_text() == str(prod_parquet):
        print(f'  panel up to date ({prod_parquet.name})')
        return
    print(f'  building factor panel from {prod_parquet.name} (~5-10 min) ...', flush=True)
    r = subprocess.run([sys.executable, str(HERE / 'build_factor_panel.py'), str(prod_parquet), str(panel), '--force'],
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'ABORT: panel build failed\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    stamp.write_text(str(prod_parquet))


def update_f19_scores(panel_path, scores_path):
    """Score every COMPLETED month-end not scored yet; never touch stored scores."""
    keep = pd.read_csv(FACTORS).iloc[:, 0].tolist()
    assert len(keep) == 19, keep
    P = pd.read_parquet(panel_path)
    P = P[P['Date'] >= '2010-01-01'].copy()
    P['sector'] = P['GICSSectorName'].astype('category').cat.codes
    P[keep] = P.groupby('Date')[keep].rank(pct=True)
    P['y'] = P.groupby('Date')['fwd90'].rank(pct=True)
    snaps = sorted(P['Date'].unique())
    # the latest snapshot is a completed month-end only if the data has moved into the next month
    last_data = pd.Timestamp(snaps[-1])
    complete = [pd.Timestamp(s) for s in snaps
                if pd.Timestamp(s).to_period('M') < last_data.to_period('M') and pd.Timestamp(s) >= SEED_FROM - pd.DateOffset(months=1)]
    S = pd.read_parquet(scores_path) if scores_path.exists() else pd.DataFrame(columns=['snap', 'Symbol', 'score'])
    done = set(pd.to_datetime(S['snap']).unique()) if len(S) else set()
    todo = [m for m in complete if m not in done]
    feats = keep + ['sector']
    new = []
    for m in todo:
        tr = P[(P['Date'] >= m - pd.DateOffset(months=60)) & (P['Date'] <= m - pd.DateOffset(days=135)) & P['fwd90'].notna()]
        te = P[P['Date'] == m]
        if len(tr) < 5000 or len(te) == 0:
            continue
        mdl = HistGradientBoostingRegressor(**GBM).fit(tr[feats], tr['y'])
        new.append(pd.DataFrame({'snap': m, 'Symbol': te['Symbol'].values, 'score': mdl.predict(te[feats])}))
        print(f'    19-factor model {m:%Y-%m}: trained {len(tr):,} rows -> scored {len(te):,}', flush=True)
    if new:
        S = pd.concat([S] + new, ignore_index=True)
    S['snap'] = pd.to_datetime(S['snap'])
    return S, new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prod-forecast', help='production *_forecast_only.csv (default: newest in MLData/)')
    ap.add_argument('--prod-parquet', help='production predictions parquet (default: newest in MLData/)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    fc = Path(a.prod_forecast) if a.prod_forecast else newest('*_forecast_only.csv', exclude='maxrank')
    pq = Path(a.prod_parquet) if a.prod_parquet else newest('*.parquet')
    out = HERE / 'MLData' / fc.name.replace('_forecast_only.csv', '_maxrank_forecast_only.csv')
    print(f'production forecast : {fc.name}\nproduction parquet  : {pq.name}\noutput              : MLData/{out.name}\n')

    # 1. factor panel + 19-factor scores (monthly, frozen)
    panel = STATE / 'panel_monthly.parquet'
    build_panel(pq, panel)
    scores_path = STATE / 'f19_scores.parquet'
    S, new = update_f19_scores(panel, scores_path)
    print(f'  19-factor months: {S["snap"].nunique()} stored, {len(new)} new')

    # 2. production forecast
    prod = pd.read_csv(fc)
    prod['Date'] = pd.to_datetime(prod['Date'])
    prod = prod.rename(columns={'predicted_return': 'mlf1'})

    # 3. frozen quantile reference (production's own distribution at first run)
    ref_path = STATE / 'quantile_reference.npy'
    qs = np.linspace(0, 1, 1001)
    if ref_path.exists():
        ref = np.load(ref_path)
    else:
        recent = prod[prod['Date'] >= prod['Date'].max() - pd.DateOffset(years=1)]['mlf1'].dropna()
        ref = np.quantile(recent.values, qs)
        if not a.dry_run:
            np.save(ref_path, ref)
        print(f'  quantile reference frozen from {len(recent):,} production values (last 12 months)')

    # 4. which dates are new? (previous output is preserved verbatim)
    prev_path = STATE / 'maxrank_latest.csv'
    prev = pd.read_csv(prev_path, parse_dates=['Date']) if prev_path.exists() else None
    last_done = prev['Date'].max() if prev is not None else pd.Timestamp('1900-01-01')
    first_new = max(last_done + pd.Timedelta(days=1), SEED_FROM)
    # skip stray weekend/holiday dates (a handful of rows) -- ranks on 3 names are noise
    n_day = prod.groupby('Date').size()
    real = set(n_day[n_day >= 1000].index)
    days = pd.DatetimeIndex(sorted(d for d in prod.loc[prod['Date'] >= first_new, 'Date'].unique() if d in real))
    if len(days) == 0:
        print('\nnothing new to add -- output unchanged'); return

    # 5. carry each month-end 19-factor score onto the trading days after it
    snaps = np.array(sorted(S['snap'].unique()), dtype='datetime64[ns]')
    pos = np.searchsorted(snaps, days.values, side='left') - 1            # latest snap strictly before the day
    days = days[pos >= 0]; pos = pos[pos >= 0]                             # (never wrap to the last snap)
    use = snaps[pos]
    D = pd.DataFrame({'Date': days, 'snap': use})
    G = D.merge(S, on='snap', how='left')[['Date', 'Symbol', 'score']].rename(columns={'score': 'gbm'})

    # 6. max-rank per day, mapped to the frozen reference
    B = prod[prod['Date'].isin(days)].merge(G, on=['Date', 'Symbol'], how='outer')
    ra = B.groupby('Date')['gbm'].rank(pct=True)
    rb = B.groupby('Date')['mlf1'].rank(pct=True)
    B['score'] = pd.concat([ra, rb], axis=1).max(axis=1)
    B['predicted_return'] = np.interp(B.groupby('Date')['score'].rank(pct=True), qs, ref)
    B = B.dropna(subset=['predicted_return'])
    add = B[['Symbol', 'Date', 'predicted_return']]

    full = pd.concat([prev, add], ignore_index=True) if prev is not None else add
    full = full.sort_values(['Date', 'Symbol']).drop_duplicates(['Date', 'Symbol'], keep='first')

    # 7. checks + report
    latest = B[B['Date'] == B['Date'].max()]
    both = latest['gbm'].notna() & latest['mlf1'].notna()
    print(f'\nadded {add["Date"].nunique()} trading days ({add["Date"].min():%Y-%m-%d} -> {add["Date"].max():%Y-%m-%d}), '
          f'{len(add):,} rows')
    print(f'latest day {latest["Date"].max():%Y-%m-%d}: {len(latest):,} names, {int(both.sum()):,} scored by both models')
    lag = (latest['Date'].max() - pd.Timestamp(use[-1])).days
    print(f'19-factor scores in use: month-end {pd.Timestamp(use[-1]):%Y-%m-%d} ({lag} days old)')
    if lag > 45:
        print('  WARNING: 19-factor scores are > 45 days old -- has a month-end been missed?')
    if both.sum() < 350:
        print('  WARNING: fewer than 350 names carry both scores -- check symbol coverage')
    print(f'predicted_return mean {add.predicted_return.mean():+.2f} std {add.predicted_return.std():.2f}')
    print('top 10 by max-rank today:', ', '.join(latest.nlargest(10, 'predicted_return')['Symbol']))

    if a.dry_run:
        print('\n--dry-run: nothing written'); return
    if new:
        S.to_parquet(scores_path, index=False)
    full.to_csv(prev_path, index=False)
    full.to_csv(out, index=False)
    print(f'\nwrote MLData/{out.name}  ({len(full):,} rows, {full.Date.min():%Y-%m-%d} -> {full.Date.max():%Y-%m-%d})')


if __name__ == '__main__':
    main()
