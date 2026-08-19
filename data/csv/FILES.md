# Directory Contents

Reference for every program, notebook, document, and data artifact in `data/csv/`.

## Python Programs

| File                                     | Description                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `forecast_returns_ml_walk_forward.py`    | **Production ML forecasting script.** Walk-forward (month-by-month) Histogram Gradient Boosting model that predicts forward stock returns with zero look-ahead bias. Handles 70–290+ features automatically, supports `--resume-file`, `--preserve-existing`, `--lookback-months`, `--log-features`, `--temporal-diagnostics`, `--pca`, and `--overwrite-months`. Current version: v3.3.27. |
| `forecast_returns_ml.py`                 | **Single-model exploration script.** Trains one HistGradientBoostingRegressor on all data at once (faster, but has look-ahead bias). Use only for research/exploration — never for backtesting or production. Also exposes shared CSV/Parquet I/O helpers used by the walk-forward script.                                                                                                  |
| `test_window_strategies.py`              | Walk-forward window comparison framework. Benchmarks expanding window vs. rolling windows (6/12/24/36 months) and emits per-month metrics, a summary text file, and matplotlib plots to identify the best lookback for a dataset.                                                                                                                                                           |
| `extract_symbol.py`                      | Utility to slice a prediction file (CSV or Parquet) by symbol. Supports single-symbol extraction, `ALLSYMBOLS` for the full dataset, and `--forecast-only` mode that keeps only `Symbol`, `Date`, `predicted_return` for lean trading-strategy consumption.                                                                                                                                 |
| `filter_top_500_marketcap.py`            | Filters a fundamentals CSV down to the top N companies (default 500) by `CompanyMarketCap` per date. Used to create large-cap-only training sets.                                                                                                                                                                                                                                           |
| `convert_csv_to_parquet.py`              | One-shot CSV → Parquet converter (Snappy-compressed) for 10× storage savings and 5–10× faster reads in the ML pipeline.                                                                                                                                                                                                                                                                     |
| `convert_parquet_to_csv.py`              | Reverse converter: Parquet → CSV for sharing or compatibility with non-PyArrow tooling.                                                                                                                                                                                                                                                                                                     |
| `verify_reproducibility.py`              | Compares two prediction files (Parquet or CSV) row-by-row and reports whether they are bit-identical. Used to verify the deterministic guarantees introduced in v3.2.2.                                                                                                                                                                                                                     |
| `OPTIMIZATION_MULTIINDEX_ALTERNATIVE.py` | **Reference snippet, not a runnable script.** Alternative MultiIndex-based implementation of the prediction-alignment step considered as an alternative to the v3.2.0 `merge()` approach. Kept for reference.                                                                                                                                                                               |

## Notebooks

| File | Description |
|------|-------------|
| `analyze_feature_importance.ipynb` | Interactive analysis of `logs/feature_importance_*.csv` files. Auto-detects the newest log and produces 10 visualizations: ranking-evolution heatmap, rank/importance trend lines, stability metrics, monthly top-5 bars, and category-level (Sector, Ratios, Momentum, etc.) breakdowns. Supports `FOCUS_ON_FUNDAMENTALS` mode to suppress price/market-cap features. |

## Documentation (this directory)

| File | Description |
|------|-------------|
| `README.md` | Primary, ~1.5k-line documentation for the ML forecasting system (v3.3.27). Covers architecture, features, walk-forward design, production deployment, safety guarantees, and full CLI reference. |
| `USAGE.md` | Command-line reference and recipe book. Includes detailed `--preserve-existing`, `--resume-file`, `--log-features`, and forecast-stability sections with copy-paste examples. |
| `CHANGELOG.md` | Per-version changelog from early releases through v3.3.27 with rationale, root-cause analyses, and migration notes. |
| `weekly_command_guide.md` | Production runbook for weekly/monthly updates: when to use `--preserve-existing` vs. `--overwrite-months N`, expected console output, and troubleshooting. |
| `FILES.md` | This file. |

## `Docs/` subdirectory

| File | Description |
|------|-------------|
| `Docs/INDEX.md` | Index into the deeper design docs; updated per release. |
| `Docs/FORECAST_STABILITY.md` | Technical deep-dive (~15 KB) on why predictions can drift across runs (universe changes, ranking shifts, data revisions) and how `--preserve-existing` and complete-month rankings eliminate it. |
| `Docs/LOOK_AHEAD_BIAS_AUDIT.md` | Auditor-style review of every place look-ahead bias could leak in, with the six protection layers and line-number citations into `forecast_returns_ml_walk_forward.py`. |
| `Docs/DETERMINISTIC_DESIGN.md` | Rationale for eliminating all RNG from the pipeline (v3.2.2): deterministic sampling, stable sorting, no global seeds. |
| `Docs/REPRODUCIBILITY_FIX.md` | Root-cause write-up of the 5.62 pp non-determinism bug fixed in v3.2.2 and the five distinct sources of non-determinism that were closed. |
| `Docs/MERGE_OPTIMIZATION_ANALYSIS.md` | Memory analysis of the v3.2.0 alignment optimization (merge-before-feature-engineering, 2 GB → 300 MB). |
| `Docs/SUMMARY_v3.2.2.md` | Short release summary for v3.2.2. |
| `Docs/ML_FORECASTING_VERSIONS.md` | Internal version-tracking notes. |

## `logs/` subdirectory

Runtime artifacts emitted by `forecast_returns_ml_walk_forward.py`:

| Pattern | Description |
|---------|-------------|
| `feature_importance_YYYYMMDD_HHMMSS.csv` | Per-month top-20 feature importances, written when `--log-features` is set. Consumed by `analyze_feature_importance.ipynb`. |
| `feature_stability_analysis*_YYYYMMDD_HHMMSS.csv` | Aggregate stability metrics (appearance count, avg/median rank, std, top-10/20 counts) exported from the notebook. |
| `forecast_ml_walk_forward_YYYYMMDD_HHMMSS.log` | Console-capture log of a training run (parameters, per-month timing, warnings, final stats). |

## `MLData/` subdirectory

Versioned prediction outputs from weekly walk-forward runs. Two files per run, both keyed by the input data's end date:

| Pattern | Description |
|---------|-------------|
| `20091231_YYYYMMDD_metadata_fmpdata_perdict-90-10-rolling-12.parquet` | Full prediction Parquet (all engineered features + `predicted_return`). Used as the `--resume-file` for the next week's run. Naming reflects 90-day target / 10-day forecast cadence / 12-month rolling window. |
| `20091231_YYYYMMDD_forecast_only.csv` | Lean CSV of `Symbol, Date, predicted_return` for direct trading-strategy consumption. Auto-exported alongside the Parquet. |

## Input Data CSVs (this directory root)

Three weekly snapshots are produced per input date (`YYYYMMDD_YYYYMMDD` = start–end of coverage):

| Pattern | Description |
|---------|-------------|
| `20091231_YYYYMMDD.csv` | Raw LSEG fundamentals + price (~3.1 GB, base 36 LSEG columns). |
| `20091231_YYYYMMDD_with_metadata.csv` | Same data enriched with 9 Sharadar metadata columns (~4.5 GB). Produced by `examples/lseg_fundamentals/add_sharadar_metadata_to_fundamentals.py`. |
| `20091231_YYYYMMDD_with_metadata_with_fmpdata.csv` | Full production input: LSEG + Sharadar metadata + FMP fundamentals (~6.5 GB, ~290 features). Default input for the walk-forward script. |

## Auxiliary CSVs

| File | Description |
|------|-------------|
| `vix_flag.csv` | VIX-based regime/risk flag time series consumed by the LS-ZR-ported strategy and downstream signal logic. |
| `bc_data.csv` | "BC1" signal time series used as a feature/overlay in strategies. |
| `bsi_data_7_30_20160801_20260324.csv` | Breadth/sentiment indicator dataset (7- and 30-day windows) covering 2016-08-01 → 2026-03-24. |
| `sentdata.csv` | Sentiment data feed (~25 MB). |
