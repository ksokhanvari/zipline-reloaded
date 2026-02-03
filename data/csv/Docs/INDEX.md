# ML Forecasting Documentation Index

This directory contains detailed technical documentation for the ML-based stock return forecasting system.

## 📚 Main Documentation

Start here for general usage:
- **[../USAGE.md](../USAGE.md)** - Complete command-line reference and usage examples ⭐ **START HERE**
- **[../weekly_command_guide.md](../weekly_command_guide.md)** - When to use --preserve-existing vs --overwrite-months ⭐ **PRODUCTION GUIDE**
- **[../README.md](../README.md)** - Feature overview, what's new, production guide
- **[../CHANGELOG.md](../CHANGELOG.md)** - Complete version history and release notes

---

## 🔧 Technical Deep Dives

### v3.3.14 - Forecast Stability (2026-01-20)

**Core Design Document:**
- **[FORECAST_STABILITY.md](FORECAST_STABILITY.md)** - Why predictions change & --preserve-existing flag ⭐ **PRODUCTION CRITICAL**

**Key Topics:**
- 🔒 Why historical predictions change when adding new data
- 📊 Three sources of variance: Stock universe, cross-sectional rankings, data revisions
- ✅ How --preserve-existing freezes historical forecasts
- 🎯 Production best practices for stable backtests
- ❓ FAQ: When to use --overwrite-months vs --preserve-existing

---

### v3.2.2 - Deterministic Design (2026-01-07)

**Core Design Documents:**
- **[DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md)** - Why zero randomness is better (fully deterministic sampling)
- **[REPRODUCIBILITY_FIX.md](REPRODUCIBILITY_FIX.md)** - Root cause analysis of 5 non-determinism sources
- **[MERGE_OPTIMIZATION_ANALYSIS.md](MERGE_OPTIMIZATION_ANALYSIS.md)** - Memory optimization (85% reduction)
- **[SUMMARY_v3.2.2.md](SUMMARY_v3.2.2.md)** - Version 3.2.2 summary and upgrade guide

**Key Topics:**
- ✅ Eliminated ALL random number generation
- ✅ Fixed 5.62% prediction variance between identical runs
- ✅ Merge before feature engineering (2 GB → 300 MB temp df)
- ✅ Stable sorting and duplicate detection

---

## 📖 Historical Documentation

### Version Tracking

- **[ML_FORECASTING_VERSIONS.md](ML_FORECASTING_VERSIONS.md)** - Comparison of two forecasting scripts (walk-forward vs single-model)

---

## 📋 Quick Navigation

### By Topic

**Production Stability** (⭐ CRITICAL):
- [FORECAST_STABILITY.md](FORECAST_STABILITY.md) - Why predictions change & how to freeze them
- [LOOK_AHEAD_BIAS_AUDIT.md](LOOK_AHEAD_BIAS_AUDIT.md) - Production safety verification

**Reproducibility:**
- [REPRODUCIBILITY_FIX.md](REPRODUCIBILITY_FIX.md) - Complete analysis
- [DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md) - Design rationale

**Performance:**
- [MERGE_OPTIMIZATION_ANALYSIS.md](MERGE_OPTIMIZATION_ANALYSIS.md) - Memory optimization
- [DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md) - Speed improvements

**Versions:**
- [../CHANGELOG.md](../CHANGELOG.md) - All versions
- [SUMMARY_v3.2.2.md](SUMMARY_v3.2.2.md) - Latest version
- [ML_FORECASTING_VERSIONS.md](ML_FORECASTING_VERSIONS.md) - Version tracking

**Usage:**
- [../USAGE.md](../USAGE.md) - Command-line reference
- [../README.md](../README.md) - Main usage guide
- [../CHANGELOG.md](../CHANGELOG.md) - Version-specific usage notes

---

## 🏗️ Document Structure

```
data/csv/
├── USAGE.md                           # Command-line reference (START HERE) ⭐
├── README.md                          # Feature overview and what's new
├── CHANGELOG.md                       # Version history
│
├── Docs/                              # Detailed documentation
│   ├── INDEX.md                       # This file
│   │
│   ├── FORECAST_STABILITY.md          # ⭐ Production stability (v3.3.14)
│   ├── LOOK_AHEAD_BIAS_AUDIT.md       # Production safety verification
│   │
│   ├── DETERMINISTIC_DESIGN.md        # Core design philosophy (v3.2.2)
│   ├── REPRODUCIBILITY_FIX.md         # Technical analysis (v3.2.2)
│   ├── MERGE_OPTIMIZATION_ANALYSIS.md # Performance details (v3.2.2)
│   ├── SUMMARY_v3.2.2.md              # Version 3.2.2 summary
│   │
│   └── ML_FORECASTING_VERSIONS.md     # Script comparison guide
│
├── logs/                              # Log files (auto-generated)
└── convert_*.py                       # Utility scripts
```

---

## 🚀 Recommended Reading Order

### For New Users:
1. [../USAGE.md](../USAGE.md) - How to run the script (all flags and examples) ⭐ **START HERE**
2. [../README.md](../README.md) - Overview and feature highlights
3. [../CHANGELOG.md](../CHANGELOG.md) - What's new in latest version

### For Understanding Design:
1. [FORECAST_STABILITY.md](FORECAST_STABILITY.md) - Why predictions change & how to prevent it ⭐
2. [DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md) - Why fully deterministic
3. [REPRODUCIBILITY_FIX.md](REPRODUCIBILITY_FIX.md) - What problems were fixed
4. [MERGE_OPTIMIZATION_ANALYSIS.md](MERGE_OPTIMIZATION_ANALYSIS.md) - How memory was optimized

### For Production Deployment:
1. [../USAGE.md](../USAGE.md) - Command-line reference and common workflows
2. [FORECAST_STABILITY.md](FORECAST_STABILITY.md) - **CRITICAL**: Freeze historical forecasts ⭐
3. [LOOK_AHEAD_BIAS_AUDIT.md](LOOK_AHEAD_BIAS_AUDIT.md) - Safety verification
4. [../README.md](../README.md) - Feature overview and production guide
5. [REPRODUCIBILITY_FIX.md](REPRODUCIBILITY_FIX.md) - Verification checklist
6. [DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md) - Production considerations

---

**Last Updated:** 2026-01-20
**Current Version:** v3.3.24
