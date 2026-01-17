# ML Forecasting Documentation Index

This directory contains detailed technical documentation for the ML-based stock return forecasting system.

## 📚 Main Documentation

Start here for general usage:
- **[../USAGE.md](../USAGE.md)** - Complete command-line reference and usage examples ⭐ **START HERE**
- **[../README.md](../README.md)** - Feature overview, what's new, production guide
- **[../CHANGELOG.md](../CHANGELOG.md)** - Complete version history and release notes

---

## 🔧 Technical Deep Dives

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
- [../README.md](../README.md) - Main usage guide
- [../CHANGELOG.md](../CHANGELOG.md) - Version-specific usage notes

---

## 🏗️ Document Structure

```
data/csv/
├── USAGE.md                           # Command-line reference (START HERE) ⭐
├── README.md                          # Feature overview and what's new
├── CHANGELOG.md                       # Version history
├── LOOK_AHEAD_BIAS_AUDIT.md           # Production safety verification
│
├── Docs/                              # Detailed documentation
│   ├── INDEX.md                       # This file
│   │
│   ├── DETERMINISTIC_DESIGN.md        # Core design philosophy (v3.2.2)
│   ├── REPRODUCIBILITY_FIX.md         # Technical analysis (v3.2.2)
│   ├── MERGE_OPTIMIZATION_ANALYSIS.md # Performance details (v3.2.2)
│   ├── SUMMARY_v3.2.2.md              # Latest version summary
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
1. [DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md) - Why fully deterministic
2. [REPRODUCIBILITY_FIX.md](REPRODUCIBILITY_FIX.md) - What problems were fixed
3. [MERGE_OPTIMIZATION_ANALYSIS.md](MERGE_OPTIMIZATION_ANALYSIS.md) - How memory was optimized

### For Production Deployment:
1. [../USAGE.md](../USAGE.md) - Command-line reference and common workflows
2. [../README.md](../README.md) - Feature overview and production guide
3. [../LOOK_AHEAD_BIAS_AUDIT.md](../LOOK_AHEAD_BIAS_AUDIT.md) - Safety verification
4. [REPRODUCIBILITY_FIX.md](REPRODUCIBILITY_FIX.md) - Verification checklist
5. [DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md) - Production considerations

---

**Last Updated:** 2026-01-16
**Current Version:** v3.3.12
