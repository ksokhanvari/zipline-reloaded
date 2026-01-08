# ML Forecasting Documentation Index

This directory contains detailed technical documentation for the ML-based stock return forecasting system.

## 📚 Main Documentation

Start here for general usage:
- **[../README.md](../README.md)** - Main documentation, quick start, usage examples
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

### v3.2.0 - Checkpoint & Resume (Earlier 2026-01-07)

- **[CHECKPOINT_RESUME_GUIDE.md](CHECKPOINT_RESUME_GUIDE.md)** - Original checkpoint/resume workflow (now simplified)

### Version Tracking

- **[ML_FORECASTING_VERSIONS.md](ML_FORECASTING_VERSIONS.md)** - Complete version history and changelog

### Session Summaries

- **[SESSION_SUMMARY_2024-12-17.md](SESSION_SUMMARY_2024-12-17.md)** - Session notes from Dec 17, 2024
- **[SESSION_SUMMARY_2024-12-16.md](SESSION_SUMMARY_2024-12-16.md)** - Session notes from Dec 16, 2024

---

## 🧪 Testing Documentation

- **[TESTING_FRAMEWORK_README.md](TESTING_FRAMEWORK_README.md)** - Testing framework and validation procedures

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
- [CHECKPOINT_RESUME_GUIDE.md](CHECKPOINT_RESUME_GUIDE.md) - Resume workflow (legacy)

---

## 🏗️ Document Structure

```
data/csv/
├── README.md                          # Main documentation (START HERE)
├── CHANGELOG.md                       # Version history
│
└── Docs/                              # Detailed documentation
    ├── INDEX.md                       # This file
    │
    ├── DETERMINISTIC_DESIGN.md        # Core design philosophy
    ├── REPRODUCIBILITY_FIX.md         # Technical analysis
    ├── MERGE_OPTIMIZATION_ANALYSIS.md # Performance details
    ├── SUMMARY_v3.2.2.md              # Latest version summary
    │
    ├── CHECKPOINT_RESUME_GUIDE.md     # Legacy checkpoint system
    ├── ML_FORECASTING_VERSIONS.md     # Version tracking
    ├── TESTING_FRAMEWORK_README.md    # Testing documentation
    │
    ├── SESSION_SUMMARY_2024-12-17.md  # Session notes
    └── SESSION_SUMMARY_2024-12-16.md  # Session notes
```

---

## 🚀 Recommended Reading Order

### For New Users:
1. [../README.md](../README.md) - Overview and quick start
2. [../CHANGELOG.md](../CHANGELOG.md) - What's new in latest version
3. [SUMMARY_v3.2.2.md](SUMMARY_v3.2.2.md) - Current version details

### For Understanding Design:
1. [DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md) - Why fully deterministic
2. [REPRODUCIBILITY_FIX.md](REPRODUCIBILITY_FIX.md) - What problems were fixed
3. [MERGE_OPTIMIZATION_ANALYSIS.md](MERGE_OPTIMIZATION_ANALYSIS.md) - How memory was optimized

### For Production Deployment:
1. [../README.md](../README.md) - Basic usage
2. [REPRODUCIBILITY_FIX.md](REPRODUCIBILITY_FIX.md) - Verification checklist
3. [DETERMINISTIC_DESIGN.md](DETERMINISTIC_DESIGN.md) - Production considerations

---

**Last Updated:** 2026-01-07
**Current Version:** v3.2.2
