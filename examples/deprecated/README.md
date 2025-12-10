# Deprecated Examples and Files

This directory contains files and examples that have been superseded by newer implementations or are no longer actively maintained. Files are organized by category and preserved for historical reference.

## Directory Structure

```
deprecated/
├── README.md                    # This file
├── strategies/                  # Deprecated strategy files
│   ├── README.md
│   ├── test_*.py               # Test files
│   ├── LS-ZR-ported-FILTERED.py
│   ├── LS-QR-code.py
│   └── *.ipynb                 # Old tutorial notebooks
├── session_summaries/          # Historical session documentation
│   ├── README.md
│   ├── *_FIX.md                # Fix documentation
│   ├── *_SUMMARY.md            # Implementation summaries
│   └── SESSION_SUMMARY_*.md    # Session logs
├── lseg_fundamentals/          # Deprecated LSEG files
│   ├── README.md
│   └── FIX_INSTRUCTIONS.md
└── multi_source_data/          # Deprecated multi-source files
    ├── README.md
    └── debug_multi_source.py
```

## Purpose

This directory serves several purposes:

1. **Historical Reference**: Preserve the evolution of the codebase
2. **Learning Resource**: Show what approaches were tried and why they were superseded
3. **Avoid Regression**: Document known issues and their solutions
4. **Clean Active Directories**: Keep main directories focused on current, maintained code

## Deprecation Policy

Files are moved to this directory when they:

1. **Have been superseded** by better implementations
2. **Are test/debug files** no longer needed
3. **Are temporary fixes** that have been incorporated into permanent solutions
4. **Are outdated tutorials** replaced by newer examples
5. **Are session logs** preserved for historical reference only

## What's NOT Deprecated

The following types of files are NOT moved here:

- ✅ **Active strategy implementations** (e.g., `LS-ZR-ported.py`)
- ✅ **Current examples and tutorials** (e.g., `getting_started/`)
- ✅ **Utility scripts in active use** (e.g., `utils/backtest_helpers.py`)
- ✅ **Current documentation** (e.g., main READMEs, active guides)
- ✅ **Production data workflows** (e.g., LSEG fundamentals loading)

## Migration History

### December 10, 2025 - Initial Organization
**Moved from** `examples/strategies/`:
- test_date_issue.py
- test_sharadar_filters_simple.py
- test_sharadar_filters.ipynb
- startegy-log.txt (typo in name)
- LS-ZR-ported-FILTERED.py (intermediate version)
- LS-QR-code.py (QuantRocket reference)
- 03_moving_average_crossover.ipynb (old tutorial)
- 06_strategy_comparison.ipynb (old tutorial)
- 8 session summary/fix documentation files

**Moved from** `examples/lseg_fundamentals/`:
- FIX_INSTRUCTIONS.md (temporary fix incorporated)

**Moved from** `examples/multi_source_data/`:
- debug_multi_source.py (debug file)

## Quick Links to Active Code

### Strategies
- **Long-Short Strategy**: `examples/strategies/LS-ZR-ported.py`
- **FCF Yield**: `examples/strategies/fcf_yield_strategy.py`
- **Momentum**: `examples/strategies/momentum_strategy_with_flightlog.py`
- **Documentation**: `examples/strategies/SHARADAR_FILTERS_README.md`

### Fundamentals
- **LSEG Workflow**: `examples/lseg_fundamentals/README.md`
- **Sharadar Guide**: `docs/SHARADAR_FUNDAMENTALS_GUIDE.md`

### Multi-Source Data
- **Guide**: `docs/MULTI_SOURCE_DATA.md`
- **Examples**: `examples/multi_source_data/`

### Getting Started
- **Quickstart**: `examples/getting_started/01_quickstart_sharadar_flightlog.ipynb`
- **Buy & Hold**: `examples/getting_started/02_buy_and_hold_strategy.ipynb`

## Cleanup Note

This directory was created as part of a comprehensive repository cleanup on December 10, 2025, to:
- Organize historical files
- Clean up main directories
- Improve discoverability of active code
- Preserve institutional knowledge

---

**Last Updated**: 2025-12-10
**Maintained By**: Hidden Point Capital
**Project**: Zipline-Reloaded
