# Deprecated Strategy Files

This directory contains strategy files and examples that have been superseded by newer implementations or are no longer actively maintained.

## Deprecated Files

### Test Files
- **test_date_issue.py** - Test file for debugging date issues
- **test_sharadar_filters_simple.py** - Simple test for Sharadar filters (superseded by production implementations)
- **test_sharadar_filters.ipynb** - Notebook test for Sharadar filters

### Old Strategy Implementations
- **LS-ZR-ported-FILTERED.py** - Early filtered version of LS-ZR strategy
  - **Superseded by**: `LS-ZR-ported.py` (main implementation)
  - **Reason**: This was an intermediate version during development

- **LS-QR-code.py** - Reference QuantRocket long-short strategy code
  - **Purpose**: Original QuantRocket implementation for comparison
  - **Note**: Kept for reference to understand differences between QuantRocket and Zipline implementations

### Tutorial Notebooks
- **03_moving_average_crossover.ipynb** - Basic moving average crossover strategy tutorial
- **06_strategy_comparison.ipynb** - Strategy comparison tutorial
  - **Status**: Old tutorial notebooks from original Zipline examples
  - **Recommended**: Use `getting_started/` notebooks and current strategy examples instead

### Log Files
- **startegy-log.txt** - Old strategy execution log (typo in filename)
  - **Note**: Superseded by FlightLog system for real-time monitoring

## Current Strategy Files

For current, actively maintained strategy files, see:
- **Main Strategies Directory**: `examples/strategies/`
- **Active Strategies**:
  - `LS-ZR-ported.py` - Main long-short equity strategy
  - `fcf_yield_strategy.py` - Free cash flow yield strategy
  - `top_enterprise_value.py` - Enterprise value based strategy
  - `momentum_strategy_with_flightlog.py` - Momentum with real-time logging
  - `strategy_top5_roe.py` - Top 5 ROE strategy

## Session Summaries and Fix Documentation

For historical session summaries and fix documentation, see:
- `examples/deprecated/session_summaries/` - Consolidated historical documentation

---

**Last Updated**: 2025-12-10
**Deprecation Date**: 2025-12-10
