# FCF Data Source Fix - Sharadar vs LSEG

## Date: 2025-12-01

## Summary
Fixed critical data source mismatch in LS-ZR-ported.py strategy. The strategy was using LSEG FCF data instead of Sharadar FCF, causing performance discrepancies with the QuantRocket reference implementation.

---

## Problem Identified

### Root Cause
**Line 1031** of LS-ZR-ported.py was using LSEG FCF data from CustomFundamentals instead of Sharadar FCF.

### Why This Matters
The cash return formula is:
```python
cash_return = (fcf - interest_expense) / enterprise_value
```

This is the **PRIMARY ranking metric** in the strategy. Using different FCF data sources produces:
- Different cash return values
- Different stock rankings
- Different position selection
- **Different backtest performance**

---

## The Fix

### Before (Line 1031):
```python
# Use Sharadar FCF
#columns['fcf'] = SharadarFundamentals.fcf.latest
# Commented out - using Sharadar FCF instead
columns['fcf'] = CustomFundamentals.FOCFExDividends_Discrete.latest  # ❌ LSEG data
```

**Issue**: Comment claimed to use Sharadar FCF, but code used LSEG FCF.

### After (Line 1029):
```python
# Use Sharadar FCF to match QuantRocket implementation
columns['fcf'] = SharadarFundamentals.fcf.latest  # ✅ Sharadar data
# OLD: Was using LSEG FCF (different data source)
# columns['fcf'] = CustomFundamentals.FOCFExDividends_Discrete.latest
```

**Result**: Now correctly uses Sharadar FCF, matching QuantRocket implementation.

---

## Verification Steps

### 1. Confirmed SharadarFundamentals Availability
```python
from zipline.pipeline.data.sharadar import SharadarFundamentals
```

**Result**: ✅ Import successful (already present at line 65)

### 2. Confirmed fcf Column Exists
Checked available columns in SharadarFundamentals:
- Total: 68 columns
- FCF-related: `fcf`, `fcfps`, `cashneq`

**Result**: ✅ `fcf` column available

### 3. Test Script Execution
Created `/tmp/check_sharadar_fcf.py` to verify:
- SharadarFundamentals can be imported
- fcf column is accessible
- Data can be loaded via Pipeline

**Result**: ✅ All checks passed

---

## Impact on Strategy

### Critical Ranking Formula
The strategy uses this ranking in normal market conditions:
```python
df['estrank'] = (
    (df['entval'].rank() * 2) +           # Enterprise value rank
    (df['cash_return'].rank()) +          # ← FCF-based cash return rank (CRITICAL)
    df['eps_gr_mean'].rank() * weight +   # Earnings growth rank
    (other_factors / 3)
)
```

**cash_return** is directly impacted by FCF data source, affecting:
1. Individual stock cash return values
2. Relative rankings between stocks
3. Long position selection
4. Short position selection

### Expected Performance Impact
With Sharadar FCF (now matching QuantRocket):
- ✅ Same fundamental data as QuantRocket reference
- ✅ Same cash return calculations
- ✅ Same stock rankings (assuming other factors equal)
- ✅ Aligned backtest performance

---

## Remaining Differences to Address

Even with FCF aligned, other differences still exist:

### 1. Sector Filtering
**QuantRocket**:
```python
df = df[df['sector'] != 'Financials']
```

**Zipline-Reloaded**:
```python
df = df[~df['sector'].astype(str).replace('nan', 'Unknown')
        .isin(['Financial Services', 'Financials'])]
```

**Impact**: Zipline excludes more stocks (both 'Financials' AND 'Financial Services')

**Recommendation**: Change Zipline to match QuantRocket exactly:
```python
df = df[df['sector'] != 'Financials']
```

### 2. Tradeability Checks
**QuantRocket**: No `data.can_trade()` checks

**Zipline-Reloaded**: Added tradeability checks before all order placements
```python
if data.can_trade(sid_val):
    order_target_percent(sid_val, w * port_weight_factor)
```

**Impact**:
- QuantRocket trades delisted stocks until forced liquidation
- Zipline skips delisted stocks earlier
- Different position entry/exit timing

**Recommendation**: Keep tradeability checks for production (more realistic), but be aware this differs from QuantRocket.

### 3. Missing Technical Indicators
**QuantRocket has** (Zipline missing):
- Money Flow Index
- Chaikin Money Flow
- Simple Money Flow
- Volume-Weighted Money Flow
- Composite Money Flow
- Ichimoku Signal

**Impact**: Unknown - these aren't used in ranking formula, but may affect universe filtering

---

## Testing Recommendations

### 1. Quick Sanity Check
Run backtest for a short period (e.g., 1 month) and verify:
```python
# In before_trading_start():
print(f"Mean FCF: {df['fcf'].mean():,.0f}")
print(f"Mean cash_return: {df['cash_return'].mean():.4f}")
print(f"Top 10 by cash_return:")
print(df.nlargest(10, 'cash_return')[['name', 'fcf', 'int', 'entval', 'cash_return']])
```

### 2. Compare Universes
For the same date, compare:
- Universe size
- Top 10 stocks by estrank
- Top 10 stocks by cash_return

If these differ significantly from QuantRocket, investigate other factors.

### 3. Full Backtest Comparison
Run identical date ranges:
- Same start date
- Same end date
- Compare final returns
- Compare Sharpe ratio
- Compare max drawdown

---

## Files Modified

1. **examples/strategies/LS-ZR-ported.py**
   - Line 1029: Changed FCF source from LSEG to Sharadar
   - Line 65: SharadarFundamentals import already present

2. **examples/strategies/PERFORMANCE_DELTA_CHECKLIST.md**
   - Updated to reflect FCF fix status
   - Marked Issue #1 as FIXED

3. **examples/strategies/FCF_DATA_SOURCE_FIX.md** (this file)
   - New documentation of fix

---

## Git Commit Ready

Files ready to commit:
```bash
git add examples/strategies/LS-ZR-ported.py
git add examples/strategies/PERFORMANCE_DELTA_CHECKLIST.md
git add examples/strategies/FCF_DATA_SOURCE_FIX.md
```

Suggested commit message:
```
fix: Use Sharadar FCF instead of LSEG FCF to match QuantRocket

- Changed line 1029 to use SharadarFundamentals.fcf.latest
- Previously used CustomFundamentals.FOCFExDividends_Discrete (LSEG)
- FCF is critical for cash_return calculation and stock ranking
- Aligns with QuantRocket reference implementation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Key Takeaways

1. **Always verify data sources match** when porting strategies between platforms
2. **FCF definitions vary** between data providers (Sharadar, LSEG, etc.)
3. **Comments can be misleading** - verify actual code behavior
4. **Data source mismatches cascade** - different FCF → different rankings → different positions → different performance
5. **Test data availability** before assuming imports will work in production

---

## Next Steps

1. ✅ FCF data source fixed
2. ⏳ Consider aligning sector filtering (remove 'Financial Services' exclusion)
3. ⏳ Decide whether to keep tradeability checks (recommended: keep)
4. ⏳ Optionally add missing technical indicators
5. ⏳ Run full backtest comparison with QuantRocket
6. ⏳ Commit changes to git
