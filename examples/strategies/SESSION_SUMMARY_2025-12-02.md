# Session Summary: Sharadar Dimension Configuration & Performance Delta Analysis

## Date: 2025-12-02

---

## Session Overview

This session continued work on porting and debugging the LS-ZR-ported.py long-short equity strategy from QuantRocket to Zipline-Reloaded. The focus was on understanding Sharadar data dimensions (ARQ vs MRQ), configuring the bundle for MRQ ingestion, and performing a comprehensive comparison to identify return discrepancies.

---

## Part 1: Sharadar Dimension Investigation

### User Question 1: "when we download sharadar are we downloading ARQ or what"

**Investigation Steps**:
1. Checked stored Sharadar data at `~/.zipline/data/sharadar/*/fundamentals/sf1.h5`
2. Analyzed the dimension column in the HDF5 file

**Findings**:
- **Current Data**: ARQ only (635,546 records)
- **Date Range**: 1998-03-31 to 2025-09-30
- **Columns**: 114 fundamental metrics including fcf, revenue, assets, etc.

**Dimension Types Explained**:
- **ARQ** (As-Reported Quarterly): Original reported values, point-in-time accurate, NO look-ahead bias
- **MRQ** (Most Recent Quarterly): Includes restated/updated values, HAS look-ahead bias
- **ART** (As-Reported Trailing 12 Months): Rolling TTM from ARQ
- **MRT** (Most Recent Trailing 12 Months): Rolling TTM from MRQ

**Conclusion**: User currently has ARQ data, which matches QuantRocket's usage: `Fundamentals.slice('ARQ', period_offset=0)`

---

## Part 2: Sharadar Bundle Ingestion File

### User Question 2: "when we use zipline ingest -b sharadar which file do we use to do teh download ?"

**Answer**: `/Users/kamran/Documents/Code/Docker/zipline-reloaded/src/zipline/data/bundles/sharadar_bundle.py`

**Sharadar Bundle Components**:
The bundle downloads 5 tables from NASDAQ Data Link:

1. **TICKERS** (`SHARADAR/TICKERS`): Security master list
2. **SEP** (`SHARADAR/SEP`): Stock prices (daily OHLCV)
3. **SFP** (`SHARADAR/SFP`): Institutional holdings
4. **SF1** (`SHARADAR/SF1`): Fundamentals (focus of our investigation)
5. **ACTIONS** (`SHARADAR/ACTIONS`): Corporate actions (splits, dividends)

**SF1 Fundamentals Configuration** (Lines 1420-1440):
- **Table Code**: `SHARADAR/SF1`
- **Stored In**: `fundamentals/sf1.h5`
- **Columns**: 114 fundamental metrics
- **Dimension Note**: Code comment at line 1426 says "filter to MRQ", but actual implementation was missing the filter

**Discrepancy Found**: Code comment claimed MRQ filtering, but no actual `dimension` filter was applied in the API calls.

---

## Part 3: Configure MRQ Dimension Download

### User Request: "I wanr to test MRQ. MAke sure we are downloading and ingesting MRQ. I will redownload"

**Changes Made**:

Modified `sharadar_bundle.py` to add MRQ dimension filters at API level for efficiency.

### Change 1: nasdaqdatalink API Path (Line 1064)

**Before**:
```python
elif table == 'SF1':
    # SF1 uses 'calendardate' for filtering (quarter end date)
    if start_date:
        filters['calendardate.gte'] = start_date
    if end_date:
        filters['calendardate.lte'] = end_date
```

**After**:
```python
elif table == 'SF1':
    # SF1 uses 'calendardate' for filtering (quarter end date)
    if start_date:
        filters['calendardate.gte'] = start_date
    if end_date:
        filters['calendardate.lte'] = end_date
    # Filter to MRQ dimension at API level for efficiency
    filters['dimension'] = 'MRQ'  # ✅ ADDED
```

### Change 2: Bulk Export API Path (Line 1264)

**Before**:
```python
elif table == 'SF1':
    # SF1 uses 'calendardate' for filtering
    if start_date:
        params['calendardate.gte'] = start_date
    if end_date:
        params['calendardate.lte'] = end_date
```

**After**:
```python
elif table == 'SF1':
    # SF1 uses 'calendardate' for filtering
    if start_date:
        params['calendardate.gte'] = start_date
    if end_date:
        params['calendardate.lte'] = end_date
    # Filter to MRQ dimension
    params['dimension'] = 'MRQ'  # ✅ ADDED
```

**Rationale**: Filtering at API level (instead of post-processing) is more efficient because:
- Reduces data transfer (only downloads MRQ records)
- Reduces processing time (no need to filter millions of records locally)
- Reduces storage space (smaller HDF5 file)

**Commit Details**:
- **Branch**: claude/continue-session-011-011CUzneiQ5d1tV3Y3r29tCA
- **Commit Hash**: f224fd84
- **Message**: "fix: Add dimension filter to Sharadar bundle for MRQ data"

**Next Steps for User**:
```bash
# 1. Remove old data
rm -rf ~/.zipline/data/sharadar/*

# 2. Re-ingest with MRQ filter
zipline ingest -b sharadar

# 3. Verify MRQ data
docker exec zipline-reloaded-jupyter python3 -c "
import pandas as pd
sf1 = pd.read_hdf('/root/.zipline/data/sharadar/*/fundamentals/sf1.h5', 'sf1')
print('Dimension counts:', sf1['dimension'].value_counts())
print('Expected: Only MRQ records')
"
```

---

## Part 4: Comprehensive Performance Delta Analysis

### User Request:
"I have done my best to find out teh return diffrence between these two versions. See if you see any key areas that could be causing a diffrence in returns..."

User provided full QuantRocket and Zipline-Reloaded strategy code for comparison.

### Analysis: 9 Key Differences Identified

---

### 🔴 CRITICAL ISSUE #1: Hardcoded Cash Return Override

**Location**: `LS-ZR-ported.py` Line 1627

**The Bug**:
```python
df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
df['cash_return'] = df['cash_return'].replace([np.inf, -np.inf], np.nan)
df['cash_return'] = 0.03  # ❌ CRITICAL: Overrides all calculations!
```

**Impact**:
- ALL stocks get the same cash_return value (0.03)
- Completely defeats the strategy's ranking logic
- Strategy becomes effectively random instead of FCF-based

**Why This Destroys Performance**:
The cash_return is the PRIMARY ranking metric:
```python
df['estrank'] = (
    (df['entval'].rank() * 2) +
    (df['cash_return'].rank()) +          # ← Uses hardcoded 0.03 for ALL stocks!
    df['eps_gr_mean'].rank() * weight +
    (other_factors / 3)
)
```

When all stocks have `cash_return = 0.03`, the `df['cash_return'].rank()` term becomes meaningless, fundamentally changing which stocks are selected for long/short positions.

**QuantRocket Version** (correct):
```python
df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
df['cash_return'] = df['cash_return'].replace([np.inf, -np.inf], np.nan)
# NO override - uses calculated values
```

---

### 🔴 CRITICAL ISSUE #2: Missing .shift() on Fundamentals

**Location**: `LS-ZR-ported.py` Lines 1032-1033

**The Issue**:
```python
# Zipline version - NO .shift()
columns['int'] = CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest
columns['entval'] = CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest
```

**QuantRocket Version**:
```python
# Has .shift() to prevent look-ahead bias
'int': CustomFundamentals.InterestExpense...latest.shift(),
'entval': CustomFundamentals.EnterpriseValue...latest.shift(),
```

**Impact**:
- **Look-ahead bias**: Using current-day fundamentals that wouldn't be available yet
- **Optimistic backtest**: Artificially improves returns by using future information
- **Production failure**: Real-world trading wouldn't have access to this data

**Why .shift() Matters**:
The `.shift()` method delays the data by one trading day, ensuring you only use information that was available at the time of the trading decision.

Without `.shift()`:
- Monday: Use Monday's fundamental data (not available until Tuesday)
- Result: Strategy performance looks better than reality

With `.shift()`:
- Monday: Use Friday's fundamental data (available on Monday)
- Result: Realistic, tradeable strategy

---

### 🔴 CRITICAL ISSUE #3: Different Sector Filtering

**Location**: `LS-ZR-ported.py` Line 1651

**Zipline Version**:
```python
# Excludes BOTH 'Financials' AND 'Financial Services'
df = df[~df['sector'].astype(str).replace('nan', 'Unknown')
        .isin(['Financial Services', 'Financials'])]
```

**QuantRocket Version**:
```python
# Excludes ONLY 'Financials'
df = df[df['sector'] != 'Financials']
```

**Impact**:
- Zipline excludes MORE stocks than QuantRocket
- Different universe → different rankings → different positions
- If excluded stocks are high-performing, Zipline misses gains

**Example Scenario**:
- Stock ABC has sector = 'Financial Services'
- QuantRocket: Includes ABC (might go long if high cash_return)
- Zipline: Excludes ABC (misses potential gain)

**Recommendation**: Change Zipline to match QuantRocket exactly:
```python
df = df[df['sector'] != 'Financials']
```

---

### 🔴 HIGH IMPACT #4: Missing Money Flow Factors

**Location**: `LS-ZR-ported.py` Lines 1084-1089

**The Issue**:
```python
# ALL COMMENTED OUT in Zipline:
# columns['mfi'] = money_flow_index
# columns['cmf'] = chaikin_money_flow
# columns['simple_mf'] = simple_money_flow
# columns['vw_mf'] = volume_weighted_mf
# columns['composite_mf'] = composite_money_flow
```

**QuantRocket Has**:
```python
'mfi': money_flow_index,
'cmf': chaikin_money_flow,
'simple_mf': simple_money_flow,
'vw_mf': volume_weighted_mf,
'composite_mf': composite_money_flow,
```

**Impact**:
- These factors may be used in universe filtering or screening
- Missing factors → different universe → different positions
- Unknown magnitude of impact (needs testing to quantify)

**What Are Money Flow Indicators?**
- **Money Flow Index (MFI)**: Volume-weighted RSI, identifies overbought/oversold
- **Chaikin Money Flow (CMF)**: Measures buying/selling pressure
- **Simple/Volume-Weighted/Composite**: Various combinations for robust signals

---

### 🔴 HIGH IMPACT #5: Missing Ichimoku Signal

**Location**: `LS-ZR-ported.py` Line ~1090

**The Issue**:
```python
# COMMENTED OUT in Zipline:
# columns['ich_signal'] = IchimokuSignal()
```

**QuantRocket Has**:
```python
'ich_signal': IchimokuSignal(),
```

**Impact**:
- Ichimoku Cloud is a trend-following indicator
- May affect universe filtering or position sizing
- Unknown magnitude of impact (needs testing)

**What Is Ichimoku?**
Japanese technical indicator that identifies:
- Trend direction (bullish/bearish)
- Support/resistance levels
- Momentum strength

---

### 🟠 MODERATE IMPACT #6: New Tradeability Checks

**Location**: `LS-ZR-ported.py` Lines 1462-1470

**New in Zipline** (not in QuantRocket):
```python
for sid_val, w in zip(longs, longs_mcw['cash_return'].values):
    w = abs(w)
    # Check tradeability before ordering
    if data.can_trade(sid_val):  # ← NEW CHECK
        order_target_percent(sid_val, w * port_weight_factor)
    else:
        print(f"  WARNING: Cannot trade {sid_val} - skipping")
```

**QuantRocket Version** (no check):
```python
for sid_val, w in zip(longs, longs_mcw['cash_return'].values):
    w = abs(w)
    order_target_percent(sid_val, w * port_weight_factor)  # Always trades
```

**Impact**:
- **QuantRocket**: Trades delisted stocks until forced liquidation
- **Zipline**: Skips delisted stocks earlier
- **Result**: Different position entry/exit timing

**Which Is Better?**
- **Zipline approach (with checks)**: More realistic for production
- **QuantRocket approach (no checks)**: May allow riding momentum until the end

**Recommendation**: Keep tradeability checks for production, but be aware this causes backtest differences.

---

### 🟢 LOW IMPACT #7: Beta Adjustment (Commented Out)

**Location**: `LS-ZR-ported.py` Lines 1687-1693

**The Issue**:
```python
# COMMENTED OUT in Zipline:
# if context.spyprice >= context.spyma80:
#     context.longs['cash_return'] /= winsorize(context.longs['beta60IWM'], limits=[0.005, 0.4])
# else:
#     context.longs['cash_return'] /= winsorize(context.longs['beta60SPY'], limits=[0.005, 0.1])
```

**QuantRocket Has This Active**:
```python
if context.spyprice >= context.spyma80:
    context.longs['cash_return'] /= winsorize(context.longs['beta60IWM'], limits=[0.005, 0.4])
else:
    context.longs['cash_return'] /= winsorize(context.longs['beta60SPY'], limits=[0.005, 0.1])
```

**Impact**:
- Adjusts position sizing based on market conditions (SPY above/below 80-day MA)
- Uses beta to IWM (small caps) or SPY (large caps)
- Affects risk exposure and diversification

**Why Low Impact?**
This is applied AFTER initial ranking, so it affects position sizing but not stock selection.

---

### 🟢 LOW IMPACT #8: Slope Adjustment (Commented Out)

**Location**: `LS-ZR-ported.py` Line 1698

**The Issue**:
```python
slopefact = 1+(context.longs['slope120'] / context.longs['slope120'].sum())
# COMMENTED OUT:
# context.longs['cash_return'] = context.longs['cash_return'] * slopefact**2
```

**QuantRocket Has This Active**:
```python
slopefact = 1+(context.longs['slope120'] / context.longs['slope120'].sum())
context.longs['cash_return'] = context.longs['cash_return'] * slopefact**2
```

**Impact**:
- Boosts cash_return for stocks with stronger price momentum (slope120)
- Affects final position sizing

**Why Low Impact?**
Applied after stock selection, affects sizing but not which stocks are chosen.

---

### 🟢 LOW IMPACT #9: FCF Data Source (NOW ALIGNED)

**Status**: ✅ **FIXED** in previous session

**Previous Issue** (Line 1031):
```python
columns['fcf'] = CustomFundamentals.FOCFExDividends_Discrete.latest  # LSEG data
```

**Current Code** (Line 1029):
```python
columns['fcf'] = SharadarFundamentals.fcf.latest  # ✅ Sharadar data
```

**Result**: Now matches QuantRocket's data source.

---

## Summary of Differences by Impact

### 🔴 Critical (Must Fix):
1. **Hardcoded cash_return = 0.03** (Line 1627) - Destroys ranking logic
2. **Missing .shift() on fundamentals** (Lines 1032-1033) - Look-ahead bias
3. **Different sector filtering** (Line 1651) - Excludes more stocks

### 🔴 High Impact (Should Fix):
4. **Missing money flow factors** (Lines 1084-1089) - May affect universe
5. **Missing Ichimoku signal** (Line ~1090) - May affect universe

### 🟠 Moderate Impact (Consider):
6. **Tradeability checks** (Lines 1462-1470) - Different exit timing

### 🟢 Low Impact (Optional):
7. **Beta adjustment** (Lines 1687-1693) - Affects sizing only
8. **Slope adjustment** (Line 1698) - Affects sizing only
9. **FCF data source** - ✅ Already fixed

---

## Recommended Action Plan

### Phase 1: Critical Fixes (Required for Meaningful Backtest)
1. **Remove hardcoded cash_return** (Line 1627)
   ```python
   # DELETE THIS LINE:
   df['cash_return'] = 0.03
   ```

2. **Add .shift() to fundamentals** (Lines 1032-1033)
   ```python
   columns['int'] = CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest.shift()
   columns['entval'] = CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest.shift()
   ```

3. **Align sector filtering** (Line 1651)
   ```python
   df = df[df['sector'] != 'Financials']  # Match QuantRocket
   ```

### Phase 2: High Impact Fixes (For Full Alignment)
4. Uncomment and implement money flow factors
5. Uncomment and implement Ichimoku signal

### Phase 3: Moderate Impact Decisions
6. Decide on tradeability checks (recommend: keep for production realism)

### Phase 4: Low Impact Enhancements (Optional)
7. Uncomment beta adjustment
8. Uncomment slope adjustment

---

## Files Modified in This Session

1. **`src/zipline/data/bundles/sharadar_bundle.py`**
   - Added MRQ dimension filter at line 1064 (nasdaqdatalink API)
   - Added MRQ dimension filter at line 1264 (bulk export API)

2. **`examples/strategies/MRQ_CONFIGURATION_SUMMARY.md`** (created)
   - Documents the MRQ configuration changes
   - Provides verification steps

---

## Testing Recommendations

### After MRQ Re-ingestion:

1. **Verify MRQ Data**:
   ```python
   import pandas as pd
   sf1 = pd.read_hdf('~/.zipline/data/sharadar/*/fundamentals/sf1.h5', 'sf1')
   print(sf1['dimension'].value_counts())  # Should show only MRQ
   ```

2. **Quick Sanity Check** (1-month backtest):
   ```python
   # In before_trading_start():
   print(f"Universe size: {len(df)}")
   print(f"Mean FCF: {df['fcf'].mean():,.0f}")
   print(f"Mean cash_return: {df['cash_return'].mean():.4f}")
   print(f"Cash_return std: {df['cash_return'].std():.4f}")
   ```

3. **Compare Top 10 Stocks** (same date in both systems):
   ```python
   print(df.nlargest(10, 'estrank')[['name', 'cash_return', 'estrank', 'sector']])
   ```

4. **Full Backtest Comparison**:
   - Run identical date ranges
   - Compare: Total return, Sharpe ratio, max drawdown, number of trades

---

## User Response

When asked "Would you like me to create a fix for these issues?", user responded:

> "no thats fine for now"

This indicates user wants to:
1. Test MRQ data first
2. Understand the differences before applying fixes
3. Potentially fix issues selectively based on testing results

---

## Next Steps (When User Is Ready)

1. **Re-ingest with MRQ**: `rm -rf ~/.zipline/data/sharadar/* && zipline ingest -b sharadar`
2. **Test current strategy** with MRQ data
3. **Apply Phase 1 critical fixes** (especially hardcoded cash_return)
4. **Re-test** and compare performance
5. **Iterate** through remaining fixes based on impact

---

## Key Takeaways

1. **Data Source Matters**: ARQ vs MRQ can create look-ahead bias
2. **Comments Can Mislead**: Code claimed MRQ but downloaded ARQ
3. **Hidden Bugs Are Deadly**: Hardcoded cash_return override completely changes strategy behavior
4. **Look-Ahead Bias Is Subtle**: Missing .shift() silently improves backtest but fails in production
5. **Incremental Testing**: Fix critical issues first, then measure impact of each subsequent change

---

## Git Commit History (This Session)

```bash
f224fd84 fix: Add dimension filter to Sharadar bundle for MRQ data
266237a4 docs: Update documentation to reflect proper custom_loader approach
5235d380 refactor: Replace monkey-patching with clean custom_loader approach
71b556eb fix: Set working directory to /notebooks for strategy execution
bd93b2e9 feat: Add Sharadar bundle registration for Jupyter notebook compatibility
```

---

## Files Ready for Commit

```bash
git add examples/strategies/SESSION_SUMMARY_2025-12-02.md
```

---

## End of Session Summary
