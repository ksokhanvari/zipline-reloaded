# QuantRocket vs Zipline-Reloaded LS Strategy Comparison

## Date: 2025-12-01

## Overview
Comparison of the original QuantRocket implementation vs our Zipline-Reloaded port to identify differences that may explain backtest result discrepancies.

---

## 1. UNIVERSE SELECTION DIFFERENCES

### QuantRocket Version:
```python
# Uses top 150 by market cap + static IBM
tradable_filter = (
    CustomFundamentals.CompanyMarketCap.latest.shift().top(UNIVERSE_SIZE) |
    StaticAssets([symbol('IBM')])
)
```

### Zipline-Reloaded Version:
```python
# Same universe definition
tradable_filter = (
    CustomFundamentals.CompanyMarketCap.latest.shift().top(UNIVERSE_SIZE) |
    StaticAssets([symbol('IBM')])
)
```

**Status**: ✅ **IDENTICAL**

---

## 2. PIPELINE FACTOR DIFFERENCES

### QuantRocket Has Additional Factors:

#### A. **Money Flow Factors** (MISSING in Zipline version)
```python
# QuantRocket has these factors:
'mfi': money_flow_index,                    # Money Flow Index
'cmf': chaikin_money_flow,                  # Chaikin Money Flow
'simple_mf': simple_money_flow,             # Simple Money Flow
'vw_mf': volume_weighted_mf,                # Volume-Weighted Money Flow
'composite_mf': composite_money_flow,       # Composite of all money flows
```

**Impact**: Money flow factors measure buying/selling pressure. Their absence could significantly affect stock selection.

#### B. **Ichimoku Cloud Signal** (MISSING in Zipline version)
```python
'ich_signal': IchimokuSignal(),  # Technical trend indicator
```

**Impact**: Ichimoku provides additional trend confirmation. Missing this could affect momentum stock selection.

#### C. **Weighted Alpha** (PRESENT in both)
```python
'walpha': WeightedAlpha(),  # Weighted excess return vs SPY
```

**Status**: ✅ **PRESENT** in both versions

---

## 3. CASH RETURN CALCULATION DIFFERENCES

### QuantRocket Version:
```python
# Uses FCF from Sharadar fundamentals
'fcf': s_fundamentals.FCF.latest,

# Cash return calculation
df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
```

### Zipline-Reloaded Version:
```python
# Uses LSEG FOCFExDividends_Discrete
'fcf': CustomFundamentals.FOCFExDividends_Discrete.latest.shift(),

# Same cash return formula
df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
```

**Status**: ⚠️ **DIFFERENT DATA SOURCE**
- QuantRocket: Sharadar FCF (Free Cash Flow)
- Zipline: LSEG FOCFExDividends_Discrete

**Impact**: **CRITICAL** - Different FCF definitions can lead to significantly different cash return values and rankings.

---

## 4. SECTOR FILTERING DIFFERENCES

### QuantRocket Version:
```python
# Filters out Financials sector
df = df[df['sector'] != 'Financials']

# Optional sector selection (commented out):
# df = df[(df['sector'] == 'Communication Services') |
#         (df['sector'] == 'Information Technology') |
#         (df['sector'] == 'Industrials') |
#         (df['sector'] == 'Consumer Discretionary')]
```

### Zipline-Reloaded Version:
```python
# Filters out BOTH Financials AND Financial Services
df = df[~df['sector'].astype(str).replace('nan', 'Unknown')
        .isin(['Financial Services', 'Financials'])]
```

**Status**: ⚠️ **SLIGHTLY DIFFERENT**
- QuantRocket: Removes only 'Financials'
- Zipline: Removes both 'Financials' AND 'Financial Services'

**Impact**: MODERATE - May exclude additional stocks in Zipline version.

---

## 5. RANKING FORMULA DIFFERENCES

### QuantRocket Version:
```python
if context.spyprice <= context.spyma80:
    # Defensive ranking
    df['estrank'] = df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1)
else:
    # Normal ranking
    df['estrank'] = (
        (df['entval'].rank() * 2) +
        (df['cash_return'].rank()) +
        df['eps_gr_mean'].rank() * (4 if context.season == 1 else 1) +
        (df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1) / 3)
    )
```

### Zipline-Reloaded Version:
```python
if context.spyprice <= context.spyma80:
    # Same defensive ranking
    df['estrank'] = df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1)
else:
    # Normal ranking
    df['estrank'] = (
        (df['entval'].rank() * 2) +
        (df['cash_return'].rank()) +
        df['eps_gr_mean'].rank() * (4 if context.season == 1 else 1) +
        (df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1) / 3)
    )
```

**Status**: ✅ **IDENTICAL**

---

## 6. ENHANCED RISK CONTROLS

### QuantRocket Has Additional Functions (ALL COMMENTED OUT):

```python
# ENHANCED: Apply risk controls
#dfl = enhanced_stock_screening(dfl)
#dfl = apply_market_cap_diversification(context, dfl)
#dfl = apply_volatility_adjustment(context, dfl, data)

# ENHANCED: Apply sector and correlation constraints
#longs_mcw, shorts_mcw = apply_sector_constraints(context, longs_mcw, shorts_mcw)
#longs_mcw = apply_correlation_constraints(context, data, longs_mcw)
```

**Status**: ⚠️ **ALL DISABLED** in QuantRocket version (commented out)

**Impact**: NONE currently - these are not active in either version.

---

## 7. PORTFOLIO CONSTRUCTION DIFFERENCES

### Position Sizing - QuantRocket:
```python
port_weight_factor = 0.90  # 90% to LS selection
spy_weight_factor = 0.10   # 10% to SPY

# Execute long orders
total_wl = 0
for sid, w in zip(longs, longs_mcw['cash_return'].values):
    w = abs(w)
    algo.order_target_percent(sid, w * port_weight_factor)
    total_wl = total_wl + w

# SPY allocation
algo.order_target_percent(context.spysym, total_wl * spy_weight_factor)
```

### Position Sizing - Zipline-Reloaded:
```python
# Should be the same - need to verify your version
```

**Status**: ❓ **NEED TO VERIFY** in your Zipline version

---

## 8. CRITICAL DIFFERENCES SUMMARY

### 🔴 **HIGH IMPACT DIFFERENCES**:

1. **FCF Data Source** (CRITICAL)
   - QuantRocket: `s_fundamentals.FCF.latest` (Sharadar)
   - Zipline: `CustomFundamentals.FOCFExDividends_Discrete.latest.shift()` (LSEG)
   - **Impact**: Different FCF values → Different cash_return → Different rankings → Different stock selection

2. **Missing Money Flow Factors** (HIGH)
   - QuantRocket has: MFI, CMF, Simple MF, Volume-Weighted MF, Composite MF
   - Zipline: Missing all money flow factors
   - **Impact**: Missing technical indicators that could influence stock selection

3. **Missing Ichimoku Signal** (MODERATE)
   - QuantRocket has: IchimokuSignal() factor
   - Zipline: Missing
   - **Impact**: Missing trend confirmation indicator

### 🟡 **MODERATE IMPACT DIFFERENCES**:

4. **Sector Filtering**
   - QuantRocket: Excludes 'Financials' only
   - Zipline: Excludes both 'Financials' AND 'Financial Services'
   - **Impact**: Slightly different universe

### 🟢 **LOW IMPACT DIFFERENCES**:

5. **Categorical Dtype Handling**
   - Zipline: Added `.astype(str).replace('nan', 'Unknown')` to handle categoricals
   - QuantRocket: Direct `.fillna('Unknown')`
   - **Impact**: Implementation detail, should produce same results

---

## 9. RECOMMENDATIONS TO FIX DISCREPANCIES

### Priority 1: FCF Data Alignment
```python
# Option A: Use Sharadar FCF in Zipline (if available)
'fcf': sharadar.Fundamentals.slice('ARQ').FCF.latest,

# Option B: Verify LSEG vs Sharadar FCF correlation
# Run analysis to see if FOCFExDividends_Discrete ≈ Sharadar FCF
```

### Priority 2: Add Money Flow Factors
```python
# Add to Zipline pipeline:
money_flow_index = MoneyFlowIndexFactor(mask=tradable_filter, window_length=90)
chaikin_money_flow = ChaikinMoneyFlowFactor(mask=tradable_filter, window_length=90)
simple_money_flow = SimpleMoneyFlowFactor(mask=tradable_filter, window_length=90)
volume_weighted_mf = VolumeWeightedMoneyFlowFactor(mask=tradable_filter, window_length=90)

composite_money_flow = (
    money_flow_index.zscore() +
    chaikin_money_flow.zscore() +
    simple_money_flow.zscore() +
    volume_weighted_mf.zscore()
) / 4.0

# Add to pipeline columns:
'mfi': money_flow_index,
'cmf': chaikin_money_flow,
'simple_mf': simple_money_flow,
'vw_mf': volume_weighted_mf,
'composite_mf': composite_money_flow,
```

### Priority 3: Add Ichimoku Signal
```python
# Add to pipeline:
'ich_signal': IchimokuSignal(),
```

### Priority 4: Align Sector Filtering
```python
# Change Zipline to match QuantRocket:
df = df[df['sector'] != 'Financials']
# Remove the Financial Services exclusion
```

---

## 10. DATA COMPARISON CHECKLIST

To verify data alignment, check:

- [ ] FCF values for same stocks on same dates (Sharadar vs LSEG)
- [ ] Enterprise Value values (should be identical)
- [ ] Interest Expense values (should be identical)
- [ ] Resulting cash_return rankings
- [ ] Universe size on same dates
- [ ] Top 10 stocks by estrank on same dates

---

## 11. IMPLEMENTATION DIFFERENCES TO VERIFY

Check your Zipline version for:

1. **Port weight factor**: Is it 0.90 or different?
2. **SPY weight factor**: Is it 0.10 or different?
3. **Tradeability checks**: QuantRocket doesn't have `data.can_trade()` checks
4. **Delisting protection**: Zipline has added checks, QuantRocket doesn't

---

## CONCLUSION

The **most likely cause** of backtest discrepancies is:

1. **Different FCF data** (Sharadar vs LSEG) → Different cash returns → Different stock selection
2. **Missing technical factors** (Money Flow, Ichimoku) → Different rankings

**Recommended Action**:
1. First, verify if Sharadar FCF data is available in your Zipline setup
2. If yes, switch to Sharadar FCF to match QuantRocket
3. Add the missing Money Flow factors
4. Add the Ichimoku signal
5. Re-run backtest and compare results

**If Sharadar FCF is NOT available**:
- You'll need to accept that the Zipline version uses different fundamental data
- Results will differ, but strategy logic remains sound
- Consider this a "variant" of the original strategy
