# Performance Delta Troubleshooting Checklist

## Date: 2025-12-01
## UPDATED: 2025-12-01 - FCF Data Source Fixed!

✅ **FIXED**: FCF data source changed from LSEG to Sharadar (line 1029)

---

## ✅ ISSUE #1: FCF Data Source (FIXED!)

### Previous Code (Line 1031) - WRONG:
```python
columns['fcf'] = CustomFundamentals.FOCFExDividends_Discrete.latest  # LSEG data
```

### Current Code (Line 1029) - CORRECT:
```python
from zipline.pipeline.data.sharadar import SharadarFundamentals  # Line 65
columns['fcf'] = SharadarFundamentals.fcf.latest  # Now matches QuantRocket!
```

**Status**: ✅ **FIXED** - Now using Sharadar FCF to match QuantRocket implementation

### Verification:
- ✅ SharadarFundamentals import exists (line 65)
- ✅ SharadarFundamentals.fcf column available (confirmed via test)
- ✅ Line 1029 now uses `SharadarFundamentals.fcf.latest`
- ✅ Old LSEG code commented out for reference

---

## OTHER POTENTIAL CAUSES OF DELTA

### 1. ⚠️ Data Lag/Shift Differences

**QuantRocket**:
```python
'fcf': s_fundamentals.FCF.latest,  # No .shift()
'int': CustomFundamentals.InterestExpense...latest.shift(),
'entval': CustomFundamentals.EnterpriseValue...latest.shift(),
```

**Your Zipline**:
```python
'fcf': CustomFundamentals.FOCFExDividends_Discrete.latest,  # No .shift()
'int': CustomFundamentals.InterestExpense...latest,  # No .shift()
'entval': CustomFundamentals.EnterpriseValue...latest,  # No .shift()
```

**Issue**: You may have removed `.shift()` from some columns. The `.shift()` prevents look-ahead bias.

---

### 2. ⚠️ Tradeability Checks (NEW in your version)

**Your Zipline** has `data.can_trade()` checks that **QuantRocket DOES NOT have**:

```python
# Your version (lines 1462-1470)
if data.can_trade(sid_val):
    order_target_percent(sid_val, w * port_weight_factor)
else:
    print(f"  WARNING: Cannot trade {sid_val} - skipping")
```

**Impact**: This means:
- QuantRocket trades delisted stocks until forced liquidation
- Your version skips delisted stocks earlier

**Result**: Different position entries/exits → Different returns

**Question**: Is this intentional or should you match QuantRocket behavior?

---

### 3. ⚠️ Sector Filtering Differences

**QuantRocket** (single line):
```python
df = df[df['sector'] != 'Financials']
```

**Your Zipline** (more complex):
```python
df = df[~df['sector'].astype(str).replace('nan', 'Unknown')
        .isin(['Financial Services', 'Financials'])]
```

**Impact**: You're excluding MORE stocks (both 'Financials' and 'Financial Services')

**Action**: Change to match QuantRocket exactly:
```python
df = df[df['sector'] != 'Financials']
```

---

### 4. ⚠️ Missing Technical Indicators

Your pipeline is **MISSING** these factors that QuantRocket has:

```python
# QuantRocket has (you don't):
'mfi': money_flow_index,
'cmf': chaikin_money_flow,
'simple_mf': simple_money_flow,
'vw_mf': volume_weighted_mf,
'composite_mf': composite_money_flow,
'ich_signal': IchimokuSignal(),
```

**Impact**: Unknown - these aren't used in ranking formula, but may affect universe

---

### 5. ⚠️ Universe Size Filter

Check if you're using the same `FILTERED_UNIVERSE_SIZE`:

**QuantRocket**:
```python
FILTERED_UNIVERSE_SIZE = 500
df = df.sort_values(by=['market_cap'], ascending=[False])[0:FILTERED_UNIVERSE_SIZE].copy()
```

**Your version**: Verify this is 500, not different

---

### 6. ⚠️ Benchmark and Slippage Settings

**QuantRocket**:
```python
algo.set_benchmark(algo.sid(symbol('SPY').real_sid))
algo.set_slippage(algo.slippage.FixedSlippage(spread=SLIPPAGE_SPREAD))
algo.set_commission(algo.commission.PerShare(cost=COMMISSION_COST, min_trade_cost=MIN_TRADE_COST))
```

Where:
```python
SLIPPAGE_SPREAD = 0.05
COMMISSION_COST = 0.01
MIN_TRADE_COST = 1.00
```

**Check**: Are your values identical?

---

### 7. ⚠️ Date Range Differences

**Critical Question**: Are you running the EXACT SAME backtest period?

- Start date?
- End date?
- Same calendar days?

Small date shifts can cause large performance differences due to rebalancing timing.

---

### 8. ⚠️ Price Data Source

**QuantRocket** uses:
- zipline.pipeline.data.USEquityPricing (from QuantRocket's data)

**Your Zipline** uses:
- zipline.pipeline.data.USEquityPricing (from Sharadar bundle?)

**Question**: Are you using the SAME price data source?

---

## STEP-BY-STEP DIAGNOSTIC PLAN

### Step 1: Fix FCF Data Source
```python
# Change line 1031 from:
columns['fcf'] = CustomFundamentals.FOCFExDividends_Discrete.latest

# To:
from zipline.pipeline.data.sharadar import Fundamentals as SharadarFundamentals
s_fundamentals = SharadarFundamentals.slice('ARQ', period_offset=0)
columns['fcf'] = s_fundamentals.fcf.latest
```

### Step 2: Align Sector Filtering
```python
# Change from:
df = df[~df['sector'].astype(str).replace('nan', 'Unknown')
        .isin(['Financial Services', 'Financials'])]

# To:
df = df[df['sector'] != 'Financials']
```

### Step 3: Verify Data Shifts
Check all your `.latest` calls - do they need `.shift()` to avoid look-ahead bias?

### Step 4: Remove Tradeability Checks (optional)
If you want to exactly match QuantRocket, remove all `data.can_trade()` checks.

**WARNING**: This will allow trading delisted stocks, which may not be realistic.

### Step 5: Compare Universe on Same Date

Run both strategies and print:
```python
print("Universe size:", len(df))
print("Top 10 stocks by estrank:")
print(df.nlargest(10, 'estrank')[['name', 'cash_return', 'estrank', 'sector']])
```

Compare output for the SAME date between QuantRocket and Zipline.

### Step 6: Compare Cash Returns

For the same date, print:
```python
print("Cash return distribution:")
print(df['cash_return'].describe())
print("\nTop 10 by cash_return:")
print(df.nlargest(10, 'cash_return')[['name', 'fcf', 'int', 'entval', 'cash_return']])
```

If cash returns differ, the FCF data is still wrong.

---

## CRITICAL QUESTIONS TO ANSWER

1. **Are you using Sharadar bundle for price data?**
   - QuantRocket uses their own price data
   - You need to verify prices match

2. **What is your backtest date range?**
   - Exact start and end dates?
   - Same as QuantRocket backtest?

3. **Have you verified FCF values match?**
   - Pick a stock (e.g., AAPL) on a specific date
   - Compare FCF value in QuantRocket vs your Zipline
   - If they differ, data source is wrong

4. **Are you using the SAME rebalancing schedule?**
   - QuantRocket: `date_rules.week_start(days_offset=context.days_offset)`
   - Your Zipline: Same?
   - `context.days_offset = 1` (Tuesday)?

---

## IMMEDIATE ACTION ITEMS

### Priority 1 (CRITICAL):
- [ ] Fix FCF to use Sharadar data (line 1031)
- [ ] Verify Sharadar FCF is available in your setup
- [ ] If not available, you CANNOT match QuantRocket results

### Priority 2 (HIGH):
- [ ] Change sector filter to match QuantRocket exactly
- [ ] Verify backtest date range matches
- [ ] Check slippage and commission settings match

### Priority 3 (MEDIUM):
- [ ] Decide: Keep or remove `data.can_trade()` checks?
- [ ] Verify `.shift()` usage on all data columns
- [ ] Compare universe size on same date

### Priority 4 (LOW):
- [ ] Add missing technical indicators (if desired)
- [ ] Compare price data sources

---

## HOW TO CHECK IF YOU HAVE SHARADAR FCF

Run this in a notebook:

```python
from zipline.pipeline.data.sharadar import Fundamentals as SharadarFundamentals

# Check if fcf column exists
s_fund = SharadarFundamentals.slice('ARQ', period_offset=0)
print("Available columns:", dir(s_fund))
print("Has FCF?", hasattr(s_fund, 'fcf'))

# Try to get FCF data
try:
    fcf_factor = s_fund.fcf.latest
    print("✅ Sharadar FCF is available!")
except Exception as e:
    print(f"❌ Sharadar FCF NOT available: {e}")
```

---

## BOTTOM LINE

If Sharadar FCF is NOT available in your Zipline setup:
- **You CANNOT match QuantRocket results exactly**
- The fundamental data is different
- Accept this as a "variant strategy" with different (possibly better or worse) performance

If Sharadar FCF IS available:
- Fix line 1031 to use it
- Follow the diagnostic plan above
- Results should converge
