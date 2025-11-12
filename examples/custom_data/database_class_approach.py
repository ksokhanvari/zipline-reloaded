"""
Custom Fundamentals Database Definition

This module demonstrates the PROPER way to create custom data sources
that work seamlessly with Zipline Pipeline factors.

Based on the Database class pattern, this allows you to:
1. Define columns with proper types
2. Use .latest, .shift(), .rank() etc. directly
3. Integrate with CustomFactors automatically
4. Work with all Pipeline operations
"""

from zipline.pipeline.data.db import Database, Column


class CustomFundamentals(Database):
    """
    Custom fundamentals database for quarterly fundamental data.

    This replaces the make_custom_dataset_class() approach with a
    cleaner, more Pythonic class-based definition.

    Usage:
        # In your pipeline:
        roe = CustomFundamentals.ROE.latest
        high_roe = (roe > 10.0)

        # Or with shifts for point-in-time correctness:
        roe = CustomFundamentals.ROE.latest.shift()
    """

    # Database configuration
    CODE = "fundamentals"  # Must match your database code
    LOOKBACK_WINDOW = 240  # Days to look back for data

    # ========================================================================
    # COLUMN DEFINITIONS
    # ========================================================================

    # Income Statement Metrics
    Revenue = Column(float)
    NetIncome = Column(float)

    # Balance Sheet Metrics
    TotalAssets = Column(float)
    TotalEquity = Column(float)
    SharesOutstanding = Column(float)

    # Per-Share Metrics
    EPS = Column(float)
    BookValuePerShare = Column(float)

    # Financial Ratios
    ROE = Column(float)  # Return on Equity
    DebtToEquity = Column(float)  # Debt/Equity ratio
    CurrentRatio = Column(float)  # Current Assets/Current Liabilities
    PERatio = Column(float)  # Price-to-Earnings ratio

    # Metadata
    Sector = Column(str)  # Use str instead of object for Python 3


# ============================================================================
# EXAMPLE: Using CustomFundamentals in Pipeline
# ============================================================================

def make_pipeline_with_database_class():
    """
    Example pipeline using the Database class approach.

    This is MUCH cleaner than the previous approach!
    """
    from zipline.pipeline import Pipeline
    from zipline.pipeline.data import EquityPricing
    from zipline.pipeline.factors import Returns

    # ========================================================================
    # GET DATA FROM CUSTOM DATABASE
    # ========================================================================

    # Get latest values (most common usage)
    roe = CustomFundamentals.ROE.latest
    pe_ratio = CustomFundamentals.PERatio.latest
    debt_to_equity = CustomFundamentals.DebtToEquity.latest
    sector = CustomFundamentals.Sector.latest

    # Use .shift() for point-in-time correctness
    # (gets data from N days ago)
    roe_shifted = CustomFundamentals.ROE.latest.shift()  # 1 day ago
    roe_shifted_5 = CustomFundamentals.ROE.latest.shift(5)  # 5 days ago

    # ========================================================================
    # GET DATA FROM BUNDLE (Pricing)
    # ========================================================================

    close_price = EquityPricing.close.latest
    volume = EquityPricing.volume.latest

    # Calculate returns
    returns_60d = Returns(window_length=60)

    # ========================================================================
    # SCREENING (Combining fundamental + pricing)
    # ========================================================================

    # Fundamental quality
    high_roe = (roe > 5.0)
    reasonable_pe = (pe_ratio < 50.0)
    manageable_debt = (debt_to_equity < 5.0)

    # Liquidity (from pricing)
    avg_volume = EquityPricing.volume.mavg(20)
    liquid = (avg_volume > 100000)

    # Combined screen
    universe = high_roe & reasonable_pe & manageable_debt & liquid

    # ========================================================================
    # RANKING
    # ========================================================================

    # Rank by ROE (higher is better)
    roe_rank = roe.rank(mask=universe, ascending=False)

    # Normalize using zscore
    roe_zscore = roe.zscore(mask=universe)

    # Demean by sector
    roe_demeaned = roe.demean(groupby=sector, mask=universe)

    # ========================================================================
    # RETURN PIPELINE
    # ========================================================================

    return Pipeline(
        columns={
            'roe': roe,
            'pe_ratio': pe_ratio,
            'debt_to_equity': debt_to_equity,
            'sector': sector,
            'roe_rank': roe_rank,
            'roe_zscore': roe_zscore,
            'returns_60d': returns_60d,
            'close': close_price,
        },
        screen=universe
    )


# ============================================================================
# ADVANTAGES OF DATABASE CLASS APPROACH
# ============================================================================

"""
ADVANTAGES:

1. **Cleaner Syntax**:
   - Old: Fundamentals = make_custom_dataset_class(...)
   - New: class CustomFundamentals(Database): ...

2. **Direct Column Access**:
   - CustomFundamentals.ROE.latest
   - CustomFundamentals.PERatio.latest.shift()
   - CustomFundamentals.Sector.latest.rank()

3. **All Pipeline Operations Work**:
   - .latest, .shift(N), .rank(), .zscore()
   - .demean(groupby=...), .winsorize()
   - .top(N), .bottom(N), .percentile_between()

4. **Type Safety**:
   - Columns are properly typed (float, str, etc.)
   - Better IDE autocomplete
   - Clearer documentation

5. **CustomFactor Integration**:
   - Can use directly as CustomFactor inputs
   - Works with all built-in factors
   - Composable with other factors

EXAMPLE USAGE IN CUSTOMFACTOR:

class QualityScore(CustomFactor):
    inputs = [
        CustomFundamentals.ROE,         # Direct column reference!
        CustomFundamentals.PERatio,
        CustomFundamentals.DebtToEquity,
    ]
    window_length = 1

    def compute(self, today, assets, out, roe, pe, debt):
        # Normalize metrics - use [-1] for most recent data
        roe_score = (roe[-1] - np.nanmin(roe[-1])) / (np.nanmax(roe[-1]) - np.nanmin(roe[-1]))
        pe_score = 1 - ((pe[-1] - np.nanmin(pe[-1])) / (np.nanmax(pe[-1]) - np.nanmin(pe[-1])))
        debt_score = 1 - ((debt[-1] - np.nanmin(debt[-1])) / (np.nanmax(debt[-1]) - np.nanmin(debt[-1])))

        out[:] = (roe_score + pe_score + debt_score) / 3.0

# Use in pipeline:
quality = QualityScore()
pipe = Pipeline(columns={'quality': quality})
"""


# ============================================================================
# MIGRATION GUIDE: OLD → NEW
# ============================================================================

"""
OLD APPROACH (make_custom_dataset_class):

    Fundamentals = make_custom_dataset_class(
        db_code='fundamentals',
        columns={'ROE': 'float', 'PERatio': 'float', ...},
        base_name='Fundamentals'
    )

    roe = Fundamentals.ROE.latest


NEW APPROACH (Database class):

    class CustomFundamentals(Database):
        CODE = "fundamentals"
        LOOKBACK_WINDOW = 240

        ROE = Column(float)
        PERatio = Column(float)
        # ...

    roe = CustomFundamentals.ROE.latest


MIGRATION STEPS:

1. Replace make_custom_dataset_class() with class definition
2. Move db_code → CODE
3. Convert column dict → Column attributes
4. Update import: from zipline.pipeline.data.db import Database, Column
5. Test your pipeline - all operations should work the same!
"""


# ============================================================================
# INTEGRATION WITH LOADER
# ============================================================================

"""
The Database class automatically integrates with CustomSQLiteLoader.

You don't need to do anything special - just use the same CODE:

    from zipline.data.custom import CustomSQLiteLoader

    class CustomFundamentals(Database):
        CODE = "fundamentals"  # Loader will use this
        # ...

    # In your algorithm:
    def initialize(context):
        pipe = make_pipeline()
        attach_pipeline(pipe, 'my_pipeline')

    # Zipline automatically uses CustomSQLiteLoader for CustomFundamentals
    # based on the CODE attribute!

The magic happens in Pipeline's loader dispatch:
- It sees CustomFundamentals.ROE
- Looks up CustomFundamentals.CODE
- Creates CustomSQLiteLoader('fundamentals')
- Fetches data automatically!
"""
