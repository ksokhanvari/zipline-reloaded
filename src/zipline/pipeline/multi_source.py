"""
Multi-Source Data Integration for Zipline Pipeline

This module provides a simple, clean API for combining multiple data sources
(Sharadar fundamentals, custom databases, etc.) in Zipline backtests.

Quick Start
-----------
>>> from zipline.pipeline import multi_source
>>> from zipline import run_algorithm
>>>
>>> # Define your custom database
>>> class MyFundamentals(multi_source.Database):
...     CODE = "my_database"
...     MyMetric = multi_source.Column(float)
>>>
>>> # Use in pipeline with Sharadar
>>> def make_pipeline():
...     s_roe = multi_source.SharadarFundamentals.roe.latest
...     my_metric = MyFundamentals.MyMetric.latest
...     return multi_source.Pipeline(columns={'roe': s_roe, 'my': my_metric})
>>>
>>> # Run backtest - auto-detects all data sources!
>>> results = run_algorithm(
...     start='2023-01-01',
...     end='2024-01-01',
...     initialize=initialize,
...     bundle='sharadar',
...     custom_loader=multi_source.setup_auto_loader(),
... )

Components
----------
Database : class
    Base class for defining custom data schemas
Column : class
    Column definition for Database classes
Pipeline : class
    Pipeline for defining data queries and screens
sharadar : module
    Sharadar fundamentals data
setup_auto_loader : function
    One-line setup for multi-source backtests

Example: Multi-Source Strategy
-------------------------------
>>> from zipline.pipeline import multi_source as ms
>>> from zipline.api import attach_pipeline, pipeline_output
>>>
>>> class CustomFundamentals(ms.Database):
...     CODE = "fundamentals"
...     LOOKBACK_WINDOW = 252
...     ROE = ms.Column(float)
...     PEG = ms.Column(float)
>>>
>>> def make_pipeline():
...     # Sharadar data
...     s_roe = ms.SharadarFundamentals.roe.latest
...     s_fcf = ms.SharadarFundamentals.fcf.latest
...     s_marketcap = ms.SharadarFundamentals.marketcap.latest
...
...     # Custom data
...     c_roe = CustomFundamentals.ROE.latest
...     c_peg = CustomFundamentals.PEG.latest
...
...     # Universe
...     universe = s_marketcap.top(100)
...
...     # Consensus: both sources agree
...     both_quality = (s_roe > 15) & (c_roe > 15)
...
...     # Selection
...     selection = s_roe.top(10, mask=universe & both_quality)
...
...     return ms.Pipeline(
...         columns={
...             's_roe': s_roe,
...             's_fcf': s_fcf,
...             'c_roe': c_roe,
...             'c_peg': c_peg,
...         },
...         screen=selection,
...     )
>>>
>>> def initialize(context):
...     attach_pipeline(make_pipeline(), 'my_pipeline')
>>>
>>> def before_trading_start(context, data):
...     context.pipeline_data = pipeline_output('my_pipeline')
>>>
>>> # Run with auto loader
>>> from zipline import run_algorithm
>>> results = run_algorithm(
...     start='2023-01-01',
...     end='2024-01-01',
...     initialize=initialize,
...     before_trading_start=before_trading_start,
...     bundle='sharadar',
...     custom_loader=ms.setup_auto_loader(),
... )

Notes
-----
- The auto loader automatically detects and routes Sharadar and custom data columns
- SID translation is enabled by default for run_algorithm() compatibility
- Custom databases must define a CODE attribute matching the SQLite database name
- All data sources can be mixed freely in the same pipeline
- No complex loader setup required - just import and use!

See Also
--------
zipline.pipeline.loaders.auto_loader : Automatic loader implementation
zipline.data.custom : Custom database creation and management
zipline.pipeline.data.sharadar : Sharadar data reference
"""

# Core Pipeline components
from zipline.pipeline import Pipeline
from zipline.pipeline.data.db import Database, Column

# Sharadar data
from zipline.pipeline.data.sharadar import SharadarFundamentals

# Auto loader for easy multi-source integration
from zipline.pipeline.loaders.auto_loader import setup_auto_loader, AutoLoader

# Re-export for convenience
__all__ = [
    # Pipeline basics
    'Pipeline',
    'Database',
    'Column',
    # Data sources
    'SharadarFundamentals',
    # Auto loader
    'setup_auto_loader',
    'AutoLoader',
]


# Module-level docstring aliases for better help()
def help_quick_start():
    """
    Quick start guide for multi-source data integration.

    Returns
    -------
    str
        Quick start documentation
    """
    return """
    Multi-Source Data - Quick Start
    ================================

    1. Define your custom database:

        from zipline.pipeline import multi_source as ms

        class CustomFundamentals(ms.Database):
            CODE = "fundamentals"
            LOOKBACK_WINDOW = 252
            ROE = ms.Column(float)
            PEG = ms.Column(float)

    2. Create a pipeline mixing Sharadar and custom data:

        def make_pipeline():
            # Sharadar
            s_roe = ms.SharadarFundamentals.roe.latest

            # Custom
            c_roe = CustomFundamentals.ROE.latest

            # Mix them!
            both_quality = (s_roe > 15) & (c_roe > 15)

            return ms.Pipeline(
                columns={'s_roe': s_roe, 'c_roe': c_roe},
                screen=both_quality,
            )

    3. Run your backtest:

        from zipline import run_algorithm

        results = run_algorithm(
            start='2023-01-01',
            end='2024-01-01',
            initialize=initialize,
            bundle='sharadar',
            custom_loader=ms.setup_auto_loader(),  # Magic!
        )

    That's it! The auto loader handles all the complexity.

    See also:
    - help(ms.setup_auto_loader) for configuration options
    - help(ms.Database) for database definition
    - help(ms.SharadarFundamentals) for Sharadar data reference
    """


def help_database():
    """
    Guide for defining custom databases.

    Returns
    -------
    str
        Database definition documentation
    """
    return """
    Defining Custom Databases
    =========================

    Custom databases allow you to use your own fundamental data alongside
    Sharadar in Zipline backtests.

    Basic Structure
    ---------------
    from zipline.pipeline import multi_source as ms

    class CustomFundamentals(ms.Database):
        CODE = "fundamentals"           # SQLite database name
        LOOKBACK_WINDOW = 252           # Days to look back

        # Define columns matching your database
        ROE = ms.Column(float)
        PEG = ms.Column(float)
        MarketCap = ms.Column(float)
        Sector = ms.Column(object)      # For text columns

    Required Attributes
    -------------------
    CODE : str
        Database identifier. Must match the SQLite database filename
        (without .sqlite extension) in ~/.zipline/data/custom/

    LOOKBACK_WINDOW : int
        Number of days to look back when querying data

    Column Types
    ------------
    ms.Column(float)   - Numeric data (ROE, prices, ratios, etc.)
    ms.Column(int)     - Integer data (counts, flags)
    ms.Column(object)  - Text data (sector names, tickers, etc.)

    Database Location
    -----------------
    Custom databases should be located at:
        ~/.zipline/data/custom/{CODE}.sqlite

    For example:
        CODE = "fundamentals"
        Location: ~/.zipline/data/custom/fundamentals.sqlite

    Using in Pipeline
    -----------------
    Once defined, use columns in your pipeline:

        roe = CustomFundamentals.ROE.latest
        sector = CustomFundamentals.Sector.latest

        high_roe = roe > 15.0
        tech_stocks = sector == 'Technology'

        pipeline = ms.Pipeline(
            columns={'roe': roe, 'sector': sector},
            screen=high_roe & tech_stocks,
        )

    See also:
    - help(ms.Column) for column definition details
    - help(ms.setup_auto_loader) for loader configuration
    """


def help_sharadar():
    """
    Guide for using Sharadar fundamentals.

    Returns
    -------
    str
        Sharadar fundamentals documentation
    """
    return """
    Sharadar Fundamentals
    =====================

    Access Sharadar SF1 fundamentals data in your pipelines.

    Basic Usage
    -----------
    from zipline.pipeline import multi_source as ms

    # Access Sharadar fundamentals directly
    roe = ms.SharadarFundamentals.roe.latest
    fcf = ms.SharadarFundamentals.fcf.latest
    revenue = ms.SharadarFundamentals.revenue.latest
    marketcap = ms.SharadarFundamentals.marketcap.latest

    Data Frequency
    --------------
    Sharadar fundamentals are quarterly (As-Reported Quarterly - ARQ)
    and point-in-time correct based on filing dates. Data is automatically
    forward-filled between quarters.

    Common Fields
    -------------
    Financial Metrics:
        roe, roa, roic - Return ratios
        fcf, fcfps - Free cash flow
        revenue, revenueusd - Revenue
        ebitda, ebitdausd - EBITDA
        eps, epsusd - Earnings per share

    Valuation:
        marketcap - Market capitalization
        pe, pe1 - Price to earnings
        pb - Price to book
        ps, ps1 - Price to sales

    Balance Sheet:
        assets - Total assets
        cashnequsd - Cash and equivalents
        debtusd - Total debt
        equity, equityusd - Total equity

    Example Pipeline
    ----------------
    def make_pipeline():
        # Get metrics
        roe = ms.SharadarFundamentals.roe.latest
        fcf = ms.SharadarFundamentals.fcf.latest
        marketcap = ms.SharadarFundamentals.marketcap.latest
        pe = ms.SharadarFundamentals.pe.latest

        # Universe: top 500 by market cap
        universe = marketcap.top(500)

        # Quality screen
        quality = (
            (roe > 15) &
            (fcf > 0) &
            (pe > 0) &
            (pe < 25)
        )

        # Selection
        selection = roe.top(20, mask=universe & quality)

        return ms.Pipeline(
            columns={
                'roe': roe,
                'fcf': fcf,
                'pe': pe,
                'marketcap': marketcap,
            },
            screen=selection,
        )

    Combining with Custom Data
    ---------------------------
    # Mix Sharadar and custom data seamlessly!
    s_roe = ms.SharadarFundamentals.roe.latest
    c_roe = CustomFundamentals.ROE.latest

    # Both sources agree = higher confidence
    consensus = (s_roe > 15) & (c_roe > 15)

    See also:
    - Sharadar SF1 documentation: https://data.nasdaq.com/databases/SF1/documentation
    - help(ms.Pipeline) for pipeline construction
    """


# Add helper functions to module
quick_start = help_quick_start
database_guide = help_database
sharadar_guide = help_sharadar
