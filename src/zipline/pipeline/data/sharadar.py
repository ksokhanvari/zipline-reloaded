"""
Sharadar Fundamentals Dataset for Pipeline.

This module provides access to Sharadar SF1 fundamental data through the Pipeline API.
The data includes 150+ quarterly fundamental metrics sourced from company filings.

Data Source
-----------
Sharadar SF1 table (As-Reported Quarterly - ARQ dimension)
- Downloaded and stored during bundle ingestion
- Point-in-time correct using 'datekey' field
- Quarterly frequency with automatic forward-filling

Key Metrics Available
---------------------
Income Statement: revenue, netinc, ebitda, ebit, opex, grossprofit
Balance Sheet: assets, equity, debt, cashnequiv, workingcapital
Cash Flow: ncf, capex, fcf, ncfdebt, ncfinv
Ratios: roe, roa, pe, pb, ps, ev_ebitda, de, currentratio
Per-Share: eps, bvps, sps, fcfps, dps

Usage
-----
>>> from zipline.pipeline import Pipeline
>>> from zipline.pipeline.data.sharadar import SharadarFundamentals
>>> from zipline.pipeline.factors import Latest
>>>
>>> # Get latest revenue
>>> revenue = SharadarFundamentals.revenue.latest
>>>
>>> # Create pipeline
>>> pipe = Pipeline(
...     columns={
...         'revenue': revenue,
...         'net_income': SharadarFundamentals.netinc.latest,
...         'roe': SharadarFundamentals.roe.latest,
...     }
... )

Point-in-Time Correctness
--------------------------
All data is point-in-time correct using Sharadar's 'datekey' field, which represents
the date when the fundamental data became publicly available. This prevents look-ahead
bias in backtests.

For example, Q4 2022 earnings (calendardate = 2022-12-31) might have been filed on
2023-02-15 (datekey = 2023-02-15). The Pipeline will only make this data available
from 2023-02-15 onwards, not from 2022-12-31.
"""

from zipline.utils.numpy_utils import float64_dtype

from ..domain import US_EQUITIES
from .dataset import Column, DataSet


class SharadarFundamentals(DataSet):
    """
    Dataset containing Sharadar SF1 fundamental data.

    All metrics are quarterly (As-Reported Quarterly - ARQ dimension) and
    point-in-time correct based on filing dates.

    See Also
    --------
    zipline.pipeline.loaders.sharadar_fundamentals : Loader for this dataset
    """

    # ==================== INCOME STATEMENT ====================

    # Revenue and Profitability
    revenue = Column(float64_dtype, missing_value=float('nan'))
    """Total revenue (sales)"""

    netinc = Column(float64_dtype, missing_value=float('nan'))
    """Net income (bottom line profit)"""

    ebitda = Column(float64_dtype, missing_value=float('nan'))
    """Earnings before interest, taxes, depreciation, and amortization"""

    ebit = Column(float64_dtype, missing_value=float('nan'))
    """Earnings before interest and taxes (operating income)"""

    grossprofit = Column(float64_dtype, missing_value=float('nan'))
    """Gross profit (revenue - cost of revenue)"""

    opinc = Column(float64_dtype, missing_value=float('nan'))
    """Operating income"""

    # Expenses
    opex = Column(float64_dtype, missing_value=float('nan'))
    """Operating expenses"""

    cor = Column(float64_dtype, missing_value=float('nan'))
    """Cost of revenue"""

    sgna = Column(float64_dtype, missing_value=float('nan'))
    """Selling, general & administrative expenses"""

    rnd = Column(float64_dtype, missing_value=float('nan'))
    """Research & development expenses"""

    intexp = Column(float64_dtype, missing_value=float('nan'))
    """Interest expense"""

    taxexp = Column(float64_dtype, missing_value=float('nan'))
    """Tax expense"""

    # ==================== BALANCE SHEET ====================

    # Assets
    assets = Column(float64_dtype, missing_value=float('nan'))
    """Total assets"""

    assetsc = Column(float64_dtype, missing_value=float('nan'))
    """Current assets"""

    assetsnc = Column(float64_dtype, missing_value=float('nan'))
    """Non-current assets (long-term)"""

    cashneq = Column(float64_dtype, missing_value=float('nan'))
    """Cash and equivalents"""

    investments = Column(float64_dtype, missing_value=float('nan'))
    """Investments"""

    ppnenet = Column(float64_dtype, missing_value=float('nan'))
    """Property, plant & equipment (net)"""

    intangibles = Column(float64_dtype, missing_value=float('nan'))
    """Intangible assets"""

    # Liabilities
    liabilities = Column(float64_dtype, missing_value=float('nan'))
    """Total liabilities"""

    liabilitiesc = Column(float64_dtype, missing_value=float('nan'))
    """Current liabilities"""

    liabilitiesnc = Column(float64_dtype, missing_value=float('nan'))
    """Non-current liabilities (long-term)"""

    debt = Column(float64_dtype, missing_value=float('nan'))
    """Total debt"""

    debtc = Column(float64_dtype, missing_value=float('nan'))
    """Current debt (short-term)"""

    debtnc = Column(float64_dtype, missing_value=float('nan'))
    """Non-current debt (long-term)"""

    # Equity
    equity = Column(float64_dtype, missing_value=float('nan'))
    """Shareholders' equity"""

    equityusd = Column(float64_dtype, missing_value=float('nan'))
    """Shareholders' equity in USD"""

    # Working Capital
    workingcapital = Column(float64_dtype, missing_value=float('nan'))
    """Working capital (current assets - current liabilities)"""

    # ==================== CASH FLOW ====================

    ncf = Column(float64_dtype, missing_value=float('nan'))
    """Net cash flow"""

    ncfo = Column(float64_dtype, missing_value=float('nan'))
    """Net cash flow from operations"""

    ncfi = Column(float64_dtype, missing_value=float('nan'))
    """Net cash flow from investing"""

    ncff = Column(float64_dtype, missing_value=float('nan'))
    """Net cash flow from financing"""

    capex = Column(float64_dtype, missing_value=float('nan'))
    """Capital expenditures"""

    fcf = Column(float64_dtype, missing_value=float('nan'))
    """Free cash flow"""

    ncfdebt = Column(float64_dtype, missing_value=float('nan'))
    """Net cash flow from debt"""

    ncfdiv = Column(float64_dtype, missing_value=float('nan'))
    """Cash from dividends"""

    # ==================== RATIOS ====================

    # Profitability
    roe = Column(float64_dtype, missing_value=float('nan'))
    """Return on equity (%)"""

    roa = Column(float64_dtype, missing_value=float('nan'))
    """Return on assets (%)"""

    roic = Column(float64_dtype, missing_value=float('nan'))
    """Return on invested capital (%)"""

    grossmargin = Column(float64_dtype, missing_value=float('nan'))
    """Gross profit margin (%)"""

    netmargin = Column(float64_dtype, missing_value=float('nan'))
    """Net profit margin (%)"""

    ebitdamargin = Column(float64_dtype, missing_value=float('nan'))
    """EBITDA margin (%)"""

    # Valuation
    pe = Column(float64_dtype, missing_value=float('nan'))
    """Price to earnings ratio"""

    pb = Column(float64_dtype, missing_value=float('nan'))
    """Price to book ratio"""

    ps = Column(float64_dtype, missing_value=float('nan'))
    """Price to sales ratio"""

    ev = Column(float64_dtype, missing_value=float('nan'))
    """Enterprise value"""

    evebitda = Column(float64_dtype, missing_value=float('nan'))
    """EV/EBITDA ratio"""

    marketcap = Column(float64_dtype, missing_value=float('nan'))
    """Market capitalization"""

    # Leverage
    de = Column(float64_dtype, missing_value=float('nan'))
    """Debt to equity ratio"""

    debttoassets = Column(float64_dtype, missing_value=float('nan'))
    """Total debt to total assets"""

    # Liquidity
    currentratio = Column(float64_dtype, missing_value=float('nan'))
    """Current ratio (current assets / current liabilities)"""

    quickratio = Column(float64_dtype, missing_value=float('nan'))
    """Quick ratio ((current assets - inventory) / current liabilities)"""

    # Efficiency
    assetturnover = Column(float64_dtype, missing_value=float('nan'))
    """Asset turnover (revenue / average total assets)"""

    invturnover = Column(float64_dtype, missing_value=float('nan'))
    """Inventory turnover"""

    # ==================== PER-SHARE METRICS ====================

    eps = Column(float64_dtype, missing_value=float('nan'))
    """Earnings per share (basic)"""

    epsdil = Column(float64_dtype, missing_value=float('nan'))
    """Earnings per share (diluted)"""

    bvps = Column(float64_dtype, missing_value=float('nan'))
    """Book value per share"""

    sps = Column(float64_dtype, missing_value=float('nan'))
    """Sales per share"""

    fcfps = Column(float64_dtype, missing_value=float('nan'))
    """Free cash flow per share"""

    dps = Column(float64_dtype, missing_value=float('nan'))
    """Dividends per share"""

    # Share Counts
    shareswa = Column(float64_dtype, missing_value=float('nan'))
    """Weighted average shares outstanding (basic)"""

    shareswadil = Column(float64_dtype, missing_value=float('nan'))
    """Weighted average shares outstanding (diluted)"""

    sharesoutstanding = Column(float64_dtype, missing_value=float('nan'))
    """Shares outstanding at period end"""

    # ==================== GROWTH METRICS ====================

    revenuegrowth = Column(float64_dtype, missing_value=float('nan'))
    """Year-over-year revenue growth (%)"""

    netincgrowth = Column(float64_dtype, missing_value=float('nan'))
    """Year-over-year net income growth (%)"""

    epsgrowth = Column(float64_dtype, missing_value=float('nan'))
    """Year-over-year EPS growth (%)"""


# Backwards compat alias for US equities
USSharadarFundamentals = SharadarFundamentals.specialize(US_EQUITIES)
