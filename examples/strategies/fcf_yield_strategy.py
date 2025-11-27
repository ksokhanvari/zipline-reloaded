"""
Free Cash Flow Yield Strategy with Sharadar Universe Filtering

This strategy:
1. Filters to NYSE/NASDAQ/NYSEMKT domestic common stocks (no ADRs)
2. Selects top 1500 by market cap
3. Calculates FCF Yield = (Free Cash Flow - Interest Expense) / Enterprise Value
4. Goes long top 100 by FCF Yield
5. Rebalances monthly
"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/examples/strategies')

from zipline.pipeline import Pipeline
from zipline.pipeline.data import DataSet, Column
from zipline.pipeline.factors import CustomFactor, Returns
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    schedule_function,
    date_rules,
    time_rules,
    set_commission,
    set_slippage,
)
from zipline.finance import commission, slippage

# Import Sharadar filters
from sharadar_filters import (
    ExchangeFilter,
    CategoryFilter,
    ADRFilter,
    create_sharadar_universe,
)


# ============================================================================
# Database Definitions
# ============================================================================

class CustomFundamentals(DataSet):
    """Custom fundamentals database from LSEG data."""

    CODE = "fundamentals"
    LOOKBACK_WINDOW = 252

    # Price and Volume
    RefPriceClose = Column(float)
    RefVolume = Column(float)

    # Company Info
    CompanyCommonName = Column(object, missing_value='')
    GICSSectorName = Column(object, missing_value='')

    # Valuation Metrics
    EnterpriseValue_DailyTimeSeries_ = Column(float)
    CompanyMarketCap = Column(float)
    PricetoBookValue_DailyTimeSeries_ = Column(float)
    PricetoEarnings_DailyTimeSeries_ = Column(float)

    # Income Statement
    TotalRevenue = Column(float)
    GrossProfit = Column(float)
    OperatingIncome = Column(float)
    NetIncome = Column(float)
    EBITDA = Column(float)
    BasicEPS = Column(float)
    DilutedEPS = Column(float)

    # Balance Sheet
    TotalAssets = Column(float)
    TotalLiabilities = Column(float)
    TotalEquity = Column(float)

    # Cash Flow
    FOCFExDividends_Discrete = Column(float)  # Free Operating Cash Flow

    # Financial Ratios
    ReturnOnEquity = Column(float)
    ReturnOnAssets = Column(float)
    ReturnOnInvestedCapital = Column(float)
    DebttoEquityRatio = Column(float)
    CurrentRatio = Column(float)

    # Smart Estimates
    EBITDA_SmartEstimate = Column(float)
    EBIT_SmartEstimate = Column(float)
    ReturnOnEquity_SmartEstimat = Column(float)

    # Interest
    InterestExpense_NetofCapitalizedInterest = Column(float)

    # Signals
    pred = Column(float)
    bc1 = Column(float)


# SharadarTickers DataSet is now imported from sharadar_filters.py


# ============================================================================
# Pipeline Construction
# ============================================================================

def make_pipeline():
    """
    Create pipeline with Sharadar universe filtering and FCF Yield alpha.
    """

    # ========== Universe Filtering ==========

    # TODO: Add Sharadar universe filtering once static metadata loading is fixed
    # sharadar_universe = create_sharadar_universe(
    #     exchanges=['NYSE', 'NASDAQ', 'NYSEMKT'],
    #     include_adrs=False,
    # )

    # Market cap filter: top 1500
    market_cap = CustomFundamentals.CompanyMarketCap.latest
    top_1500_by_mcap = market_cap.top(1500)  # No mask for now

    # ========== Alpha Factor: FCF Yield ==========

    # Components
    fcf = CustomFundamentals.FOCFExDividends_Discrete.latest
    interest_expense = CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest
    enterprise_value = CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest

    # FCF Yield = (FCF - Interest Expense) / EV
    # Higher is better (more cash flow per dollar of enterprise value)
    fcf_yield = (fcf - interest_expense) / enterprise_value

    # ========== Final Screening ==========

    TOP_N = 100
    top_n_by_fcf_yield = fcf_yield.top(TOP_N, mask=top_1500_by_mcap)

    # Rank for position sizing
    alpha_rank = fcf_yield.rank(mask=top_n_by_fcf_yield, ascending=False)

    # ========== Pipeline Columns ==========

    return Pipeline(
        columns={
            # Alpha factor and rank
            'fcf_yield': fcf_yield,
            'alpha_rank': alpha_rank,

            # Components (for debugging)
            'fcf': fcf,
            'interest_expense': interest_expense,
            'enterprise_value': enterprise_value,
            'market_cap': market_cap,
        },
        screen=top_n_by_fcf_yield,
    )


# ============================================================================
# Algorithm Functions
# ============================================================================

def initialize(context):
    """
    Initialize algorithm.
    """
    # Attach pipeline
    attach_pipeline(make_pipeline(), 'fcf_yield_strategy')

    # Set commission and slippage
    set_commission(commission.PerDollar(cost=0.001))  # 10 bps
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))

    # Schedule rebalance
    schedule_function(
        rebalance,
        date_rules.month_start(),
        time_rules.market_open(hours=1)
    )

    context.longs = []


def before_trading_start(context, data):
    """
    Run pipeline and get today's universe.
    """
    context.output = pipeline_output('fcf_yield_strategy')
    context.longs = context.output.index.tolist()

    # Log universe size
    from zipline.utils.flightlog_client import log_to_flightlog
    log_to_flightlog(
        f"Universe: {len(context.longs)} stocks (FCF Yield)",
        level='INFO'
    )


def rebalance(context, data):
    """
    Rebalance portfolio - equal weight long-only.
    """
    from zipline.utils.flightlog_client import log_to_flightlog

    # Equal weight for longs
    n_longs = len(context.longs)

    if n_longs == 0:
        log_to_flightlog("No stocks to trade", level='WARNING')
        return

    target_weight = 0.95 / n_longs  # 95% invested, 5% cash buffer

    # Get current positions
    current_positions = set(context.portfolio.positions.keys())
    target_positions = set(context.longs)

    # Exit positions not in target
    to_exit = current_positions - target_positions
    for asset in to_exit:
        if data.can_trade(asset):
            order_target_percent(asset, 0.0)

    # Enter/adjust target positions
    for asset in context.longs:
        if data.can_trade(asset):
            order_target_percent(asset, target_weight)

    log_to_flightlog(
        f"Rebalanced: {n_longs} longs @ {target_weight*100:.2f}% each, "
        f"exited {len(to_exit)} positions",
        level='INFO'
    )


def analyze(context, perf):
    """
    Analyze backtest results.
    """
    from zipline.utils.flightlog_client import log_to_flightlog

    total_return = perf['algorithm_period_return'].iloc[-1] * 100
    sharpe = perf['sharpe'].iloc[-1] if 'sharpe' in perf.columns else 0
    max_dd = perf['max_drawdown'].min() * 100

    log_to_flightlog(
        f"Backtest Complete - Total Return: {total_return:+.2f}%, "
        f"Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}%",
        level='INFO'
    )
