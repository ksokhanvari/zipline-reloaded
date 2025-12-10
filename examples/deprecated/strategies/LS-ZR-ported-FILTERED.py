"""
Long-Short Equity Trading Algorithm - Zipline-Reloaded Port with Proper Universe Filtering
--------------------------------------------------------------------------------------------
This algorithm implements a long-short equity strategy with proper US equity universe filtering:
- Only US exchanges (NYSE, NASDAQ, NYSEMKT)
- Only domestic common stocks (no ADRs, ETFs, preferred shares, etc.)
- Excludes delisted securities

NOTE: This is a minimal version for testing universe filtering.
For the full strategy with all CustomFactors and algorithm functions,
use LS-ZR-ported.py and modify its make_pipeline() function.
"""

import pandas as pd
import numpy as np

# Zipline-Reloaded imports
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    set_slippage,
    set_commission,
    set_benchmark,
    get_datetime,
)
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import SimpleMovingAverage, SimpleBeta
from zipline.pipeline.filters import StaticAssets
# Sharadar fundamentals for FCF
from zipline.pipeline.data.sharadar import SharadarFundamentals
from zipline.finance import commission, slippage

# Multi-source module for custom databases
from zipline.pipeline import multi_source as ms

# Import universe filtering tools
import sys
sys.path.insert(0, '/app/examples/strategies')
from sharadar_filters import (
    TickerMetadata,
    create_sharadar_universe,
)

# FlightLog for real-time monitoring
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog

#################################################
# GLOBAL CONFIGURATION VARIABLES
#################################################

# Symbol-SID Cache for performance
SYM_SID_CACHE_DICT = {}

# Asset finder reference (set during initialization)
ASSET_FINDER = None

# Portfolio Construction Parameters
UNIVERSE_SIZE = 1500      # Top N stocks by market cap to consider initially

# Risk Management Parameters
SLIPPAGE_SPREAD = 0.05           # Fixed slippage in dollars
COMMISSION_COST = 0.01          # Per-share commission
MIN_TRADE_COST = 1.00          # Minimum commission per trade

#################################################
# DATABASE DEFINITIONS (Multi-Source Pattern)
#################################################

class CustomFundamentals(ms.Database):
    """Primary fundamentals database with core company financial data."""

    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240

    # Company identifiers
    Symbol = ms.Column(object)
    Instrument = ms.Column(object)
    CompanyCommonName = ms.Column(object)
    GICSSectorName = ms.Column(object)

    # Valuation metrics
    EnterpriseValue_DailyTimeSeries_ = ms.Column(float)
    CompanyMarketCap = ms.Column(float)

    # Cash flow metrics
    InterestExpense_NetofCapitalizedInterest = ms.Column(float)

    # Debt metrics
    CashCashEquivalents_Total = ms.Column(float)

    # Earnings metrics
    EarningsPerShare_ActualSurprise = ms.Column(float)
    LongTermGrowth_Mean = ms.Column(float)

    # Prediction columns
    pred = ms.Column(float)
    bc1 = ms.Column(float)  # BC signal

    # Price and volume reference data
    RefPriceClose = ms.Column(float)
    RefVolume = ms.Column(float)


#################################################
# MAIN ALGORITHM FUNCTIONS
#################################################

def initialize(context):
    """
    Initialize the trading algorithm.
    """
    global ASSET_FINDER

    # Store asset finder reference for symbol lookup
    from zipline.data import bundles
    bundle = bundles.load('sharadar')
    ASSET_FINDER = bundle.asset_finder

    # Attach our main stock selection pipeline
    pipeline = make_pipeline()
    attach_pipeline(pipeline, 'my_pipeline')

    # Set the benchmark
    spy = symbol('SPY')
    if spy:
        set_benchmark(spy)

    # Set trading costs
    set_slippage(slippage.FixedSlippage(spread=SLIPPAGE_SPREAD))
    set_commission(commission.PerShare(cost=COMMISSION_COST, min_trade_cost=MIN_TRADE_COST))

    # Enable FlightLog for real-time monitoring
    try:
        enable_flightlog(host='localhost', port=9020)
        log_to_flightlog('Universe filter test initialized', level='INFO')
    except:
        print("FlightLog not available - continuing without real-time monitoring")

    print("[INFO] Strategy initialized with US equities universe filter")


def make_pipeline():
    """
    Creates the stock selection pipeline with proper US equity universe filtering.

    NOTE: Due to SQLite TEXT-in-REAL column issues, we filter by market cap first,
    then apply exchange/ADR/category filters in before_trading_start().
    """
    # STEP 1: Get top stocks by market cap (wider universe)
    # We'll filter by metadata columns in before_trading_start()
    market_cap_filter = CustomFundamentals.CompanyMarketCap.latest.top(
        UNIVERSE_SIZE * 2  # Get 2x to account for filtering
    )

    # STEP 2: Add benchmark assets (SPY, QQQ, IWM, etc.)
    benchmark_assets = []
    for sym in ['SPY', 'QQQ', 'IWM', 'IBM', 'DIA', 'TLT']:
        asset = symbol(sym)
        if asset:
            benchmark_assets.append(asset)

    # Final tradable filter: market cap + benchmarks
    tradable_filter = market_cap_filter | StaticAssets(benchmark_assets)

    print("[INFO] Created pipeline (will filter by exchange/ADR/category in before_trading_start)")

    # STEP 4: Build pipeline columns
    columns = {}

    # Basic identifiers
    columns['name'] = CustomFundamentals.Symbol.latest
    columns['compname'] = CustomFundamentals.CompanyCommonName.latest
    columns['sector'] = CustomFundamentals.GICSSectorName.latest

    # Add metadata columns for verification
    columns['sharadar_exchange'] = TickerMetadata.sharadar_exchange.latest
    columns['sharadar_category'] = TickerMetadata.sharadar_category.latest
    columns['sharadar_is_adr'] = TickerMetadata.sharadar_is_adr.latest

    # Fundamental data
    columns['market_cap'] = CustomFundamentals.CompanyMarketCap.latest
    columns['entval'] = CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest
    columns['price'] = USEquityPricing.close.latest
    columns['volume'] = USEquityPricing.volume.latest

    # Cash flow
    columns['CashCashEquivalents_Total'] = CustomFundamentals.CashCashEquivalents_Total.latest
    columns['fcf'] = SharadarFundamentals.fcf.latest
    columns['int'] = CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest

    # Earnings
    columns['eps_ActualSurprise_prev_Q_percent'] = CustomFundamentals.EarningsPerShare_ActualSurprise.latest
    columns['eps_gr_mean'] = CustomFundamentals.LongTermGrowth_Mean.latest

    # Technical indicators
    columns['smav'] = SimpleMovingAverage(inputs=[USEquityPricing.volume], window_length=10)

    # VIX and BC signals
    columns['vixflag'] = CustomFundamentals.pred.latest
    columns['bc1'] = CustomFundamentals.bc1.latest

    pipe = ms.Pipeline(
        screen=tradable_filter,
        columns=columns
    )

    return pipe


def before_trading_start(context, data):
    """Daily preprocessing before trading begins."""
    # Get pipeline output
    df = pipeline_output('my_pipeline')

    print(f"\n{'='*80}")
    print(f"Date: {get_datetime()}")
    print(f"Raw pipeline output size: {df.shape[0]}")

    # APPLY UNIVERSE FILTERS in pandas (after data is loaded)
    # This avoids the SQLite REAL/TEXT dtype issues

    # Filter 1: Exchange (NYSE, NASDAQ, NYSEMKT only)
    valid_exchanges = {'NYSE', 'NASDAQ', 'NYSEMKT'}
    if 'sharadar_exchange' in df.columns:
        df = df[df['sharadar_exchange'].isin(valid_exchanges)]
        print(f"After exchange filter: {len(df)} stocks")

    # Filter 2: Category (Domestic Common Stock only)
    if 'sharadar_category' in df.columns:
        df = df[df['sharadar_category'] == 'Domestic Common Stock']
        print(f"After category filter: {len(df)} stocks")

    # Filter 3: Exclude ADRs
    if 'sharadar_is_adr' in df.columns:
        # sharadar_is_adr is 0.0 for non-ADRs, 1.0 for ADRs
        df = df[df['sharadar_is_adr'] == 0.0]
        print(f"After ADR filter: {len(df)} stocks")

    # Trim to final universe size
    df = df.nlargest(UNIVERSE_SIZE, 'market_cap')
    print(f"Final universe (top {UNIVERSE_SIZE} by market cap): {len(df)} stocks")

    # Verify filtering worked
    if 'sharadar_exchange' in df.columns:
        print(f"Exchanges: {df['sharadar_exchange'].value_counts().to_dict()}")

    if 'sharadar_category' in df.columns:
        categories = df['sharadar_category'].value_counts().to_dict()
        print(f"Categories: {categories}")

    adr_count = df['sharadar_is_adr'].sum() if 'sharadar_is_adr' in df.columns else 0
    print(f"ADRs in final universe: {int(adr_count)} (should be 0)")

    # Show top 10 by market cap
    print(f"\nTop 10 stocks by market cap:")
    top10 = df.nlargest(10, 'market_cap')[['name', 'market_cap', 'sharadar_exchange', 'sharadar_category']]
    print(top10)
    print(f"{'='*80}\n")

    # Log to FlightLog
    try:
        log_to_flightlog(f'Universe: {len(df)} stocks, ADRs: {int(adr_count)}', level='INFO')
    except:
        pass


def symbol(sym):
    """Gets the security ID for a symbol using Zipline's asset finder."""
    global SYM_SID_CACHE_DICT
    global ASSET_FINDER

    if ASSET_FINDER is None:
        return None

    if SYM_SID_CACHE_DICT.get(sym) is None:
        try:
            asset = ASSET_FINDER.lookup_symbol(sym, as_of_date=None)
            SYM_SID_CACHE_DICT.update({sym: asset})
        except Exception as e:
            print(f"Error getting symbol {sym}: {e}")
            return None

    return SYM_SID_CACHE_DICT.get(sym)
