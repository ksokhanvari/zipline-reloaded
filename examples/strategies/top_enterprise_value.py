"""
Top Enterprise Value Strategy

Simple strategy that:
1. Filters to top 1500 stocks by market cap
2. Selects top 100 by enterprise value
3. Equal weight allocation (1% each)
4. Rebalances weekly (every Monday)
"""

import pandas as pd
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    schedule_function,
    get_datetime,
    symbol,
)

import statsmodels.api as sm
from zipline.pipeline.factors import (Latest, Returns, RollingLinearRegressionOfReturns,
                                     SimpleMovingAverage, SimpleBeta, PercentChange, RSI,
                                     MACDSignal, DailyReturns, AnnualizedVolatility,
                                     AverageDollarVolume, RateOfChangePercentage, VWAP)

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data.db import Database, Column
from zipline.pipeline.filters import StaticAssets
from zipline import run_algorithm
import pandas as pd
import numpy as np 
import pytz

from zipline.pipeline.data.sharadar import SharadarFundamentals





# Define CustomFundamentals Database
class CustomFundamentals(Database):
      """Custom fundamentals database."""

      CODE = "fundamentals"
      LOOKBACK_WINDOW = 252

      # Price and Volume
      RefPriceClose = Column(float)
      RefVolume = Column(float)

      # Company Info
      CompanyCommonName = Column(str)
      GICSSectorName = Column(str)

      # Valuation Metrics
      EnterpriseValue_DailyTimeSeries_ = Column(float)
      CompanyMarketCap = Column(float)

      # Cash Flow
      FOCFExDividends_Discrete = Column(float)

      # Debt and Interest
      InterestExpense_NetofCapitalizedInterest = Column(float)
      Debt_Total = Column(float)

      # Earnings
      EarningsPerShare_Actual = Column(float)
      EarningsPerShare_SmartEstimate_prev_Q = Column(float)
      EarningsPerShare_ActualSurprise = Column(float)
      EarningsPerShare_SmartEstimate_current_Q = Column(float)

      # Growth and Targets
      LongTermGrowth_Mean = Column(float)
      PriceTarget_Median = Column(float)
      Estpricegrowth_percent = Column(float)

      # Rankings
      CombinedAlphaModelSectorRank = Column(float)
      CombinedAlphaModelSectorRankChange = Column(float)
      CombinedAlphaModelRegionRank = Column(float)
      EarningsQualityRegionRank_Current = Column(float)

      # Valuation Ratios
      EnterpriseValueToEBIT_DailyTimeSeriesRatio_ = Column(float)
      EnterpriseValueToEBITDA_DailyTimeSeriesRatio_ = Column(float)
      EnterpriseValueToSales_DailyTimeSeriesRatio_ = Column(float)
      ForwardPEG_DailyTimeSeriesRatio_ = Column(float)
      PriceEarningsToGrowthRatio_SmartEstimate_ = Column(float)
      ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_ = Column(float)
      ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_ = Column(float)
      ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_ = Column(float)

      # Returns
      ReturnOnEquity_SmartEstimat = Column(float)
      ReturnOnAssets_SmartEstimate = Column(float)

      # Margins
      GrossProfitMargin_ActualSurprise = Column(float)

      # Analyst Recommendations
      Recommendation_Median_1_5_ = Column(float)

      # Cash
      CashCashEquivalents_Total = Column(float)

      # Dividends
      Dividend_Per_Share_SmartEstimate = Column(float)

      # Signals
      pred = Column(float)  # VIX prediction signal
      bc1 = Column(float)   # BC signal

      # Sharadar metadata columns for universe filtering (TEXT columns use str type)
      sharadar_exchange = Column(str)
      sharadar_category = Column(str)
      sharadar_location = Column(str)
      sharadar_sector = Column(str)
      sharadar_industry = Column(str)
      sharadar_sicsector = Column(str)
      sharadar_sicindustry = Column(str)
      sharadar_scalemarketcap = Column(str)
      sharadar_is_adr = Column(float)


# Import universe filtering tools
import sys
sys.path.insert(0, '/app/examples/strategies')
from sharadar_filters import (
    ExchangeFilter,
    CategoryFilter,
    ADRFilter,
)

def make_pipeline():
    """
    Create pipeline that selects top 100 stocks by enterprise value
    from the top 1500 by market cap.
    """

    global ASSET_FINDER

    from zipline.data import bundles
    bundle = bundles.load('sharadar')
    ASSET_FINDER = bundle.asset_finder

    


    # Get fundamentals - Price and Volume
    ref_price = CustomFundamentals.RefPriceClose.latest
    ref_volume = CustomFundamentals.RefVolume.latest

    # Company Info
    company_name = CustomFundamentals.CompanyCommonName.latest
    sector = CustomFundamentals.GICSSectorName.latest

    # Valuation Metrics
    enterprise_value = CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest
    market_cap = CustomFundamentals.CompanyMarketCap.latest

    # Cash Flow
    focf_ex_dividends = CustomFundamentals.FOCFExDividends_Discrete.latest

    # Debt and Interest
    interest_expense = CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest
    total_debt = CustomFundamentals.Debt_Total.latest

    # Earnings
    eps_actual = CustomFundamentals.EarningsPerShare_Actual.latest
    eps_estimate_prev_q = CustomFundamentals.EarningsPerShare_SmartEstimate_prev_Q.latest
    eps_actual_surprise = CustomFundamentals.EarningsPerShare_ActualSurprise.latest
    eps_estimate_current_q = CustomFundamentals.EarningsPerShare_SmartEstimate_current_Q.latest

    # Growth and Targets
    long_term_growth = CustomFundamentals.LongTermGrowth_Mean.latest
    price_target = CustomFundamentals.PriceTarget_Median.latest
    est_price_growth = CustomFundamentals.Estpricegrowth_percent.latest

    # Rankings
    alpha_model_sector_rank = CustomFundamentals.CombinedAlphaModelSectorRank.latest
    alpha_model_sector_rank_change = CustomFundamentals.CombinedAlphaModelSectorRankChange.latest
    alpha_model_region_rank = CustomFundamentals.CombinedAlphaModelRegionRank.latest
    earnings_quality_region_rank = CustomFundamentals.EarningsQualityRegionRank_Current.latest

    # Valuation Ratios
    ev_to_ebit = CustomFundamentals.EnterpriseValueToEBIT_DailyTimeSeriesRatio_.latest
    ev_to_ebitda = CustomFundamentals.EnterpriseValueToEBITDA_DailyTimeSeriesRatio_.latest
    ev_to_sales = CustomFundamentals.EnterpriseValueToSales_DailyTimeSeriesRatio_.latest
    forward_peg = CustomFundamentals.ForwardPEG_DailyTimeSeriesRatio_.latest
    pe_to_growth = CustomFundamentals.PriceEarningsToGrowthRatio_SmartEstimate_.latest
    forward_p_to_cf = CustomFundamentals.ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_.latest
    forward_p_to_sales = CustomFundamentals.ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_.latest
    forward_ev_to_ocf = CustomFundamentals.ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_.latest

    # Returns
    roe_estimate = CustomFundamentals.ReturnOnEquity_SmartEstimat.latest
    roa_estimate = CustomFundamentals.ReturnOnAssets_SmartEstimate.latest

    # Margins
    gross_margin_surprise = CustomFundamentals.GrossProfitMargin_ActualSurprise.latest

    # Analyst Recommendations
    rec_median = CustomFundamentals.Recommendation_Median_1_5_.latest

    # Cash
    cash_and_equivalents = CustomFundamentals.CashCashEquivalents_Total.latest

    # Dividends
    dividend_per_share_estimate = CustomFundamentals.Dividend_Per_Share_SmartEstimate.latest

    # Signals
    vix_pred = CustomFundamentals.pred.latest
    bc_signal = CustomFundamentals.bc1.latest

    # # Step 1: Filter to top 1500 by market cap
    top_1500_by_mcap = market_cap.top(150)

    # # Step 2: From those, select top 100 by enterprise value
    # top_100_by_ev = enterprise_value.top(100, mask=top_1500_by_mcap)

    # return Pipeline(
    #     columns={
    #         'market_cap': market_cap,
    #         'enterprise_value': enterprise_value,
    #         'sector': sector,
    #     },
    #     screen=top_100_by_ev,
    # )

# Step 3: Rank by alpha and select top N

    spy_asset = ASSET_FINDER.lookup_symbol('SPY', as_of_date=None)

    iwm_asset = ASSET_FINDER.lookup_symbol('IWM', as_of_date=None)

    qqq_asset = ASSET_FINDER.lookup_symbol('QQQ', as_of_date=None) 

    beta_spy = SimpleBeta(target=spy_asset, regression_length=60)
    beta_iwm = SimpleBeta(target=iwm_asset, regression_length=60)

 #test
    # #####   Key alpha factor  ######
    # eps_actual_surprise - Match
    # Slope-90 -  Match at 150 count universe ! we have some kind of marketcap universe issue
    # focf_ex_dividends - Match 
    # long_term_growth - Match 
    #  enterprise_value  Match
    #  market_cap - MAtch 
    # RelativeStrength - Match no tarding asset universe selection 
    # interest - Match 
    # SMAV - Match 
    #CashReturn - Match !




    # df['doll_vol'] = df['price'] * df['smav']

    # # Calculate cash return
    # df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
    # df['cash_return'] = df['cash_return'].replace([np.inf, -np.inf], np.nan)

    

    # Step 4: Get alpha rank (1 = highest alpha, N = lowest alpha)
    #alpha_rank = alpha.rank(mask=top_n_by_alpha, ascending=False)


    # Step 4: Get alpha rank (1 = highest alpha, N = lowest alpha)
    qqq_sid = qqq_asset = ASSET_FINDER.lookup_symbol('QQQ', as_of_date=None) 


    


    #alpha_rank = RelativeStrength(window_length=140, market_sid=qqq_sid)
    
  # Get ticker info
    

    alpha =   (CustomFundamentals.FOCFExDividends_Discrete.latest - CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest )/enterprise_value

     # Get Sharadar fundamentals for filtering
    

    # Base universe filter using Sharadar
   

    #USEquityPricing.close.latest * SimpleMovingAverage(inputs=[USEquityPricing.volume], window_length=10) #
    #CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest 
    #RelativeStrength(window_length=140, market_sid=qqq_asset.sid) #Slope(window_length=90).slope#.zscore()
    #top_1500_by_mcap = market_cap.top(1500)
    
# TEMPORARY: Disable CustomFilters to debug
    # Just use market cap filter without metadata filters
    top_1500_by_mcap = CustomFundamentals.CompanyMarketCap.latest.top(1500)

    # Add benchmark asset
    tradable_filter = top_1500_by_mcap | StaticAssets([symbol('IBM')])

    TOP_N = 100  # Change this to select different number of stocks
    top_n_by_alpha = alpha.top(TOP_N, mask=top_1500_by_mcap)

    # DEBUG: Create filters but don't apply them in the screen yet
    # We'll check the metadata values first
    exchange_filter = ExchangeFilter(CustomFundamentals.sharadar_exchange)
    category_filter = CategoryFilter(CustomFundamentals.sharadar_category)
    adr_filter = ADRFilter(CustomFundamentals.sharadar_is_adr)

    return Pipeline(
        columns={
            # Core metrics
            'market_cap': market_cap,
            'enterprise_value': enterprise_value,
            'sector': sector,

            # Alpha and rank
            'alpha': alpha,
           # 'alpha_rank': alpha_rank,

            # Add any other columns you want to see
            'roe': roe_estimate,
            'roa': roa_estimate,
            'ev_to_ebitda': ev_to_ebitda,
            'long_term_growth': long_term_growth,
            'price_target': price_target,


            'price': USEquityPricing.close.latest,
            'volume':USEquityPricing.volume.latest,
            'beta60SPY': beta_spy,

            'beta60IWM': beta_iwm,

            'smav': SimpleMovingAverage(
                inputs=[USEquityPricing.volume],
                window_length=10
            ),

            'slope120': Slope(window_length=120, mask=tradable_filter).slope.zscore(),
            'slope220': Slope(window_length=220, mask=tradable_filter).slope.zscore(),
            'slope90':  Slope(window_length=90, mask=tradable_filter).slope.zscore(),
            'slope30':  Slope(window_length=30, mask=tradable_filter).slope.zscore(),

            
           

            'RS140_QQQ': RelativeStrength(window_length=140, market_sid=qqq_asset.sid),
            'RS160_QQQ': RelativeStrength(window_length=160, market_sid=qqq_asset.sid),
            'RS180_QQQ': RelativeStrength(window_length=180, market_sid=qqq_asset.sid),

            'Ret60': Returns(window_length=60, mask=tradable_filter),
            'Ret120': Returns(window_length=120, mask=tradable_filter),
            'Ret220': Returns(window_length=220, mask=tradable_filter),

           
            'vol': Volatility(window_length=10, mask=tradable_filter),

            # VIX & BC from custom fundamentals
            'vixflag': CustomFundamentals.pred.latest, #PreviousValue(inputs=[CustomFundamentals.pred]),
            'vixflag0': CustomFundamentals.pred.latest,
            'bc1': CustomFundamentals.bc1.latest,

            # Sharadar metadata columns - MUST be in columns dict for Pipeline to load them!
            'sharadar_exchange': CustomFundamentals.sharadar_exchange.latest,
            'sharadar_category': CustomFundamentals.sharadar_category.latest,
            'sharadar_is_adr': CustomFundamentals.sharadar_is_adr.latest,
        },
        screen=top_n_by_alpha,
    )

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Attach pipeline
    attach_pipeline(make_pipeline(), 'top_ev')

    # Schedule rebalancing every Monday at market open
    schedule_function(
        rebalance,
        date_rule=date_rules.week_start()
       
    )

    # Track some metrics
    context.rebalance_count = 0

    print("Strategy initialized: Top Enterprise Value")
    print("  Universe: Top 1500 by market cap")
    print("  Selection: Top 100 by enterprise value")
    print("  Rebalance: Weekly (Mondays)")
    print("  Allocation: Equal weight (1% each)")


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    # Get pipeline output
    context.output = pipeline_output('top_ev')

    # Filter to only tradeable stocks
    all_stocks = context.output.index.tolist()
    tradeable_stocks = [stock for stock in all_stocks if data.can_trade(stock)]

    # Debug logging - FIRST Monday only for detailed diagnostics
    if get_datetime().weekday() == 0 and context.rebalance_count == 0:  # First Monday
        print(f"\n=== PIPELINE DIAGNOSTIC (First Monday: {get_datetime().date()}) ===")
        print(f"Pipeline returned {len(all_stocks)} stocks")
        print(f"Pipeline columns: {context.output.columns.tolist()}")
        if len(context.output) > 0:
            print(f"\nFirst 3 rows of Pipeline output:")
            print(context.output.head(3))
            print(f"\nMetadata column values:")
            print(f"  sharadar_exchange: {context.output['sharadar_exchange'].value_counts().to_dict()}")
            print(f"  sharadar_category: {context.output['sharadar_category'].value_counts().to_dict()}")
            print(f"  sharadar_is_adr: {context.output['sharadar_is_adr'].value_counts().to_dict()}")
        else:
            print("\nPipeline output is EMPTY!")
            print("This means CustomFilters are rejecting ALL stocks.")
            print("Likely cause: metadata columns are empty/None in the database for the test date.")

    # Regular Monday logging
    if get_datetime().weekday() == 0:  # Monday
        print(f"{get_datetime().date()}: Pipeline returned {len(all_stocks)} stocks, {len(tradeable_stocks)} tradeable")
        if len(all_stocks) > 0 and len(tradeable_stocks) == 0:
            print(f"  WARNING: All {len(all_stocks)} stocks filtered out by can_trade()")
        if len(all_stocks) == 0:
            print(f"  WARNING: Pipeline returned ZERO stocks!")

    # Update target positions for today
    context.target_positions = tradeable_stocks


def rebalance(context, data):
    """
    Execute rebalancing logic.
    Called every Monday.
    """

    # Sort output by market cap (highest first)
    sorted_output = context.output.sort_values(by='market_cap', ascending=False)

    print(f"\n{get_datetime().date()} REBALANCE:")
    print(f"  Pipeline output: {len(sorted_output)} stocks")
    print(f"  Target positions: {len(context.target_positions)} stocks")
    if len(sorted_output) > 0:
        print(f"  Top 10 by market cap: {sorted_output.index[:10].tolist()}")

    target_positions = context.target_positions

    if len(target_positions) == 0:
        print(f"  WARNING: No positions to rebalance - exiting")
        return

    # Equal weight allocation
    target_weight = 1.0 / len(target_positions)

    # Get current positions
    current_positions = list(context.portfolio.positions.keys())

    # Close positions not in target
    for asset in current_positions:
        if asset not in target_positions:
            # Check if we can trade it before trying to close
            if data.can_trade(asset):
                order_target_percent(asset, 0.0)
            else:
                # Can't trade it - it will auto-liquidate on delisting
                print(f"  WARNING: Cannot close {asset.symbol} - not tradeable (likely delisted)")

    # Open/adjust target positions (already filtered for tradeability in before_trading_start)
    for asset in target_positions:
        # Double-check tradeability before ordering (defensive programming)
        if data.can_trade(asset):
            try:
                order_target_percent(asset, target_weight)
            except Exception as e:
                print(f"{get_datetime()}: Could not order {asset.symbol}: {e}")

    context.rebalance_count += 1

    # Log every 4 weeks (monthly)
    if context.rebalance_count % 4 == 0:
        print(f"{get_datetime()}: Rebalanced {len(target_positions)} positions "
              f"at {target_weight*100:.2f}% each (rebalance #{context.rebalance_count})")


def analyze(context, perf):
    """
    Called once at the end of the backtest.
    """
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)

    try:
        total_return = (perf['portfolio_value'].iloc[-1] / perf['portfolio_value'].iloc[0] - 1) * 100
        print(f"Total Return: {total_return:.2f}%")
    except Exception as e:
        print(f"Total Return: Unable to calculate ({e})")

    try:
        print(f"Final Portfolio Value: ${perf['portfolio_value'].iloc[-1]:,.2f}")
    except Exception as e:
        print(f"Final Portfolio Value: Unable to calculate ({e})")

    print(f"Total Rebalances: {context.rebalance_count}")

    try:
        avg_positions = perf['positions'].apply(len).mean()
        print(f"Avg Positions: {avg_positions:.1f}")
    except Exception as e:
        print(f"Avg Positions: Unable to calculate ({e})")

    print("="*70)


# Make date_rules and time_rules available
from zipline.api import date_rules, time_rules


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Top Enterprise Value Strategy')
    parser.add_argument('--algo-name', default='TopEV', help='Algorithm name for logging')
    parser.add_argument('--output-file', default='top_ev_results.pkl', help='Output filename')
    parser.add_argument('--start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Starting capital')

    args = parser.parse_args()

    # Run the backtest
    start = pd.Timestamp(args.start, tz='UTC')
    end = pd.Timestamp(args.end, tz='UTC')
    capital_base = args.capital

    print(f"Starting backtest: {args.algo_name}")
    print(f"  Period: {start.date()} to {end.date()}")
    print(f"  Capital: ${capital_base:,.0f}")
    print()

    results = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        before_trading_start=before_trading_start,
        capital_base=capital_base,
        bundle='sharadar',
    )

    # Save results
    output_path = f'/data/backtest_results/{args.output_file}'
    results.to_pickle(output_path)
    print(f"\nResults saved to: {output_path}")



class Slope(CustomFactor):
    """Linear regression slope of price data"""
    inputs = [USEquityPricing.close]
    outputs = ['slope', 'rsq']

    def compute(self, today, assets, out, closes):
        try:
            mask = np.isnan(closes)
            closes[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), closes[~mask])
        except:
            pass

        lr = sm.OLS(closes, sm.add_constant(range(-len(closes) + 1, 1))).fit()
        out.slope[:] = lr.params[-1]
        out.rsq[:] = 0

class RelativeStrength(CustomFactor):
    """Relative strength against a benchmark"""
    params = ('market_sid',)
    inputs = [USEquityPricing.close]
    window_safe = True

    def compute(self, today, assets, out, close, market_sid):
        rsRankTable = pd.DataFrame(index=assets)
        returns = (close[-22] - close[0]) / close[0]
        market_idx = assets.get_loc(market_sid)
        rsRankTable["RS"] = (((returns + 1) / (returns[market_idx] + 1)) - 1) * 100
        out[:] = rsRankTable["RS"]


class Volatility(CustomFactor):
    """Price volatility"""
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close_prices):
        daily_returns = np.diff(close_prices, axis=0) / close_prices[:-1]
        volatility = np.std(daily_returns, axis=0)
        out[:] = volatility
