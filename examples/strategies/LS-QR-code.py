"""
Long-Short Equity Trading Algorithm
-----------------------------------
This algorithm implements a long-short equity strategy that selects securities based on:
1. Fundamental factors (cash return, enterprise value, growth metrics)
2. Technical factors (price momentum, relative strength)
3. Sentiment indicators
4. Machine learning predictions

The algorithm dynamically adjusts exposure based on market trends, volatility conditions,
and drawdown protection mechanisms.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats.mstats import gmean, winsorize
from datetime import timedelta, datetime
from pytz import timezone
import math
import logging

# Machine Learning & Statistical Models
from sklearn.svm import SVR, LinearSVR
from sklearn.mixture import GaussianMixture
from sklearn import (linear_model, decomposition, ensemble, preprocessing, isotonic, metrics, impute)
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor

# Zipline & QuantRocket imports
import zipline.api as algo
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.utils.numpy_utils import repeat_first_axis, repeat_last_axis, rolling_window
from zipline.pipeline.data.db import Database, Column
from zipline.pipeline.factors import (Latest, Returns, RollingLinearRegressionOfReturns, 
                                     SimpleMovingAverage, SimpleBeta, PercentChange, RSI, 
                                     MACDSignal, DailyReturns, AnnualizedVolatility, 
                                     AverageDollarVolume, RateOfChangePercentage, VWAP)
from zipline.pipeline.classifiers import CustomClassifier
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline import sharadar
from zipline.pipeline.filters.master import Universe
from zipline.finance import commission, slippage
from zipline.pipeline.periodic import (PeriodicCAGR, AllPeriodsIncreasing, PeriodicAverage, 
                                      PeriodicPercentChange, AllPeriodsAbove)
from zipline.pipeline.data import master

from quantrocket.history import download_history_file
from quantrocket.master import get_securities
from quantrocket import get_prices
from quantrocket.zipline import set_default_bundle
from quantrocket.flightlog import FlightlogHandler

from operator import itemgetter


from zipline.pipeline import sharadar



#################################################
# GLOBAL CONFIGURATION VARIABLES
#################################################

# ML Learning Configuration
ML_GLOBAL_COUNTER = 0
ML_MODEL_REUSE_LIMIT = 1
ML_CLASSIFIER_GLOBAL = 0

# Symbol-SID Cache for performance
SYM_SID_CACHE_DICT = {}

# Portfolio Construction Parameters
UNIVERSE_SIZE = 1500        # Top N stocks by market cap to consider initially
FILTERED_UNIVERSE_SIZE = 500 # Final universe size after screening
TOP_MOMENTUM_STOCKS = 20     # Number of momentum stocks to include
LONG_PORTFOLIO_SIZE = 50    # Total number of long positions
SHORT_PORTFOLIO_SIZE = 50    # Total number of short positions
MAX_POSITION_SIZE_LONG = 0.06 # Maximum weight for a single long position
MAX_POSITION_SIZE_SHORT = 0.02 # Maximum weight for a single short position
MAX_SECTOR_EXPOSURE = 0.5
MAX_MARKET_CAP_BUCKET_EXPOSURE = 0.33
CORRELATION_THRESHOLD = 0.7
MIN_DOLLAR_VOLUME = 10e7
# Risk Management Parameters
DRAWDOWN_FACTOR_MULTIPLIER = 5.0  # Multiplier for drawdown protection
SLIPPAGE_SPREAD = 0.05           # Fixed slippage in dollars
COMMISSION_COST = 0.01          # Per-share commission
MIN_TRADE_COST = 1.00          # Minimum commission per trade

# Market Regime Parameters
TREND_UPWARD_FACTOR = 1.5       # Position sizing factor for upward trends
TREND_NEUTRAL_FACTOR = 1.0      # Position sizing factor for neutral trends
TREND_DOWNWARD_FACTOR = 0.5     # Position sizing factor for downward trends
BETA_NORMAL_THRESHOLD = 0.4     # Target beta for IWM in normal conditions

# Moving Average Parameters
HULL_MA_PERIOD = 80            # Period for Hull moving average
MA_SHORT_PERIOD = 21           # Short-term MA period
MA_MEDIUM_PERIOD = 50          # Medium-term MA period
MA_LONG_PERIOD = 85            # Long-term MA period

# Seasonal Parameters
GROWTH_SEASON_MONTHS = {4, 5, 6, 7, 8, 9, 10, 11, 12}  # Months to overweight growth
SHORT_RESTRICTED_MONTHS = {1, 2, 3, 5, 7, 8, 9, 10, 11, 12}  # Months with reduced short exposure

#################################################
# DATABASE DEFINITIONS
#################################################

class CustomFundamentals(Database):
    """Primary fundamentals database with core company financial data."""
    
    CODE = "refe-fundamentals"
    LOOKBACK_WINDOW = 240
    
    # Company identifiers
    Symbol = Column(object)
    Inobjectument = Column(object)
    CompanyCommonName = Column(object)
    GICSSectorName = Column(object)
    
    # Price and volume reference data
    RefPriceClose = Column(float)
    RefVolume = Column(float)
    
    # Valuation metrics
    EnterpriseValue_DailyTimeSeries_ = Column(float)
    CompanyMarketCap = Column(float)
    
    # Cash flow metrics
    FOCFExDividends_Discrete = Column(float)
    InterestExpense_NetofCapitalizedInterest = Column(float)
    CashFlowComponent_Current = Column(float)
    CashFlowPerShare_BrokerEstimate = Column(float)
    FreeCashFlowPerShare_BrokerEstimate = Column(float)
    
    # Debt metrics
    Debt_Total = Column(float)
    CashCashEquivalents_Total = Column(float)
    
    # Earnings metrics
    EarningsPerShare_Actual = Column(float)
    EarningsPerShare_SmartEstimate_prev_Q = Column(float)
    EarningsPerShare_ActualSurprise = Column(float)
    EarningsPerShare_SmartEstimate_current_Q = Column(float)
    EPS_SurpirsePrct_prev_Q = Column(float)
    
    # Growth and target metrics
    LongTermGrowth_Mean = Column(float)
    PriceTarget_Median = Column(float)
    Estpricegrowth_percent = Column(float)
    
    # Alpha model rankings
    CombinedAlphaModelSectorRank = Column(float)
    CombinedAlphaModelSectorRankChange = Column(float)
    CombinedAlphaModelRegionRank = Column(float)
    EarningsQualityRegionRank_Current = Column(float)
    
    # Valuation ratios
    EnterpriseValueToEBIT_DailyTimeSeriesRatio_= Column(float)
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_= Column(float)
    EnterpriseValueToSales_DailyTimeSeriesRatio_ = Column(float)
    ForwardPEG_DailyTimeSeriesRatio_= Column(float)
    PriceEarningsToGrowthRatio_SmartEstimate_= Column(float)
    ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_ = Column(float)
    ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_ = Column(float)
    ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_ = Column(float)
    
    # Other metrics
    TradeDate = Column(object)
    Dividend_Per_Share_SmartEstimate= Column(float)
    ReturnOnInvestedCapital_BrokerEstimate = Column(float)
    
    # Analyst metrics
    Recommendation_NumberOfTotal = Column(float)
    Recommendation_Median_1_5_= Column(float)
    Recommendation_NumberOfStrongBuy = Column(float)
    Recommendation_NumberOfBuy = Column(float)
    Recommendation_Mean_1_5_ = Column(float)
    ReturnOnCapitalEmployed_Mean = Column(float)
    
    # Return metrics
    ReturnOnEquity_SmartEstimat = Column(float)
    ReturnOnAssets_SmartEstimate = Column(float)
    
    # Profitability metrics
    GrossProfitMargin_ = Column(float)
    GrossProfitMargin_ActualSurprise = Column(float)
    
    # Prediction columns
    pred = Column(float)
    forcast = Column(float)

class CustomFundamentals2(Database):
    """Sentiment data database."""
    
    CODE = "refe-fundamentals-sent"
    LOOKBACK_WINDOW = 200
    
    sent2pol = Column(float)
    sent2sub = Column(float)
    sentvad = Column(float)
    sentvad_neg = Column(float)
    ChatGPT = Column(float)
    SentVal1 = Column(float)
    SentVal2 = Column(float)
    SentVal3 = Column(float)
    SentVal4 = Column(float)

class CustomFundamentals3(Database):
    """Alternative sentiment metrics database."""
    
    CODE = "refe-fundamentals-alex"
    LOOKBACK_WINDOW = 200
     
    Sentiment = Column(float)
    Confidence = Column(float)
    Novelty = Column(float)
    Relevance = Column(float)
    MarketImpactScore = Column(float)
    Prob_POS = Column(float)
    Prob_NTR = Column(float)
    Prob_NEG = Column(float)
    pred = Column(float)

class CustomFundamentals4(Database):
    """VIX data database."""
    
    CODE = "vixdata"
    LOOKBACK_WINDOW = 200
    
    pred = Column(float)
    
   
class CustomFundamentals5(Database):
    """Revenue surprise database."""
    
    CODE = "refe-fund-test-revesur"
    LOOKBACK_WINDOW = 240

    Revenue_ActualSurprise = Column(float)

class CustomFundamentals6(Database):
    """Earnings calendar database."""
    
    CODE = "refe-fundamentals-ecal"
    LOOKBACK_WINDOW = 240

    Earn_Date = Column(object)
    Earn_Collection_Date = Column(object)

class CustomFundamentals7(Database):
    """Financial ratios database."""
    
    CODE = "refe-fundamentals-finratios"
    LOOKBACK_WINDOW = 240

    # Identifiers
    period = Column(object)
    companyName = Column(object)
    calendarYear = Column(float)
    
    # Liquidity ratios
    currentRatio = Column(float)
    quickRatio = Column(float)
    cashRatio = Column(float)
    
    # Efficiency ratios
    daysOfSalesOutstanding = Column(float)
    daysOfInventoryOutstanding = Column(float)
    operatingCycle = Column(float)
    daysOfPayablesOutstanding = Column(float)
    cashConversionCycle = Column(float)
    receivablesTurnover = Column(float)
    payablesTurnover = Column(float)
    inventoryTurnover = Column(float)
    fixedAssetTurnover = Column(float)
    assetTurnover = Column(float)
    
    # Profitability ratios
    grossProfitMargin = Column(float)
    operatingProfitMargin = Column(float)
    pretaxProfitMargin = Column(float)
    netProfitMargin = Column(float)
    effectiveTaxRate = Column(float)
    returnOnAssets = Column(float)
    returnOnEquity = Column(float)
    returnOnCapitalEmployed = Column(float)
    netIncomePerEBT = Column(float)
    ebtPerEbit = Column(float)
    ebitPerRevenue = Column(float)
    
    # Debt ratios
    debtRatio = Column(float)
    debtEquityRatio = Column(float)
    longTermDebtToCapitalization = Column(float)
    totalDebtToCapitalization = Column(float)
    interestCoverage = Column(float)
    cashFlowToDebtRatio = Column(float)
    companyEquityMultiplier = Column(float)
    
    # Cash flow ratios
    operatingCashFlowPerShare = Column(float)
    freeCashFlowPerShare = Column(float)
    cashPerShare = Column(float)
    payoutRatio = Column(float)
    operatingCashFlowSalesRatio = Column(float)
    freeCashFlowOperatingCashFlowRatio = Column(float)
    cashFlowCoverageRatios = Column(float)
    shortTermCoverageRatios = Column(float)
    capitalExpenditureCoverageRatio = Column(float)
    dividendPaidAndCapexCoverageRatio = Column(float)
    dividendPayoutRatio = Column(float)
    
    # Valuation ratios
    priceBookValueRatio = Column(float)
    priceToBookRatio = Column(float)
    priceToSalesRatio = Column(float)
    priceEarningsRatio = Column(float)
    priceToFreeCashFlowsRatio = Column(float)
    priceToOperatingCashFlowsRatio = Column(float)
    priceCashFlowRatio = Column(float)
    priceEarningsToGrowthRatio = Column(float)
    priceSalesRatio = Column(float)
    dividendYield = Column(float)
    enterpriseValueMultiple = Column(float)
    priceFairValue = Column(float)

class CustomFundamentals8(Database):
    """Forward EV/FCF database."""
    
    CODE = "refe-fundamentals-eps"
    LOOKBACK_WINDOW = 240

    ForwardEVFreeCashFlow_SmartEstimate_ = Column(float)

class CustomFundamentals9(Database):
    """BC data database."""
    
    CODE = "bcdata"
    LOOKBACK_WINDOW = 200
    
    bc1 = Column(float)

# class CustomFundamentals10(Database):
#     """BC data database IWM."""
    
#     CODE = "bcdataiwm"
#     LOOKBACK_WINDOW = 200
    
#     bc2 = Column(float

#################################################
# CUSTOM FACTORS
#################################################
from zipline.pipeline import Factor, CustomFactor
from zipline.pipeline.data import USEquityPricing
import numpy as np
import pandas as pd

class MoneyFlowFactor(CustomFactor):
    """
    Money Flow Factor - measures the relationship between price and volume
    
    Money Flow combines price and volume to identify buying/selling pressure:
    - Positive money flow: volume on up days
    - Negative money flow: volume on down days
    - Money Flow Index: ratio of positive to total money flow
    
    Higher values indicate stronger buying pressure
    """
    
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
    window_length = 14  # Default 14-day period
    
    def compute(self, today, assets, out, high, low, close, volume):
        """
        Compute Money Flow Index
        
        Formula:
        1. Typical Price = (High + Low + Close) / 3
        2. Money Flow = Typical Price * Volume
        3. Positive Money Flow = sum of money flow when typical price increases
        4. Negative Money Flow = sum of money flow when typical price decreases
        5. Money Flow Index = 100 - (100 / (1 + (Positive MF / Negative MF)))
        """
        
        # Calculate typical price for each day
        typical_price = (high + low + close) / 3.0
        
        # Calculate money flow for each day
        money_flow = typical_price * volume
        
        # Initialize output
        out[:] = np.nan
        
        for i in range(len(assets)):
            asset_typical_price = typical_price[:, i]
            asset_money_flow = money_flow[:, i]
            
            # Skip if we don't have enough data
            if len(asset_typical_price) < 2:
                continue
                
            # Calculate price changes
            price_changes = np.diff(asset_typical_price)
            
            # Separate positive and negative money flows
            positive_mf = 0.0
            negative_mf = 0.0
            
            for j in range(len(price_changes)):
                if price_changes[j] > 0:
                    positive_mf += asset_money_flow[j + 1]
                elif price_changes[j] < 0:
                    negative_mf += asset_money_flow[j + 1]
            
            # Calculate Money Flow Index
            if negative_mf == 0:
                out[i] = 100.0  # All positive flow
            elif positive_mf == 0:
                out[i] = 0.0    # All negative flow
            else:
                money_ratio = positive_mf / negative_mf
                out[i] = 100.0 - (100.0 / (1.0 + money_ratio))


class SimpleMoneyFlowFactor(CustomFactor):
    """
    Simplified Money Flow Factor - easier to understand and compute
    
    This version calculates the cumulative money flow over the window period:
    Money Flow = Price Change * Volume
    
    Positive values indicate net buying pressure
    Negative values indicate net selling pressure
    """
    
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 14
    
    def compute(self, today, assets, out, close, volume):
        """
        Compute simple money flow as sum of (price_change * volume)
        """
        
        out[:] = np.nan
        
        for i in range(len(assets)):
            asset_close = close[:, i]
            asset_volume = volume[:, i]
            
            if len(asset_close) < 2:
                continue
            
            # Calculate price changes
            price_changes = np.diff(asset_close)
            
            # Calculate money flow for each period
            money_flows = price_changes * asset_volume[1:]  # Volume corresponding to price change
            
            # Sum all money flows in the window
            out[i] = np.sum(money_flows)


class VolumeWeightedMoneyFlowFactor(CustomFactor):
    """
    Volume Weighted Money Flow - focuses on volume-weighted price movements
    
    This factor weights price changes by their corresponding volume,
    giving more importance to high-volume movements.
    """
    
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 20
    
    def compute(self, today, assets, out, close, volume):
        """
        Compute volume-weighted money flow
        """
        
        out[:] = np.nan
        
        for i in range(len(assets)):
            asset_close = close[:, i]
            asset_volume = volume[:, i]
            
            if len(asset_close) < 2 or np.sum(asset_volume) == 0:
                continue
            
            # Calculate percentage price changes
            price_returns = np.diff(asset_close) / asset_close[:-1]
            
            # Weight by volume
            volume_weights = asset_volume[1:]  # Volume corresponding to returns
            
            # Calculate volume-weighted average return
            if np.sum(volume_weights) > 0:
                vwap_return = np.sum(price_returns * volume_weights) / np.sum(volume_weights)
                out[i] = vwap_return * np.sum(volume_weights)  # Scale by total volume
            else:
                out[i] = 0.0


class ChaikinMoneyFlowFactor(CustomFactor):
    """
    Chaikin Money Flow (CMF) - measures money flow volume over a period
    
    CMF uses the close location value (CLV) to determine money flow:
    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = CLV * Volume
    CMF = Sum of Money Flow Volume / Sum of Volume
    
    Values range from -1 to +1:
    - Positive values indicate buying pressure
    - Negative values indicate selling pressure
    """
    
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
    window_length = 21  # Standard 21-day period
    
    def compute(self, today, assets, out, high, low, close, volume):
        """
        Compute Chaikin Money Flow
        """
        
        out[:] = np.nan
        
        for i in range(len(assets)):
            asset_high = high[:, i]
            asset_low = low[:, i]
            asset_close = close[:, i]
            asset_volume = volume[:, i]
            
            # Calculate Close Location Value (CLV)
            high_low_diff = asset_high - asset_low
            
            # Avoid division by zero
            clv = np.zeros_like(asset_close)
            mask = high_low_diff != 0
            
            clv[mask] = ((asset_close[mask] - asset_low[mask]) - 
                        (asset_high[mask] - asset_close[mask])) / high_low_diff[mask]
            
            # Calculate Money Flow Volume
            money_flow_volume = clv * asset_volume
            
            # Calculate CMF
            total_volume = np.sum(asset_volume)
            if total_volume > 0:
                out[i] = np.sum(money_flow_volume) / total_volume
            else:
                out[i] = 0.0


class MoneyFlowIndexFactor(CustomFactor):
    """
    Classic Money Flow Index (MFI) - the "Volume RSI"
    
    Similar to RSI but incorporates volume:
    - Typical Price = (High + Low + Close) / 3
    - Money Flow = Typical Price * Volume
    - Positive/Negative Money Flow based on typical price changes
    - MFI = 100 - (100 / (1 + Money Ratio))
    
    Values range from 0 to 100:
    - Above 80: Potentially overbought
    - Below 20: Potentially oversold
    """
    
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
    window_length = 14
    
    def compute(self, today, assets, out, high, low, close, volume):
        """
        Compute Money Flow Index
        """
        
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        
        out[:] = np.nan
        
        for i in range(len(assets)):
            asset_typical_price = typical_price[:, i]
            asset_volume = volume[:, i]
            
            if len(asset_typical_price) < 2:
                continue
            
            # Calculate money flow
            money_flow = asset_typical_price * asset_volume
            
            # Calculate price changes
            price_changes = np.diff(asset_typical_price)
            
            # Separate positive and negative money flows
            positive_mf = 0.0
            negative_mf = 0.0
            
            for j in range(len(price_changes)):
                if price_changes[j] > 0:
                    positive_mf += money_flow[j + 1]
                elif price_changes[j] < 0:
                    negative_mf += money_flow[j + 1]
            
            # Calculate Money Flow Index
            if negative_mf == 0:
                out[i] = 100.0
            elif positive_mf == 0:
                out[i] = 0.0
            else:
                money_ratio = positive_mf / negative_mf
                out[i] = 100.0 - (100.0 / (1.0 + money_ratio))




class Above200DMA(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 200
    def compute(self, today, assets, out, close):
        # Get the most recent closing price (last row in window)
        latest_close = close[-1]
        # Calculate the 200-day moving average
        ma_200 = np.mean(close, axis=0)
        # Set 1 if price > MA, else 0
        out[:] = (latest_close > ma_200).astype(int)

class StochasticOscillator(CustomFactor):
    """
    Custom Zipline factor to compute the Stochastic Oscillator.
    Default lookback period is 14 days.
    """
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 14  # Default lookback period

    def compute(self, today, assets, out, high, low, close):
        highest_high = np.max(high, axis=0)
        lowest_low = np.min(low, axis=0)
        out[:] = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100

class StochasticOscillatorWeekly(CustomFactor):
    """
    Custom Zipline factor to compute the Stochastic Oscillator on weekly data.
    It aggregates daily prices into weekly high, low, and close before computation.
    """
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 20 * 5  # 20 weeks of daily data

    def compute(self, today, assets, out, high, low, close):
        # Reshape data to weekly (assuming 5 trading days per week)
        high_weekly = high.reshape(-1, 5, high.shape[1]).max(axis=1)
        low_weekly = low.reshape(-1, 5, low.shape[1]).min(axis=1)
        close_weekly = close.reshape(-1, 5, close.shape[1])[:, -1]
        
        highest_high = np.max(high_weekly, axis=0)
        lowest_low = np.min(low_weekly, axis=0)
        out[:] = ((close_weekly[-1] - lowest_low) / (highest_high - lowest_low)) * 100



class TenkanSen(CustomFactor):
    """
    Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    """
    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 9
    window_safe = True
    
    def compute(self, today, assets, out, highs, lows):
        highest_high = np.max(highs, axis=0)
        lowest_low = np.min(lows, axis=0)
        out[:] = (highest_high + lowest_low) / 2


class KijunSen(CustomFactor):
    """
    Kijun-sen (Base Line): (26-period high + 26-period low)/2
    """
    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 26
    window_safe = True
    
    def compute(self, today, assets, out, highs, lows):
        highest_high = np.max(highs, axis=0)
        lowest_low = np.min(lows, axis=0)
        out[:] = (highest_high + lowest_low) / 2


class SenkouSpanA(CustomFactor):
    """
    Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    """
    inputs = [TenkanSen(), KijunSen()]
    window_length = 1
    window_safe = True
    
    def compute(self, today, assets, out, tenkan_sen, kijun_sen):
        out[:] = (tenkan_sen[0] + kijun_sen[0]) / 2


class SenkouSpanB(CustomFactor):
    """
    Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    """
    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 52
    window_safe = True
    
    def compute(self, today, assets, out, highs, lows):
        highest_high = np.max(highs, axis=0)
        lowest_low = np.min(lows, axis=0)
        out[:] = (highest_high + lowest_low) / 2


class ChikouSpan(CustomFactor):
    """
    Chikou Span (Lagging Span): Current closing price plotted 26 periods behind
    """
    inputs = [USEquityPricing.close]
    window_length = 26
    window_safe = True
    
    def compute(self, today, assets, out, closes):
        out[:] = closes[-1]


class CloudDisplacement(CustomFactor):
    """
    Helper class to implement the 26-period displacement of the cloud
    """
    inputs = [SenkouSpanA(), SenkouSpanB()]
    window_length = 26
    window_safe = True
    
    def compute(self, today, assets, out, span_a, span_b):
        # Use values from 26 periods ago
        out[:] = np.column_stack((span_a[0], span_b[0]))


class IchimokuSignal(CustomFactor):
    """
    Creates a single factor that ranks assets from most bullish (highest) to most bearish (lowest).
    
    The score is composed of several components of the Ichimoku system:
    1. Position relative to cloud (above, inside, below)
    2. Tenkan-Kijun relationship (above, crossing, below)
    3. Cloud color (green, flat, red)
    4. Chikou span position
    
    Higher values indicate more bullish signals, lower values indicate more bearish.
    """
    inputs = [
        USEquityPricing.close, 
        TenkanSen(), 
        KijunSen(), 
        SenkouSpanA(), 
        SenkouSpanB(), 
        ChikouSpan(),
    ]
    window_length = 1
    window_safe = True
    
    def compute(self, today, assets, out, closes, tenkan, kijun, span_a, span_b, chikou):
        # Extract the values from the inputs
        close = closes[0]
        tenkan_val = tenkan[0]
        kijun_val = kijun[0]
        span_a_val = span_a[0]
        span_b_val = span_b[0]
        chikou_val = chikou[0]
        
        # Initialize base scores
        out[:] = 0.0
        
        # 1. Position relative to cloud (range: -2 to +2)
        # Above cloud: +2, Below cloud: -2, Inside cloud: Score based on position within cloud
        above_cloud = (close > span_a_val) & (close > span_b_val)
        below_cloud = (close < span_a_val) & (close < span_b_val)
        inside_cloud = ~(above_cloud | below_cloud)
        
        # Set cloud position scores
        out[above_cloud] += 2.0
        out[below_cloud] -= 2.0
        
        # For inside cloud, score based on relative position within the cloud
        for i in np.where(inside_cloud)[0]:
            cloud_top = max(span_a_val[i], span_b_val[i])
            cloud_bottom = min(span_a_val[i], span_b_val[i])
            cloud_range = cloud_top - cloud_bottom
            if cloud_range > 0:
                # Normalize position within cloud to range -1 to +1
                relative_pos = (close[i] - cloud_bottom) / cloud_range * 2 - 1
                out[i] += relative_pos
        
        # 2. Tenkan-Kijun relationship (range: -1.5 to +1.5)
        tk_diff = tenkan_val - kijun_val
        # Normalize to a reasonable scale (e.g., % of price)
        # Add small epsilon to prevent division by zero
        tk_norm = tk_diff / (close + 1e-8) * 100  # Convert to percentage of price
        # Cap at [-1.5, 1.5] range
        out += np.clip(tk_norm, -1.5, 1.5)
        
        # 3. Cloud color (range: -1 to +1)
        cloud_diff = span_a_val - span_b_val
        # Normalize to a reasonable scale
        cloud_color = cloud_diff / (close + 1e-8) * 100  # Convert to percentage of price
        # Cap at [-1, 1] range
        out += np.clip(cloud_color, -1.0, 1.0)
        
        # 4. Chikou span position (range: -1.5 to +1.5)
        chikou_diff = chikou_val - close
        # Normalize to a reasonable scale
        chikou_norm = chikou_diff / (close + 1e-8) * 100  # Convert to percentage of price
        # Cap at [-1.5, 1.5] range
        out += np.clip(chikou_norm, -1.5, 1.5)




class Slope(CustomFactor):
    """
    Calculates the linear regression slope of price data.
    
    Returns both slope coefficient and R-squared value of the regression.
    """
    inputs = [USEquityPricing.close]
    outputs = ['slope', 'rsq']
    
    def compute(self, today, assets, out, closes): 
        try:
            # Fill NaN values with interpolation
            mask = np.isnan(closes)
            closes[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), closes[~mask])
        except:
            pass
        
        # Run linear regression
        lr = sm.OLS(closes, sm.add_constant(range(-len(closes) + 1, 1))).fit()
        out.slope[:] = lr.params[-1]
        out.rsq[:] = 0  # Not currently using R-squared

class RelativeStrength(CustomFactor):
    """
    Computes relative strength against a benchmark:
    (% change in stock / % change in benchmark - 1) * 100
    """
    params = ('market_sid',)
    inputs = [USEquityPricing.close]
    window_safe = True
    
    def compute(self, today, assets, out, close, market_sid):
        rsRankTable = pd.DataFrame(index=assets)
        
        # Calculate returns over window length
        returns = (close[-22] - close[0]) / close[0]
        
        # Find the benchmark index and compute relative performance
        market_idx = assets.get_loc(market_sid)
        rsRankTable["RS"] = (((returns + 1) / (returns[market_idx] + 1)) - 1) * 100
        
        out[:] = rsRankTable["RS"]

class Volatility(CustomFactor):
    """Computes price volatility as the standard deviation of daily returns."""
    inputs = [USEquityPricing.close]
    
    def compute(self, today, assets, out, close_prices):
        # Calculate daily returns
        daily_returns = np.diff(close_prices, axis=0) / close_prices[:-1]
        
        # Compute standard deviation
        volatility = np.std(daily_returns, axis=0)
        
        out[:] = volatility

class PublicSince(CustomFactor):
    """
    Identifies how long a security has been publicly traded by examining
    early price data. Higher values indicate established securities.
    """
    inputs = [USEquityPricing.close]
    
    def compute(self, today, assets, out, prices):
        prices = np.nan_to_num(prices)
        # Sum of early price data serves as a proxy for established securities
        out[:] = ((prices[0] + prices[1] + prices[2] + prices[3] + 
                  prices[4] + prices[5] + prices[6] + prices[7]))

class SumFactor(CustomFactor):
    """
    Sums a factor over the window length.
    Useful for accumulating sentiment scores over time.
    """
    window_safe = True
    
    def compute(self, today, assets, out, factordata):
        out[:] = np.sum(factordata, axis=0)

class MLFactor(CustomFactor):
    """
    Machine learning factor that predicts future returns based on input features.
    Uses scikit-learn regression models to generate predictions.
    """
    params = ('shift_target',)
    window_safe = True

    def compute(self, today, assets, out, target, *features, shift_target):
        global ML_GLOBAL_COUNTER
        global ML_MODEL_REUSE_LIMIT
        global ML_CLASSIFIER_GLOBAL

        # Prepare data preprocessing components
        self.imputer = impute.SimpleImputer(strategy='constant', fill_value=0)
        self.scaler = preprocessing.RobustScaler()
        self.scaler_2 = preprocessing.MinMaxScaler()
        
        self.imputer_Y = impute.SimpleImputer(strategy='constant', fill_value=0)
        self.scaler_Y = preprocessing.MinMaxScaler()
        
        # Create linear regression model
        self.clf = linear_model.LinearRegression(n_jobs=-1)  
        #self.clf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

        # Stack features for training
        X = np.dstack(features)  # (time, stocks, factors)
        Y = target  # (time, stocks)
        
        print(algo.get_datetime(timezone("America/Los_Angeles")))
        print('X raw', X.shape)
        print('Y raw', Y.shape)
          
        n_time, n_stocks, n_factors = X.shape
        print('t', n_time, 'stk', n_stocks, 'factors', n_factors)
        
        # Shift data for future prediction
        n_fwd_days = shift_target
        shift_index = n_time - n_fwd_days
        X = X[0:shift_index, :, :]
        n_time, n_stocks, n_factors = X.shape
        print('t', n_time, 'stk', n_stocks, 'factors', n_factors)
        
        # Reshape data for scikit-learn
        X = X.reshape(n_time * n_stocks, n_factors)
        print('X post reshape', X.shape)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        
        Y = Y[-shift_index:]
        Y = Y.reshape(n_time * n_stocks, 1)
        print('Y post reshape', Y.shape)
        
        Y = self.imputer_Y.fit_transform(Y)
        Y = self.scaler_Y.fit_transform(Y)
       
        # Fit model or reuse existing model
        if ML_GLOBAL_COUNTER == 0:
            self.clf.fit(X, np.array(Y)[:, 0])
            print('Model fitted -- reuse count:', ML_GLOBAL_COUNTER)
            ML_CLASSIFIER_GLOBAL = self.clf
        
        self.clf = ML_CLASSIFIER_GLOBAL
        
        try:
            predict_error = 0
            X_test = np.dstack(features)[-1, :, :]
            X_test_df = pd.DataFrame(X_test)
            print('X_test shape', X_test.shape)
            X_test = X_test_df.values
            X_test = self.imputer.fit_transform(X_test)
            X_test = self.scaler.fit_transform(X_test)
            
            Y_pred = self.clf.predict(X_test)
            print('Model reuse count:', ML_GLOBAL_COUNTER)
        except:
            predict_error = 1
            ML_GLOBAL_COUNTER = 0
            print("predict error!!!!")
            print('resetting model reuse to 0')
            print('X used for training')
            print(pd.DataFrame(X).head(50))
            print('Xtest used during failure')
            X_test = np.dstack(features)[-1, :, :]
            X_test_df = pd.DataFrame(X_test)
            X_test = X_test_df.values
            X_test = self.imputer.fit_transform(X_test)
            X_test = self.scaler.fit_transform(X_test)
            print(pd.DataFrame(X_test).head(50))
            Y_pred = np.zeros([1, 2])
        
        ML_GLOBAL_COUNTER = ML_GLOBAL_COUNTER + 1
        if ML_GLOBAL_COUNTER >= ML_MODEL_REUSE_LIMIT:
            ML_GLOBAL_COUNTER = 0

        print('X_flat.shape', X.shape, 'X.shape', np.dstack(features).shape)
        try:
            print('coef', self.clf.coef_)
        except:
            pass
        
        if predict_error == 0:
            print(Y_pred.shape)
            out[:] = Y_pred.flatten()
        else:
            out[:] = 0



class WeightedAlpha(CustomFactor):
    """
    Weighted excess return (alpha) vs SPY benchmark:
      - 50% weight on 30-day excess return
      - 30% weight on 90-day excess return
      - 20% weight on 252-day excess return
    """
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):
        spy_sid = symbol("SPY").sid

        # Try to locate SPY in the asset universe
        # try:
        #     spy_idx = np.where(assets == spy_sid)[0][0]
        # except IndexError:
        #     # SPY not present, assign NaN
        #     out[:] = np.nan
        #     return
        spy_idx = assets.get_loc(spy_sid) 
        spy_close = close[:, spy_idx]

        # Exclude SPY from computation
        is_asset = np.arange(len(assets)) != spy_idx
        asset_close = close[:, is_asset]

        # Compute asset returns
        ret_30 = asset_close[-1] / asset_close[-22] - 1  #22 tarding days --> 1 month 
        ret_90 = asset_close[-1] / asset_close[-66] - 1  #66 trading days --> 3 months 
        ret_252 = asset_close[-1] / asset_close[0] - 1

        # Compute SPY returns
        spy_ret_30 = spy_close[-1] / spy_close[-22] - 1
        spy_ret_90 = spy_close[-1] / spy_close[-66] - 1
        spy_ret_252 = spy_close[-1] / spy_close[0] - 1

        # Compute alpha (excess return)
        alpha_30 = ret_30 - spy_ret_30
        alpha_90 = ret_90 - spy_ret_90
        alpha_252 = ret_252 - spy_ret_252

        # Weighted alpha
        weighted_alpha = 0.15 * alpha_30 + 0.5 * alpha_90 + 0.35 * alpha_252

        # Write output: match full asset list shape
        out_vals = np.empty(len(assets))
        out_vals[:] = np.nan
        out_vals[is_asset] = weighted_alpha

        out[:] = out_vals


class SumVolume(CustomFactor):
    """
    Custom factor to compute 30-day average volume for assets.
    """
    inputs = [USEquityPricing.volume]
    
    
    def compute(self, today, assets, out, volume):
        """
        Compute  volume.
        
        Parameters:
        -----------
        today : pd.Timestamp
            Current date
        assets : pd.Index
            Assets to compute for
        out : np.array
            Output array to fill with results
        volume : np.array
            30-day window of volume data (window_length x num_assets)
        """
        # Calculate sum volume across the window for each asset
        out[:] = np.sum(volume, axis=0)

#################################################
# MAIN ALGORITHM FUNCTIONS
#################################################

def initialize(context):
    """
    Initialize the trading algorithm. Sets up pipelines,
    schedules, parameters, and basic configuration.
    """
    # Attach our main stock selection pipeline
    algo.attach_pipeline(make_pipeline(), 'my_pipeline')
    
    # Set the benchmark
    algo.set_benchmark(algo.sid(symbol('SPY').real_sid))
    
    # Initialize portfolio and risk parameters
    context.longfact = 1.0           # Long exposure factor
    context.shortfact = 1.0          # Short exposure factor
    context.order_id = {}            # Track order IDs
    context.topMom = 9               # Number of top momentum stocks to consider
    context.max_liquid = context.portfolio.starting_cash  # Initial capital
    context.cash_adjustment = 0      # Adjustment for external cash flows
    context.dd_factor = 1.0          # Drawdown adjustment factor
    context.draw_down = 0.0          # Current drawdown
    context.print_set_delta = False  # Debug printing control
    context.days_offset = 1          # Day of week offset for trading (Tuesday)
    context.initialized = 0          # Flag for initialization
    context.sids_initialized = 0     # Flag for SID initialization
    context.verbose = 1              # Verbosity level
    context.qqq_ratio_prev = 0       # Previous QQQ ratio
    context.spy_ratio_prev = 0       # Previous SPY ratio
    context.total_ws = 0             # Total weight short
    context.iwm_w = 0                # IWM weight
    context.spy_below80ma = False    # Flag for SPY below 80-day MA
    context.vixflag = 0              # VIX signal flag
    context.vixflag_prev = 0         # Previous VIX signal flag
    context.clip = 1.0               # Position size clipping value

    # Set trading costs
    algo.set_slippage(algo.slippage.FixedSlippage(spread=SLIPPAGE_SPREAD))
    algo.set_commission(algo.commission.PerShare(cost=COMMISSION_COST, min_trade_cost=MIN_TRADE_COST))

    # Schedule functions
    algo.schedule_function(
        initial_allocation,
        date_rule=algo.date_rules.every_day()
    )
    
    algo.schedule_function(
        regular_allocation,
        date_rule=algo.date_rules.week_start(days_offset=context.days_offset)
    )
    
    algo.schedule_function(
        exit_positions,
        date_rule=algo.date_rules.week_start(days_offset=context.days_offset)
    )

def make_pipeline():
    """
    Creates the stock selection pipeline that identifies candidates
    for both long and short positions based on various factors.
    """
    # Initial universe filter - top stocks by market cap
    tradable_filter = (CustomFundamentals.CompanyMarketCap.latest.shift().top(UNIVERSE_SIZE)) | StaticAssets([symbol('IBM')])
    

#2021
#     tickers = [
#     "AHCO", "AMRS", "AMSC", "APPS", "AQMS", "ARCT", "ATOM", "AWH", "AZRE", "BEAM",
#     "BEEM", "BEEMW", "BILI", "BLFS", "BLNK", "CDNA", "CELH", "CLCT", "CLSK", "CMT",
#     "CPST", "CRDF", "CRNC", "CRSP", "CRWD", "CURIW", "DNLI", "DNM.W", "DQ", "ENPH",
#     "ETSY", "EXPCW", "EXPI", "FATE", "FLGT", "FRHC", "FSM", "FTCH", "FUTU", "FVRR",
#     "GNOGW", "GROW", "GRVY", "GRWG", "IEA", "IMAB", "IPA", "IPV.W", "KIRK", "KSPN",
#     "LEU", "LL", "MP.W", "MSTR", "MTLS", "MVIS", "NET", "NFE", "NGD", "NIO", "NMCI",
#     "NOVA", "NTLA", "OCUL", "OIIM", "OPRX", "OSTK", "PACB", "PDD", "PEIX", "PINS",
#     "PLUG", "PNTG", "PRPH", "PRPL", "PRTS", "PSNL", "PTON", "REGI", "RIOT", "RSI.W",
#     "RUN", "SE", "SI", "SITM", "SMMCW", "SMTI", "SPWR", "STKL", "THCAW", "TPIC",
#     "TRIL", "TSLA", "TTCFW", "TWST", "UMC", "UTZ.W", "VERI", "ZM", "ZS"
# ]

#     tickers = [
#     "META", "GOOGL", "PLTR",  # 5 Themes
#     "GRMN", "MSFT",           # 4 Themes
#     "AAPL", "PANW", "NVDA", "CRWD",  # 3 Themes
#     "CDNS", "AXP", "ISRG", "CAT", "AMZN", "TSLA", "EMR", "JPM", "PYPL", "PGR",  # 2 Themes
#     "GEV", "ETN", "BK", "SPGI", "HOOD", "ABT", "MNST", "PWR", "NFLX", "MSTR",
#     "ORCL", "GE", "FICO", "GS", "KLAC", "COST", "AXON", "ANET", "AMD"  # 1 Theme
# ]
#     tickers = [
#         "RDDT",
#         "NFLX",
#         "AMZN",
#         "AVGO",
#         "WFC",
#         "LNG",
#         "COST",
#         "AAPL",
#         "MSFT",
#         "NVDA",
#         "GOOGL",
#         "META",
#         "AZN",
#         "GS",
#         "BRK B",
#         "ORCL",
#         "PLTR",
#         "RBLX",
#         "GE",
#         "NEM"
#     ]
#     tickers.append("IBM")
#     asset_list = [sid for sid in (symbol(ticker) for ticker in tickers) if sid is not None]

#     tradable_filter = (
#     #CustomFundamentals.CompanyMarketCap.latest.shift().top(35) |
#     StaticAssets(asset_list)
#  )
    
    # Example usage in a pipeline

    # Create different money flow factors
    money_flow_index = MoneyFlowIndexFactor(mask=tradable_filter,window_length=90)
    chaikin_money_flow = ChaikinMoneyFlowFactor(mask=tradable_filter,window_length=90)
    simple_money_flow = SimpleMoneyFlowFactor(mask=tradable_filter,window_length=90)
    volume_weighted_mf = VolumeWeightedMoneyFlowFactor(mask=tradable_filter,window_length=90)
    comp = ( 
                (USEquityPricing.close.latest * SimpleMovingAverage(inputs=[USEquityPricing.volume], window_length=10)).zscore(mask=tradable_filter) + 
         
                ((CustomFundamentals.FOCFExDividends_Discrete.latest.shift() - CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest.shift()) / 
                CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest.shift()).zscore(mask=tradable_filter) +
         
                Slope(window_length=120, mask=tradable_filter).slope.zscore(mask=tradable_filter)
         )
    
    # Create composite money flow score
    composite_money_flow = (
        money_flow_index.zscore() + 
        chaikin_money_flow.zscore() + 
        simple_money_flow.zscore() + 
        volume_weighted_mf.zscore()
    ) / 4.0

   
    #have_low_enterprise_multiples = sharadar.Fundamentals.slice(dimension="ARQ").EVEBITDA.latest.percentile_between(0, 20) # Enterprise Value to EBITDA
    s_fundamentals = sharadar.Fundamentals.slice('MRQ', period_offset=0) # sharadar fundamentals
    #   previous_fundamentals = sharadar.Fundamentals.slice('ARQ', period_offset=-1)

    # total_assets = current_fundamentals.ASSETS.latest
    # previous_total_assets = previous_fundamentals.ASSETS.latest
    # assets_increased = total_assets > previous_total_assets 
                   
    # Build pipeline with selected columns
    pipe = Pipeline(

        # cash_return = CashReturnFromDatabase(
        #     inputs=[
        #         CustomFundamentals.FOCFExDividends_Discrete.latest,
        #         CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest,
        #         CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest
        #     ]
        # )
        
    
        
        # # Rolling average for smoother signals
        # cash_return_smooth = CashReturnRolling(
        #     fcf_column=CustomFundamentals.FOCFExDividends_Discrete.latest,
        #     interest_column=CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest,
        #     ev_column=CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest,
        #     window_length=4  # 4-quarter average
        # )
        
        # # Quality-adjusted version
        # cash_return_quality = CashReturnQuality(
        #     fcf_column=CustomFundamentals.FOCFExDividends_Discrete.latest,
        #     interest_column=CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest,
        #     ev_column=CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest
        # )

        screen=tradable_filter,
        columns={
            # Company information
            'name': CustomFundamentals.Symbol.latest,
            'compname': CustomFundamentals.CompanyCommonName.latest,
           'sector': CustomFundamentals.GICSSectorName.latest,
           #   'sector': master.SecuritiesMaster.usstock_Sector.latest,
            
            # Market data
        
            'market_cap': CustomFundamentals.CompanyMarketCap.latest.shift().shift(),
            #'market_cap': s_fundamentals.SHARESWADIL.latest * USEquityPricing.close.latest.shift(),

            'entval': CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest.shift(),
            'price': USEquityPricing.close.latest.shift(),
            'volume': USEquityPricing.volume.latest.shift(),
            'fs_price': CustomFundamentals.RefPriceClose.latest.shift(),
            'fs_volume': CustomFundamentals.RefVolume.latest.shift(),
            'sumvolume': SumVolume(window_length=3),
         
            
            # Earnings and growth metrics
            'eps_ActualSurprise_prev_Q_percent': CustomFundamentals.EarningsPerShare_ActualSurprise.latest.shift(),
            'eps_gr_mean': CustomFundamentals.LongTermGrowth_Mean.latest.shift(),
            
            # Financial metrics
            'CashCashEquivalents_Total': CustomFundamentals.CashCashEquivalents_Total.latest.shift(),
            #'fcf': CustomFundamentals.FOCFExDividends_Discrete.latest.shift(),
            #'fcf': (CustomFundamentals.FOCFExDividends_Discrete.latest.shift() + s_fundamentals.FCF.latest)/2,
            'fcf': s_fundamentals.FCF.latest,
            'int': CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest.shift(),
            # 'crpipe':  ((CustomFundamentals.FOCFExDividends_Discrete.latest.shift() - CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest.shift()) / 
            #             CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest.shift()), 

            # 'comp': comp,
            # Risk metrics
            'beta60SPY': SimpleBeta(target=symbol('SPY'), regression_length=60).shift(),
            'beta60IWM': SimpleBeta(target=symbol('IWM'), regression_length=60).shift(),
            
            # Technical indicators
            'smav': SimpleMovingAverage(inputs=[USEquityPricing.volume], window_length=10),
            'slope120': Slope(window_length=120, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            'slope220': Slope(window_length=220, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            'slope90': Slope(window_length=90, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            'slope30': Slope(window_length=30, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),

            # 'mfi': money_flow_index, #*
            # 'cmf': chaikin_money_flow, #*
            # 'simple_mf': simple_money_flow, #*
            # 'vw_mf': volume_weighted_mf, #*
            # 'composite_mf': composite_money_flow, #*
            
            
           
            #'ich_signal': IchimokuSignal(),
            'stk20w' : StochasticOscillatorWeekly(),
            'above_200dma':  Above200DMA(mask=tradable_filter),
            'walpha': WeightedAlpha(),
            
    
            
            # Relative strength metrics
            'RS140_QQQ': RelativeStrength(window_length=140, market_sid=symbol('QQQ').sid).shift(),
            'RS160_QQQ': RelativeStrength(window_length=160, market_sid=symbol('QQQ').sid).shift(),
            'RS180_QQQ': RelativeStrength(window_length=180, market_sid=symbol('QQQ').sid).shift(),
            
            # Return metrics
            'Ret60': Returns(window_length=60, mask=tradable_filter),
            'Ret120': Returns(window_length=120, mask=tradable_filter),
            'Ret220': Returns(window_length=220, mask=tradable_filter),
            
            # Other factors
            'publicdays': PublicSince(window_length=121),
            'vol': Volatility(window_length=10, mask=tradable_filter),
            
            # VIX signal
            'vixflag': CustomFundamentals4.pred.latest.shift(),
            'vixflag0': CustomFundamentals4.pred.latest,
          

            #bcdata
            'bc1': CustomFundamentals9.bc1.latest,
           
            
            # ML Factor
            'MLfactor': MLFactor(
                inputs=[
                    Returns(window_length=90, mask=tradable_filter),
                    CustomFundamentals.EnterpriseValueToEBITDA_DailyTimeSeriesRatio_,
                    CustomFundamentals.LongTermGrowth_Mean,
                    CustomFundamentals.CombinedAlphaModelSectorRank,
                    CustomFundamentals.ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_,
                    #comp
                ],
                window_length=180,
                mask=tradable_filter,
                shift_target=10
            ).shift(),

    
            
            # Sentiment factors
            'sentcomb': (
                SumFactor(CustomFundamentals2.sentvad_neg, window_length=18).zscore() +
                SumFactor(CustomFundamentals2.sent2sub, window_length=18).zscore() +
                (1/SumFactor(CustomFundamentals2.sent2pol, window_length=18).zscore())
            ),
            
            'sentest': 1/SumFactor(CustomFundamentals2.sent2pol, window_length=18),

        }
    )
    
    return pipe

def before_trading_start(context, data):
    """
    Daily preprocessing before trading begins. Analyzes market conditions,
    selects longs and shorts, and prepares the trading universe.
    """
    # Initialize security IDs if needed
    initialize_sids(context, data, algo)

    # Get pipeline output
    df = algo.pipeline_output('my_pipeline')
    
    print(f"Raw stock universe size {df.shape}")
    print(algo.get_datetime(timezone("America/Los_Angeles")))
    
    # Update market condition indicators
    update_market_indicators(context, data)
    
    # Update VIX signal
    context.vixflag_prev = context.vixflag
    context.vixflag = df.loc[context.ibm_sid].vixflag.copy()
    context.vixflag0 = df.loc[context.ibm_sid].vixflag0.copy()
   
    print('IBM-vixdata', context.vixflag)

    #barchart data
    context.bc1 = df.loc[context.ibm_sid].bc1.copy()

    #barchart data IWM
    #context.bc2 = df.loc[context.ibm_sid].bc2.copy()
    
    #Get IWM 20 week Stochastic
    df_iwm = data.history(symbol('IWM'), ['high','low','price'],20*5, '1d')
    context.iwm_stk20w = compute_weekly_stochastic(df_iwm, lookback_weeks=20)
    print('IWM_stk20w', context.iwm_stk20w)
    
    # Determine market trend based on VIX and other indicators
    compute_trend(context, data)
    
    # Initialize daily tracking variables
    context.daily_flag = 0
    context.daily_print_flag = 0
    
    # Filter universe and generate alpha signals
    df = process_universe(context, df, data)
    
    return

def process_universe(context, df, data):
    """
    Processes the universe of stocks, calculates alpha signals,
    and selects securities for long and short positions.
    """
    # Remove stocks from excluded sectors and companies undergoing acquisitions
    df = filter_symbols_sectors_universe(df)

    # Calculate dollar volume for liquidity filtering
    df['doll_vol'] = df['price'] * df['smav'] 

    # Calculate cash return as a key alpha signal
    df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
    df['cash_return'] = df['cash_return'].replace([np.inf, -np.inf], np.nan)
    
    # Remove stocks with invalid cash return values
    df.dropna(subset=['cash_return'], inplace=True)
    # Display highest-returning sectors
    sorted_df = df.groupby('sector')['cash_return'].agg(['mean']).sort_values(by='mean', ascending=False)
    print(sorted_df.iloc[0].name)

    # Calculate momentum-based ranking
    # Use different weightings based on time period
    current_year = algo.get_datetime().date().year
    current_month = algo.get_datetime().date().month
    
    # if (current_year >= 2022 and current_month >= 3) or current_year >= 2023:
    #     print("sentcomb--->>>>>> ", df.sort_values(by='market_cap', ascending=False)[0:10][['sentcomb']], '\n')
    #     df['myrs'] = (df.slope120.rank() + df['sentcomb'].rank() )
    #     #df['myrs'] = df.slope120.rank() + df.RS140_QQQ.rank() 
    # else:
    #     df['myrs'] =  df.slope120.rank() + df.RS140_QQQ.rank()
    #     df['sentcomb'] = 0

    df['myrs'] =  df.slope120.rank() + df.RS140_QQQ.rank()
    # Display top stocks by market cap with key metrics
    print("cash ret, myrs --->>>>>> ", df.sort_values(by='market_cap', ascending=False)[0:10][['cash_return', 'myrs', 'slope90', 'above_200dma','walpha']], '\n')
   
    # Create temporary ranking for initial filtering
    # df['estrank_temp'] = (
    # #     df['doll_vol'].rank()+ 
    # #(df['slope90']).rank() * 2 
        
        
    # #     #+ 
    # #     #df['eps_ActualSurprise_prev_Q_percent'].rank() + 
    # df['eps_gr_mean'].rank()#+
    # #    # df['comp'].rank()
    # #    # df['volume'].rank()     
    # ) 
    # df = df.sort_values(by=['estrank_temp'], ascending=[False])[0:1000].copy()

    df = df.sort_values(by=['market_cap'], ascending=[False])[0:FILTERED_UNIVERSE_SIZE].copy()
   
    # Calculate final ranking based on market conditions
    # Use different factor weights depending on market conditions
    if context.spyprice <= context.spyma80:
        # Defensive ranking when market is below 80-day MA
        df['estrank'] = df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1)
        print('switch estrank to doll_vol spy below spyma80 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', 
              'spyprice:', context.spyprice, 'spyma80:', context.spyma80)
    else:
        # Normal ranking with multiple factors
        # Determine growth season multiplier based on current month
        if algo.get_datetime().date().month in GROWTH_SEASON_MONTHS:
            context.season = 1  # Growth season
        else:
            context.season = 0  # Value season
            
        df['estrank'] = (
            df['MLfactor'].rank() +
            (df['entval'].rank() * 2) +
            (df['cash_return'].rank()) + 
            df['eps_gr_mean'].rank() * (4 if context.season == 1 else 1) +
            #(df['eps_ActualSurprise_prev_Q_percent'].rank() / 3) +#+
           (df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1) / 3) #+#+
          #(1/df['vol']).rank()

        )
    
    #print(df['above_200dma'].rank())
    print(f"Filtered stock universe size {df.shape}")
    
    # Create long and short portfolios
    select_long_portfolio(context, df, data)
    select_short_portfolio(context, df, data)
    context.topmcap = df.sort_values(by=['market_cap'], ascending=[False])[0:7].copy()
    
    # Combine longs and shorts to form complete universe
    context.universe = np.union1d(context.longs.index.values, context.shorts.index.values)
   
    # Calculate portfolio beta to benchmark
    context.beta_ratio = max(1.3,compute_beta(context, data))

    print(f'Beta ratio: {context.beta_ratio:.4f}')
    
    return df

def select_long_portfolio(context, df, data):
    """
    Selects securities for the long portfolio based on multiple alpha signals.
    Creates separate buckets for value and momentum longs.
    """
    # Create a copy for long selection
    dfl = df.copy()

    # ENHANCED: Apply risk controls
    #dfl = enhanced_stock_screening(dfl)
    #dfl = apply_market_cap_diversification(context, dfl)
    #dfl = apply_volatility_adjustment(context, dfl, data)
    
    # Determine number of stocks from each strategy
    num_momentum_stocks = TOP_MOMENTUM_STOCKS
    num_value_stocks = LONG_PORTFOLIO_SIZE - num_momentum_stocks
    #dfl['cash_return'] = dfl['cash_return'] * (1+dfl['slope120'] / dfl['slope120'].sum()) 
    # Select value stocks based on cash return
    context.longs_c = (
        dfl.sort_values(by=['estrank'], ascending=[False])[0:150]
           .sort_values(by=['cash_return'], ascending=[False])[0:num_value_stocks]
           .copy()
    )
   
    # Select momentum stocks based on market trend
    if context.vix_uptrend_flag:
        context.longs_m = (
            dfl.sort_values(by=['RS140_QQQ'], ascending=[False])[0:500]
               .sort_values(by=['myrs'], ascending=[False])[0:num_momentum_stocks]
               .copy()
        )
    else:
        # In downtrends, select more value-oriented stocks
        context.longs_m = (
            dfl.sort_values(by=['estrank'], ascending=[False])[0:150]
               .sort_values(by=['cash_return'], ascending=[False])[num_value_stocks:LONG_PORTFOLIO_SIZE]
               .copy()
        )
    
    # Combine the two sets of longs
    c_set = set(context.longs_c.index)
    m_set = set(context.longs_m.index)
    context.longs = dfl[dfl.index.isin(c_set.union(m_set))].copy()
    print(f'Long portfolio size: {len(context.longs)}')
   
    # Sort longs by cash return
    context.longs = context.longs.sort_values(by=['cash_return'], ascending=[False]).copy()
    
    # Adjust cash return by risk factors depending on market conditions
    if context.spyprice >= context.spyma80:
        # Normal market - normalize by IWM beta
        context.longs['cash_return'] /= winsorize(context.longs['beta60IWM'], limits=[0.005, 0.4])
    else:
        # Defensive market - normalize by SPY beta
        context.longs['cash_return'] /= winsorize(context.longs['beta60SPY'], limits=[0.005, 0.1])
        pass

    # Set minimum cash return to avoid negative weights
    context.longs['cash_return'] = context.longs['cash_return'].clip(lower=0.005)
    
    
    # proportional weighting based on slope strength:
    slopefact = 1+(context.longs['slope120'] / context.longs['slope120'].sum())
    context.longs['cash_return'] = context.longs['cash_return'] * slopefact**2 #slopefact squared !

    # Final sorting of long portfolio
    context.longs = context.longs.sort_values(by=['cash_return'], ascending=[False])
    
    print(f'Long portfolio calculated with total cash return: {context.longs["cash_return"].sum()}')
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    #     print(context.longs[['cash_return','slope120','eps_gr_mean','eps_ActualSurprise_prev_Q_percent','walpha']][0:10])
    
    return

def select_short_portfolio(context, df,data):
    """
    Selects securities for the short portfolio based on poor fundamentals and technicals.
    Excludes sectors with strong momentum.
    """
    # Create a copy for short selection
    dfs = df.copy()
    
    # Remove top momentum stocks to avoid shorting momentum leaders
    #dfs.drop(dfs.sort_values(by=['myrs'], ascending=[False]).index[0:50], inplace=True)
    
    # Get sector momentum ranking
    mom_list = GenerateMomentumList(context, data, context.sector_etf, 242)
    mom_list = [item[0] for item in mom_list]
    
    # Select top momentum sector to avoid shorting
    top_momentum_sector = mom_list[0]
    context.mometf = top_momentum_sector
    
    # Remove top momentum sector from short candidates
    dfs = RemoveSectors(context, top_momentum_sector, dfs, "not shorting! %s")
    print('Bottom momentum sector:', mom_list[-1])
    
    # Select shorts based on lowest cash return
    context.shorts = (dfs.sort_values(by=['cash_return'], ascending=[True])[0:SHORT_PORTFOLIO_SIZE]
                    .copy()
    )
    return

def update_market_indicators(context, data):
    """
    Updates market indicators such as moving averages and trend signals.
    These are used to adjust position sizing and exposure.
    """
    # Get SPY price history and calculate moving averages
    context.price_history_spy100 = data.history(symbol('SPY'), 'price', 200, '1d')
    context.spyprice = context.price_history_spy100.values[-1]
    context.spyma21 = np.mean(context.price_history_spy100.tail(21).values)
    context.spyma50 = np.mean(context.price_history_spy100.tail(50).values)
    context.spyma80 = np.mean(context.price_history_spy100.tail(80).values)
    context.spyma85 = np.mean(context.price_history_spy100.tail(85).values)
    context.spyma150 = np.mean(context.price_history_spy100.tail(150).values)
    context.spyma200 = np.mean(context.price_history_spy100.tail(200).values)

    # Get IWM price history and calculate Hull moving averages
    context.price_history_iwm250 = data.history(symbol('IWM'), 'price', 250, '1d')
    context.iwmprice = context.price_history_iwm250.values[-1]
    context.iwmma50 = hull_moving_average(context.price_history_iwm250.values, 50)[-1]
    context.iwmma10 = hull_moving_average(context.price_history_iwm250.values, 10)[-1]
    context.hulltrend = hull_ma_trend(context.price_history_iwm250.values, 80, lookback=7)

def handle_data(context, data):
    """
    Function called on each trading bar. Handles intraday monitoring
    and records key metrics.
    """
    # Check if we're at end of day for summary
    time_minute = algo.get_datetime(timezone("America/Los_Angeles")).minute
    time_hour = algo.get_datetime(timezone("America/Los_Angeles")).hour
    
    # Set leverage for account
    context.account_leverage = 2.0
    
    # Flag for end-of-day reporting
    if (time_hour == 12 and time_minute == 59):
        printflag = True
    else:
        printflag = False
        
    # Update and report account metrics once daily
    if context.daily_flag == 0 or printflag:
        # Calculate net liquidation with adjustment for cash withdrawals
        my_net_liquidation = context.account.net_liquidation + context.cash_adjustment
        
        # Track new equity high watermark
        if my_net_liquidation > context.max_liquid:
            context.max_liquid = my_net_liquidation
            if context.daily_print_flag == 0 or printflag:
                print(f"New equity high! Max liquidation for the trading period: {context.max_liquid:.0f}")
                
        # Calculate drawdown from high
        context.draw_down = (context.max_liquid - my_net_liquidation) / context.max_liquid
        if context.draw_down != 0 and (context.daily_print_flag == 0 or printflag):
            print(f"Current account drawdown from high of: {context.max_liquid:.0f}")
            print(f'DD= {context.draw_down:.3%}')
        
        print(" ")
        print(" ")
            
        context.daily_print_flag = 1
        if context.cash_adjustment == 0:
            context.daily_flag = 1
            
    return



def initial_allocation(context, data):
    """
    Handles initial portfolio allocation and intraday
    adjustments based on market signals.
    """
    # Get current and previous SPY prices
    spyprice_1 = context.price_history_spy100.iloc[-1]
    spyprice_2 = context.price_history_spy100.iloc[-2]

    # Setup logging
    logger = logging.getLogger('LS-Prod-Algo')
    logger.setLevel(logging.DEBUG)
    handler = FlightlogHandler()
    logger.addHandler(handler)

    # Set initial IWM weight if not set
    if context.iwm_w == 0:
        context.iwm_w = -0.4
        
    # Respond to VIX signal changes
    if (context.vixflag_prev > 0 and context.vixflag <= 0 and 
        algo.get_datetime().date().weekday() != context.days_offset):
        
        if spyprice_1 < context.spyma80:
            # VIX signal turned negative but SPY below MA80 - reduce short exposure
            if context.shortfact != 0:
                # Reduce IWM short position
                new_weight = min(context.iwm_w / (2 * context.shortfact), -0.4 * context.shortfact * context.bcfactor)
                algo.order_target_percent(context.iwm_sid, new_weight)
                
                # Log the signal change
                logger.info(' ')
                logger.info('ALERT ALERT ALERT !!!! ')
                logger.info(f'vix long exit for IWM at {algo.get_datetime(timezone("America/Los_Angeles"))}')
                logger.info(f'switching out to less short IWM -- weight = {new_weight}')

                # # Execute full reallocation
                #regular_allocation(context, data)
                #exit_positions(context, data)
                
                # Print alert to console
                print("\n\n\n\n\n")
                print(">"*100)
                print(" ALERT ALERT ALERT !!!! ")
                print("")
                print(algo.get_datetime(timezone("America/Los_Angeles")))
                print("vix long exit for IWM")
                print(f"switching out to less short IWM -- weight = {new_weight}")
                print("\n\n\n\n\n")
      
        if spyprice_1 >= context.spyma80:
            # VIX signal turned negative and SPY above MA80 - reallocate portfolio
            logger.info(' ')
            logger.info('ALERT ALERT ALERT !!!! ')
            logger.info(f'vix long exit for IWM at {algo.get_datetime(timezone("America/Los_Angeles"))}')
            logger.info('executing reallocation')
            
            # Execute full reallocation
            regular_allocation(context, data)
            exit_positions(context, data)
            
        logger.info(' ')
       
    # Initialize portfolio if first run
    if context.initialized == 0:
        context.initialized = 1
        regular_allocation(context, data)

    # Handle SPY crossing above 80-day MA
    if (spyprice_1 > context.spyma80 and 
        spyprice_2 <= context.spyma80 and 
        context.vix_uptrend_flag and 
        context.spy_below80ma and 
        algo.get_datetime().date().weekday() != context.days_offset):
        
        print("\n\n\n\n\n")
        if context.shortfact != 0:
            # Reduce IWM short position
            new_weight = min(context.iwm_w / 2, -0.4 * context.bcfactor)
            algo.order_target_percent(context.iwm_sid, new_weight)
            
            # Print alert to console
            print(">"*100)
            print("ALERT ALERT ALERT !!!! ")
            print("")
            print(algo.get_datetime(timezone("America/Los_Angeles")))
            print(f"spyma cross over, spyprice_1, spyprice_2, spyma80: {spyprice_1}, {spyprice_2}, {context.spyma80}")
            print(f"spyma cross over --- switching out to less short IWM -- weight = {new_weight}")
            context.spy_below80ma = False 
            print(f'weekday: {algo.get_datetime().date().weekday()}')
            print("\n\n\n\n\n")
            
            # Log the signal change
            logger.info(' ')
            logger.info('ALERT ALERT ALERT !!!! ')
            logger.info(f'spyma cross over, spyprice_1: {spyprice_1}, spyprice_2: {spyprice_2}, spyma80: {context.spyma80} at {algo.get_datetime(timezone("America/Los_Angeles"))}')
            logger.info(f"spyma cross over --- switching out to less short IWM -- weight = {new_weight}")
            logger.info(' ')
    
    return

def regular_allocation(context, data):
    """
    Main portfolio allocation function. Assigns weights to long and
    short positions based on alpha signals and market conditions.
    """
    # Get the longs and shorts
    longs = context.longs.index
    shorts = context.shorts.index
    
    # Update trend signals
    compute_trend(context, data)
    context.initialized = 1
   

    # Adjust for drawdown protection
    context.dd_factor = min([context.clip, (1 + context.draw_down * DRAWDOWN_FACTOR_MULTIPLIER)])

    # Print portfolio status
    try:
        print(algo.get_datetime(timezone("America/Los_Angeles")))
        print(f'Beta ratio: {context.beta_ratio:.4f}')
        print(f"Drawdown factor: {context.dd_factor:.4f}")
        print(f"Long factor: {context.longfact:.2f}")
        print(f"Net liquidation: {context.account.net_liquidation:.2f}")
    except:
        print("print ERROR >>>>>")
        pass
    
    # Get normalized position weights
    longs_mcw, shorts_mcw = get_normalized_weights(context, data, 'cash_return')

    #ENHANCED: Apply sector and correlation constraints
    #longs_mcw, shorts_mcw = apply_sector_constraints(context, longs_mcw, shorts_mcw)
   
    # try:
    #     longs_mcw = apply_correlation_constraints(context, data, longs_mcw)
    # except Exception as e:
    #     print(f"Correlation constraints failed: {e}")
    
    # Calculate trend-based adjustment factors
    if context.vix_uptrend_flag:
        trend_longfact_multiplier = 0.625  # More conservative in uptrends
        trend_spy_gt_ma21 = 1
    else:
        trend_longfact_multiplier = 1.625  # More aggressive in downtrends
        trend_spy_gt_ma21 = 1.3 if context.spyprice > context.spyma21 else 1

    # Market condition adjustment factor
    if context.spyprice < context.spyma80:
        adjust_fact = 1.05  # More conservative below 80-day MA
    else:
        print('SPY above 80-day MA: switch adjust_fact to 1.102 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', 
              'spyprice:', context.spyprice, 'spyma:', context.spyma80)
        adjust_fact = 1.102  # More aggressive above 80-day MA
        
    print('adjust factor', adjust_fact)
    
    # Apply adjustments to long weights
    longs_mcw['cash_return'] = (
        longs_mcw['cash_return'] * 
        context.longfact * 
        context.dd_factor * 
        0.637 * 
        adjust_fact * 
        1.03
    )
    
    # Additional factor for uptrends
    # if the vix flag is not triggred 
    if context.vix_uptrend_flag:
        longs_mcw['cash_return'] = longs_mcw['cash_return'] * 1.6

    #reduce long weights based on if we are below the 150 MA and barchart trend spoter is not negative (red) (i,e, no red dots)
    if context.spyprice < context.spyma150 and context.bc1 == 1:
        context.bcfactor = 0.65
        longs_mcw['cash_return'] = longs_mcw['cash_return'] * context.bcfactor
    else:
         context.bcfactor = 0.65

    # data = context.longs['beta60SPY']
    # normalized = (data - data.min()) / (data.max() - data.min())
    # context.longs['beta60SPY'] = normalized
    # context.longs['beta60SPY'].fillna(value=1,  inplace=True)

    # data = context.longs['cash_return']
    # normalized = (data - data.min()) / (data.max() - data.min())
    # context.longs['cash_return'] = normalized
    # context.longs['cash_return'].fillna(value=1,  inplace=True)

    port_weight_factor = 0.90 # the perecentage to allocate to LS algo selection 
    spy_weight_factor = 1 - port_weight_factor # the perecentage to allocate to SPY selection 

    # Print positions if verbose mode
    if context.verbose == 1:
        print_positions(longs, longs_mcw, 'cash_return',port_weight_factor)

    # Execute orders for long position
    total_wl = 0

    for sid, w in zip(longs, longs_mcw['cash_return'].values):
        w = abs(w)
        algo.order_target_percent(sid, w * port_weight_factor)
        
        if w > 1.0:
            print(sid, w)
        total_wl = total_wl + w

    algo.order_target_percent(context.spysym, total_wl * spy_weight_factor)


    # context.topsyms = context.topmcap.index 
   
    # context.topsyms = [
    # algo.sid(symbol('NVDA').real_sid),
    # algo.sid(symbol('GOOGL').real_sid),
    # # # algo.sid(symbol('MSFT').real_sid),
    # # # algo.sid(symbol('AAPL').real_sid),
    # algo.sid(symbol('AMZN').real_sid),
    # # # algo.sid(symbol('AVGO').real_sid),
    # # # algo.sid(symbol('META').real_sid),
    # # # algo.sid(symbol('BRK.B').real_sid),
    # # # algo.sid(symbol('JPM').real_sid),
    #  ]
    # weights = 0.065 #(1/len(context.topsyms)) * total_wl * spy_weight_factor
    
    # for symbol_sid in context.topsyms:
    #     algo.order_target_percent(symbol_sid, weights)
    #     print('placing order for',symbol_sid, weights )
    # print(f'Top weights: {weights:.4f}')
    # print('Top syms:', context.topsyms)
   

    print(f'Total SPY weight: {total_wl * spy_weight_factor:.4f}')
    print(f'Total port long weight: {total_wl * port_weight_factor:.4f}')
    print(f'Total long weight: {total_wl:.4f}')
    
    # Limit short position sizes
    shorts_mcw[shorts_mcw['cash_return'] > 0.15] = 0.15
    
    # Apply adjustments to short weights
    shorts_mcw['cash_return'] = (
        -shorts_mcw['cash_return'] * 
        context.shortfact * 
        context.beta_ratio * 
        trend_longfact_multiplier * 
        trend_spy_gt_ma21 * 
        0.637 * 
        adjust_fact * 
        1.03
    )

    # Additional factor for market conditions
    if context.vix_uptrend_flag and context.spyprice > context.spyma21:
        shorts_mcw['cash_return'] = shorts_mcw['cash_return'] * 1.4
    else:
        shorts_mcw['cash_return'] = shorts_mcw['cash_return'] * 1.4

    # Calculate total short weight
    total_ws = 0
    for sid, w in zip(shorts, shorts_mcw['cash_return'].values):
        
        w = abs(w)
        w = w * -1
        if w > 1.0:
            print(sid, w)
        total_ws = total_ws + w
    
    # Ensure total_ws is negative
    if total_ws > 0:
        print(total_ws)
        print('Error: total short weight is positive')

    context.total_ws = total_ws
    print(f'Total short weight: {total_ws:.4f}')
    
    # Update SPY MA crossing flag
    context.spy_below80ma = context.spyprice < context.spyma80
    
    # Execute IWM short position or place individual short orders
    if (context.vix_uptrend_flag and context.spy_below80ma):
        # Set IWM weight based on long exposure
        iwm_w = min(-1 * 0.384 * total_wl, total_ws)
        
        place_short_orders(algo, context, context.short_symbol_weights, iwm_w)
        print('Short Position Factors: shortfact:', context.shortfact, 'beta_ratio:', context.beta_ratio, 'trend_multiplier:', trend_longfact_multiplier, 'spy_ma21_factor:', trend_spy_gt_ma21, 'base_multiplier:', 0.637, 'adjust_factor:', adjust_fact, 'final_multiplier:', 1.03)
        print(f'Min active-> SPY below MA80 - IWM weight {iwm_w:.4f}')
        context.iwm_w = iwm_w
    else:
        place_short_orders(algo, context, context.short_symbol_weights, total_ws)
        print('Short Position Factors: shortfact:', context.shortfact, 'beta_ratio:', context.beta_ratio, 'trend_multiplier:', trend_longfact_multiplier, 'spy_ma21_factor:', trend_spy_gt_ma21, 'base_multiplier:', 0.637, 'adjust_factor:', adjust_fact, 'final_multiplier:', 1.03)   
        print(f'IWM weight {total_ws:.4f}')
        context.iwm_w = total_ws
    
    # Warning if too many positions
    if len(pd.Series(tuple(context.portfolio.positions.keys()))) > 52:
        print("WARNING: Too many positions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>+++++++++++++++++++++++++++++++++")
        
    print(" ")
    return

def exit_positions(context, data):
    """
    Exits positions that are no longer in the desired portfolio.
    """
    # Get set of desired securities
    desired_sids = set(context.longs.index) #| set(context.topmcap.index)
    
    # Identify positions to exit - exclude ETFs and benchmark instruments
    getting_the_boot = [
        sid for sid in context.portfolio.positions.keys() 
        if sid not in desired_sids 
        and sid != context.iwm_sid 
        and sid != context.spysym 
        and sid != context.dia_sid 
        and sid != context.qqq_sid 
        and sid != context.tlt_sid 
        and sid != algo.sid(symbol('SPLV').real_sid)
        #and sid not in context.topmcap
        #and sid != algo.sid(symbol('NVDA').real_sid)
        #and sid != algo.sid(symbol('GOOGL').real_sid)
        # and sid != algo.sid(symbol('MSFT').real_sid)
        # and sid != algo.sid(symbol('AAPL').real_sid)
        #and  sid != algo.sid(symbol('AMZN').real_sid)
        # and sid != algo.sid(symbol('AVGO').real_sid)
        # and sid != algo.sid(symbol('META').real_sid)
        # and sid != algo.sid(symbol('BRK.B').real_sid)
        # and sid != algo.sid(symbol('JPM').real_sid)
    ]
    
    # Print log if exiting positions
    if context.verbose == 1 and getting_the_boot:
        print('Exiting positions not in longs and not IWM.')
        
    # Exit each position
    for sid in getting_the_boot:
        if context.verbose == 1:
            print('Exiting', sid)
        try:
            algo.order_target(sid, 0)
        except Exception as e:
            print(f'Failed to exit {sid}:', e)
            
    return

#################################################
# UTILITY FUNCTIONS
#################################################

def symbol(sym):
    """
    Gets the security ID for a symbol with caching for performance.
    """
    global SYM_SID_CACHE_DICT
    if SYM_SID_CACHE_DICT.get(sym) is None:
        try:
            securities = get_securities(vendors="usstock", fields=["Sid", "Symbol"])
            sid_val = algo.sid(sid=securities[securities.Symbol == sym].index.values[0])
            SYM_SID_CACHE_DICT.update({sym: sid_val})
        except Exception as e:
            print(f"Error getting symbol {sym}: {e}")
            sid_val = None
    else:
        sid_val = SYM_SID_CACHE_DICT[sym]
        
    return sid_val

def symbols(syms):
    """
    Gets security IDs for a list of symbols.
    """
    securities = get_securities(vendors="usstock", fields=["Sid", "Symbol"])
    securities = securities.reset_index()
    sidlist = []
    for sym in syms:
        sidlist.append(securities[securities.Symbol == sym].Sid)
    return sidlist

def get_tickers(sidlist):
    """
    Gets ticker symbols from a list of security IDs.
    """
    ticker_list = []
    for siditem in sidlist:
        ticker_list.append(siditem.symbol)
    return ticker_list

def get_sidids(sidlist):
    """
    Gets numeric IDs from a list of security IDs.
    """
    sidid_list = []
    for siditem in sidlist:
        sidid_list.append(siditem.real_sid)
    return sidid_list

def initialize_sids(context, data, algo):
    """
    Initialize security IDs for ETFs and sector indexes.
    """
    if context.sids_initialized == 0:
        print('Looking up security IDs >>>>>>')
        
        # Initialize ETF security IDs
        context.benchmarkSecurity = algo.sid(symbol('IWM').real_sid)
        context.iwm_sid = algo.sid(symbol('IWM').real_sid)
        context.dia_sid = algo.sid(symbol('DIA').real_sid)
        context.ibm_sid = algo.sid(symbol('IBM').real_sid)
        context.spysym = algo.sid(symbol('SPY').real_sid)
        context.qqq_sid = algo.sid(symbol('QQQ').real_sid)
        context.rwm_sid = algo.sid(symbol('RWM').real_sid)
        context.tlt_sid = algo.sid(symbol('TLT').real_sid)
        context.iwb_sid = algo.sid(symbol('IWB').real_sid)
        
        # Define non-tradable symbols
        context.no_trade_sym = symbols([
            'OEF', 'QQQ', 'IWM', 'SPY', 'TLT', 
            'MTUM', 'SPYG', 'QUAL', 'DIA','SPHB'
        ])
        print('Non-tradable symbols:', context.no_trade_sym)
        
        # Initialize sector ETFs
        context.sector_etf = []
        context.sector_etf_dict = {}
        for sym in ['IYZ', 'XLF', 'XLE', 'XLK', 'XLB', 'XLY', 'XLI', 'XLV', 'XLP', 'XLU']:
            sid_var = algo.sid(symbol(sym).real_sid)
            context.sector_etf.append(sid_var)
            context.sector_etf_dict.update({sym: sid_var})
        context.sector_etf = pd.Series(context.sector_etf)
        print("Sector ETF dictionary:", context.sector_etf_dict)
        
        # Initialize index ETFs
        context.index_etf = []
        context.index_etf_dict = {}
        for sym in ['IWM', 'QQQ']:
            sid_var = algo.sid(symbol(sym).real_sid)
            context.index_etf.append(sid_var)
            context.index_etf_dict.update({sym: sid_var})
        context.index_etf = pd.Series(context.index_etf)
        print("Index ETF dictionary:", context.index_etf_dict)
         
        # Define short symbol weights
        context.short_symbol_weights = {
            context.iwm_sid: 1,
        }
        
        context.sids_initialized = 1

    return

def filter_symbols_sectors_universe(df):
    """
    Filters the universe by removing certain sectors and stocks.
    """
    # Fetch current datetime once for efficiency
    current_date = algo.get_datetime().date()
    current_year = current_date.year
    current_month = current_date.month

    # Drop stocks based on acquisition dates or other exclusions
    to_drop = []
    if (current_year > 2022) or (current_year == 2022 and current_month >= 6):
        to_drop.append('GBT')
    if (current_year > 2022) or (current_year == 2022 and current_month >= 11):
        to_drop.append('ABMD')
    if (current_year > 2023) or (current_year == 2023 and current_month >= 7):
        to_drop.append('XM')
    if (current_year > 2023) or (current_year == 2023 and current_month >= 11):
        to_drop.append('SPLK')
    if (current_year > 2010):
        to_drop.append('MSTR')
    if (current_year > 2025) or (current_year == 2025 and current_month >= 3):
        to_drop.append('ITCI')

    # Drop stocks that appear in the to_drop list
    for stock_name in to_drop:
        df = df[df['name'] != stock_name]

    # Filter out Financials sector
    df = df[df['sector'] != 'Financials']
    #df = df[df['sector'] != 'Health Care']
    
    # df = df[(df['sector'] == 'Communication Services') | (df['sector'] == 'Information Technology') | 
    #     (df['sector'] == 'Industrials')| (df['sector'] == 'Consumer Discretionary') ]
    
#  context.sector_etf_dict['XLB']: 'Materials',
#         context.sector_etf_dict['XLY']: 'Consumer Discretionary',
#         context.sector_etf_dict['XLF']: 'Financials',
#         context.sector_etf_dict['XLP']: 'Consumer Staples',
#         context.sector_etf_dict['XLV']: 'Health Care',
#         context.sector_etf_dict['XLU']: 'Utilities',
#         context.sector_etf_dict['IYZ']: 'Communication Services',
#         context.sector_etf_dict['XLE']: 'Energy',
#         context.sector_etf_dict['XLI']: 'Industrials',
#         context.sector_etf_dict['XLK']: 'Information Technology',
    # Limit number of stocks from Energy and Real Estate sectors
    df_energy = df[df['sector'] == 'Energy'].sort_values(by='market_cap', ascending=False)[:20]
    df_real_estate = df[df['sector'] == 'Real Estate'].sort_values(by='market_cap', ascending=False)[:15]
    
    # Combine filtered sectors with remaining stocks
    df = pd.concat([
        df[~df['sector'].isin(['Energy', 'Real Estate'])],
        df_energy,
        df_real_estate
    ])

    # Remove stocks that are too new to the public
    df = df[df['publicdays'] > 0]
   
    

    return df

def get_normalized_weights(context, data, target):
    """
    Normalizes position weights for both long and short portfolios.
    Applies position size limits and ensures proper diversification.
    
    Parameters:
        context: Algorithm context
        data: Market data
        target: Column name containing the target alpha score
    
    Returns:
        longs_mcw: Normalized weights for long positions
        shorts_mcw: Normalized weights for short positions
    """
    # Ensure no missing values in the target column
    context.longs[target].fillna(value=0.001, inplace=True)  
    context.shorts[target].fillna(value=0.001, inplace=True)
    
    # Display sector statistics
    df1 = context.longs.groupby('sector')[target].agg(['median', 'mean', 'count'])
    print('Mean of target', target)
    print(df1)
    print('Length of target column:', len(context.longs[[target]]), 'Sum:', context.longs[[target]].sum())
    
    # Normalize long weights
    longs_mcw = abs(context.longs[[target]]) / abs(context.longs[[target]]).sum()

    # Apply position size limits to long positions
    longs_mcw[longs_mcw[target] >= MAX_POSITION_SIZE_LONG] = MAX_POSITION_SIZE_LONG
    longs_mcw[longs_mcw[target] <= 0.002] = 0.02
    longs_mcw = abs(longs_mcw[[target]]) / abs(longs_mcw[[target]]).sum()
    
    # Apply position size limits again (for boundary cases)
    longs_mcw[longs_mcw[target] >= MAX_POSITION_SIZE_LONG] = MAX_POSITION_SIZE_LONG
    longs_mcw[longs_mcw[target] <= 0.002] = 0.02
    longs_mcw = abs(longs_mcw[[target]]) / abs(longs_mcw[[target]]).sum()
    
    # Normalize short weights
    shorts_mcw = abs(context.shorts[[target]]) / abs(context.shorts[[target]]).sum()
    shorts_mcw[shorts_mcw[target] > MAX_POSITION_SIZE_SHORT] = MAX_POSITION_SIZE_SHORT
    shorts_mcw = abs(shorts_mcw[[target]]) / abs(shorts_mcw[[target]]).sum()
    
    return longs_mcw, shorts_mcw

def print_positions(port, port_w, target,factor=1):
    """
    Prints current positions with weights.
    
    Parameters:
        port: Portfolio of positions
        port_w: Weights for positions
        target: Column name containing weights
    """
   
    print(algo.get_datetime(timezone("America/Los_Angeles")))
    l = sorted(
        [list(c) for c in zip(port[0:], (port_w[target]*factor).round(6).astype(str).values[0:])], 
        key=lambda x: x[1], 
        reverse=True
    )
    print(pd.Series(l).values)
    return

def place_short_orders(algo, context, symbol_weights, total_weight):
    """
    Places orders for short positions based on target weights.
    
    Parameters:
        algo: Algorithm context
        context: Context object containing portfolio information
        symbol_weights: Dictionary mapping symbols to their weights
        total_weight: Total target weight for all short positions
    """
    logger = logging.getLogger('LS-Prod-Algo')
    logger.setLevel(logging.DEBUG)
    handler = FlightlogHandler()
    logger.addHandler(handler)
    
    sum_of_weights = sum(symbol_weights.values())
    
    for symbol, weight in symbol_weights.items():
        # Calculate proportional weight
        target_percent = (weight / sum_of_weights) * total_weight
        print(f"Executing short target {symbol}, {target_percent * context.shortfact:.4f}")
        
        # Place order
        algo.order_target_percent(symbol, target_percent * context.shortfact)
    
    return

def GenerateMomentumList(context, data, etf_list, momlength):
    """
    Generates a list of ETFs ranked by momentum.
    
    Parameters:
        context: Algorithm context
        data: Market data
        etf_list: List of ETFs to rank
        momlength: Lookback period for momentum calculation
    
    Returns:
        List of ETFs sorted by momentum (highest first)
    """
    # Fetch price history for all ETFs in one call
    price_history = data.history(etf_list, 'price', momlength, '1d')
    
    # Calculate percent change for the whole DataFrame
    pct_change = (price_history.iloc[-1] - price_history.iloc[0]) / price_history.iloc[0]
    
    # Convert to DataFrame for easy sorting
    momentum_df = pct_change.to_frame(name='momentum').reset_index()
    
    # Sort by momentum in descending order
    momentum_df = momentum_df.sort_values(by='momentum', ascending=False)
    
    # Select the top securities
    top_momentum_list = momentum_df.head(context.topMom).values.tolist()

    return top_momentum_list

def RemoveSectors(context, etf, dfs, prt_str):
    """
    Removes stocks from specific sectors based on ETF mapping.
    
    Parameters:
        context: Algorithm context
        etf: ETF representing sector to remove
        dfs: DataFrame of stocks
        prt_str: Format string for printing
    
    Returns:
        Filtered DataFrame with sector removed
    """
    # Map ETFs to sectors that should be removed
    etf_sector_map = {
        context.sector_etf_dict['XLB']: 'Materials',
        context.sector_etf_dict['XLY']: 'Consumer Discretionary',
        context.sector_etf_dict['XLF']: 'Financials',
        context.sector_etf_dict['XLP']: 'Consumer Staples',
        context.sector_etf_dict['XLV']: 'Health Care',
        context.sector_etf_dict['XLU']: 'Utilities',
        context.sector_etf_dict['IYZ']: 'Communication Services',
        context.sector_etf_dict['XLE']: 'Energy',
        context.sector_etf_dict['XLI']: 'Industrials',
        context.sector_etf_dict['XLK']: 'Information Technology',
    }
    
    # Check if the ETF is in the map and filter the dataframe accordingly
    if etf in etf_sector_map:
        sector_to_remove = etf_sector_map[etf]
        dfs = dfs[dfs['sector'] != sector_to_remove]
        print(prt_str % etf)

    return dfs

def compute_trend(context, data):
    """
    Determines market trend based on VIX signal and other indicators.
    Sets position sizing factors accordingly.
    """
    print('VIX flag:', context.vixflag)
    
    # Store previous longfact
    context.longfact_last = context.longfact
    
    # Set parameters based on VIX flag
    if context.vixflag <= 0:
        # Bullish trend
        print('Trend mode: 1.5')
        context.vix_uptrend_flag = True
        context.longfact = 1.5
        
        # Adjust short factor based on seasonality
        if algo.get_datetime().date().month in SHORT_RESTRICTED_MONTHS:
            context.shortfact = 0.45  # Reduced short exposure in certain months
        else:
             context.shortfact = 0.9
        context.clip = 1.2  # Max drawdown adjustment
        
    else:
        # Bearish trend
        print('Trend mode: 1')
        context.vix_uptrend_flag = False
        context.longfact = 0.0 if context.spy_below80ma else abs(context.iwm_w) #long exposure depends on trend of spy vs. 80ma assign the short weight when above 80ma
        context.shortfact = 0.5  # Reduced short exposure
        context.clip = 1.6  # Max drawdown adjustment
        
    return

def compute_beta(context, data):
    """
    Computes the beta ratio between long and short portfolios
    relative to the benchmark.
    
    Returns:
        float: Beta ratio for position sizing adjustment
    """
    # Create array with benchmark security
    benchmark_array = np.array([context.benchmarkSecurity])

    # Combine with universe assets
    assets_array = np.concatenate((context.universe, benchmark_array))

    # Fetch price history
    prices = data.history(assets_array, 'price', 120, '1d')

    # Extract prices for portfolio components
    prices_longs = prices[context.longs.index.intersection(prices.columns)]
    prices_shorts = prices[context.shorts.index.intersection(prices.columns)]
    prices_spy = prices[context.benchmarkSecurity]

    # Calculate returns
    rets_long_port = prices_longs.pct_change().sum(axis=1)
    rets_short_port = prices_shorts.pct_change().sum(axis=1)
    rets_spy = prices_spy.pct_change()
    
    beta_span = 120

    # Compute covariances and variances using exponentially weighted moving average
    long_cov = rets_long_port.ewm(span=beta_span, adjust=True).cov(rets_spy)
    short_cov = rets_short_port.ewm(span=beta_span, adjust=True).cov(rets_spy)
    bench_var = rets_spy.ewm(span=beta_span, adjust=True).var()

    # Calculate betas
    long_beta = long_cov.iloc[-1] / bench_var.iloc[-1]
    short_beta = short_cov.iloc[-1] / bench_var.iloc[-1]
    
    # Calculate beta ratio
    beta_ratio = long_beta / short_beta

    print("long_beta, short_beta, beta_ratio:", long_beta, short_beta, beta_ratio)
    
    return beta_ratio

def weighted_moving_average(prices, period):
    """
    Calculates the Weighted Moving Average (WMA) for a given period.
    
    Parameters:
        prices: Price series
        period: Lookback period
    
    Returns:
        numpy.array: Weighted Moving Average values
    """
    weights = np.arange(1, period + 1)
    wma = np.convolve(prices, weights / weights.sum(), mode='valid')
    return wma

def hull_moving_average(prices, period):
    """
    Calculates the Hull Moving Average (HMA) for a given period.
    
    The Hull Moving Average is designed to reduce lag while improving smoothness.
    Formula: HMA = WMA(2 * WMA(price, n/2) - WMA(price, n), sqrt(n))
    
    Parameters:
        prices: Price series
        period: Lookback period
    
    Returns:
        numpy.array: Hull Moving Average values
    """
    # Step 1: Calculate WMA(n)
    wma_n = weighted_moving_average(prices, period)
    
    # Step 2: Calculate WMA(n/2)
    wma_half_n = weighted_moving_average(prices, period // 2)
    
    # Step 3: Calculate 2 * WMA(n/2) - WMA(n)
    raw_hma = 2 * wma_half_n[-len(wma_n):] - wma_n  # Align lengths
    
    # Step 4: Calculate WMA(sqrt(n)) of the raw HMA
    sqrt_n = int(np.sqrt(period))
    hma = weighted_moving_average(raw_hma, sqrt_n)
    
    return hma

def hull_ma_trend(prices, period, lookback=3):
    """
    Determines if the trend of the Hull Moving Average is positive or negative.
    
    Parameters:
        prices: Price series
        period: Period for Hull MA
        lookback: Number of periods to compare
    
    Returns:
        str: 'positive' or 'negative'
    """
    # Calculate Hull Moving Average
    hma = hull_moving_average(prices, period)
    
    # Get recent values
    recent_hma = hma[-lookback:]
    
    # Determine trend direction
    trend = "positive" if recent_hma[-1] > recent_hma[0] else "negative"
    
    return trend

def compute_weekly_stochastic(df, lookback_weeks=14):
    """
    Computes the weekly stochastic oscillator for a single symbol.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns and daily data.
        lookback_weeks (int): Number of weeks to look back (default: 14 weeks).
        
    Returns:
        float: Stochastic oscillator value.
    """
    # Resample daily data to weekly
    weekly_high = df['high'].resample('W-FRI').max()
    weekly_low = df['low'].resample('W-FRI').min()
    weekly_close = df['price'].resample('W-FRI').last()
    
    # Get the rolling highest high and lowest low over the lookback period
    highest_high = weekly_high.rolling(lookback_weeks).max()
    lowest_low = weekly_low.rolling(lookback_weeks).min()
    
    # Compute the stochastic oscillator for the latest week
    stochastic = ((weekly_close.iloc[-1] - lowest_low.iloc[-1]) /
                  (highest_high.iloc[-1] - lowest_low.iloc[-1])) * 100
    
    return stochastic

def enhanced_stock_screening(df):
   """Apply additional filters to reduce idiosyncratic risk"""
   print(f"Before enhanced screening: {len(df)} stocks")
   
   df = df[df['cash_return'].between(df['cash_return'].quantile(0.05), 
                                    df['cash_return'].quantile(0.95))]
   df = df[df['doll_vol'] >= MIN_DOLLAR_VOLUME]
   df = df[df['beta60SPY'].between(-0.5, 3.0)]
   
   required_columns = ['eps_gr_mean', 'fcf', 'entval']
   df = df.dropna(subset=required_columns)
   print(f"After enhanced screening: {len(df)} stocks")
   
   return df

def apply_market_cap_diversification(context, df):
   """Ensure diversification across market cap ranges"""
   print("Applying market cap diversification...")
   
   df['market_cap_bucket'] = pd.cut(df['market_cap'], 
                                  bins=[0, 2e9, 10e9, 50e9, np.inf],
                                  labels=['Small', 'Mid', 'Large', 'Mega'])
   
   filtered_df = pd.DataFrame()
   
   for bucket in df['market_cap_bucket'].unique():
       bucket_positions = df[df['market_cap_bucket'] == bucket]
       max_positions = min(len(bucket_positions), 
                          int(len(df) * MAX_MARKET_CAP_BUCKET_EXPOSURE))
       
       if len(bucket_positions) > max_positions:
           top_positions = bucket_positions.nlargest(max_positions, 'estrank')
           filtered_df = pd.concat([filtered_df, top_positions])
       else:
           filtered_df = pd.concat([filtered_df, bucket_positions])
   
   print(f"After market cap diversification: {len(filtered_df)} stocks")
   return filtered_df.drop(columns=['market_cap_bucket'])

def apply_volatility_adjustment(context, df, data):
   """Adjust position sizes based on individual stock volatility"""
   print("Applying volatility adjustment...")
   
   try:
       prices = data.history(df.index, 'price', 60, '1d')
       returns = prices.pct_change().dropna()
       volatilities = returns.std()
       
       TARGET_VOLATILITY = volatilities.median()
       df['vol_adjustment'] = TARGET_VOLATILITY / volatilities
       df['vol_adjustment'] = df['vol_adjustment'].clip(0.5, 2.0)
       df['cash_return'] = df['cash_return'] * df['vol_adjustment']
       
       print(f"Volatility adjustment applied. Range: {df['vol_adjustment'].min():.2f} to {df['vol_adjustment'].max():.2f}")
       
   except Exception as e:
       print(f"Volatility adjustment failed: {e}")
       df['vol_adjustment'] = 1.0
   
   return df

def apply_correlation_constraints(context, data, selected_positions):
   """Reduce positions that are highly correlated"""
   print("Applying correlation constraints...")
   
   try:
       prices = data.history(selected_positions.index, 'price', 60, '1d')
       returns = prices.pct_change().dropna()
       corr_matrix = returns.corr()
       
       high_corr_pairs = []
       for i in range(len(corr_matrix.columns)):
           for j in range(i+1, len(corr_matrix.columns)):
               if abs(corr_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
                   high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                         corr_matrix.iloc[i, j]))
       
       print(f"Found {len(high_corr_pairs)} highly correlated pairs")
       
       for pair in high_corr_pairs:
           sid1, sid2, corr_val = pair
           if sid1 in selected_positions.index and sid2 in selected_positions.index:
               if selected_positions.loc[sid1, 'cash_return'] > selected_positions.loc[sid2, 'cash_return']:
                   selected_positions.loc[sid2, 'cash_return'] *= 0.7
               else:
                   selected_positions.loc[sid1, 'cash_return'] *= 0.7
       
   except Exception as e:
       print(f"Correlation constraints failed: {e}")
   
   return selected_positions

def apply_sector_constraints(context, longs_mcw, shorts_mcw):
   """Apply sector-level position limits"""
   print("Applying sector constraints...")
   
   if hasattr(context, 'longs') and 'sector' in context.longs.columns:
       longs_with_sectors = context.longs.copy()
       longs_with_sectors['weight'] = longs_mcw['cash_return'].values
       
       for sector in longs_with_sectors['sector'].unique():
           sector_mask = longs_with_sectors['sector'] == sector
           sector_weight = longs_with_sectors.loc[sector_mask, 'weight'].sum()
           
           if sector_weight > MAX_SECTOR_EXPOSURE:
               scale_factor = MAX_SECTOR_EXPOSURE / sector_weight
               sector_positions = longs_with_sectors.loc[sector_mask]
               
               for idx in sector_positions.index:
                   if idx in longs_mcw.index:
                       longs_mcw.loc[idx, 'cash_return'] *= scale_factor
               
               print(f"Scaled down {sector} sector by {scale_factor:.3f}")
   
   return longs_mcw, shorts_mcw
