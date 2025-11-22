"""
Long-Short Equity Trading Algorithm - Zipline-Reloaded Port
-----------------------------------------------------------
This algorithm implements a long-short equity strategy that selects securities based on:
1. Fundamental factors (cash return, enterprise value, growth metrics)
2. Technical factors (price momentum, relative strength)
3. Sentiment indicators
4. Machine learning predictions

Ported from QuantRocket to Zipline-Reloaded with multi_source Pipeline support.

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

# Zipline-Reloaded imports (replacing QuantRocket)
from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    order_target_percent,
    order_target,
    schedule_function,
    date_rules,
    time_rules,
    set_slippage,
    set_commission,
    set_benchmark,
    symbol as zp_symbol,
    sid,
    get_datetime,
)
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.utils.numpy_utils import repeat_first_axis, repeat_last_axis, rolling_window
from zipline.pipeline.factors import (Latest, Returns, RollingLinearRegressionOfReturns,
                                     SimpleMovingAverage, SimpleBeta, PercentChange, RSI,
                                     MACDSignal, DailyReturns, AnnualizedVolatility,
                                     AverageDollarVolume, RateOfChangePercentage, VWAP)
from zipline.pipeline.classifiers import CustomClassifier
from zipline.pipeline.filters import StaticAssets
# Sharadar fundamentals for FCF
from zipline.pipeline.data.sharadar import SharadarFundamentals
from zipline.finance import commission, slippage

# Multi-source module for custom databases
from zipline.pipeline import multi_source as ms

# FlightLog for real-time monitoring (Zipline-Reloaded version)
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog

from operator import itemgetter

#################################################
# GLOBAL CONFIGURATION VARIABLES
#################################################

# ML Learning Configuration
ML_GLOBAL_COUNTER = 0
ML_MODEL_REUSE_LIMIT = 1
ML_CLASSIFIER_GLOBAL = 0

# Symbol-SID Cache for performance
SYM_SID_CACHE_DICT = {}

# Asset finder reference (set during initialization)
ASSET_FINDER = None

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
# DATABASE DEFINITIONS (Multi-Source Pattern)
#################################################

class CustomFundamentals(ms.Database):
    """Primary fundamentals database with core company financial data."""

    CODE = "fundamentals"
    LOOKBACK_WINDOW = 240

    # Company identifiers
    Symbol = ms.Column(object)
    Inobjectument = ms.Column(object)
    CompanyCommonName = ms.Column(object)
    GICSSectorName = ms.Column(object)

    # Price and volume reference data
    RefPriceClose = ms.Column(float)
    RefVolume = ms.Column(float)

    # Valuation metrics
    EnterpriseValue_DailyTimeSeries_ = ms.Column(float)
    CompanyMarketCap = ms.Column(float)

    # Cash flow metrics
    FOCFExDividends_Discrete = ms.Column(float)
    InterestExpense_NetofCapitalizedInterest = ms.Column(float)
    CashFlowComponent_Current = ms.Column(float)
    CashFlowPerShare_BrokerEstimate = ms.Column(float)
    FreeCashFlowPerShare_BrokerEstimate = ms.Column(float)

    # Debt metrics
    Debt_Total = ms.Column(float)
    CashCashEquivalents_Total = ms.Column(float)

    # Earnings metrics
    EarningsPerShare_Actual = ms.Column(float)
    EarningsPerShare_SmartEstimate_prev_Q = ms.Column(float)
    EarningsPerShare_ActualSurprise = ms.Column(float)
    EarningsPerShare_SmartEstimate_current_Q = ms.Column(float)
    EPS_SurpirsePrct_prev_Q = ms.Column(float)

    # Growth and target metrics
    LongTermGrowth_Mean = ms.Column(float)
    PriceTarget_Median = ms.Column(float)
    Estpricegrowth_percent = ms.Column(float)

    # Alpha model rankings
    CombinedAlphaModelSectorRank = ms.Column(float)
    CombinedAlphaModelSectorRankChange = ms.Column(float)
    CombinedAlphaModelRegionRank = ms.Column(float)
    EarningsQualityRegionRank_Current = ms.Column(float)

    # Valuation ratios
    EnterpriseValueToEBIT_DailyTimeSeriesRatio_= ms.Column(float)
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_= ms.Column(float)
    EnterpriseValueToSales_DailyTimeSeriesRatio_ = ms.Column(float)
    ForwardPEG_DailyTimeSeriesRatio_= ms.Column(float)
    PriceEarningsToGrowthRatio_SmartEstimate_= ms.Column(float)
    ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_ = ms.Column(float)
    ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_ = ms.Column(float)
    ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_ = ms.Column(float)

    # Other metrics
    TradeDate = ms.Column(object)
    Dividend_Per_Share_SmartEstimate= ms.Column(float)
    ReturnOnInvestedCapital_BrokerEstimate = ms.Column(float)

    # Analyst metrics
    Recommendation_NumberOfTotal = ms.Column(float)
    Recommendation_Median_1_5_= ms.Column(float)
    Recommendation_NumberOfStrongBuy = ms.Column(float)
    Recommendation_NumberOfBuy = ms.Column(float)
    Recommendation_Mean_1_5_ = ms.Column(float)
    ReturnOnCapitalEmployed_Mean = ms.Column(float)

    # Return metrics
    ReturnOnEquity_SmartEstimat = ms.Column(float)
    ReturnOnAssets_SmartEstimate = ms.Column(float)

    # Profitability metrics
    GrossProfitMargin_ = ms.Column(float)
    GrossProfitMargin_ActualSurprise = ms.Column(float)

    # Prediction columns
    pred = ms.Column(float)
    forcast = ms.Column(float)
    bc1 = ms.Column(float)  # BC signal

class CustomFundamentals2(ms.Database):
    """Sentiment data database."""

    CODE = "refe-fundamentals-sent"
    LOOKBACK_WINDOW = 200

    sent2pol = ms.Column(float)
    sent2sub = ms.Column(float)
    sentvad = ms.Column(float)
    sentvad_neg = ms.Column(float)
    ChatGPT = ms.Column(float)
    SentVal1 = ms.Column(float)
    SentVal2 = ms.Column(float)
    SentVal3 = ms.Column(float)
    SentVal4 = ms.Column(float)

class CustomFundamentals3(ms.Database):
    """Alternative sentiment metrics database."""

    CODE = "refe-fundamentals-alex"
    LOOKBACK_WINDOW = 200

    Sentiment = ms.Column(float)
    Confidence = ms.Column(float)
    Novelty = ms.Column(float)
    Relevance = ms.Column(float)
    MarketImpactScore = ms.Column(float)
    Prob_POS = ms.Column(float)
    Prob_NTR = ms.Column(float)
    Prob_NEG = ms.Column(float)
    pred = ms.Column(float)

class CustomFundamentals4(ms.Database):
    """VIX data database."""

    CODE = "vixdata"
    LOOKBACK_WINDOW = 200

    pred = ms.Column(float)


class CustomFundamentals5(ms.Database):
    """Revenue surprise database."""

    CODE = "refe-fund-test-revesur"
    LOOKBACK_WINDOW = 240

    Revenue_ActualSurprise = ms.Column(float)

class CustomFundamentals6(ms.Database):
    """Earnings calendar database."""

    CODE = "refe-fundamentals-ecal"
    LOOKBACK_WINDOW = 240

    Earn_Date = ms.Column(object)
    Earn_Collection_Date = ms.Column(object)

class CustomFundamentals7(ms.Database):
    """Financial ratios database."""

    CODE = "refe-fundamentals-finratios"
    LOOKBACK_WINDOW = 240

    # Identifiers
    period = ms.Column(object)
    companyName = ms.Column(object)
    calendarYear = ms.Column(float)

    # Liquidity ratios
    currentRatio = ms.Column(float)
    quickRatio = ms.Column(float)
    cashRatio = ms.Column(float)

    # Efficiency ratios
    daysOfSalesOutstanding = ms.Column(float)
    daysOfInventoryOutstanding = ms.Column(float)
    operatingCycle = ms.Column(float)
    daysOfPayablesOutstanding = ms.Column(float)
    cashConversionCycle = ms.Column(float)
    receivablesTurnover = ms.Column(float)
    payablesTurnover = ms.Column(float)
    inventoryTurnover = ms.Column(float)
    fixedAssetTurnover = ms.Column(float)
    assetTurnover = ms.Column(float)

    # Profitability ratios
    grossProfitMargin = ms.Column(float)
    operatingProfitMargin = ms.Column(float)
    pretaxProfitMargin = ms.Column(float)
    netProfitMargin = ms.Column(float)
    effectiveTaxRate = ms.Column(float)
    returnOnAssets = ms.Column(float)
    returnOnEquity = ms.Column(float)
    returnOnCapitalEmployed = ms.Column(float)
    netIncomePerEBT = ms.Column(float)
    ebtPerEbit = ms.Column(float)
    ebitPerRevenue = ms.Column(float)

    # Debt ratios
    debtRatio = ms.Column(float)
    debtEquityRatio = ms.Column(float)
    longTermDebtToCapitalization = ms.Column(float)
    totalDebtToCapitalization = ms.Column(float)
    interestCoverage = ms.Column(float)
    cashFlowToDebtRatio = ms.Column(float)
    companyEquityMultiplier = ms.Column(float)

    # Cash flow ratios
    operatingCashFlowPerShare = ms.Column(float)
    freeCashFlowPerShare = ms.Column(float)
    cashPerShare = ms.Column(float)
    payoutRatio = ms.Column(float)
    operatingCashFlowSalesRatio = ms.Column(float)
    freeCashFlowOperatingCashFlowRatio = ms.Column(float)
    cashFlowCoverageRatios = ms.Column(float)
    shortTermCoverageRatios = ms.Column(float)
    capitalExpenditureCoverageRatio = ms.Column(float)
    dividendPaidAndCapexCoverageRatio = ms.Column(float)
    dividendPayoutRatio = ms.Column(float)

    # Valuation ratios
    priceBookValueRatio = ms.Column(float)
    priceToBookRatio = ms.Column(float)
    priceToSalesRatio = ms.Column(float)
    priceEarningsRatio = ms.Column(float)
    priceToFreeCashFlowsRatio = ms.Column(float)
    priceToOperatingCashFlowsRatio = ms.Column(float)
    priceCashFlowRatio = ms.Column(float)
    priceEarningsToGrowthRatio = ms.Column(float)
    priceSalesRatio = ms.Column(float)
    dividendYield = ms.Column(float)
    enterpriseValueMultiple = ms.Column(float)
    priceFairValue = ms.Column(float)

class CustomFundamentals8(ms.Database):
    """Forward EV/FCF database."""

    CODE = "refe-fundamentals-eps"
    LOOKBACK_WINDOW = 240

    ForwardEVFreeCashFlow_SmartEstimate_ = ms.Column(float)

class CustomFundamentals9(ms.Database):
    """BC data database."""

    CODE = "bcdata"
    LOOKBACK_WINDOW = 200

    bc1 = ms.Column(float)


#################################################
# CUSTOM FACTORS
#################################################

class MoneyFlowFactor(CustomFactor):
    """
    Money Flow Factor - measures the relationship between price and volume
    """

    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
    window_length = 14

    def compute(self, today, assets, out, high, low, close, volume):
        typical_price = (high + low + close) / 3.0
        money_flow = typical_price * volume
        out[:] = np.nan

        for i in range(len(assets)):
            asset_typical_price = typical_price[:, i]
            asset_money_flow = money_flow[:, i]

            if len(asset_typical_price) < 2:
                continue

            price_changes = np.diff(asset_typical_price)
            positive_mf = 0.0
            negative_mf = 0.0

            for j in range(len(price_changes)):
                if price_changes[j] > 0:
                    positive_mf += asset_money_flow[j + 1]
                elif price_changes[j] < 0:
                    negative_mf += asset_money_flow[j + 1]

            if negative_mf == 0:
                out[i] = 100.0
            elif positive_mf == 0:
                out[i] = 0.0
            else:
                money_ratio = positive_mf / negative_mf
                out[i] = 100.0 - (100.0 / (1.0 + money_ratio))


class SimpleMoneyFlowFactor(CustomFactor):
    """Simplified Money Flow Factor"""

    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 14

    def compute(self, today, assets, out, close, volume):
        out[:] = np.nan

        for i in range(len(assets)):
            asset_close = close[:, i]
            asset_volume = volume[:, i]

            if len(asset_close) < 2:
                continue

            price_changes = np.diff(asset_close)
            money_flows = price_changes * asset_volume[1:]
            out[i] = np.sum(money_flows)


class VolumeWeightedMoneyFlowFactor(CustomFactor):
    """Volume Weighted Money Flow"""

    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 20

    def compute(self, today, assets, out, close, volume):
        out[:] = np.nan

        for i in range(len(assets)):
            asset_close = close[:, i]
            asset_volume = volume[:, i]

            if len(asset_close) < 2 or np.sum(asset_volume) == 0:
                continue

            price_returns = np.diff(asset_close) / asset_close[:-1]
            volume_weights = asset_volume[1:]

            if np.sum(volume_weights) > 0:
                vwap_return = np.sum(price_returns * volume_weights) / np.sum(volume_weights)
                out[i] = vwap_return * np.sum(volume_weights)
            else:
                out[i] = 0.0


class ChaikinMoneyFlowFactor(CustomFactor):
    """Chaikin Money Flow (CMF)"""

    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
    window_length = 21

    def compute(self, today, assets, out, high, low, close, volume):
        out[:] = np.nan

        for i in range(len(assets)):
            asset_high = high[:, i]
            asset_low = low[:, i]
            asset_close = close[:, i]
            asset_volume = volume[:, i]

            high_low_diff = asset_high - asset_low
            clv = np.zeros_like(asset_close)
            mask = high_low_diff != 0

            clv[mask] = ((asset_close[mask] - asset_low[mask]) -
                        (asset_high[mask] - asset_close[mask])) / high_low_diff[mask]

            money_flow_volume = clv * asset_volume
            total_volume = np.sum(asset_volume)
            if total_volume > 0:
                out[i] = np.sum(money_flow_volume) / total_volume
            else:
                out[i] = 0.0


class MoneyFlowIndexFactor(CustomFactor):
    """Classic Money Flow Index (MFI)"""

    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
    window_length = 14

    def compute(self, today, assets, out, high, low, close, volume):
        typical_price = (high + low + close) / 3.0
        out[:] = np.nan

        for i in range(len(assets)):
            asset_typical_price = typical_price[:, i]
            asset_volume = volume[:, i]

            if len(asset_typical_price) < 2:
                continue

            money_flow = asset_typical_price * asset_volume
            price_changes = np.diff(asset_typical_price)

            positive_mf = 0.0
            negative_mf = 0.0

            for j in range(len(price_changes)):
                if price_changes[j] > 0:
                    positive_mf += money_flow[j + 1]
                elif price_changes[j] < 0:
                    negative_mf += money_flow[j + 1]

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
        latest_close = close[-1]
        ma_200 = np.mean(close, axis=0)
        out[:] = (latest_close > ma_200).astype(int)


class StochasticOscillator(CustomFactor):
    """Stochastic Oscillator"""
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 14

    def compute(self, today, assets, out, high, low, close):
        highest_high = np.max(high, axis=0)
        lowest_low = np.min(low, axis=0)
        out[:] = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100


class StochasticOscillatorWeekly(CustomFactor):
    """Weekly Stochastic Oscillator"""
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 20 * 5

    def compute(self, today, assets, out, high, low, close):
        high_weekly = high.reshape(-1, 5, high.shape[1]).max(axis=1)
        low_weekly = low.reshape(-1, 5, low.shape[1]).min(axis=1)
        close_weekly = close.reshape(-1, 5, close.shape[1])[:, -1]

        highest_high = np.max(high_weekly, axis=0)
        lowest_low = np.min(low_weekly, axis=0)
        out[:] = ((close_weekly[-1] - lowest_low) / (highest_high - lowest_low)) * 100


class TenkanSen(CustomFactor):
    """Tenkan-sen (Conversion Line)"""
    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 9
    window_safe = True

    def compute(self, today, assets, out, highs, lows):
        highest_high = np.max(highs, axis=0)
        lowest_low = np.min(lows, axis=0)
        out[:] = (highest_high + lowest_low) / 2


class KijunSen(CustomFactor):
    """Kijun-sen (Base Line)"""
    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 26
    window_safe = True

    def compute(self, today, assets, out, highs, lows):
        highest_high = np.max(highs, axis=0)
        lowest_low = np.min(lows, axis=0)
        out[:] = (highest_high + lowest_low) / 2


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


class PublicSince(CustomFactor):
    """How long a security has been publicly traded"""
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, prices):
        prices = np.nan_to_num(prices)
        out[:] = ((prices[0] + prices[1] + prices[2] + prices[3] +
                  prices[4] + prices[5] + prices[6] + prices[7]))


class SumFactor(CustomFactor):
    """Sums a factor over the window length"""
    window_safe = True

    def compute(self, today, assets, out, factordata):
        out[:] = np.sum(factordata, axis=0)


class PreviousValue(CustomFactor):
    """Returns the previous day's value (1-day lag)"""
    window_length = 2
    window_safe = True

    def compute(self, today, assets, out, data):
        # data[0] is previous day, data[1] is current day
        out[:] = data[0]


class MLFactor(CustomFactor):
    """Machine learning factor that predicts future returns"""
    params = ('shift_target',)
    window_safe = True

    def compute(self, today, assets, out, target, *features, shift_target):
        global ML_GLOBAL_COUNTER
        global ML_MODEL_REUSE_LIMIT
        global ML_CLASSIFIER_GLOBAL

        self.imputer = impute.SimpleImputer(strategy='constant', fill_value=0)
        self.scaler = preprocessing.RobustScaler()
        self.scaler_2 = preprocessing.MinMaxScaler()

        self.imputer_Y = impute.SimpleImputer(strategy='constant', fill_value=0)
        self.scaler_Y = preprocessing.MinMaxScaler()

        self.clf = linear_model.LinearRegression(n_jobs=-1)

        X = np.dstack(features)
        Y = target

        print(get_datetime(timezone("America/Los_Angeles")))
        print('X raw', X.shape)
        print('Y raw', Y.shape)

        n_time, n_stocks, n_factors = X.shape
        print('t', n_time, 'stk', n_stocks, 'factors', n_factors)

        n_fwd_days = shift_target
        shift_index = n_time - n_fwd_days
        X = X[0:shift_index, :, :]
        n_time, n_stocks, n_factors = X.shape
        print('t', n_time, 'stk', n_stocks, 'factors', n_factors)

        X = X.reshape(n_time * n_stocks, n_factors)
        print('X post reshape', X.shape)

        # Guard against empty arrays (no samples after filtering)
        if X.shape[0] == 0:
            import logging
            logging.info(f'MLFactor EMPTY ARRAY: date={today}, n_time={n_time}, n_stocks={n_stocks}, shape={X.shape}')
            print(f'WARNING: No samples after filtering on {today}, returning zeros')
            out[:] = 0.0
            return

        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        Y = Y[-shift_index:]
        Y = Y.reshape(n_time * n_stocks, 1)
        print('Y post reshape', Y.shape)

        Y = self.imputer_Y.fit_transform(Y)
        Y = self.scaler_Y.fit_transform(Y)

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
    """Weighted excess return (alpha) vs SPY benchmark"""
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):
        global ASSET_FINDER

        # Get SPY SID using asset finder
        try:
            spy_asset = ASSET_FINDER.lookup_symbol('SPY', as_of_date=None)
            spy_sid = spy_asset.sid
            spy_idx = assets.get_loc(spy_sid)
        except:
            out[:] = np.nan
            return

        spy_close = close[:, spy_idx]
        is_asset = np.arange(len(assets)) != spy_idx
        asset_close = close[:, is_asset]

        ret_30 = asset_close[-1] / asset_close[-22] - 1
        ret_90 = asset_close[-1] / asset_close[-66] - 1
        ret_252 = asset_close[-1] / asset_close[0] - 1

        spy_ret_30 = spy_close[-1] / spy_close[-22] - 1
        spy_ret_90 = spy_close[-1] / spy_close[-66] - 1
        spy_ret_252 = spy_close[-1] / spy_close[0] - 1

        alpha_30 = ret_30 - spy_ret_30
        alpha_90 = ret_90 - spy_ret_90
        alpha_252 = ret_252 - spy_ret_252

        weighted_alpha = 0.15 * alpha_30 + 0.5 * alpha_90 + 0.35 * alpha_252

        out_vals = np.empty(len(assets))
        out_vals[:] = np.nan
        out_vals[is_asset] = weighted_alpha
        out[:] = out_vals


class SumVolume(CustomFactor):
    """Sum of volume over window"""
    inputs = [USEquityPricing.volume]

    def compute(self, today, assets, out, volume):
        out[:] = np.sum(volume, axis=0)


#################################################
# SYMBOL LOOKUP FUNCTIONS (Zipline-Reloaded)
#################################################

def symbol(sym):
    """
    Gets the security ID for a symbol using Zipline's asset finder.
    Replaces QuantRocket's get_securities approach.
    """
    global SYM_SID_CACHE_DICT
    global ASSET_FINDER

    if ASSET_FINDER is None:
        # Return None if asset finder not yet initialized
        return None

    if SYM_SID_CACHE_DICT.get(sym) is None:
        try:
            asset = ASSET_FINDER.lookup_symbol(sym, as_of_date=None)
            SYM_SID_CACHE_DICT.update({sym: asset})
        except Exception as e:
            print(f"Error getting symbol {sym}: {e}")
            return None

    return SYM_SID_CACHE_DICT.get(sym)


def symbols(syms):
    """Gets security IDs for a list of symbols."""
    return [symbol(sym) for sym in syms]


def get_tickers(sidlist):
    """Gets ticker symbols from a list of security IDs."""
    return [s.symbol for s in sidlist if s is not None]


#################################################
# MAIN ALGORITHM FUNCTIONS
#################################################

def initialize(context):
    """
    Initialize the trading algorithm.
    """
    global ASSET_FINDER

    print("[DEBUG] initialize() starting...")

    # Store asset finder reference for symbol lookup
    from zipline.data import bundles
    bundle = bundles.load('sharadar')
    ASSET_FINDER = bundle.asset_finder
    print(f"[DEBUG] ASSET_FINDER set: {ASSET_FINDER}")

    # Attach our main stock selection pipeline
    print("[DEBUG] Calling make_pipeline()...")
    pipeline = make_pipeline()
    print("[DEBUG] make_pipeline() returned successfully")
    attach_pipeline(pipeline, 'my_pipeline')
    print("[DEBUG] Pipeline attached")

    # Set the benchmark
    spy = symbol('SPY')
    if spy:
        set_benchmark(spy)

    # Initialize portfolio and risk parameters
    context.longfact = 1.0
    context.shortfact = 1.0
    context.order_id = {}
    context.topMom = 9
    context.max_liquid = context.portfolio.starting_cash
    context.cash_adjustment = 0
    context.dd_factor = 1.0
    context.draw_down = 0.0
    context.print_set_delta = False
    context.days_offset = 1
    context.initialized = 0
    context.sids_initialized = 0
    context.verbose = 1
    context.qqq_ratio_prev = 0
    context.spy_ratio_prev = 0
    context.total_ws = 0
    context.iwm_w = 0
    context.spy_below80ma = False
    context.vixflag = 0
    context.vixflag_prev = 0
    context.clip = 1.0

    # Set trading costs
    set_slippage(slippage.FixedSlippage(spread=SLIPPAGE_SPREAD))
    set_commission(commission.PerShare(cost=COMMISSION_COST, min_trade_cost=MIN_TRADE_COST))

    # Schedule functions
    schedule_function(
        initial_allocation,
        date_rules.every_day()
    )

    schedule_function(
        regular_allocation,
        date_rules.week_start(days_offset=context.days_offset)
    )

    schedule_function(
        exit_positions,
        date_rules.week_start(days_offset=context.days_offset)
    )

    # Enable FlightLog for real-time monitoring
    try:
        enable_flightlog(host='localhost', port=9020)
        log_to_flightlog('LS Strategy initialized', level='INFO')
    except:
        print("FlightLog not available - continuing without real-time monitoring")


def make_pipeline():
    """
    Creates the stock selection pipeline.
    """
    global ASSET_FINDER

    print("[DEBUG] make_pipeline() starting...")

    # Get benchmark assets using asset_finder (available after initialize sets it)
    print("[DEBUG] Looking up SPY...")
    spy_asset = ASSET_FINDER.lookup_symbol('SPY', as_of_date=None)
    print(f"[DEBUG] SPY asset: {spy_asset}, type: {type(spy_asset)}, sid: {spy_asset.sid if spy_asset else 'None'}")

    print("[DEBUG] Looking up IWM...")
    iwm_asset = ASSET_FINDER.lookup_symbol('IWM', as_of_date=None)
    print(f"[DEBUG] IWM asset: {iwm_asset}, type: {type(iwm_asset)}, sid: {iwm_asset.sid if iwm_asset else 'None'}")

    print("[DEBUG] Looking up QQQ...")
    qqq_asset = ASSET_FINDER.lookup_symbol('QQQ', as_of_date=None)
    print(f"[DEBUG] QQQ asset: {qqq_asset}, type: {type(qqq_asset)}, sid: {qqq_asset.sid if qqq_asset else 'None'}")

    # Initial universe filter - top stocks by market cap
    print("[DEBUG] Creating tradable_filter...")
    tradable_filter = (CustomFundamentals.CompanyMarketCap.latest.top(UNIVERSE_SIZE)) | StaticAssets([symbol('IBM')])
    print("[DEBUG] tradable_filter created")

    # Money flow factors
    print("[DEBUG] Creating money flow factors...")
    money_flow_index = MoneyFlowIndexFactor(mask=tradable_filter, window_length=90)
    chaikin_money_flow = ChaikinMoneyFlowFactor(mask=tradable_filter, window_length=90)
    simple_money_flow = SimpleMoneyFlowFactor(mask=tradable_filter, window_length=90)
    volume_weighted_mf = VolumeWeightedMoneyFlowFactor(mask=tradable_filter, window_length=90)
    print("[DEBUG] Money flow factors created")

    # Create SimpleBeta factors with debug
    print("[DEBUG] Creating SimpleBeta for SPY...")
    try:
        beta_spy = SimpleBeta(target=spy_asset, regression_length=60)
        print(f"[DEBUG] SimpleBeta SPY created: {beta_spy}")
    except Exception as e:
        print(f"[DEBUG] ERROR creating SimpleBeta SPY: {e}")
        raise

    print("[DEBUG] Creating SimpleBeta for IWM...")
    try:
        beta_iwm = SimpleBeta(target=iwm_asset, regression_length=60)
        print(f"[DEBUG] SimpleBeta IWM created: {beta_iwm}")
    except Exception as e:
        print(f"[DEBUG] ERROR creating SimpleBeta IWM: {e}")
        raise

    # Build pipeline columns one by one for debugging
    print("[DEBUG] Building pipeline columns dict...")
    columns = {}

    print("[DEBUG] Adding company info columns...")
    columns['name'] = CustomFundamentals.Symbol.latest
    columns['compname'] = CustomFundamentals.CompanyCommonName.latest
    columns['sector'] = CustomFundamentals.GICSSectorName.latest

    print("[DEBUG] Adding market data columns...")
    columns['market_cap'] = CustomFundamentals.CompanyMarketCap.latest
    columns['entval'] = CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest
    columns['price'] = USEquityPricing.close.latest
    columns['volume'] = USEquityPricing.volume.latest
    columns['fs_price'] = CustomFundamentals.RefPriceClose.latest
    columns['fs_volume'] = CustomFundamentals.RefVolume.latest
    columns['sumvolume'] = SumVolume(window_length=3)

    print("[DEBUG] Adding earnings columns...")
    columns['eps_ActualSurprise_prev_Q_percent'] = CustomFundamentals.EarningsPerShare_ActualSurprise.latest
    columns['eps_gr_mean'] = CustomFundamentals.LongTermGrowth_Mean.latest

    print("[DEBUG] Adding financial columns...")
    columns['CashCashEquivalents_Total'] = CustomFundamentals.CashCashEquivalents_Total.latest
    # Use Sharadar FCF
    #columns['fcf'] = SharadarFundamentals.fcf.latest
    # Commented out - using Sharadar FCF instead
    columns['fcf'] = CustomFundamentals.FOCFExDividends_Discrete.latest
    columns['int'] = CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest

    print("[DEBUG] Adding beta columns...")
    columns['beta60SPY'] = beta_spy
    columns['beta60IWM'] = beta_iwm

    print("[DEBUG] Adding technical indicators...")
    columns['smav'] = SimpleMovingAverage(inputs=[USEquityPricing.volume], window_length=10)
    columns['slope120'] = Slope(window_length=120, mask=tradable_filter).slope.zscore(mask=tradable_filter)
    columns['slope220'] = Slope(window_length=220, mask=tradable_filter).slope.zscore(mask=tradable_filter)
    columns['slope90'] = Slope(window_length=90, mask=tradable_filter).slope.zscore(mask=tradable_filter)
    columns['slope30'] = Slope(window_length=30, mask=tradable_filter).slope.zscore(mask=tradable_filter)
    columns['stk20w'] = StochasticOscillatorWeekly()
    columns['above_200dma'] = Above200DMA(mask=tradable_filter)
    columns['walpha'] = WeightedAlpha()

    print("[DEBUG] Adding relative strength columns...")
    columns['RS140_QQQ'] = RelativeStrength(window_length=140, market_sid=qqq_asset.sid)
    columns['RS160_QQQ'] = RelativeStrength(window_length=160, market_sid=qqq_asset.sid)
    columns['RS180_QQQ'] = RelativeStrength(window_length=180, market_sid=qqq_asset.sid)

    print("[DEBUG] Adding return columns...")
    columns['Ret60'] = Returns(window_length=60, mask=tradable_filter)
    columns['Ret120'] = Returns(window_length=120, mask=tradable_filter)
    columns['Ret220'] = Returns(window_length=220, mask=tradable_filter)

    print("[DEBUG] Adding other factors...")
    columns['publicdays'] = PublicSince(window_length=121)
    columns['vol'] = Volatility(window_length=10, mask=tradable_filter)

    # VIX and BC signals (loaded via IBM carrier in fundamentals database)
    print("[DEBUG] Adding VIX and BC signal columns...")
    columns['vixflag'] =  PreviousValue(inputs=[CustomFundamentals.pred])
    columns['vixflag0'] = CustomFundamentals.pred.latest  # Previous day's VIX flag

     
    columns['bc1'] = CustomFundamentals.bc1.latest

    # print("[DEBUG] Adding MLFactor...")
    # columns['MLfactor'] = MLFactor(
    #     inputs=[
    #         Returns(window_length=90, mask=tradable_filter),
    #         CustomFundamentals.EnterpriseValueToEBITDA_DailyTimeSeriesRatio_,
    #         CustomFundamentals.LongTermGrowth_Mean,
    #         CustomFundamentals.CombinedAlphaModelSectorRank,
    #         CustomFundamentals.ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_,
    #     ],
    #     window_length=180,
    #     mask=tradable_filter,
    #     shift_target=10
    # )

    # Commenting out sentiment factors - SumFactor usage needs fixing
    # The SumFactor class doesn't have inputs defined properly
    print("[DEBUG] Skipping sentiment factors (need fixing)...")
    # columns['sentcomb'] = (
    #     SumFactor(CustomFundamentals2.sentvad_neg, window_length=18).zscore() +
    #     SumFactor(CustomFundamentals2.sent2sub, window_length=18).zscore() +
    #     (1/SumFactor(CustomFundamentals2.sent2pol, window_length=18).zscore())
    # )
    # columns['sentest'] = 1/SumFactor(CustomFundamentals2.sent2pol, window_length=18)

    print("[DEBUG] Creating ms.Pipeline...")
    pipe = ms.Pipeline(
        screen=tradable_filter,
        columns=columns
    )
    print("[DEBUG] Pipeline created successfully")

    return pipe


def before_trading_start(context, data):
    """Daily preprocessing before trading begins."""
    # Initialize security IDs if needed
    initialize_sids(context, data)

    # Get pipeline output
    df = pipeline_output('my_pipeline')

    print(f"Raw stock universe size {df.shape}")
    print(get_datetime(timezone("America/Los_Angeles")))

    # DEBUG: Check if IBM is in the pipeline output
    log_to_flightlog(f"[DEBUG] IBM SID: {context.ibm_sid}", level='INFO')
    log_to_flightlog(f"[DEBUG] IBM in pipeline index: {context.ibm_sid in df.index}", level='INFO')

    # DEBUG: Check available columns
    if 'vixflag' in df.columns:
        log_to_flightlog(f"[DEBUG] vixflag column exists, dtype: {df['vixflag'].dtype}", level='INFO')
        log_to_flightlog(f"[DEBUG] vixflag non-null count: {df['vixflag'].notna().sum()}", level='INFO')
        log_to_flightlog(f"[DEBUG] vixflag sample values: {df['vixflag'].dropna().head(5).tolist()}", level='INFO')
    else:
        log_to_flightlog("[DEBUG] WARNING: vixflag column NOT in pipeline output!", level='WARNING')

    if 'bc1' in df.columns:
        log_to_flightlog(f"[DEBUG] bc1 column exists, dtype: {df['bc1'].dtype}", level='INFO')
    else:
        log_to_flightlog("[DEBUG] WARNING: bc1 column NOT in pipeline output!", level='WARNING')

    # Update market condition indicators
    update_market_indicators(context, data)

    # Update VIX signal
    context.vixflag_prev = context.vixflag
    
    context.vixflag = df.loc[context.ibm_sid].vixflag.copy()
    context.vixflag0 = df.loc[context.ibm_sid].vixflag0.copy()
    context.bc1 = df.loc[context.ibm_sid].bc1.copy()
    log_to_flightlog(f"[DEBUG] Successfully read IBM signals: vixflag={context.vixflag}, bc1={context.bc1}", level='INFO')


    print('IBM-vixdata', context.vixflag)

    # Get IWM 20 week Stochastic
    iwm_sym = symbol('IWM')
    iwm_high = data.history(iwm_sym, 'high', 20*5, '1d')
    iwm_low = data.history(iwm_sym, 'low', 20*5, '1d')
    iwm_close = data.history(iwm_sym, 'close', 20*5, '1d')
    df_iwm = pd.DataFrame({'high': iwm_high, 'low': iwm_low, 'price': iwm_close})
    context.iwm_stk20w = compute_weekly_stochastic(df_iwm, lookback_weeks=20)
    print('IWM_stk20w', context.iwm_stk20w)

    # Determine market trend
    compute_trend(context, data)

    # Initialize daily tracking variables
    context.daily_flag = 0
    context.daily_print_flag = 0

    # Filter universe and generate alpha signals
    df = process_universe(context, df, data)

    return


def process_universe(context, df, data):
    """Process the universe of stocks and generate alpha signals."""
    # Remove excluded sectors
    df = filter_symbols_sectors_universe(df)

    # Calculate dollar volume
    df['doll_vol'] = df['price'] * df['smav']

    # Calculate cash return
    df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
    df['cash_return'] = df['cash_return'].replace([np.inf, -np.inf], np.nan)

    df.dropna(subset=['cash_return'], inplace=True)

    # Display sector returns
    sorted_df = df.groupby('sector')['cash_return'].agg(['mean']).sort_values(by='mean', ascending=False)
    if len(sorted_df) > 0:
        print(sorted_df.iloc[0].name)

    # Calculate momentum ranking
    df['myrs'] = df.slope120.rank() + df.RS140_QQQ.rank()

    print("cash ret, myrs --->>>>>> ", df.sort_values(by='market_cap', ascending=False)[0:10][['cash_return', 'myrs', 'slope90', 'above_200dma', 'walpha']], '\n')

    df = df.sort_values(by=['market_cap'], ascending=[False])[0:FILTERED_UNIVERSE_SIZE].copy()

    # Calculate final ranking
    if context.spyprice <= context.spyma80:
        df['estrank'] = df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1)
        print('switch estrank - SPY below spyma80')
    else:
        if get_datetime().date().month in GROWTH_SEASON_MONTHS:
            context.season = 1
        else:
            context.season = 0

        df['estrank'] = (
            #df['MLfactor'].rank() +
            (df['entval'].rank() * 2) +
            (df['cash_return'].rank()) +
            df['eps_gr_mean'].rank() * (4 if context.season == 1 else 1) +
            (df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1) / 3)
        )

    print(f"Filtered stock universe size {df.shape}")

    # Create portfolios
    select_long_portfolio(context, df, data)
    select_short_portfolio(context, df, data)
    context.topmcap = df.sort_values(by=['market_cap'], ascending=[False])[0:7].copy()

    # Combine universes
    context.universe = np.union1d(context.longs.index.values, context.shorts.index.values)

    # Calculate portfolio beta
    context.beta_ratio = max(1.3, compute_beta(context, data))
    print(f'Beta ratio: {context.beta_ratio:.4f}')

    return df


def select_long_portfolio(context, df, data):
    """Select securities for long portfolio."""
    dfl = df.copy()

    num_momentum_stocks = TOP_MOMENTUM_STOCKS
    num_value_stocks = LONG_PORTFOLIO_SIZE - num_momentum_stocks

    # Select value stocks
    context.longs_c = (
        dfl.sort_values(by=['estrank'], ascending=[False])[0:150]
           .sort_values(by=['cash_return'], ascending=[False])[0:num_value_stocks]
           .copy()
    )

    # Select momentum stocks
    if context.vix_uptrend_flag:
        context.longs_m = (
            dfl.sort_values(by=['RS140_QQQ'], ascending=[False])[0:500]
               .sort_values(by=['myrs'], ascending=[False])[0:num_momentum_stocks]
               .copy()
        )
    else:
        context.longs_m = (
            dfl.sort_values(by=['estrank'], ascending=[False])[0:150]
               .sort_values(by=['cash_return'], ascending=[False])[num_value_stocks:LONG_PORTFOLIO_SIZE]
               .copy()
        )

    # Combine longs
    c_set = set(context.longs_c.index)
    m_set = set(context.longs_m.index)
    context.longs = dfl[dfl.index.isin(c_set.union(m_set))].copy()
    print(f'Long portfolio size: {len(context.longs)}')

    context.longs = context.longs.sort_values(by=['cash_return'], ascending=[False]).copy()

    # Adjust by beta
    if context.spyprice >= context.spyma80:
        context.longs['cash_return'] /= winsorize(context.longs['beta60IWM'], limits=[0.005, 0.4])
    else:
        context.longs['cash_return'] /= winsorize(context.longs['beta60SPY'], limits=[0.005, 0.1])

    context.longs['cash_return'] = context.longs['cash_return'].clip(lower=0.005)

    # Adjust by slope
    slopefact = 1 + (context.longs['slope120'] / context.longs['slope120'].sum())
    context.longs['cash_return'] = context.longs['cash_return'] * slopefact**2

    context.longs = context.longs.sort_values(by=['cash_return'], ascending=[False])

    print(f'Long portfolio calculated with total cash return: {context.longs["cash_return"].sum()}')

    return


def select_short_portfolio(context, df, data):
    """Select securities for short portfolio."""
    dfs = df.copy()

    # Get sector momentum
    mom_list = GenerateMomentumList(context, data, context.sector_etf, 242)
    mom_list = [item[0] for item in mom_list]

    top_momentum_sector = mom_list[0]
    context.mometf = top_momentum_sector

    # Remove top momentum sector
    dfs = RemoveSectors(context, top_momentum_sector, dfs, "not shorting! %s")
    print('Bottom momentum sector:', mom_list[-1])

    # Select shorts
    context.shorts = (dfs.sort_values(by=['cash_return'], ascending=[True])[0:SHORT_PORTFOLIO_SIZE]
                    .copy()
    )
    return


def update_market_indicators(context, data):
    """Update market indicators."""
    # SPY moving averages
    context.price_history_spy100 = data.history(symbol('SPY'), 'price', 200, '1d')
    context.spyprice = context.price_history_spy100.values[-1]
    context.spyma21 = np.mean(context.price_history_spy100.tail(21).values)
    context.spyma50 = np.mean(context.price_history_spy100.tail(50).values)
    context.spyma80 = np.mean(context.price_history_spy100.tail(80).values)
    context.spyma85 = np.mean(context.price_history_spy100.tail(85).values)
    context.spyma150 = np.mean(context.price_history_spy100.tail(150).values)
    context.spyma200 = np.mean(context.price_history_spy100.tail(200).values)

    # IWM Hull moving averages
    context.price_history_iwm250 = data.history(symbol('IWM'), 'price', 250, '1d')
    context.iwmprice = context.price_history_iwm250.values[-1]
    context.iwmma50 = hull_moving_average(context.price_history_iwm250.values, 50)[-1]
    context.iwmma10 = hull_moving_average(context.price_history_iwm250.values, 10)[-1]
    context.hulltrend = hull_ma_trend(context.price_history_iwm250.values, 80, lookback=7)


def handle_data(context, data):
    """Function called on each trading bar."""
    time_minute = get_datetime(timezone("America/Los_Angeles")).minute
    time_hour = get_datetime(timezone("America/Los_Angeles")).hour

    context.account_leverage = 2.0

    if (time_hour == 12 and time_minute == 59):
        printflag = True
    else:
        printflag = False

    if context.daily_flag == 0 or printflag:
        my_net_liquidation = context.account.net_liquidation + context.cash_adjustment

        if my_net_liquidation > context.max_liquid:
            context.max_liquid = my_net_liquidation
            if context.daily_print_flag == 0 or printflag:
                print(f"New equity high! Max liquidation: {context.max_liquid:.0f}")

        context.draw_down = (context.max_liquid - my_net_liquidation) / context.max_liquid
        if context.draw_down != 0 and (context.daily_print_flag == 0 or printflag):
            print(f"Current drawdown from high of {context.max_liquid:.0f}: {context.draw_down:.3%}")

        context.daily_print_flag = 1
        if context.cash_adjustment == 0:
            context.daily_flag = 1

    return


def initial_allocation(context, data):
    """Handle initial portfolio allocation."""
    spyprice_1 = context.price_history_spy100.iloc[-1]
    spyprice_2 = context.price_history_spy100.iloc[-2]
    
    if context.iwm_w == 0:
        context.iwm_w = -0.4

    # Respond to VIX signal changes
    if (context.vixflag_prev > 0 and context.vixflag <= 0 and
        get_datetime().date().weekday() != context.days_offset):

        if spyprice_1 < context.spyma80:
            if context.shortfact != 0:
                new_weight = min(context.iwm_w / (2 * context.shortfact), -0.4 * context.shortfact * context.bcfactor)
                order_target_percent(context.iwm_sid, new_weight)

                log_to_flightlog(f'VIX long exit for IWM - weight = {new_weight}', level='INFO')
                print(f"VIX long exit for IWM - weight = {new_weight}")

        if spyprice_1 >= context.spyma80:
            log_to_flightlog('VIX long exit - executing reallocation', level='INFO')
            regular_allocation(context, data)
            exit_positions(context, data)

    # Initialize portfolio
    if context.initialized == 0:
        context.initialized = 1
        regular_allocation(context, data)

    # Handle SPY crossing above 80-day MA
    if (spyprice_1 > context.spyma80 and
        spyprice_2 <= context.spyma80 and
        context.vix_uptrend_flag and
        context.spy_below80ma and
        get_datetime().date().weekday() != context.days_offset):

        if context.shortfact != 0:
            new_weight = min(context.iwm_w / 2, -0.4 * context.bcfactor)
            order_target_percent(context.iwm_sid, new_weight)

            print(f"SPY MA crossover - less short IWM - weight = {new_weight}")
            log_to_flightlog(f'SPY MA crossover - IWM weight = {new_weight}', level='INFO')
            context.spy_below80ma = False

    return


def regular_allocation(context, data):
    """Main portfolio allocation function."""
    longs = context.longs.index
    shorts = context.shorts.index

    compute_trend(context, data)
    context.initialized = 1

    # Drawdown protection
    context.dd_factor = min([context.clip, (1 + context.draw_down * DRAWDOWN_FACTOR_MULTIPLIER)])

    try:
        print(get_datetime(timezone("America/Los_Angeles")))
        print(f'Beta ratio: {context.beta_ratio:.4f}')
        print(f"Drawdown factor: {context.dd_factor:.4f}")
        print(f"Long factor: {context.longfact:.2f}")
        print(f"Net liquidation: {context.account.net_liquidation:.2f}")
    except:
        pass

    # Get normalized weights
    longs_mcw, shorts_mcw = get_normalized_weights(context, data, 'cash_return')

    # Calculate trend adjustments
    if context.vix_uptrend_flag:
        trend_longfact_multiplier = 0.625
        trend_spy_gt_ma21 = 1
    else:
        trend_longfact_multiplier = 1.625
        trend_spy_gt_ma21 = 1.3 if context.spyprice > context.spyma21 else 1

    if context.spyprice < context.spyma80:
        adjust_fact = 1.05
    else:
        print('SPY above 80-day MA: adjust_fact = 1.102')
        adjust_fact = 1.102

    # Apply adjustments to long weights
    longs_mcw['cash_return'] = (
        longs_mcw['cash_return'] *
        context.longfact *
        context.dd_factor *
        0.637 *
        adjust_fact *
        1.03
    )

    if context.vix_uptrend_flag:
        longs_mcw['cash_return'] = longs_mcw['cash_return'] * 1.6

    if context.spyprice < context.spyma150 and context.bc1 == 1:
        context.bcfactor = 0.65
        longs_mcw['cash_return'] = longs_mcw['cash_return'] * context.bcfactor
    else:
        context.bcfactor = 0.65

    port_weight_factor = 0.90
    spy_weight_factor = 1 - port_weight_factor

    if context.verbose == 1:
        print_positions(longs, longs_mcw, 'cash_return', port_weight_factor)

    # Execute long orders
    total_wl = 0
    for sid_val, w in zip(longs, longs_mcw['cash_return'].values):
        w = abs(w)
        order_target_percent(sid_val, w * port_weight_factor)
        total_wl = total_wl + w

    order_target_percent(context.spysym, total_wl * spy_weight_factor)

    print(f'Total SPY weight: {total_wl * spy_weight_factor:.4f}')
    print(f'Total port long weight: {total_wl * port_weight_factor:.4f}')
    print(f'Total long weight: {total_wl:.4f}')

    # Short weights
    shorts_mcw[shorts_mcw['cash_return'] > 0.15] = 0.15

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

    if context.vix_uptrend_flag and context.spyprice > context.spyma21:
        shorts_mcw['cash_return'] = shorts_mcw['cash_return'] * 1.4
    else:
        shorts_mcw['cash_return'] = shorts_mcw['cash_return'] * 1.4

    total_ws = 0
    for sid_val, w in zip(shorts, shorts_mcw['cash_return'].values):
        w = abs(w) * -1
        total_ws = total_ws + w

    context.total_ws = total_ws
    print(f'Total short weight: {total_ws:.4f}')

    context.spy_below80ma = context.spyprice < context.spyma80

    # Execute IWM short
    if (context.vix_uptrend_flag and context.spy_below80ma):
        iwm_w = min(-1 * 0.384 * total_wl, total_ws)
        place_short_orders(context, context.short_symbol_weights, iwm_w)
        print(f'SPY below MA80 - IWM weight {iwm_w:.4f}')
        context.iwm_w = iwm_w
    else:
        place_short_orders(context, context.short_symbol_weights, total_ws)
        print(f'IWM weight {total_ws:.4f}')
        context.iwm_w = total_ws

    if len(pd.Series(tuple(context.portfolio.positions.keys()))) > 52:
        print("WARNING: Too many positions")

    return


def exit_positions(context, data):
    """Exit positions no longer in portfolio."""
    desired_sids = set(context.longs.index)

    getting_the_boot = [
        sid_val for sid_val in context.portfolio.positions.keys()
        if sid_val not in desired_sids
        and sid_val != context.iwm_sid
        and sid_val != context.spysym
        and sid_val != context.dia_sid
        and sid_val != context.qqq_sid
        and sid_val != context.tlt_sid
    ]

    if context.verbose == 1 and getting_the_boot:
        print('Exiting positions not in longs')

    for sid_val in getting_the_boot:
        if context.verbose == 1:
            print('Exiting', sid_val)
        try:
            order_target(sid_val, 0)
        except Exception as e:
            print(f'Failed to exit {sid_val}:', e)

    return


#################################################
# UTILITY FUNCTIONS
#################################################

def initialize_sids(context, data):
    """Initialize security IDs for ETFs."""
    if context.sids_initialized == 0:
        print('Looking up security IDs')

        context.benchmarkSecurity = symbol('IWM')
        context.iwm_sid = symbol('IWM')
        context.dia_sid = symbol('DIA')
        context.ibm_sid = symbol('IBM')
        context.spysym = symbol('SPY')
        context.qqq_sid = symbol('QQQ')
        context.rwm_sid = symbol('RWM')
        context.tlt_sid = symbol('TLT')
        context.iwb_sid = symbol('IWB')

        # Define non-tradable symbols
        context.no_trade_sym = symbols([
            'OEF', 'QQQ', 'IWM', 'SPY', 'TLT',
            'MTUM', 'SPYG', 'QUAL', 'DIA', 'SPHB'
        ])

        # Initialize sector ETFs
        context.sector_etf = []
        context.sector_etf_dict = {}
        for sym in ['IYZ', 'XLF', 'XLE', 'XLK', 'XLB', 'XLY', 'XLI', 'XLV', 'XLP', 'XLU']:
            sid_var = symbol(sym)
            if sid_var:
                context.sector_etf.append(sid_var)
                context.sector_etf_dict.update({sym: sid_var})
        context.sector_etf = pd.Series(context.sector_etf)

        # Initialize index ETFs
        context.index_etf = []
        context.index_etf_dict = {}
        for sym in ['IWM', 'QQQ']:
            sid_var = symbol(sym)
            if sid_var:
                context.index_etf.append(sid_var)
                context.index_etf_dict.update({sym: sid_var})
        context.index_etf = pd.Series(context.index_etf)

        # Short symbol weights
        context.short_symbol_weights = {
            context.iwm_sid: 1,
        }

        context.sids_initialized = 1

    return


def filter_symbols_sectors_universe(df):
    """Filter universe by sectors and stocks."""
    current_date = get_datetime().date()
    current_year = current_date.year
    current_month = current_date.month

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

    for stock_name in to_drop:
        df = df[df['name'] != stock_name]

    df = df[df['sector'] != 'Financials']

    # Limit Energy and Real Estate
    df_energy = df[df['sector'] == 'Energy'].sort_values(by='market_cap', ascending=False)[:20]
    df_real_estate = df[df['sector'] == 'Real Estate'].sort_values(by='market_cap', ascending=False)[:15]

    df = pd.concat([
        df[~df['sector'].isin(['Energy', 'Real Estate'])],
        df_energy,
        df_real_estate
    ])

    df = df[df['publicdays'] > 0]

    return df


def get_normalized_weights(context, data, target):
    """Normalize position weights."""
    context.longs[target].fillna(value=0.001, inplace=True)
    context.shorts[target].fillna(value=0.001, inplace=True)

    df1 = context.longs.groupby('sector')[target].agg(['median', 'mean', 'count'])
    print('Mean of target', target)
    print(df1)

    # Normalize long weights
    longs_mcw = abs(context.longs[[target]]) / abs(context.longs[[target]]).sum()
    longs_mcw[longs_mcw[target] >= MAX_POSITION_SIZE_LONG] = MAX_POSITION_SIZE_LONG
    longs_mcw[longs_mcw[target] <= 0.002] = 0.02
    longs_mcw = abs(longs_mcw[[target]]) / abs(longs_mcw[[target]]).sum()
    longs_mcw[longs_mcw[target] >= MAX_POSITION_SIZE_LONG] = MAX_POSITION_SIZE_LONG
    longs_mcw[longs_mcw[target] <= 0.002] = 0.02
    longs_mcw = abs(longs_mcw[[target]]) / abs(longs_mcw[[target]]).sum()

    # Normalize short weights
    shorts_mcw = abs(context.shorts[[target]]) / abs(context.shorts[[target]]).sum()
    shorts_mcw[shorts_mcw[target] > MAX_POSITION_SIZE_SHORT] = MAX_POSITION_SIZE_SHORT
    shorts_mcw = abs(shorts_mcw[[target]]) / abs(shorts_mcw[[target]]).sum()

    return longs_mcw, shorts_mcw


def print_positions(port, port_w, target, factor=1):
    """Print current positions with weights."""
    print(get_datetime(timezone("America/Los_Angeles")))
    l = sorted(
        [list(c) for c in zip(port[0:], (port_w[target]*factor).round(6).astype(str).values[0:])],
        key=lambda x: x[1],
        reverse=True
    )
    print(pd.Series(l).values)
    return


def place_short_orders(context, symbol_weights, total_weight):
    """Place short orders."""
    sum_of_weights = sum(symbol_weights.values())

    for sym, weight in symbol_weights.items():
        target_percent = (weight / sum_of_weights) * total_weight
        print(f"Executing short target {sym}, {target_percent * context.shortfact:.4f}")
        order_target_percent(sym, target_percent * context.shortfact)

    return


def GenerateMomentumList(context, data, etf_list, momlength):
    """Generate list of ETFs ranked by momentum."""
    price_history = data.history(etf_list, 'price', momlength, '1d')
    pct_change = (price_history.iloc[-1] - price_history.iloc[0]) / price_history.iloc[0]
    momentum_df = pct_change.to_frame(name='momentum').reset_index()
    momentum_df = momentum_df.sort_values(by='momentum', ascending=False)
    top_momentum_list = momentum_df.head(context.topMom).values.tolist()
    return top_momentum_list


def RemoveSectors(context, etf, dfs, prt_str):
    """Remove stocks from specific sectors."""
    etf_sector_map = {
        context.sector_etf_dict.get('XLB'): 'Materials',
        context.sector_etf_dict.get('XLY'): 'Consumer Discretionary',
        context.sector_etf_dict.get('XLF'): 'Financials',
        context.sector_etf_dict.get('XLP'): 'Consumer Staples',
        context.sector_etf_dict.get('XLV'): 'Health Care',
        context.sector_etf_dict.get('XLU'): 'Utilities',
        context.sector_etf_dict.get('IYZ'): 'Communication Services',
        context.sector_etf_dict.get('XLE'): 'Energy',
        context.sector_etf_dict.get('XLI'): 'Industrials',
        context.sector_etf_dict.get('XLK'): 'Information Technology',
    }

    if etf in etf_sector_map:
        sector_to_remove = etf_sector_map[etf]
        dfs = dfs[dfs['sector'] != sector_to_remove]
        print(prt_str % etf)

    return dfs


def compute_trend(context, data):
    """Determine market trend based on VIX signal."""
    print('VIX flag:', context.vixflag)

    context.longfact_last = context.longfact

    if context.vixflag <= 0:
        log_to_flightlog(f'[DEBUG] Trend mode: AGGRESSIVE, vixflag={context.vixflag} <= 0, longfact=1.5', level='INFO')
        context.vix_uptrend_flag = True
        context.longfact = 1.5

        if get_datetime().date().month in SHORT_RESTRICTED_MONTHS:
            context.shortfact = 0.45
            log_to_flightlog(f'[DEBUG] Month {get_datetime().date().month} in SHORT_RESTRICTED, shortfact=0.45', level='INFO')
        else:
            context.shortfact = 0.9
            log_to_flightlog(f'[DEBUG] Month {get_datetime().date().month} NOT in SHORT_RESTRICTED, shortfact=0.9', level='INFO')
        context.clip = 1.2

    else:
        log_to_flightlog(f'[DEBUG] Trend mode: DEFENSIVE, vixflag={context.vixflag} > 0', level='INFO')
        context.vix_uptrend_flag = False
        context.longfact = 0.0 if context.spy_below80ma else abs(context.iwm_w)
        log_to_flightlog(f'[DEBUG] spy_below80ma={context.spy_below80ma}, longfact={context.longfact}', level='INFO')
        context.shortfact = 0.5
        context.clip = 1.6

    return


def compute_beta(context, data):
    """Compute beta ratio between portfolios."""
    benchmark_array = np.array([context.benchmarkSecurity])
    assets_array = np.concatenate((context.universe, benchmark_array))
    prices = data.history(assets_array, 'price', 120, '1d')

    prices_longs = prices[context.longs.index.intersection(prices.columns)]
    prices_shorts = prices[context.shorts.index.intersection(prices.columns)]
    prices_spy = prices[context.benchmarkSecurity]

    rets_long_port = prices_longs.pct_change().sum(axis=1)
    rets_short_port = prices_shorts.pct_change().sum(axis=1)
    rets_spy = prices_spy.pct_change()

    beta_span = 120

    long_cov = rets_long_port.ewm(span=beta_span, adjust=True).cov(rets_spy)
    short_cov = rets_short_port.ewm(span=beta_span, adjust=True).cov(rets_spy)
    bench_var = rets_spy.ewm(span=beta_span, adjust=True).var()

    long_beta = long_cov.iloc[-1] / bench_var.iloc[-1]
    short_beta = short_cov.iloc[-1] / bench_var.iloc[-1]

    beta_ratio = long_beta / short_beta
    print("long_beta, short_beta, beta_ratio:", long_beta, short_beta, beta_ratio)

    return beta_ratio


def weighted_moving_average(prices, period):
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    wma = np.convolve(prices, weights / weights.sum(), mode='valid')
    return wma


def hull_moving_average(prices, period):
    """Calculate Hull Moving Average."""
    wma_n = weighted_moving_average(prices, period)
    wma_half_n = weighted_moving_average(prices, period // 2)
    raw_hma = 2 * wma_half_n[-len(wma_n):] - wma_n
    sqrt_n = int(np.sqrt(period))
    hma = weighted_moving_average(raw_hma, sqrt_n)
    return hma


def hull_ma_trend(prices, period, lookback=3):
    """Determine Hull MA trend direction."""
    hma = hull_moving_average(prices, period)
    recent_hma = hma[-lookback:]
    trend = "positive" if recent_hma[-1] > recent_hma[0] else "negative"
    return trend


def compute_weekly_stochastic(df, lookback_weeks=14):
    """Compute weekly stochastic oscillator."""
    weekly_high = df['high'].resample('W-FRI').max()
    weekly_low = df['low'].resample('W-FRI').min()
    weekly_close = df['price'].resample('W-FRI').last()

    highest_high = weekly_high.rolling(lookback_weeks).max()
    lowest_low = weekly_low.rolling(lookback_weeks).min()

    stochastic = ((weekly_close.iloc[-1] - lowest_low.iloc[-1]) /
                  (highest_high.iloc[-1] - lowest_low.iloc[-1])) * 100

    return stochastic


def analyze(context, perf):
    """Analyze backtest results."""
    returns = perf['returns']
    total_return = (perf['portfolio_value'].iloc[-1] / perf['portfolio_value'].iloc[0] - 1) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    print("="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("="*80)

    log_to_flightlog(f'Backtest complete! Return: {total_return:.2f}%', level='INFO')

    return perf


#################################################
# RUN BACKTEST
#################################################

if __name__ == '__main__':
    # Backtest parameters
    START = pd.Timestamp('2023-01-01')
    END = pd.Timestamp('2024-11-01')
    CAPITAL = 1000000

    # Run backtest with multi-source auto loader
    results = run_algorithm(
        start=START,
        end=END,
        initialize=initialize,
        before_trading_start=before_trading_start,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=CAPITAL,
        bundle='sharadar',
        custom_loader=ms.setup_auto_loader(),  # Auto-detect custom databases
    )

    print("\n" + "="*80)
    print("Backtest complete!")
    print(f"Final portfolio value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print("="*80)
