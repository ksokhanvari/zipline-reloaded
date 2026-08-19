"""
Long-Short Equity Trading Algorithm
====================================

Overview
--------
This algorithm implements a systematic long-short equity strategy designed for
the US equity market. It combines fundamental analysis, technical indicators,
sentiment data, and machine learning predictions to construct a market-neutral
portfolio that aims to generate alpha in various market conditions.

Strategy Components
-------------------
1. **Long Portfolio Selection**:
   - Value stocks: Selected based on cash return (FCF - Interest) / Enterprise Value
   - Momentum stocks: Selected based on relative strength vs QQQ and price slope
   - Weighting: Cash return normalized by beta, adjusted by slope momentum

2. **Short Portfolio Selection**:
   - Stocks with lowest cash return from sectors not in top momentum
   - Hedged primarily through IWM (Russell 2000 ETF)

3. **Risk Management**:
   - Dynamic position sizing based on VIX regime signals
   - Drawdown protection with automatic exposure reduction
   - Beta-adjusted weighting to control market exposure
   - Sector rotation to avoid shorting momentum sectors

4. **Market Regime Detection**:
   - VIX-based trend indicator for bullish/bearish regime classification
   - SPY moving average crossovers for tactical adjustments
   - Barchart trend data for additional confirmation

Key Parameters
--------------
- Universe: Top 1500 stocks by market cap, filtered to 500
- Long positions: 50 stocks (30 value + 20 momentum)
- Short positions: 50 stocks (hedged via IWM)
- Rebalancing: Weekly (Tuesdays)
- Maximum single position: 6% (long), 2% (short)

Dependencies
------------
- Zipline/QuantRocket for backtesting and live trading
- Sharadar fundamentals database
- Custom databases: refe-fundamentals, vixdata, bcdata, sentiment data
- scikit-learn for ML factor

Author: Kamran Sokhanvari
Version: claud-noeps-cr-slope120-t-aug-sept-noesttemp-spy-.1-lw-.9-noshift-pricemc-mlzscore-MLC15-mlsent
Last Updated: Dec-2025
"""

import logging
from pytz import timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
from sklearn import impute, linear_model, preprocessing

import zipline.api as algo
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.data.db import Database, Column
from zipline.pipeline.factors import Returns, SimpleMovingAverage, SimpleBeta
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline import sharadar

from quantrocket.master import get_securities
from quantrocket.flightlog import FlightlogHandler


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# These parameters control the algorithm's behavior and can be tuned for
# optimization. Changes here affect portfolio construction, risk management,
# and trading execution.

# -----------------------------------------------------------------------------
# Portfolio Construction Parameters
# -----------------------------------------------------------------------------
UNIVERSE_SIZE = 1500
"""int: Initial universe size - top N stocks by market cap to consider.
Larger values increase computational cost but may find better opportunities."""

FILTERED_UNIVERSE_SIZE = 500
"""int: Final universe size after applying fundamental and liquidity filters.
This is the pool from which long and short candidates are selected."""

TOP_MOMENTUM_STOCKS = 30
"""int: Number of momentum-based stocks to include in the long portfolio.
These are selected based on relative strength and slope indicators."""

LONG_PORTFOLIO_SIZE = 50
"""int: Total number of long positions in the portfolio.
Split between value stocks (LONG_PORTFOLIO_SIZE - TOP_MOMENTUM_STOCKS) and momentum stocks."""

SHORT_PORTFOLIO_SIZE = 50
"""int: Number of short candidates identified for hedging calculations.
Actual shorting is done via IWM ETF rather than individual stocks."""

MAX_POSITION_SIZE_LONG = 0.06
"""float: Maximum weight for any single long position (6%).
Prevents concentration risk in individual names."""

MAX_POSITION_SIZE_SHORT = 0.02
"""float: Maximum weight for any single short position (2%).
More conservative than longs due to unlimited loss potential."""

# -----------------------------------------------------------------------------
# Risk Management Parameters
# -----------------------------------------------------------------------------
DRAWDOWN_FACTOR_MULTIPLIER = 5.0
"""float: Multiplier applied to drawdown for position sizing adjustment.
Higher values reduce exposure more aggressively during drawdowns.
Formula: dd_factor = min(clip, 1 + drawdown * DRAWDOWN_FACTOR_MULTIPLIER)"""

SLIPPAGE_SPREAD = 0.05
"""float: Fixed slippage assumption in dollars per share.
Used for realistic backtest cost modeling."""

COMMISSION_COST = 0.01
"""float: Per-share commission cost in dollars.
Combined with MIN_TRADE_COST for total transaction costs."""

MIN_TRADE_COST = 1.00
"""float: Minimum commission per trade in dollars.
Ensures small trades still incur realistic costs."""

# -----------------------------------------------------------------------------
# Seasonal and Regime Parameters
# -----------------------------------------------------------------------------
GROWTH_SEASON_MONTHS = {4, 5, 6, 7, 8, 9, 10, 11, 12}
"""set: Months when growth factor receives higher weight in ranking.
During these months, eps_gr_mean gets 4x weight vs 1x in other months."""

SHORT_RESTRICTED_MONTHS = {1, 2, 3, 5, 7, 8, 9, 10, 11, 12}
"""set: Months with reduced short exposure (shortfact = 0.45 vs 0.9).
Accounts for historical seasonal patterns in short selling effectiveness."""

# -----------------------------------------------------------------------------
# Machine Learning Configuration
# -----------------------------------------------------------------------------
ML_GLOBAL_COUNTER = 0
"""int: Global counter for ML model reuse tracking.
Resets to 0 after ML_MODEL_REUSE_LIMIT iterations."""

ML_MODEL_REUSE_LIMIT = 1
"""int: Number of days to reuse fitted ML model before refitting.
Value of 1 means refit daily; higher values reduce computation."""

ML_CLASSIFIER_GLOBAL = 0
"""object: Global storage for fitted ML classifier.
Allows model persistence across multiple factor computations."""

# -----------------------------------------------------------------------------
# Symbol Cache
# -----------------------------------------------------------------------------
SYM_SID_CACHE_DICT = {}
"""dict: Cache for symbol-to-SID lookups.
Improves performance by avoiding repeated database queries."""


# =============================================================================
# DATABASE DEFINITIONS
# =============================================================================
# These classes define connections to external data sources used by the
# algorithm. Each database provides specific fundamental, sentiment, or
# signal data that feeds into the stock selection pipeline.

class CustomFundamentals(Database):
    """
    Primary fundamentals database containing core company financial data.
    
    This database provides fundamental metrics from the 'refe-fundamentals'
    data source, including valuation ratios, earnings data, cash flow metrics,
    and proprietary alpha model rankings.
    
    Attributes
    ----------
    CODE : str
        Database identifier: 'refe-fundamentals'
    LOOKBACK_WINDOW : int
        Number of trading days of historical data to load (240 ~ 1 year)
    
    Columns
    -------
    Symbol : object
        Ticker symbol for the security
    CompanyCommonName : object
        Full company name
    GICSSectorName : object
        GICS sector classification (e.g., 'Technology', 'Healthcare')
    RefPriceClose : float
        Reference closing price from fundamental data source
    RefVolume : float
        Reference trading volume
    EnterpriseValue_DailyTimeSeries_ : float
        Enterprise value (market cap + debt - cash)
    CompanyMarketCap : float
        Market capitalization
    FOCFExDividends_Discrete : float
        Free operating cash flow excluding dividends
    InterestExpense_NetofCapitalizedInterest : float
        Net interest expense
    EarningsPerShare_ActualSurprise : float
        EPS surprise vs consensus estimates (prev quarter)
    LongTermGrowth_Mean : float
        Mean analyst long-term growth estimate
    CombinedAlphaModelSectorRank : float
        Proprietary sector-relative alpha ranking
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_ : float
        EV/EBITDA valuation multiple
    ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_ : float
        Forward EV/OCF ratio
    """
    CODE = "refe-fundamentals"
    LOOKBACK_WINDOW = 240
    
    Symbol = Column(object)
    CompanyCommonName = Column(object)
    GICSSectorName = Column(object)
    RefPriceClose = Column(float)
    RefVolume = Column(float)
    EnterpriseValue_DailyTimeSeries_ = Column(float)
    CompanyMarketCap = Column(float)
    FOCFExDividends_Discrete = Column(float)
    InterestExpense_NetofCapitalizedInterest = Column(float)
    EarningsPerShare_ActualSurprise = Column(float)
    LongTermGrowth_Mean = Column(float)
    CombinedAlphaModelSectorRank = Column(float)
    EnterpriseValueToEBITDA_DailyTimeSeriesRatio_ = Column(float)
    ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_ = Column(float)


class CustomFundamentals2(Database):
    """
    Sentiment data database containing news and social sentiment indicators.
    
    This database provides sentiment metrics derived from news articles,
    social media, and other text sources. Used to gauge market sentiment
    toward individual securities.
    
    Attributes
    ----------
    CODE : str
        Database identifier: 'refe-fundamentals-sent'
    LOOKBACK_WINDOW : int
        Historical lookback period (200 days)
    
    Columns
    -------
    sent2pol : float
        Sentiment polarity score (-1 to +1, negative to positive)
    sent2sub : float
        Sentiment subjectivity score (0 to 1, objective to subjective)
    sentvad_neg : float
        Negative sentiment intensity from VAD (Valence-Arousal-Dominance) model
    """
    CODE = "refe-fundamentals-sent"
    LOOKBACK_WINDOW = 200
    
    sent2pol = Column(float)
    sent2sub = Column(float)
    sentvad_neg = Column(float)


class CustomFundamentals4(Database):
    """
    VIX prediction/signal database for market regime detection.
    
    This database contains a proprietary VIX-based signal used to determine
    market regime (bullish vs bearish). The signal drives major allocation
    decisions including long/short exposure levels.
    
    Attributes
    ----------
    CODE : str
        Database identifier: 'vixdata'
    LOOKBACK_WINDOW : int
        Historical lookback period (200 days)
    
    Columns
    -------
    pred : float
        VIX regime prediction signal
        - pred <= 0: Bullish regime (increase long exposure)
        - pred > 0: Bearish regime (reduce long exposure, increase hedging)
    """
    CODE = "vixdata"
    LOOKBACK_WINDOW = 200
    pred = Column(float)


class CustomFundamentals9(Database):
    """
    Barchart trend data for additional market confirmation signals.
    
    This database provides trend-following signals from Barchart's
    technical analysis service, used as confirmation for position sizing.
    
    Attributes
    ----------
    CODE : str
        Database identifier: 'bcdata'
    LOOKBACK_WINDOW : int
        Historical lookback period (200 days)
    
    Columns
    -------
    bc1 : float
        Barchart trend signal
        - bc1 == 1: Trend confirmed (may reduce exposure if SPY < MA150)
        - bc1 != 1: No trend confirmation
    """
    CODE = "bcdata"
    LOOKBACK_WINDOW = 200
    bc1 = Column(float)

class CustomFundamentals10(Database):
    """ MLF1 data database """
    
    CODE = "refe-fundamentals-mlf1"
    LOOKBACK_WINDOW = 200
    
    predicted_return = Column(float)


# =============================================================================
# CUSTOM FACTORS
# =============================================================================
# These classes define computed factors used in the stock selection pipeline.
# Each factor transforms raw price/volume/fundamental data into signals
# that contribute to the final ranking and selection of securities.

class Above200DMA(CustomFactor):
    """
    Binary indicator for price position relative to 200-day moving average.
    
    This factor identifies stocks trading above their long-term trend,
    which is often associated with positive momentum and relative strength.
    
    Calculation
    -----------
    1 if current_price > 200-day SMA, else 0
    
    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    window_length : int
        200 trading days
    
    Returns
    -------
    float
        Binary indicator: 1.0 (above MA) or 0.0 (below MA)
    
    Usage
    -----
    Used as a quality filter and for identifying stocks in uptrends.
    """
    inputs = [USEquityPricing.close]
    window_length = 200
    
    def compute(self, today, assets, out, close):
        latest_close = close[-1]
        ma_200 = np.mean(close, axis=0)
        out[:] = (latest_close > ma_200).astype(int)


class StochasticOscillatorWeekly(CustomFactor):
    """
    20-week stochastic oscillator computed from daily data.
    
    The stochastic oscillator measures where the current price sits within
    the recent trading range. Weekly timeframe smooths out daily noise
    while capturing intermediate-term momentum.
    
    Calculation
    -----------
    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    
    Where Highest High and Lowest Low are computed over 20 weekly periods.
    
    Parameters
    ----------
    inputs : list
        [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length : int
        100 trading days (20 weeks * 5 days/week)
    
    Returns
    -------
    float
        Oscillator value between 0 and 100
        - > 80: Potentially overbought
        - < 20: Potentially oversold
    
    Notes
    -----
    Daily data is aggregated into weekly bars before calculation to match
    the intended weekly timeframe of the indicator.
    """
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    window_length = 20 * 5  # 20 weeks of daily data
    
    def compute(self, today, assets, out, high, low, close):
        # Reshape daily data into weekly bars
        high_weekly = high.reshape(-1, 5, high.shape[1]).max(axis=1)
        low_weekly = low.reshape(-1, 5, low.shape[1]).min(axis=1)
        close_weekly = close.reshape(-1, 5, close.shape[1])[:, -1]
        
        # Calculate stochastic
        highest_high = np.max(high_weekly, axis=0)
        lowest_low = np.min(low_weekly, axis=0)
        out[:] = ((close_weekly[-1] - lowest_low) / (highest_high - lowest_low)) * 100


class Slope(CustomFactor):
    """
    Linear regression slope of price data over the window period.
    
    Measures the trend direction and strength by fitting a linear regression
    to closing prices. Positive slope indicates uptrend, negative indicates
    downtrend. Magnitude indicates trend strength.
    
    Calculation
    -----------
    Fits OLS regression: price = alpha + beta * time
    Returns beta (slope coefficient)
    
    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    outputs : list
        ['slope', 'rsq'] - slope coefficient and R-squared (unused)
    window_length : int
        Configurable (commonly 30, 90, 120, 220 days)
    
    Returns
    -------
    slope : float
        Regression slope coefficient (price change per day)
    rsq : float
        R-squared of regression fit (currently set to 0, unused)
    
    Notes
    -----
    NaN values in price data are interpolated before regression to ensure
    stable computation. The slope is typically z-scored across the universe
    for cross-sectional comparison.
    """
    inputs = [USEquityPricing.close]
    outputs = ['slope', 'rsq']
    
    def compute(self, today, assets, out, closes):
        # Handle NaN values through interpolation
        try:
            mask = np.isnan(closes)
            closes[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), closes[~mask])
        except:
            pass
        
        # Fit linear regression
        lr = sm.OLS(closes, sm.add_constant(range(-len(closes) + 1, 1))).fit()
        out.slope[:] = lr.params[-1]
        out.rsq[:] = 0  # R-squared not currently used


class RelativeStrength(CustomFactor):
    """
    Relative strength of a security versus a benchmark index.
    
    Measures how much a stock has outperformed or underperformed a benchmark
    over the lookback period. Used to identify momentum leaders and laggards.
    
    Calculation
    -----------
    RS = ((1 + stock_return) / (1 + benchmark_return) - 1) * 100
    
    Where returns are calculated over the window period (close[-22] to close[0]).
    
    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    params : tuple
        (market_sid,) - SID of the benchmark security (e.g., QQQ)
    window_length : int
        Configurable (commonly 140, 160, 180 days)
    window_safe : bool
        True - factor can be used as input to other factors
    
    Returns
    -------
    float
        Relative strength percentage
        - Positive: Outperformed benchmark
        - Negative: Underperformed benchmark
    
    Example
    -------
    RS of 10 means the stock returned 10% more than the benchmark.
    RS of -5 means the stock returned 5% less than the benchmark.
    """
    params = ('market_sid',)
    inputs = [USEquityPricing.close]
    window_safe = True
    
    def compute(self, today, assets, out, close, market_sid):
        rsRankTable = pd.DataFrame(index=assets)
        
        # Calculate returns over approximately 1 month (22 trading days)
        returns = (close[-22] - close[0]) / close[0]
        
        # Find benchmark and compute relative performance
        market_idx = assets.get_loc(market_sid)
        rsRankTable["RS"] = (((returns + 1) / (returns[market_idx] + 1)) - 1) * 100
        
        out[:] = rsRankTable["RS"]


class Volatility(CustomFactor):
    """
    Historical volatility measured as standard deviation of daily returns.
    
    Provides a measure of price variability used for risk assessment
    and position sizing adjustments.
    
    Calculation
    -----------
    volatility = std(daily_returns) over window
    where daily_return = (close[t] - close[t-1]) / close[t-1]
    
    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    window_length : int
        Configurable (commonly 10, 20, 60 days)
    window_safe : bool
        True - factor can be used as input to other factors
    
    Returns
    -------
    float
        Standard deviation of daily returns (not annualized)
    
    Notes
    -----
    To annualize, multiply by sqrt(252). A value of 0.02 daily volatility
    corresponds to approximately 32% annualized volatility.
    """
    inputs = [USEquityPricing.close]
    window_safe = True
    
    def compute(self, today, assets, out, close_prices):
        daily_returns = np.diff(close_prices, axis=0) / close_prices[:-1]
        volatility = np.std(daily_returns, axis=0)
        out[:] = volatility


class PublicSince(CustomFactor):
    """
    Proxy for how long a security has been publicly traded.
    
    Uses the sum of early price data points as a heuristic for identifying
    established vs. newly listed securities. Securities with longer trading
    histories will have more non-zero early prices.
    
    Calculation
    -----------
    Sum of first 8 closing prices in the window
    
    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    window_length : int
        Configurable (commonly 121 days)
    
    Returns
    -------
    float
        Sum of early prices (higher = more established)
    
    Notes
    -----
    This is a simple heuristic. Newly listed stocks will have NaN or zero
    values for early dates, resulting in lower sums. Used to filter out
    very new listings that may have unreliable data.
    """
    inputs = [USEquityPricing.close]
    
    def compute(self, today, assets, out, prices):
        prices = np.nan_to_num(prices)
        out[:] = ((prices[0] + prices[1] + prices[2] + prices[3] +
                   prices[4] + prices[5] + prices[6] + prices[7]))


class SumFactor(CustomFactor):
    """
    Sums a factor's values over the window period.
    
    Generic utility factor for accumulating time-series data, commonly
    used with sentiment indicators to aggregate signals over time.
    
    Parameters
    ----------
    inputs : list
        Single factor to sum (e.g., sentiment score)
    window_length : int
        Number of days to sum over
    window_safe : bool
        True - factor can be used as input to other factors
    
    Returns
    -------
    float
        Sum of factor values over the window
    
    Example
    -------
    SumFactor(sentiment_score, window_length=18) gives the cumulative
    sentiment over the past 18 days.
    """
    window_safe = True
    
    def compute(self, today, assets, out, factordata):
        out[:] = np.sum(factordata, axis=0)


class SumVolume(CustomFactor):
    """
    Sums trading volume over the window period.
    
    Used to identify recent trading activity levels, which can indicate
    liquidity and investor interest.
    
    Parameters
    ----------
    inputs : list
        [USEquityPricing.volume]
    window_length : int
        Number of days to sum (commonly 3-5 days)
    
    Returns
    -------
    float
        Total shares traded over the window
    """
    inputs = [USEquityPricing.volume]
    
    def compute(self, today, assets, out, volume):
        out[:] = np.sum(volume, axis=0)


class WeightedAlpha(CustomFactor):
    """
    Weighted excess return (alpha) versus SPY benchmark.
    
    Combines short, medium, and long-term alpha into a single score,
    with heavier weight on medium-term performance. This balances
    recent momentum with longer-term trend strength.
    
    Calculation
    -----------
    weighted_alpha = 0.15 * alpha_30 + 0.50 * alpha_90 + 0.35 * alpha_252
    
    Where alpha_N = stock_return_N - SPY_return_N
    
    Parameters
    ----------
    inputs : list
        [USEquityPricing.close]
    window_length : int
        252 trading days (1 year)
    
    Returns
    -------
    float
        Weighted alpha score
        - Positive: Outperformed SPY on weighted basis
        - Negative: Underperformed SPY on weighted basis
    
    Notes
    -----
    The 50% weight on 90-day alpha emphasizes quarterly momentum,
    which often captures fundamental catalyst reactions while avoiding
    short-term noise and long-term mean reversion.
    """
    inputs = [USEquityPricing.close]
    window_length = 252
    
    def compute(self, today, assets, out, close):
        # Get SPY index
        spy_sid = symbol("SPY").sid
        spy_idx = assets.get_loc(spy_sid)
        spy_close = close[:, spy_idx]
        
        # Separate SPY from other assets
        is_asset = np.arange(len(assets)) != spy_idx
        asset_close = close[:, is_asset]
        
        # Calculate asset returns at different horizons
        ret_30 = asset_close[-1] / asset_close[-22] - 1   # ~1 month
        ret_90 = asset_close[-1] / asset_close[-66] - 1   # ~3 months
        ret_252 = asset_close[-1] / asset_close[0] - 1    # ~1 year
        
        # Calculate SPY returns
        spy_ret_30 = spy_close[-1] / spy_close[-22] - 1
        spy_ret_90 = spy_close[-1] / spy_close[-66] - 1
        spy_ret_252 = spy_close[-1] / spy_close[0] - 1
        
        # Calculate alpha (excess return) at each horizon
        alpha_30 = ret_30 - spy_ret_30
        alpha_90 = ret_90 - spy_ret_90
        alpha_252 = ret_252 - spy_ret_252
        
        # Weighted combination
        weighted_alpha = 0.15 * alpha_30 + 0.5 * alpha_90 + 0.35 * alpha_252
        
        # Map back to full asset list
        out_vals = np.empty(len(assets))
        out_vals[:] = np.nan
        out_vals[is_asset] = weighted_alpha
        out[:] = out_vals


class MLFactorC(CustomFactor):
    """
    Machine learning factor predicting future returns using Linear Regression.
    
    This factor trains a linear regression model on fundamental features
    to predict forward returns. The model is periodically retrained to
    adapt to changing market conditions.
    
    Model Architecture
    ------------------
    - Algorithm: Linear Regression (sklearn)
    - Features: Fundamental ratios and rankings
    - Target: Forward returns over shift_target days
    - Preprocessing: Imputation (constant=0) + Robust Scaling
    
    Parameters
    ----------
    inputs : list
        [Returns, Feature1, Feature2, ...] - First input is target returns
    params : tuple
        (shift_target,) - Number of days forward to predict
    window_length : int
        Training window size (commonly 180 days)
    window_safe : bool
        True - factor can be used as input to other factors
    
    Returns
    -------
    float
        Predicted return score (higher = more bullish prediction)
    
    Training Process
    ----------------
    1. Stack features into 3D array (time x stocks x features)
    2. Create forward return labels shifted by shift_target days
    3. Flatten and clean training data (remove NaN/Inf)
    4. Apply imputation and scaling
    5. Fit linear regression
    6. Predict on most recent data
    
    Notes
    -----
    The model is reused for ML_MODEL_REUSE_LIMIT days before refitting
    to reduce computational overhead while maintaining adaptability.
    """
    params = ('shift_target',)
    window_safe = True
    
    def compute(self, today, assets, out, target, *features, shift_target):
        global ML_GLOBAL_COUNTER, ML_MODEL_REUSE_LIMIT, ML_CLASSIFIER_GLOBAL
        
        # Initialize persistent state on first call
        if not hasattr(self, "fitted"):
            self.fitted = False
        if not hasattr(self, "reuse_counter"):
            self.reuse_counter = 0
        if not hasattr(self, "reuse_limit"):
            self.reuse_limit = 1
        if not hasattr(self, "imputer"):
            self.imputer = impute.SimpleImputer(strategy="constant", fill_value=0)
        if not hasattr(self, "scaler"):
            self.scaler = preprocessing.RobustScaler()
        if not hasattr(self, "model"):
            self.model = linear_model.LinearRegression()
        
        # Prepare data
        X = np.dstack(features)  # Shape: (time, stocks, features)
        Y = target                # Shape: (time, stocks)
        n_time, n_stocks, n_factors = X.shape
        
        # Guard: insufficient data for forward prediction
        if shift_target >= n_time:
            out[:] = 0
            return
        
        # Create training data with forward-shifted labels
        X_train = X[:-shift_target].reshape(-1, n_factors)
        Y_train = Y[shift_target:].reshape(-1)
        
        # Clean training data
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train)
        X_train = X_train[mask]
        Y_train = Y_train[mask]
        
        # Guard: insufficient clean samples
        if len(X_train) < 5:
            out[:] = 0
            return
        
        # Fit model if needed (based on reuse counter)
        if not self.fitted or self.reuse_counter == 0:
            Xt = self.imputer.fit_transform(X_train)
            Xt = self.scaler.fit_transform(Xt)
            self.model.fit(Xt, Y_train)
            self.fitted = True
            self.reuse_counter = self.reuse_limit
        
        # Generate predictions for current day
        X_test = X[-1]  # Most recent data
        X_test = self.imputer.transform(X_test)
        X_test = self.scaler.transform(X_test)
        preds = self.model.predict(X_test)
        
        out[:] = preds
        self.reuse_counter -= 1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# Helper functions for symbol lookup, technical indicators, and data
# transformation. These provide reusable functionality across the algorithm.

def symbol(sym):
    """
    Get security ID (SID) for a ticker symbol with caching.
    
    Performs a lookup in the QuantRocket securities master database and
    caches results to avoid repeated database queries.
    
    Parameters
    ----------
    sym : str
        Ticker symbol (e.g., 'SPY', 'AAPL')
    
    Returns
    -------
    zipline.assets.Equity or None
        Security object if found, None if lookup fails
    
    Notes
    -----
    Results are cached in SYM_SID_CACHE_DICT for performance.
    First lookup for a symbol queries the database; subsequent
    lookups return cached value.
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
    Get security IDs for multiple ticker symbols.
    
    Parameters
    ----------
    syms : list of str
        List of ticker symbols
    
    Returns
    -------
    list
        List of SID values (may contain None for failed lookups)
    """
    securities = get_securities(vendors="usstock", fields=["Sid", "Symbol"])
    securities = securities.reset_index()
    sidlist = []
    for sym in syms:
        sidlist.append(securities[securities.Symbol == sym].Sid)
    return sidlist


def weighted_moving_average(prices, period):
    """
    Calculate Weighted Moving Average (WMA).
    
    WMA gives more weight to recent prices, with weights increasing
    linearly from 1 to period.
    
    Parameters
    ----------
    prices : array-like
        Price series
    period : int
        Lookback period
    
    Returns
    -------
    numpy.ndarray
        WMA values (length = len(prices) - period + 1)
    
    Formula
    -------
    WMA = sum(price[i] * weight[i]) / sum(weights)
    where weight[i] = i + 1 for i in range(period)
    """
    weights = np.arange(1, period + 1)
    wma = np.convolve(prices, weights / weights.sum(), mode='valid')
    return wma


def hull_moving_average(prices, period):
    """
    Calculate Hull Moving Average (HMA).
    
    HMA reduces lag while maintaining smoothness by using weighted
    moving averages of different periods combined in a specific way.
    
    Parameters
    ----------
    prices : array-like
        Price series
    period : int
        Base lookback period
    
    Returns
    -------
    numpy.ndarray
        HMA values
    
    Formula
    -------
    HMA = WMA(2 * WMA(price, period/2) - WMA(price, period), sqrt(period))
    
    Notes
    -----
    Developed by Alan Hull. More responsive than SMA/EMA while
    filtering out more noise. Good for trend identification.
    """
    wma_n = weighted_moving_average(prices, period)
    wma_half_n = weighted_moving_average(prices, period // 2)
    raw_hma = 2 * wma_half_n[-len(wma_n):] - wma_n
    sqrt_n = int(np.sqrt(period))
    hma = weighted_moving_average(raw_hma, sqrt_n)
    return hma


def hull_ma_trend(prices, period, lookback=3):
    """
    Determine trend direction based on Hull Moving Average.
    
    Parameters
    ----------
    prices : array-like
        Price series
    period : int
        HMA period
    lookback : int, optional
        Number of HMA values to compare (default: 3)
    
    Returns
    -------
    str
        'positive' if HMA is rising, 'negative' if falling
    """
    hma = hull_moving_average(prices, period)
    recent_hma = hma[-lookback:]
    trend = "positive" if recent_hma[-1] > recent_hma[0] else "negative"
    return trend


def compute_weekly_stochastic(df, lookback_weeks=14):
    """
    Compute weekly stochastic oscillator from daily OHLC data.
    
    Resamples daily data to weekly frequency before calculating
    the stochastic oscillator.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'high', 'low', 'price' columns and DatetimeIndex
    lookback_weeks : int, optional
        Number of weeks for stochastic calculation (default: 14)
    
    Returns
    -------
    float
        Stochastic oscillator value (0-100)
    
    Notes
    -----
    Weekly bars are created ending on Fridays. The stochastic
    measures where the current close is within the N-week range.
    """
    # Resample to weekly frequency
    weekly_high = df['high'].resample('W-FRI').max()
    weekly_low = df['low'].resample('W-FRI').min()
    weekly_close = df['price'].resample('W-FRI').last()
    
    # Calculate rolling high/low
    highest_high = weekly_high.rolling(lookback_weeks).max()
    lowest_low = weekly_low.rolling(lookback_weeks).min()
    
    # Compute stochastic
    stochastic = ((weekly_close.iloc[-1] - lowest_low.iloc[-1]) /
                  (highest_high.iloc[-1] - lowest_low.iloc[-1])) * 100
    return stochastic


# =============================================================================
# MAIN ALGORITHM FUNCTIONS
# =============================================================================
# Core algorithm lifecycle functions called by the Zipline framework.
# These implement the trading logic from initialization through execution.

def initialize(context):
    """
    Initialize the trading algorithm.
    
    Called once at the start of the algorithm. Sets up the pipeline,
    benchmark, trading costs, scheduled functions, and initializes
    all context variables used throughout the algorithm.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context object for storing state
    
    Context Variables Initialized
    -----------------------------
    Portfolio State:
        longfact, shortfact : float
            Exposure multipliers for long/short positions
        max_liquid : float
            High-water mark for drawdown calculation
        dd_factor : float
            Current drawdown adjustment factor
        draw_down : float
            Current drawdown from high-water mark
    
    Trading Control:
        days_offset : int
            Day of week for weekly rebalancing (1 = Tuesday)
        initialized : int
            Flag indicating if first allocation completed
        verbose : int
            Logging verbosity level
    
    Market State:
        vixflag, vixflag_prev : float
            Current and previous VIX regime signals
        spy_below80ma : bool
            Flag for SPY below 80-day moving average
        iwm_w : float
            Current IWM hedge weight
    
    Scheduled Functions
    -------------------
    - initial_allocation: Daily, handles VIX signal changes
    - regular_allocation: Weekly (Tuesday), main rebalancing
    - exit_positions: Weekly (Tuesday), close unwanted positions
    """
    # Attach stock selection pipeline
    algo.attach_pipeline(make_pipeline(), 'my_pipeline')
    
    # Set benchmark for performance comparison
    algo.set_benchmark(algo.sid(symbol('SPY').real_sid))
    
    # Initialize portfolio state variables
    context.longfact = 1.0           # Long exposure multiplier
    context.shortfact = 1.0          # Short exposure multiplier
    context.order_id = {}            # Track pending orders
    context.topMom = 9               # Number of top momentum ETFs to track
    context.max_liquid = context.portfolio.starting_cash  # High-water mark
    context.cash_adjustment = 0      # External cash flow adjustment
    context.dd_factor = 1.0          # Drawdown adjustment factor
    context.draw_down = 0.0          # Current drawdown
    context.print_set_delta = False  # Debug flag
    context.days_offset = 1          # Rebalance day (1 = Tuesday)
    context.initialized = 0          # First allocation flag
    context.sids_initialized = 0     # SID lookup flag
    context.verbose = 1              # Logging verbosity
    context.qqq_ratio_prev = 0       # Previous QQQ allocation
    context.spy_ratio_prev = 0       # Previous SPY allocation
    context.total_ws = 0             # Total short weight
    context.iwm_w = 0                # IWM hedge weight
    context.spy_below80ma = False    # SPY below MA flag
    context.vixflag = 0              # Current VIX signal
    context.vixflag_prev = 0         # Previous VIX signal
    context.clip = 1.0               # Max DD factor clip
    
    # Set realistic trading costs
    algo.set_slippage(algo.slippage.FixedSlippage(spread=SLIPPAGE_SPREAD))
    algo.set_commission(algo.commission.PerShare(cost=COMMISSION_COST, min_trade_cost=MIN_TRADE_COST))
    
    # Schedule trading functions
    algo.schedule_function(initial_allocation, date_rule=algo.date_rules.every_day())
    algo.schedule_function(regular_allocation, date_rule=algo.date_rules.week_start(days_offset=context.days_offset))
    algo.schedule_function(exit_positions, date_rule=algo.date_rules.week_start(days_offset=context.days_offset))


def make_pipeline():
    """
    Create the stock selection pipeline.
    
    Constructs a Zipline Pipeline that computes all factors needed for
    stock selection. The pipeline filters the universe, calculates
    fundamental and technical factors, and prepares data for ranking.
    
    Returns
    -------
    zipline.pipeline.Pipeline
        Configured pipeline with screen and factor columns
    
    Pipeline Structure
    ------------------
    Screen:
        Top 1500 stocks by market cap OR IBM (for VIX signal lookup)
    
    Factor Categories:
        - Identification: symbol, company name, sector
        - Valuation: market cap, enterprise value
        - Price/Volume: close, volume, moving averages
        - Fundamentals: FCF, interest expense, EPS surprise, growth
        - Risk: beta to SPY and IWM
        - Technical: slopes at various periods, stochastic, 200DMA
        - Momentum: relative strength vs QQQ, returns
        - Quality: public trading history, volatility
        - Signals: VIX flag, Barchart trend
        - ML: machine learning return prediction
        - Sentiment: combined sentiment score
    """
    # Define tradable universe
    tradable_filter = (
        CustomFundamentals.CompanyMarketCap.latest.shift().top(UNIVERSE_SIZE) |
        StaticAssets([symbol('IBM')])  # IBM used as proxy for signal lookup
    )
    
    # Get Sharadar fundamentals for FCF
    s_fundamentals = sharadar.Fundamentals.slice('ARQ', period_offset=0)
    
    pipe = Pipeline(
        screen=tradable_filter,
        columns={
            # === Identification ===
            'name': CustomFundamentals.Symbol.latest,
            'compname': CustomFundamentals.CompanyCommonName.latest,
            'sector': CustomFundamentals.GICSSectorName.latest,
            
            # === Valuation Metrics ===
            'market_cap': CustomFundamentals.CompanyMarketCap.latest,
            'entval': CustomFundamentals.EnterpriseValue_DailyTimeSeries_.latest,
            
            # === Price and Volume ===
            'price': USEquityPricing.close.latest,
            'volume': USEquityPricing.volume.latest,
            'fs_price': CustomFundamentals.RefPriceClose.latest,
            'fs_volume': CustomFundamentals.RefVolume.latest,
            'sumvolume': SumVolume(window_length=3),
            'smav': SimpleMovingAverage(inputs=[USEquityPricing.volume], window_length=10),
            
            # === Fundamental Metrics ===
            'eps_ActualSurprise_prev_Q_percent': CustomFundamentals.EarningsPerShare_ActualSurprise.latest.shift(),
            'eps_gr_mean': CustomFundamentals.LongTermGrowth_Mean.latest.shift(),
            'fcf_sharadar': s_fundamentals.FCF.latest,
            'fcf' :CustomFundamentals.FOCFExDividends_Discrete.latest,
            'int': CustomFundamentals.InterestExpense_NetofCapitalizedInterest.latest.shift(),
            
            # === Risk Metrics ===
            'beta60SPY': SimpleBeta(target=symbol('SPY'), regression_length=60).shift(),
            'beta60IWM': SimpleBeta(target=symbol('IWM'), regression_length=60).shift(),
            
            # === Technical Indicators ===
            'slope120': Slope(window_length=120, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            #'slope220': Slope(window_length=220, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            'slope90': Slope(window_length=90, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            #'slope30': Slope(window_length=30, mask=tradable_filter).slope.zscore(mask=tradable_filter).shift(),
            'stk20w': StochasticOscillatorWeekly(),
            'above_200dma': Above200DMA(mask=tradable_filter),
            #'walpha': WeightedAlpha(),
            
            # === Relative Strength ===
            'RS140_QQQ': RelativeStrength(window_length=140, market_sid=symbol('QQQ').sid).shift(),
            'RS160_QQQ': RelativeStrength(window_length=160, market_sid=symbol('QQQ').sid).shift(),
            'RS180_QQQ': RelativeStrength(window_length=180, market_sid=symbol('QQQ').sid).shift(),
            
            # === Return Metrics ===
            'Ret60': Returns(window_length=60, mask=tradable_filter),
            'Ret120': Returns(window_length=120, mask=tradable_filter),
            'Ret220': Returns(window_length=220, mask=tradable_filter),
            
            # === Quality Filters ===
            'publicdays': PublicSince(window_length=121),
            'vol': Volatility(window_length=10, mask=tradable_filter),
            
            # === Regime Signals ===
            'vixflag': CustomFundamentals4.pred.latest.shift(),
            'vixflag0': CustomFundamentals4.pred.latest,
            'bc1': CustomFundamentals9.bc1.latest,
            
            # === Machine Learning Factor ===
            # 'MLfactor': MLFactorC(
            #     inputs=[
            #         Returns(window_length=90, mask=tradable_filter),
            #         CustomFundamentals.EnterpriseValueToEBITDA_DailyTimeSeriesRatio_,
            #         CustomFundamentals.LongTermGrowth_Mean,
            #         CustomFundamentals.CombinedAlphaModelSectorRank,
            #         CustomFundamentals.ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_,
            #     ],
            #     window_length=180,
            #     mask=tradable_filter,
            #     shift_target=15
            # ).zscore(mask=tradable_filter).shift(),
            
            # === Sentiment Factors ===
            'sentcomb': (
                SumFactor(CustomFundamentals2.sentvad_neg, window_length=18).zscore() +
                SumFactor(CustomFundamentals2.sent2sub, window_length=18).zscore() +
                (1 / SumFactor(CustomFundamentals2.sent2pol, window_length=18).zscore())
            ),
            'sentest': 1 / SumFactor(CustomFundamentals2.sent2pol, window_length=18),

             'mlf1': CustomFundamentals10.predicted_return.latest,
        }
    )
    return pipe


def before_trading_start(context, data):
    """
    Daily preprocessing before market open.
    
    Called each trading day before market open. Retrieves pipeline output,
    updates market indicators, processes signals, and prepares the universe
    of securities for potential trading.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    
    Processing Steps
    ----------------
    1. Initialize security IDs (first run only)
    2. Get pipeline output
    3. Update market indicators (moving averages, etc.)
    4. Update VIX and Barchart signals
    5. Compute IWM weekly stochastic
    6. Determine market trend/regime
    7. Process universe and select long/short candidates
    """
    # Initialize SIDs on first call
    initialize_sids(context, data, algo)
    
    # Get pipeline output
    df = algo.pipeline_output('my_pipeline')
    
    print(f"Raw stock universe size {df.shape}")
    print(algo.get_datetime(timezone("America/Los_Angeles")))
    
    # Update moving averages and other market indicators
    update_market_indicators(context, data)
    
    # Update VIX signal (store previous for change detection)
    context.vixflag_prev = context.vixflag
    context.vixflag = df.loc[context.ibm_sid].vixflag.copy()
    context.vixflag0 = df.loc[context.ibm_sid].vixflag0.copy()
    print('IBM-vixdata', context.vixflag)
    
    # Update Barchart trend signal
    context.bc1 = df.loc[context.ibm_sid].bc1.copy()
    
    # Calculate IWM weekly stochastic for regime analysis
    # df_iwm = data.history(symbol('IWM'), ['high', 'low', 'price'], 20 * 5, '1d')
    # context.iwm_stk20w = compute_weekly_stochastic(df_iwm, lookback_weeks=20)
    # print('IWM_stk20w', context.iwm_stk20w)
    
    # Determine market trend based on signals
    compute_trend(context, data)
    
    # Reset daily flags
    context.daily_flag = 0
    context.daily_print_flag = 0
    
    # Process universe and select positions
    df = process_universe(context, df, data)
    return


def process_universe(context, df, data):
    """
    Process the stock universe and select long/short candidates.
    
    Applies filters, calculates alpha scores, and selects the final
    securities for the long and short portfolios.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    df : pandas.DataFrame
        Pipeline output with all factor data
    data : zipline.protocol.BarData
        Market data accessor
    
    Returns
    -------
    pandas.DataFrame
        Filtered and ranked universe
    
    Processing Steps
    ----------------
    1. Filter by sector and exclusions
    2. Calculate dollar volume and cash return
    3. Calculate momentum ranking (myrs)
    4. Filter to top 500 by market cap
    5. Calculate final ranking (estrank) based on market regime
    6. Select long portfolio (value + momentum)
    7. Select short portfolio (lowest cash return)
    8. Calculate portfolio beta ratio
    """
    # Apply sector and stock exclusions from the universe of our stocks
    df = filter_symbols_sectors_universe(df)
    
    # Calculate dollar volume for liquidity filtering
    df['doll_vol'] = df['price'] * df['smav']
    
    # Calculate cash return: (FCF - Interest) / Enterprise Value
    df['cash_return'] = (df['fcf'] - df['int']) / df['entval']
    df['cash_return'] = df['cash_return'].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=['cash_return'], inplace=True)
    

    # “Z-scored softplus scaling with cross-sectional normalization” applied to cash_return as key alpha and later used in the weight model 
    # A convex, distribution-aware signal-to-weight transform
    cr = df['cash_return']
    z = (cr - cr.mean()) / (cr.std() + 1e-8)                # cross-sectional z-score
    k = 1.0                                                 # do NOT exceed 1.5 with this tail behavior
    alpha = np.log1p(np.exp(k * z))                         # softplus (convex) transform -- 
    alpha = np.minimum(alpha, np.percentile(alpha, 99.5))   # loose tail cap
    alpha /= alpha.mean()                                   # cross-sectional normalization

    df['cash_return_zsoft'] = alpha  # assign computed normalized cash_return back to the column as key alpha

    # Display top sectors by cash return
    sorted_df = df.groupby('sector')['cash_return_zsoft'].agg(['mean']).sort_values(by='mean', ascending=False)
    print(sorted_df.iloc[0].name)
    
    # Calculate momentum ranking
    df['myrs'] =  df.mlf1.rank() #df.slope120.rank() + df.RS140_QQQ.rank() #+ df.mlf1.rank()
    
    # Display top stocks by market cap with key metrics
    print("cash ret, myrs --->>>>>> ", 
          df.sort_values(by='market_cap', ascending=False)[0:10][['cash_return_zsoft', 'cash_return', 'slope90', 'mlf1']], '\n')
    print("MLF1, --->>>>>> ", 
          df.nlargest(150, 'market_cap') \
            .sort_values('mlf1', ascending=False) \
            .head(10)[['market_cap', 'mlf1']])


    # Filter to top stocks by market cap
    df = df.sort_values(by=['market_cap'], ascending=[False])[0:FILTERED_UNIVERSE_SIZE].copy()
    df['estrank'] = df.mlf1.rank()
    print('spy below spyma80 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
              'spyprice:', context.spyprice, 'spyma80:', context.spyma80)

    # Calculate ranking based on market regime
    # if context.spyprice <= context.spyma80:
    #     # Defensive mode: focus on liquidity and momentum
    #     df['estrank'] = df.mlf1.rank() #+ df['eps_ActualSurprise_prev_Q_percent'].rank()#+#df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1)
    #     print('switch estrank to doll_vol spy below spyma80 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',
    #           'spyprice:', context.spyprice, 'spyma80:', context.spyma80)
    # else:
    #     # Normal mode: multi-factor ranking
    #     context.season = 1 if algo.get_datetime().date().month in GROWTH_SEASON_MONTHS else 0
        
    #     df['estrank'] = (
    #         # df['MLfactor'].rank() +
    #         (df.mlf1.rank()) +
    #         (df['entval'].rank() * 2) +
    #         (df['cash_return_zsoft'].rank()) + 
    #         df['eps_gr_mean'].rank() * (4 if context.season == 1 else 1) +
    #         (df[['doll_vol', 'slope90', 'eps_ActualSurprise_prev_Q_percent']].rank().sum(axis=1) / 3) 
    #     )
    
    print(f"Filtered stock universe size {df.shape}")
    
    # Select portfolios
    select_long_portfolio(context, df, data)
    select_short_portfolio(context, df, data)
    context.topmcap = df.sort_values(by=['market_cap'], ascending=[False])[0:7].copy()
    
    # Combine for universe tracking
    context.universe = np.union1d(context.longs.index.values, context.shorts.index.values)
    
    # Calculate portfolio beta ratio
    context.beta_ratio = max(1.3, compute_beta(context, data))
    print(f'Beta ratio: {context.beta_ratio:.4f}')
    
    return df


def select_long_portfolio(context, df, data):
    """
    Select securities for the long portfolio.
    
    Combines value and momentum selection strategies to build a diversified
    long portfolio. Weights are adjusted by beta and slope factors.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    df : pandas.DataFrame
        Filtered universe with rankings
    data : zipline.protocol.BarData
        Market data accessor
    
    Selection Process
    -----------------
    1. Value Selection (30 stocks):
       - From top 150 by estrank, select top 30 by cash_return_zsoft
    
    2. Momentum Selection (20 stocks):
       - If bullish: From top 500 by RS140_QQQ, select top 20 by myrs
       - If bearish: Additional value stocks by cash_return_zsoft
    
    3. Combine and adjust:
       - Merge value and momentum selections
       - Sort by cash_return_zsoft
       - Adjust by beta (IWM or SPY depending on regime)
       - Apply slope factor boost (slope^2 adjustment)
    
    Result
    ------
    Sets context.longs with selected securities and adjusted cash_return_zsoft weights
    """
    dfl = df.copy()
    print('drop top two volatility outliers')
    dfl = drop_top_vol_outliers(dfl) # outlier volatility top 2 stocks removal function to help with random MEME stocks
   
    
    num_momentum_stocks = TOP_MOMENTUM_STOCKS
    num_value_stocks = LONG_PORTFOLIO_SIZE - num_momentum_stocks
    
    # Select value stocks: high cash return from top-ranked stocks
    context.longs_c = (
        dfl.sort_values(by=['estrank'], ascending=[False])[0:150] # this is the same as mlf1 as assigned 
           .sort_values(by=['cash_return_zsoft'], ascending=[False])[0:num_value_stocks]
           .copy()
    )
   
    # Select momentum stocks based on market trend
    if context.vix_uptrend_flag:
        context.longs_m = (
            dfl.sort_values(by=['mlf1'], ascending=[False])[0:500]
               .sort_values(by=['myrs'], ascending=[False])[0:num_momentum_stocks] # this is the same as mlf1 as assigned 
               .copy()
        )
    else:
        # In downtrends, select more value-oriented stocks
        context.longs_m = (
            dfl.sort_values(by=['estrank'], ascending=[False])[0:150]
               .sort_values(by=['cash_return_zsoft'], ascending=[False])[num_value_stocks:LONG_PORTFOLIO_SIZE]
               .copy()
        )
    
    # Combine value and momentum selections
    c_set = set(context.longs_c.index)
    m_set = set(context.longs_m.index)
    context.longs = dfl[dfl.index.isin(c_set.union(m_set))].copy()
    print(f'Long portfolio size: {len(context.longs)}')
    print(context.longs.sort_values(by=['vol'], ascending=[False])['vol'].head(10))
    
    # Sort by cash return
    context.longs = context.longs.sort_values(by=['cash_return_zsoft'], ascending=[False]).copy()
    
    # Adjust by beta based on market regime
    if context.spyprice >= context.spyma80:
        # Normal: normalize by IWM beta
        context.longs['cash_return_zsoft'] /= winsorize(context.longs['beta60IWM'], limits=[0.005, 0.4])
    else:
        # Defensive: normalize by SPY beta
        context.longs['cash_return_zsoft'] /= winsorize(context.longs['beta60SPY'], limits=[0.005, 0.1])
    
    # Ensure positive weights
    context.longs['cash_return_zsoft'] = context.longs['cash_return_zsoft'].clip(lower=0.005)
    
    ## proportional weighting based on mlf1 strength:
    mlf1 = winsorize(context.longs['mlf1'], limits=[0.1, 0.05])

    mlf1_adjusted = np.where(mlf1 > 0 , mlf1 ** 1.8 , mlf1)
    context.longs['cash_return_zsoft'] = context.longs['cash_return_zsoft'] * mlf1_adjusted  

    # Final sorting of long portfolio
    context.longs = context.longs.sort_values(by=['cash_return_zsoft'], ascending=[False])
    
    print(f'Long portfolio calculated with total cash_return_zsoft {context.longs["cash_return_zsoft"].sum()}')
    
    return


def select_short_portfolio(context, df, data):
    """
    Select securities for the short portfolio.
    
    Identifies weak stocks to inform hedging decisions. Avoids shorting
    stocks in the top momentum sector to prevent fighting strong trends.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    df : pandas.DataFrame
        Filtered universe with rankings
    data : zipline.protocol.BarData
        Market data accessor
    
    Selection Process
    -----------------
    1. Calculate sector momentum using ETF returns
    2. Identify top momentum sector (to exclude from shorts)
    3. Remove top momentum sector from candidates
    4. Select stocks with lowest cash return
    
    Result
    ------
    Sets context.shorts with selected securities
    Note: Actual shorting is done via IWM, not individual stocks
    """
    dfs = df.copy()
    
    # Get sector momentum rankings
    mom_list = GenerateMomentumList(context, data, context.sector_etf, 242)
    mom_list = [item[0] for item in mom_list]
    
    # Identify top sector to avoid shorting
    top_momentum_sector = mom_list[0]
    context.mometf = top_momentum_sector
    
    # Remove top momentum sector
    dfs = RemoveSectors(context, top_momentum_sector, dfs, "not shorting! %s")
    print('Bottom momentum sector:', mom_list[-1])
    
    # Select shorts with lowest cash return
    context.shorts = (dfs.sort_values(by=['cash_return_zsoft'], ascending=[True])[0:SHORT_PORTFOLIO_SIZE]
                      .copy())
    return


def update_market_indicators(context, data):
    """
    Update market-level technical indicators.
    
    Calculates moving averages and trend indicators for SPY and IWM
    used in regime detection and position sizing.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    
    Indicators Updated
    ------------------
    SPY:
        - spyprice: Current price
        - spyma21/50/80/85/150/200: Simple moving averages
    
    IWM:
        - iwmprice: Current price
        - iwmma50/10: Hull moving averages
        - hulltrend: HMA trend direction
    
    Price History:
        - price_history_spy100: 200-day SPY prices (for signal detection)
        - price_history_iwm250: 250-day IWM prices (for HMA calculation)
    """
    # SPY indicators
    context.price_history_spy100 = data.history(symbol('SPY'), 'price', 200, '1d')
    context.spyprice = context.price_history_spy100.values[-1]
    context.spyma21 = np.mean(context.price_history_spy100.tail(21).values)
    context.spyma50 = np.mean(context.price_history_spy100.tail(50).values)
    context.spyma80 = np.mean(context.price_history_spy100.tail(80).values)
    context.spyma85 = np.mean(context.price_history_spy100.tail(85).values)
    context.spyma150 = np.mean(context.price_history_spy100.tail(150).values)
    context.spyma200 = np.mean(context.price_history_spy100.tail(200).values)

    # Update SPY MA flag
    context.spy_below80ma = context.spyprice < context.spyma80
    context.spy_below150ma = context.spyprice < context.spyma150
    
    # IWM indicators with Hull MA
    context.price_history_iwm250 = data.history(symbol('IWM'), 'price', 250, '1d')
    context.iwmprice = context.price_history_iwm250.values[-1]
    context.iwmma50 = hull_moving_average(context.price_history_iwm250.values, 50)[-1]
    context.iwmma10 = hull_moving_average(context.price_history_iwm250.values, 10)[-1]
    context.hulltrend = hull_ma_trend(context.price_history_iwm250.values, 80, lookback=7)


def handle_data(context, data):
    """
    Intraday monitoring and metrics recording.
    
    Called on each trading bar. Monitors account metrics, tracks
    drawdown, and records performance statistics.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    
    Metrics Tracked
    ---------------
    - Net liquidation value (with cash adjustments)
    - High-water mark for drawdown calculation
    - Current drawdown percentage
    - Account leverage
    """
    time_minute = algo.get_datetime(timezone("America/Los_Angeles")).minute
    time_hour = algo.get_datetime(timezone("America/Los_Angeles")).hour
    
    context.account_leverage = 2.0
    
    # End-of-day reporting flag
    printflag = (time_hour == 12 and time_minute == 59)
    
    if context.daily_flag == 0 or printflag:
        # Calculate net liquidation with adjustments
        my_net_liquidation = context.account.net_liquidation + context.cash_adjustment
        
        # Update high-water mark
        if my_net_liquidation > context.max_liquid:
            context.max_liquid = my_net_liquidation
            if context.daily_print_flag == 0 or printflag:
                print(f"New equity high! Max liquidation for the trading period: {context.max_liquid:.0f}")
        
        # Calculate drawdown
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
    Handle initial portfolio allocation and intraday signal changes.
    
    Called daily. Monitors for VIX signal changes and SPY moving average
    crossovers that require immediate action rather than waiting for
    the weekly rebalance.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    
    Signal Handling
    ---------------
    1. VIX Signal Change (vixflag turns negative):
       - If SPY below MA80: Reduce IWM short
       - If SPY above MA80: Full reallocation
    
    2. SPY MA80 Crossover (price crosses above):
       - Reduce IWM short position
       - Only acts on non-rebalance days
    
    3. Initial Portfolio:
       - Triggers regular_allocation on first run
    """
    spyprice_1 = context.price_history_spy100.iloc[-1]
    spyprice_2 = context.price_history_spy100.iloc[-2]
    
    # Setup logging
    logger = logging.getLogger('LS-Prod-Algo')
    logger.setLevel(logging.DEBUG)
    handler = FlightlogHandler()
    logger.addHandler(handler)
    
    # Initialize IWM weight
    if context.iwm_w == 0:
        context.iwm_w = -0.4
    
    # Handle VIX signal change (bearish -> bullish)
    if (context.vixflag_prev > 0 and context.vixflag <= 0 and
        algo.get_datetime().date().weekday() != context.days_offset):
        
        if spyprice_1 < context.spyma80:
            # SPY below MA80: cautiously reduce short
            if context.shortfact != 0:
                new_weight = min(context.iwm_w / (2 * context.shortfact), -0.4 * context.shortfact * context.bcfactor)
                algo.order_target_percent(context.iwm_sid, new_weight)
                
                logger.info(' ')
                logger.info('ALERT ALERT ALERT !!!! ')
                logger.info(f'vix long exit for IWM at {algo.get_datetime(timezone("America/Los_Angeles"))}')
                logger.info(f'switching out to less short IWM -- weight = {new_weight}')
                
                print("\n\n\n\n\n")
                print(">" * 100)
                print(" ALERT ALERT ALERT !!!! ")
                print("")
                print(algo.get_datetime(timezone("America/Los_Angeles")))
                print("vix long exit for IWM")
                print(f"switching out to less short IWM -- weight = {new_weight}")
                print("\n\n\n\n\n")
        
        if spyprice_1 >= context.spyma80:
            # SPY above MA80: full reallocation
            logger.info(' ')
            logger.info('ALERT ALERT ALERT !!!! ')
            logger.info(f'vix long exit for IWM at {algo.get_datetime(timezone("America/Los_Angeles"))}')
            logger.info('executing reallocation')
            
            regular_allocation(context, data)
            exit_positions(context, data)
        
        logger.info(' ')
    
    # Initial portfolio setup
    if context.initialized == 0:
        context.initialized = 1
        regular_allocation(context, data)
    
    # Handle SPY crossing above MA80
    if (spyprice_1 > context.spyma80 and
        spyprice_2 <= context.spyma80 and
        context.vix_uptrend_flag and
        context.spy_below80ma and
        algo.get_datetime().date().weekday() != context.days_offset):
        
        print("\n\n\n\n\n")
        if context.shortfact != 0:
            new_weight = min(context.iwm_w / 2, -0.4 * context.bcfactor)
            algo.order_target_percent(context.iwm_sid, new_weight)
            
            print(">" * 100)
            print("ALERT ALERT ALERT !!!! ")
            print("")
            print(algo.get_datetime(timezone("America/Los_Angeles")))
            print(f"spyma cross over, spyprice_1, spyprice_2, spyma80: {spyprice_1}, {spyprice_2}, {context.spyma80}")
            print(f"spyma cross over --- switching out to less short IWM -- weight = {new_weight}")
            context.spy_below80ma = False
            print(f'weekday: {algo.get_datetime().date().weekday()}')
            print("\n\n\n\n\n")
            
            logger.info(' ')
            logger.info('ALERT ALERT ALERT !!!! ')
            logger.info(f'spyma cross over, spyprice_1: {spyprice_1}, spyprice_2: {spyprice_2}, spyma80: {context.spyma80} at {algo.get_datetime(timezone("America/Los_Angeles"))}')
            logger.info(f"spyma cross over --- switching out to less short IWM -- weight = {new_weight}")
            logger.info(' ')
    
    return


def regular_allocation(context, data):
    """
    Main portfolio allocation and rebalancing function.
    
    Called weekly (Tuesday) to rebalance the portfolio. Calculates weights
    for all positions based on alpha signals and market conditions.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    
    Allocation Process
    ------------------
    1. Update trend signals and drawdown factor
    2. Normalize long weights by cash_return_zsoft
    3. Apply regime-based multipliers:
       - Long factor (1.5 bullish, variable bearish)
       - Drawdown factor (reduces in drawdowns)
       - Market condition factor (1.05 defensive, 1.102 normal)
       - VIX factor (1.6x in uptrend)
       - BC factor (0.65x if SPY < MA150 and bc1 triggered)
    4. Execute long orders (90% individual, 10% SPY)
    5. Calculate short weights with beta adjustment
    6. Execute short via IWM
    
    Position Sizing
    ---------------
    Long weights: cash_return_zsoft * longfact * dd_factor * 0.637 * adjust * 1.03 * vix_mult * bc_mult
    Short weights: -cash_return_zsoft * shortfact * beta_ratio * trend_mult * spy_mult * 0.637 * adjust * 1.03 * 1.4
    """
    longs = context.longs.index
    shorts = context.shorts.index
    
    # Update trend signals
    compute_trend(context, data)
    context.initialized = 1
    
    # Calculate drawdown adjustment
    context.dd_factor = min([context.clip, (1 + context.draw_down * DRAWDOWN_FACTOR_MULTIPLIER)])
    
    try:
        print(algo.get_datetime(timezone("America/Los_Angeles")))
        print(f'Beta ratio: {context.beta_ratio:.4f}')
        print(f"Drawdown factor: {context.dd_factor:.4f}")
        print(f"Long factor: {context.longfact:.2f}")
        print(f"Net liquidation: {context.account.net_liquidation:.2f}")
    except:
        print("print ERROR >>>>>")
    
    # Get normalized weights
    longs_mcw, shorts_mcw = get_normalized_weights(context, data, 'cash_return_zsoft')
    
    # Calculate regime-based multipliers
    if context.vix_uptrend_flag:
        trend_longfact_multiplier = 0.625
        trend_spy_gt_ma21 = 1
    else:
        trend_longfact_multiplier = 1.625
        trend_spy_gt_ma21 = 1.3 if context.spyprice > context.spyma21 else 1
    
    # Market condition adjustment
    if context.spyprice < context.spyma80:
        adjust_fact = 1.05  # More conservative below 80-day MA
    else:
        print('SPY above 80-day MA: switch adjust_fact to 1.102 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', 
              'spyprice:', context.spyprice, 'spyma:', context.spyma80)
        adjust_fact = 1.102  # More aggressive above 80-day MA
    
    print('adjust factor', adjust_fact)
    
    # Apply long weight multipliers
    longs_mcw['cash_return_zsoft'] = (
        longs_mcw['cash_return_zsoft'] * 
        context.longfact * 
        context.dd_factor * 
        0.637 * 
        #adjust_fact * 
        1.03
    )
    
    # VIX uptrend boost
    if context.vix_uptrend_flag:
        longs_mcw['cash_return_zsoft'] = longs_mcw['cash_return_zsoft'] * 1.6
    
    # Barchart trend adjustment
    if context.spyprice < context.spyma150 and context.bc1 == 1:
        context.bcfactor = 0.82
        longs_mcw['cash_return_zsoft'] = longs_mcw['cash_return_zsoft'] * context.bcfactor
    else:
        context.bcfactor = 0.82
    
    # Portfolio split: 90% individual stocks, 10% SPY
    port_weight_factor = 0.99
    spy_weight_factor = 1 - port_weight_factor
    
    if context.verbose == 1:
        print_positions(longs, longs_mcw, 'cash_return_zsoft', port_weight_factor)
    
    # Execute long orders
    total_wl = 0
    for sid, w in zip(longs, longs_mcw['cash_return_zsoft'].values):
        w = abs(w)
        algo.order_target_percent(sid, w * port_weight_factor)
        if w > 1.0:
            print(sid, w)
        total_wl = total_wl + w
    
    # SPY allocation
    algo.order_target_percent(context.spysym, total_wl * spy_weight_factor)
    
    print(f'Total SPY weight: {total_wl * spy_weight_factor:.4f}')
    print(f'Total port long weight: {total_wl * port_weight_factor:.4f}')
    print(f'Total long weight: {total_wl:.4f}')
    
    # Calculate short weights
    #shorts_mcw[shorts_mcw['cash_return_zsoft'] > 0.15] = 0.15
    
    shorts_mcw['cash_return_zsoft'] = (
        -shorts_mcw['cash_return_zsoft'] *
        context.shortfact *
        context.beta_ratio *
        trend_longfact_multiplier *
        trend_spy_gt_ma21 *
        0.637 *
        adjust_fact *
        1.03
    )
    
    # Apply short multiplier (same for both regimes currently)
    # if context.vix_uptrend_flag and context.spyprice > context.spyma21:
    #     shorts_mcw['cash_return_zsoft'] = shorts_mcw['cash_return_zsoft'] * 1.4
    # else:
    #     shorts_mcw['cash_return_zsoft'] = shorts_mcw['cash_return_zsoft'] * 1.4
    
    # Calculate total short weight
    total_ws = 0
    for sid, w in zip(shorts, shorts_mcw['cash_return_zsoft'].values):
        w = abs(w)
        w = w * -1
        if w > 1.0:
            print(sid, w)
        total_ws = total_ws + w
    
    if total_ws > 0:
        print(total_ws)
        print('Error: total short weight is positive')
    
    context.total_ws = total_ws
    print(f'Total short weight: {total_ws:.4f}')
    
    
    
    # Execute short orders via IWM
    if context.vix_uptrend_flag and context.spy_below80ma:
        iwm_w = min(-1 * 0.384 * total_wl, total_ws)
        place_short_orders(algo, context, context.short_symbol_weights, iwm_w)
        print('Short Position Factors: shortfact:', context.shortfact, 'beta_ratio:', context.beta_ratio, 
              'trend_multiplier:', trend_longfact_multiplier, 'spy_ma21_factor:', trend_spy_gt_ma21, 
              'base_multiplier:', 0.637, 'adjust_factor:', adjust_fact, 'final_multiplier:', 1.03)
        print(f'Min active-> SPY below MA80 - IWM weight {iwm_w:.4f}')
        context.iwm_w = iwm_w
    else:
        place_short_orders(algo, context, context.short_symbol_weights, total_ws)
        print('Short Position Factors: shortfact:', context.shortfact, 'beta_ratio:', context.beta_ratio, 
              'trend_multiplier:', trend_longfact_multiplier, 'spy_ma21_factor:', trend_spy_gt_ma21, 
              'base_multiplier:', 0.637, 'adjust_factor:', adjust_fact, 'final_multiplier:', 1.03)
        print(f'IWM weight {total_ws:.4f}')
        context.iwm_w = total_ws
    
    # Position count warning
    if len(pd.Series(tuple(context.portfolio.positions.keys()))) > 52:
        print("WARNING: Too many positions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>+++++++++++++++++++++++++++++++++")
    
    print(" ")
    return


def exit_positions(context, data):
    """
    Exit positions no longer in the target portfolio.
    
    Called weekly after regular_allocation. Closes any positions that
    are not in the current long portfolio or excluded ETFs.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    
    Protected Positions
    -------------------
    The following are never automatically closed:
    - IWM (hedge position)
    - SPY (core allocation)
    - DIA, QQQ, TLT, SPLV (other ETFs)
    """
    desired_sids = set(context.longs.index)
    
    # Find positions to close
    getting_the_boot = [
        sid for sid in context.portfolio.positions.keys()
        if sid not in desired_sids
        and sid != context.iwm_sid
        and sid != context.spysym
        and sid != context.dia_sid
        and sid != context.qqq_sid
        and sid != context.tlt_sid
        and sid != algo.sid(symbol('SPLV').real_sid)
    ]
    
    if context.verbose == 1 and getting_the_boot:
        print('Exiting positions not in longs and not IWM.')
    
    # Close each position
    for sid in getting_the_boot:
        if context.verbose == 1:
            print('Exiting', sid)
        try:
            algo.order_target(sid, 0)
        except Exception as e:
            print(f'Failed to exit {sid}:', e)
    
    return


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# Supporting functions for SID initialization, universe filtering,
# weight normalization, and other algorithm utilities.

def initialize_sids(context, data, algo):
    """
    Initialize security IDs for ETFs and benchmark instruments.
    
    Called once on first trading day. Looks up SIDs for all ETFs
    and indexes used by the algorithm.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    algo : module
        Zipline API module
    
    SIDs Initialized
    ----------------
    - benchmarkSecurity: IWM (for beta calculation)
    - iwm_sid, dia_sid, qqq_sid, tlt_sid: ETF positions
    - ibm_sid: Used for VIX signal lookup
    - spysym: SPY for core allocation
    - sector_etf: List of sector ETFs for momentum ranking
    - sector_etf_dict: Mapping of symbol to SID
    - short_symbol_weights: Hedge instrument weights
    """
    if context.sids_initialized == 0:
        print('Looking up security IDs >>>>>>')
        
        # Core ETFs
        context.benchmarkSecurity = algo.sid(symbol('IWM').real_sid)
        context.iwm_sid = algo.sid(symbol('IWM').real_sid)
        context.dia_sid = algo.sid(symbol('DIA').real_sid)
        context.ibm_sid = algo.sid(symbol('IBM').real_sid)
        context.spysym = algo.sid(symbol('SPY').real_sid)
        context.qqq_sid = algo.sid(symbol('QQQ').real_sid)
        context.rwm_sid = algo.sid(symbol('RWM').real_sid)
        context.tlt_sid = algo.sid(symbol('TLT').real_sid)
        context.iwb_sid = algo.sid(symbol('IWB').real_sid)
        
        # Non-tradable symbols (excluded from stock selection)
        context.no_trade_sym = symbols([
            'OEF', 'QQQ', 'IWM', 'SPY', 'TLT',
            'MTUM', 'SPYG', 'QUAL', 'DIA', 'SPHB'
        ])
        print('Non-tradable symbols:', context.no_trade_sym)
        
        # Sector ETFs for momentum ranking
        context.sector_etf = []
        context.sector_etf_dict = {}
        for sym in ['IYZ', 'XLF', 'XLE', 'XLK', 'XLB', 'XLY', 'XLI', 'XLV', 'XLP', 'XLU']:
            sid_var = algo.sid(symbol(sym).real_sid)
            context.sector_etf.append(sid_var)
            context.sector_etf_dict.update({sym: sid_var})
        context.sector_etf = pd.Series(context.sector_etf)
        print("Sector ETF dictionary:", context.sector_etf_dict)
        
        # Index ETFs
        context.index_etf = []
        context.index_etf_dict = {}
        for sym in ['IWM', 'QQQ']:
            sid_var = algo.sid(symbol(sym).real_sid)
            context.index_etf.append(sid_var)
            context.index_etf_dict.update({sym: sid_var})
        context.index_etf = pd.Series(context.index_etf)
        print("Index ETF dictionary:", context.index_etf_dict)
        
        # Short instrument weights (100% IWM)
        context.short_symbol_weights = {
            context.iwm_sid: 1,
        }
        
        context.sids_initialized = 1
    
    return


def filter_symbols_sectors_universe(df):
    """
    Filter universe by sector exclusions and special cases.
    
    Removes certain stocks and sectors from consideration based on:
    - Merger/acquisition dates (stocks delisted)
    - Sector exclusions (Financials)
    - Sector limits (Energy, Real Estate)
    - Trading history requirements
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw pipeline output
    
    Returns
    -------
    pandas.DataFrame
        Filtered universe
    
    Exclusions
    ----------
    - GBT: After June 2022 (acquisition)
    - ABMD: After November 2022 (acquisition)
    - XM: After July 2023
    - SPLK: After November 2023 (acquisition)
    - MSTR: After 2010 (crypto exposure)
    - ITCI: After March 2025
    - Financials sector: Excluded entirely
    - Energy: Limited to top 20 by market cap
    - Real Estate: Limited to top 15 by market cap
    - New listings: publicdays must be > 0
    """
    current_date = algo.get_datetime().date()
    current_year = current_date.year
    current_month = current_date.month
    
    # Build exclusion list based on dates
    to_drop = []
    if (current_year > 2022) or (current_year == 2022 and current_month >= 6):
        to_drop.append('GBT')
    if (current_year > 2022) or (current_year == 2022 and current_month >= 11):
        to_drop.append('ABMD')
    if (current_year > 2023) or (current_year == 2023 and current_month >= 7):
        to_drop.append('XM')
    if (current_year > 2023) or (current_year == 2023 and current_month >= 11):
        to_drop.append('SPLK')
    if current_year > 2010:
        to_drop.append('MSTR')
    if (current_year > 2025) or (current_year == 2025 and current_month >= 3):
        to_drop.append('ITCI')
    if (current_year > 2026) or (current_year == 2026 and current_month >= 2):
        to_drop.append('RNA')
    if (current_year > 2026) or (current_year == 2026 and current_month >= 3):
        to_drop.append('EXAS')
    # if (current_year > 2026) or (current_year == 2026 and current_month >= 4):
    #     to_drop.append('CAR')
       
    
    # Remove excluded stocks
    for stock_name in to_drop:
        df = df[df['name'] != stock_name]
    
    # Remove Financials sector
    df = df[df['sector'] != 'Financials']
    
    # Limit Energy and Real Estate exposure
    df_energy = df[df['sector'] == 'Energy'].sort_values(by='market_cap', ascending=False)[:20]
    df_real_estate = df[df['sector'] == 'Real Estate'].sort_values(by='market_cap', ascending=False)[:15]
    
    df = pd.concat([ 
        df[~df['sector'].isin(['Energy', 'Real Estate'])],
        df_energy,
        df_real_estate
    ])
    
    # Require trading history
    df = df[df['publicdays'] > 0]
    
    return df


def get_normalized_weights(context, data, target):
    """
    Normalize position weights with size limits.
    
    Converts raw alpha scores into portfolio weights, applying
    position size constraints to ensure diversification.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    target : str
        Column name containing alpha scores
    
    Returns
    -------
    tuple
        (longs_mcw, shorts_mcw) - DataFrames with normalized weights
    
    Normalization Process
    ---------------------
    1. Fill NaN values with small positive number
    2. Normalize to sum to 1
    3. Apply position limits (max 6% long, 2% short)
    4. Clip minimum weights to avoid tiny positions
    5. Re-normalize to sum to 1
    6. Repeat clipping/normalization for stability
    """
    # Fill missing values
    context.longs[target].fillna(value=0.001, inplace=True)
    context.shorts[target].fillna(value=0.001, inplace=True)
    
    # Display sector statistics
    df1 = context.longs.groupby('sector')[target].agg(['median', 'mean', 'count'])
    print('Mean of target', target)
    print(df1)
    print('Length of target column:', len(context.longs[[target]]), 'Sum:', context.longs[[target]].sum())
    
    # Normalize long weights
    longs_mcw = abs(context.longs[[target]]) / abs(context.longs[[target]]).sum()
    
    # Apply position limits (two passes for stability)
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
    """
    Print current position weights for debugging.
    
    Parameters
    ----------
    port : pandas.Index
        Security IDs
    port_w : pandas.DataFrame
        Weight DataFrame
    target : str
        Column name with weights
    factor : float, optional
        Multiplier for display (default: 1)
    """
    print(algo.get_datetime(timezone("America/Los_Angeles")))
    l = sorted(
        [list(c) for c in zip(port[0:], (port_w[target] * factor).round(6).astype(str).values[0:])],
        key=lambda x: x[1],
        reverse=True
    )
    print(pd.Series(l).values)
    return


def place_short_orders(algo, context, symbol_weights, total_weight):
    """
    Execute short position orders.
    
    Places orders for short hedging instruments (primarily IWM).
    Weights are distributed proportionally among instruments.
    
    Parameters
    ----------
    algo : module
        Zipline API module
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    symbol_weights : dict
        Mapping of SID to relative weight
    total_weight : float
        Total short weight to allocate (negative)
    """
    logger = logging.getLogger('LS-Prod-Algo')
    logger.setLevel(logging.DEBUG)
    handler = FlightlogHandler()
    logger.addHandler(handler)
    
    sum_of_weights = sum(symbol_weights.values())
    
    for symbol, weight in symbol_weights.items():
        target_percent = (weight / sum_of_weights) * total_weight
        print(f"Executing short target {symbol}, {target_percent * context.shortfact:.4f}")
        algo.order_target_percent(symbol, target_percent * context.shortfact)
    
    return


def GenerateMomentumList(context, data, etf_list, momlength):
    """
    Generate momentum ranking for sector ETFs.
    
    Ranks ETFs by price momentum over the specified period.
    Used to identify strong/weak sectors for rotation.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    etf_list : list
        List of ETF SIDs to rank
    momlength : int
        Lookback period in trading days
    
    Returns
    -------
    list
        List of [SID, momentum] pairs sorted by momentum (descending)
    """
    price_history = data.history(etf_list, 'price', momlength, '1d')
    pct_change = (price_history.iloc[-1] - price_history.iloc[0]) / price_history.iloc[0]
    momentum_df = pct_change.to_frame(name='momentum').reset_index()
    momentum_df = momentum_df.sort_values(by='momentum', ascending=False)
    top_momentum_list = momentum_df.head(context.topMom).values.tolist()
    return top_momentum_list


def RemoveSectors(context, etf, dfs, prt_str):
    """
    Remove stocks from a specific sector based on ETF mapping.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    etf : zipline.assets.Equity
        Sector ETF SID
    dfs : pandas.DataFrame
        Stock universe to filter
    prt_str : str
        Format string for logging
    
    Returns
    -------
    pandas.DataFrame
        Filtered universe with sector removed
    
    Sector Mapping
    --------------
    XLB -> Materials
    XLY -> Consumer Discretionary
    XLF -> Financials
    XLP -> Consumer Staples
    XLV -> Health Care
    XLU -> Utilities
    IYZ -> Communication Services
    XLE -> Energy
    XLI -> Industrials
    XLK -> Information Technology
    """
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
    
    if etf in etf_sector_map:
        sector_to_remove = etf_sector_map[etf]
        dfs = dfs[dfs['sector'] != sector_to_remove]
        print(prt_str % etf)
    
    return dfs


def compute_trend(context, data):
    """
    Determine market trend and set exposure factors.
    
    Uses VIX signal to classify market regime and sets appropriate
    exposure multipliers for long and short positions.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    
    Regime Classification
    ---------------------
    VIX signal <= 0 (Bullish):
        - vix_uptrend_flag = True
        - longfact = 1.5
        - shortfact = 0.45 or 0.9 (seasonal)
        - clip = 1.2
    
    VIX signal > 0 (Bearish):
        - vix_uptrend_flag = False
        - longfact = 0.0 (if SPY < MA80) or IWM weight
        - shortfact = 0.5
        - clip = 1.6
    """
    print('VIX flag:', context.vixflag)
    
    # Store previous value
    context.longfact_last = context.longfact
    
    if context.vixflag <= 0:
        # Bullish regime
        print('Trend mode: 1.5')
        context.vix_uptrend_flag = True
        context.longfact = 1.5
        
        # Seasonal short adjustment
        if algo.get_datetime().date().month in SHORT_RESTRICTED_MONTHS and context.bc1 == 0: #not context.spy_below80ma:
            context.shortfact = 0.45
        else:
            context.shortfact = 0.9
        context.clip = 1.2
    else:
        # Bearish regime
        print('Trend mode: 1')
        context.vix_uptrend_flag = False
        context.longfact = 0.0 if context.spy_below80ma else abs(context.iwm_w)
        context.shortfact = 0.5
        context.clip = 1.6
    
    return


def compute_beta(context, data):
    """
    Compute portfolio beta ratio versus benchmark.
    
    Calculates the ratio of long portfolio beta to short portfolio beta,
    used for adjusting hedge sizing.
    
    Parameters
    ----------
    context : zipline.algorithm.TradingAlgorithm
        Algorithm context
    data : zipline.protocol.BarData
        Market data accessor
    
    Returns
    -------
    float
        Beta ratio (long_beta / short_beta)
    
    Calculation
    -----------
    Uses exponentially weighted covariance over 120 days:
    beta = cov(portfolio_returns, benchmark_returns) / var(benchmark_returns)
    
    Notes
    -----
    Minimum return value is 1.3 to ensure adequate hedging.
    """
    benchmark_array = np.array([context.benchmarkSecurity])
    assets_array = np.concatenate((context.universe, benchmark_array))
    
    prices = data.history(assets_array, 'price', 120, '1d')
    
    prices_longs = prices[context.longs.index.intersection(prices.columns)]
    prices_shorts = prices[context.shorts.index.intersection(prices.columns)]
    prices_spy = prices[context.benchmarkSecurity]
    
    # Calculate portfolio returns (sum across positions)
    rets_long_port = prices_longs.pct_change().sum(axis=1)
    rets_short_port = prices_shorts.pct_change().sum(axis=1)
    rets_spy = prices_spy.pct_change()
    
    beta_span = 120
    
    # Calculate exponentially weighted covariances and variance
    long_cov = rets_long_port.ewm(span=beta_span, adjust=True).cov(rets_spy)
    short_cov = rets_short_port.ewm(span=beta_span, adjust=True).cov(rets_spy)
    bench_var = rets_spy.ewm(span=beta_span, adjust=True).var()
    
    # Calculate betas
    long_beta = long_cov.iloc[-1] / bench_var.iloc[-1]
    short_beta = short_cov.iloc[-1] / bench_var.iloc[-1]
    
    # Calculate ratio
    beta_ratio = long_beta / short_beta
    
    print("long_beta, short_beta, beta_ratio:", long_beta, short_beta, beta_ratio)
    
    return beta_ratio

def drop_top_vol_outliers(df, col='vol', n=2, k=1.5, verbose=True):
    """
    Drop the top-n rows of `df` ranked by `col`, but only if they exceed
    the upper fence (p90 + k*IDR), where IDR is the inter-decile range
    (p90 - p10). Rows within the normal range are kept.

    Using the 10th/90th percentiles instead of the 25th/75th widens the
    "normal" band — useful when the middle 50% is too narrow to reflect
    the full spread of the data (common for fat-tailed series like vol).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Not modified in place.
    col : str, default 'vol'
        Column used to rank rows and compute the outlier threshold.
    n : int, default 2
        Number of top-ranked candidates to consider for removal.
    k : float, default 1.5
        Fence multiplier applied to the inter-decile range.
    verbose : bool, default True
        If True, print the fence, the dropped indexes, and their values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with qualifying outliers removed. The original `df`
        is untouched.
    """
    # Compute the 10th and 90th percentiles of the target column.
    # p10 = value below which 10% of observations fall
    # p90 = value below which 90% of observations fall
    p10, p90 = df[col].quantile([0.10, 0.90])

    # Inter-decile range — the spread of the middle 80% of the data.
    # Wider than IQR, so the fence sits further out.
    idr = p90 - p10

    # Upper fence: threshold above which values are flagged as outliers.
    upper = p90 + k * idr

    # Index labels of the top-n candidates by `col` value.
    top_idx = df[col].nlargest(n).index

    # Keep only candidates that actually exceed the fence.
    drop_idx = [i for i in top_idx if df.loc[i, col] > upper]

    # Build the cleaned frame (leaves original untouched).
    cleaned = df.drop(drop_idx)

    if verbose:
        print(f"p10={p10:.4f}, p90={p90:.4f}, IDR={idr:.4f}")
        print(f"Fence ({col} > p90 + {k}*IDR) = {upper:.4f}")
        if drop_idx:
            print(f"Dropped {len(drop_idx)} outlier(s):")
            for idx in drop_idx:
                print(f"  index={idx!r}, {col}={df.loc[idx, col]:.4f}")
        else:
            print("Dropped 0 outlier(s) — top values within normal range.")

    return cleaned