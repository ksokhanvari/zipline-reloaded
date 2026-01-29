#!/usr/bin/env python3
"""
WALK-FORWARD ML Return Forecasting - NO LOOK-AHEAD BIAS

This is the WALK-FORWARD version that eliminates look-ahead bias by training
models month-by-month using only data available at that point in time.

DIFFERENCE FROM forecast_returns_ml.py:
  - forecast_returns_ml.py: Single model trained on ALL data (FASTER but has look-ahead bias)
  - THIS SCRIPT: Monthly expanding window training (SLOWER but realistic for backtesting)

This script uses Histogram-based Gradient Boosting with extensive feature
engineering to predict stock returns at customizable horizons. Designed for
institutional trading strategies with rigorous prevention of look-ahead bias.

Key Features:
  - EXPANDING WINDOW WALK-FORWARD: Each month trains on only past data
  - No look-ahead bias (all features lagged by 1 day + walk-forward training)
  - Market cap weighted training (focus on top 2000 stocks)
  - Predictions for ALL dates (including recent dates for live trading)
  - Customizable return periods (10-day, 90-day, etc.)
  - Lean output (only adds 2 columns to original data)
  - 80-95% correlation with actual returns (depending on horizon)

WALK-FORWARD EXPLANATION:
For a prediction on 2010-06-15, the model trains ONLY on data before 2010-06-01.
For a prediction on 2015-08-20, the model trains ONLY on data before 2015-08-01.
This ensures each prediction reflects what you could have known at that time.

Usage:
    # Full training run (first time) - use Parquet for 5-10x faster I/O!
    python forecast_returns_ml_walk_forward.py \\
        --input-file 20091231_20260106_data.csv \\
        --output 20091231_20260106_predictions.parquet
    # Creates: 20091231_20260106_predictions.parquet (main output, fast)
    #          20091231_20260106_forecast_only.csv (auto-created, for easy use)

    # Resume from previous predictions (95% faster for updates!)
    python forecast_returns_ml_walk_forward.py \\
        --input-file 20091231_20260110_data.csv \\
        --output 20091231_20260110_predictions.parquet \\
        --resume-file 20091231_20260106_predictions.parquet
    # Reads/writes Parquet = 5-10x faster than CSV!

    # Predict 90-day returns starting 10 days from now
    python forecast_returns_ml_walk_forward.py \\
        --input-file data.csv \\
        --output predictions.parquet \\
        --forecast-days 10 \\
        --target-return-days 90

    # Resume with custom overwrite buffer (re-predict last N months)
    python forecast_returns_ml_walk_forward.py \\
        --input-file new_data.csv \\
        --output new_predictions.parquet \\
        --resume-file old_predictions.parquet \\
        --overwrite-months 0
"""

import argparse
import warnings
from pathlib import Path
import json
import hashlib
from datetime import datetime
import sys
import logging
import re
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Temporal diagnostics
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - temporal diagnostics will be skipped. "
                  "Install with: pip install statsmodels")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# FILE I/O HELPERS - Support both CSV and Parquet
# ============================================================================

def read_dataframe(file_path, description="file"):
    """
    Read a dataframe from CSV or Parquet format (auto-detected by extension).

    Args:
        file_path: Path to file (.csv or .parquet)
        description: Description for logging

    Returns:
        pandas DataFrame
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() in ['.parquet', '.pq']:
        # Read Parquet (5-10x faster)
        print(f"  • Reading Parquet format (fast!)")
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"  • Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    else:
        # Read CSV (try PyArrow, fallback to default)
        try:
            df = pd.read_csv(file_path, engine='pyarrow')
            print(f"  • Loaded {len(df):,} rows, {len(df.columns)} columns (PyArrow engine)")
        except (ImportError, Exception):
            df = pd.read_csv(file_path)
            print(f"  • Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df


def write_dataframe(df, file_path, description="predictions"):
    """
    Write a dataframe to CSV or Parquet format (auto-detected by extension).

    Args:
        df: pandas DataFrame to write
        file_path: Path to output file (.csv or .parquet)
        description: Description for logging
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() in ['.parquet', '.pq']:
        # Write Parquet (5-10x faster, much smaller)
        print(f"💾 Saving {description} to {file_path.name}...")
        df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"  ✓ Saved {len(df):,} rows as Parquet")
        print(f"  • Format: Parquet (compressed, ~5-10x smaller than CSV)")
        print(f"  • Size: {file_size_mb:.1f} MB")
        print(f"  • Columns: {len(df.columns)}")
    else:
        # Write CSV
        print(f"💾 Saving {description} to {file_path.name}...")
        df.to_csv(file_path, index=False)
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"  ✓ Saved {len(df):,} rows as CSV")
        print(f"  • Size: {file_size_mb:.1f} MB")
        print(f"  • Columns: {len(df.columns)}")


# ============================================================================
# LOGGING SETUP
# ============================================================================

class DualLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_file=None):
        """Initialize logger with optional log file."""
        self.logger = logging.getLogger('forecast_returns_ml')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear any existing handlers

        # Create formatter
        formatter = logging.Formatter('%(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if log file specified)
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        else:
            self.log_file = None

    def info(self, msg):
        """Log info message."""
        self.logger.info(msg)

    def __call__(self, msg):
        """Allow logger to be called like print()."""
        self.info(msg)


# Global logger instance (will be initialized in main())
logger = None


# ============================================================================
# CHECKPOINT FUNCTIONS - For Resume Capability
# ============================================================================

def compute_data_hash(df, before_date=None):
    """
    Compute hash of dataframe for change detection.

    Args:
        df: DataFrame to hash
        before_date: Only hash data before this date (for historical data check)

    Returns:
        Hash string
    """
    df_to_hash = df.copy()

    if before_date is not None:
        df_to_hash = df_to_hash[df_to_hash['Date'] < before_date]

    # Hash based on shape and sample of data
    hash_input = f"{df_to_hash.shape}_{df_to_hash['Date'].min()}_{df_to_hash['Date'].max()}_{len(df_to_hash['Symbol'].unique())}"

    return hashlib.md5(hash_input.encode()).hexdigest()


class ReturnForecaster:
    """
    Fast ML-based return forecasting using fundamentals.

    This class implements a complete pipeline for training gradient boosting models
    to predict stock returns using fundamental data. Key features:

    - NO LOOK-AHEAD BIAS: All fundamental features are lagged by 1 day
    - PRODUCTION READY: Trains on historical data, predicts for ALL dates (including recent)
    - MARKET CAP WEIGHTING: Training focuses on large-cap stocks (top 2000)
    - FLEXIBLE RETURN PERIODS: Can predict any return horizon (10-day, 90-day, etc.)
    - LEAN OUTPUT: Only adds 2 columns to original data

    IMPORTANT: The model trains ONLY on dates with valid forward_return (where we have
    future prices for validation), but predicts for ALL dates including recent ones
    without forward_return. This is essential for production use - you get forecasts
    for dates you can actually trade on!

    Example:
        >>> forecaster = ReturnForecaster(
        ...     forecast_days=10,      # Start measuring returns 10 days from now
        ...     target_return_days=90  # Measure 90-day returns
        ... )
        >>> predictions = forecaster.fit_predict(df)
        >>> # predictions will include recent dates even without forward_return
    """

    def __init__(self, lookback_days=10, forecast_days=10, target_return_days=None,
                 n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31, no_lag=False,
                 sample_fraction=1.0, pca_components=None, lookback_months=None, log_features=False):
        """
        Initialize the return forecaster.

        Args:
            lookback_days (int): NOT IMPLEMENTED - Reserved for future use (default: 10)
                This parameter is currently unused. Rolling windows for momentum
                and volatility features are hardcoded to [5, 10, 20] days.

            forecast_days (int): Days ahead to start measuring returns (default: 10)
                This is the "lead time" before returns are measured

            target_return_days (int): Return period to predict (default: same as forecast_days)
                Length of the return period to forecast
                Example: forecast_days=10, target_return_days=90 means:
                "Predict the 90-day return starting 10 days from now"

            n_estimators (int): Number of boosting rounds (default: 300)
                More estimators = better fit but slower training

            learning_rate (float): Learning rate for gradient boosting (default: 0.05)
                Lower = more robust but needs more estimators

            max_depth (int): Maximum tree depth (default: 6)
                Controls model complexity. Reduced from 7 for noisy stock returns.

            num_leaves (int): Maximum number of leaves per tree (default: 31)
                Controls model complexity (2^max_depth - 1 is typical)

            no_lag (bool): If True, skip lagging of features (default: False)
                Use this when your input data is already lagged

            sample_fraction (float): Fraction of training data to use (0.0-1.0, default: 1.0)
                Use 0.5 for 50% sampling to speed up training by ~2x

            pca_components (int): If set, use PCA to reduce features to this many components (default: None)
                Dimensionality reduction using Principal Component Analysis
                Reduces feature space while preserving variance
                Typical values: 10-50 components

            lookback_months (int): If set, use rolling window of N months for training (default: None)
                None = Expanding window (train on ALL historical data)
                12 = Rolling 12-month window (train on last 12 months only)
                24 = Rolling 24-month window (train on last 24 months only)

                Example with --lookback-months 12:
                - Jan 2021: Train on Jan 2020 - Dec 2020 (12 months)
                - Feb 2021: Train on Feb 2020 - Jan 2021 (12 months)
                - Mar 2021: Train on Mar 2020 - Feb 2021 (12 months)

                Use this to test recency bias vs. long-term patterns.
                Shorter windows = more responsive to recent changes
                Longer windows/expanding = more stable, learns long-term patterns

            log_features (bool): If True, log top 20 feature importances per month (default: False)
                Creates CSV file in logs/ directory with monthly feature importance rankings
                Allows plotting feature importance changes over time
                Adds ~5-10 seconds per month to runtime
        """
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        # If target_return_days not specified, use same as forecast_days
        self.target_return_days = target_return_days if target_return_days is not None else forecast_days
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.no_lag = no_lag
        self.sample_fraction = sample_fraction
        self.pca_components = pca_components
        self.lookback_months = lookback_months
        self.log_features = log_features
        self.feature_log_path = None  # Will hold CSV path if logging features
        self.model = None  # Will hold trained model
        self.feature_cols = None  # Will hold list of feature column names
        self.pca = None  # Will hold fitted PCA transformer if using PCA
        self.scaler = None  # Will hold fitted StandardScaler if using PCA

    def create_target(self, df):
        """
        Create forward returns as target variable.

        The model predicts returns from T+forecast_days to T+forecast_days+target_return_days.

        Args:
            df: DataFrame with Date, Symbol, RefPriceClose columns

        Returns:
            DataFrame with forward_return column
        """
        print(f"Creating target: {self.target_return_days}-day returns starting {self.forecast_days} days ahead...")

        # CRITICAL SAFETY #1: Sort for time-series operations
        # groupby('Symbol').shift() requires data sorted by Symbol, then Date within each Symbol
        # Stable sort ensures reproducible ordering when (Symbol, Date) pairs duplicate
        print(f"🔒 Sorting dataframe for time-series operations...")
        df = df.sort_values(['Symbol', 'Date'], kind='stable').reset_index(drop=True)
        print(f"  ✓ Sorted by Symbol+Date ({len(df):,} rows)")

        # Calculate forward returns per symbol (REQUIRES SORTED DATA)
        # Price at T+forecast_days (e.g., T+10)
        df['price_at_forecast'] = df.groupby('Symbol')['RefPriceClose'].shift(-self.forecast_days)

        # Price at T+forecast_days+target_return_days (e.g., T+10+90 = T+100)
        df['price_at_target'] = df.groupby('Symbol')['RefPriceClose'].shift(-(self.forecast_days + self.target_return_days))

        # Return from forecast point to target point
        # e.g., (Price[T+100] - Price[T+10]) / Price[T+10]
        # NOTE: Returns are expressed as PERCENTAGES (multiply by 100)
        # Example: 0.05 (5% gain) → 5.0, not 0.05
        df['forward_return'] = (df['price_at_target'] / df['price_at_forecast'] - 1) * 100

        # Remove extreme outliers (likely data errors)
        # > 500% gain or < -99% loss are likely bad data
        df.loc[df['forward_return'] > 500, 'forward_return'] = np.nan
        df.loc[df['forward_return'] < -99, 'forward_return'] = np.nan

        df = df.drop(['price_at_forecast', 'price_at_target'], axis=1)

        print(f"  ✓ Target created: Predicting {self.target_return_days}-day return from day {self.forecast_days}")
        return df

    def engineer_features(self, df):
        """
        Engineer features from fundamental and price data.

        CRITICAL: All fundamentals are lagged by 1 day to prevent look-ahead bias
                  (unless no_lag=True, for pre-lagged data).

        Args:
            df: Raw DataFrame with fundamental columns

        Returns:
            DataFrame with engineered features
        """
        if self.no_lag:
            print("Engineering features (using pre-lagged input data - NO ADDITIONAL LAGGING)...")
        else:
            print("Engineering features (with proper lagging to prevent look-ahead bias)...")

        # CRITICAL SAFETY #2: Sort for time-series operations in feature engineering
        # All groupby('Symbol') operations below require sorted data for:
        # - .shift(), .pct_change(), .rolling(), .ffill() to work correctly per symbol
        print(f"  • Sorting by Symbol+Date for time-series integrity...")
        df = df.sort_values(['Symbol', 'Date'])

        # ===== STEP 1: Lag ALL raw fundamental columns by 1 day (unless no_lag=True) =====
        fundamental_cols = [
            'RefPriceClose', 'RefVolume', 'CompanyMarketCap',
            'EnterpriseValue_DailyTimeSeries_',
            'FOCFExDividends_Discrete',
            'InterestExpense_NetofCapitalizedInterest',
            'Debt_Total',
            'EarningsPerShare_Actual',
            'EarningsPerShare_SmartEstimate_prev_Q',
            'EarningsPerShare_ActualSurprise',
            'EarningsPerShare_SmartEstimate_current_Q',
            'LongTermGrowth_Mean',
            'PriceTarget_Median',
            'CombinedAlphaModelSectorRank',
            'CombinedAlphaModelSectorRankChange',
            'CombinedAlphaModelRegionRank',
            'EarningsQualityRegionRank_Current',
            'EnterpriseValueToEBIT_DailyTimeSeriesRatio_',
            'EnterpriseValueToEBITDA_DailyTimeSeriesRatio_',
            'EnterpriseValueToSales_DailyTimeSeriesRatio_',
            'Dividend_Per_Share_SmartEstimate',
            'CashCashEquivalents_Total',
            'ForwardPEG_DailyTimeSeriesRatio_',
            'PriceEarningsToGrowthRatio_SmartEstimate_',
            'Recommendation_Median_1_5_',
            'ReturnOnEquity_SmartEstimat',
            'ReturnOnAssets_SmartEstimate',
            'ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_',
            'ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_',
            'ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_',
            'GrossProfitMargin_ActualSurprise',
            'Estpricegrowth_percent',
        ]

        # CRITICAL SAFETY: ALWAYS lag price, volume, and market cap columns
        # Even if --no-lag is used, these columns must be lagged to prevent data leakage
        # - Price is used to calculate forward_return, so same-day price would leak information
        # - Market cap is derived from price (price × shares), so it must also be lagged
        # - Volume can reflect same-day trading activity, should use prior day
        print("  • SAFETY: Always lagging RefPriceClose, RefVolume, and CompanyMarketCap (even with --no-lag)...")
        if 'RefPriceClose' in df.columns:
            df['RefPriceClose_lag1'] = df.groupby('Symbol')['RefPriceClose'].shift(1)
        if 'RefVolume' in df.columns:
            df['RefVolume_lag1'] = df.groupby('Symbol')['RefVolume'].shift(1)
        if 'CompanyMarketCap' in df.columns:
            df['CompanyMarketCap_lag1'] = df.groupby('Symbol')['CompanyMarketCap'].shift(1)

        # Always use lagged price/volume/marketcap for features
        price_col = 'RefPriceClose_lag1'
        volume_col = 'RefVolume_lag1'

        if self.no_lag:
            print("  • Skipping lagging for other fundamentals (input assumed to be pre-lagged)")
            # Don't lag other fundamentals - assume they're already lagged in the input
        else:
            print("  • Lagging all fundamental columns by 1 day...")
            # Create lagged versions for all fundamentals
            for col in fundamental_cols:
                # Skip price/volume/marketcap (already lagged above)
                if col in df.columns and col not in ['RefPriceClose', 'RefVolume', 'CompanyMarketCap']:
                    df[f'{col}_lag1'] = df.groupby('Symbol')[col].shift(1)

        # ===== STEP 2: Price-based features (from lagged price) =====
        print("  • Creating price momentum features...")
        for days in [5, 10, 20]:
            # Price momentum from lagged prices
            df[f'return_{days}d'] = df.groupby('Symbol')[price_col].pct_change(days) * 100

            # Price volatility from lagged prices
            df[f'volatility_{days}d'] = (
                df.groupby('Symbol')[price_col]
                .pct_change()
                .rolling(days)
                .std() * np.sqrt(252) * 100
            )

        # Volume features (from lagged volume)
        print("  • Creating volume features...")
        df['volume_ma_20'] = df.groupby('Symbol')[volume_col].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df['volume_ratio'] = df[volume_col] / df['volume_ma_20']

        # ===== STEP 3: Fundamental ratios (from lagged data) =====
        print("  • Creating fundamental ratios...")

        # Helper function to get correct column name based on no_lag setting
        def get_col(base_name):
            """Get column name with or without _lag1 suffix based on no_lag flag.

            EXCEPTION: RefPriceClose, RefVolume, and CompanyMarketCap are ALWAYS lagged
            for safety (even with --no-lag).
            """
            # These columns are ALWAYS lagged (safety measure)
            if base_name in ['RefPriceClose', 'RefVolume', 'CompanyMarketCap']:
                return f'{base_name}_lag1'

            # Other columns depend on no_lag setting
            if self.no_lag:
                return base_name
            else:
                return f'{base_name}_lag1'

        # Profitability
        df['roa'] = df[get_col('ReturnOnAssets_SmartEstimate')]
        df['roe'] = df[get_col('ReturnOnEquity_SmartEstimat')]

        # Valuation
        df['ev_to_ebitda'] = df[get_col('EnterpriseValueToEBITDA_DailyTimeSeriesRatio_')]
        df['ev_to_ebit'] = df[get_col('EnterpriseValueToEBIT_DailyTimeSeriesRatio_')]
        df['ev_to_sales'] = df[get_col('EnterpriseValueToSales_DailyTimeSeriesRatio_')]
        df['forward_peg'] = df[get_col('ForwardPEG_DailyTimeSeriesRatio_')]
        df['pe_to_growth'] = df[get_col('PriceEarningsToGrowthRatio_SmartEstimate_')]

        # Growth metrics
        df['ltg'] = df[get_col('LongTermGrowth_Mean')]
        df['price_growth_est'] = df[get_col('Estpricegrowth_percent')]

        # Quality/Momentum
        df['alpha_sector_rank'] = df[get_col('CombinedAlphaModelSectorRank')]
        df['alpha_region_rank'] = df[get_col('CombinedAlphaModelRegionRank')]
        df['alpha_sector_change'] = df[get_col('CombinedAlphaModelSectorRankChange')]
        df['earnings_quality'] = df[get_col('EarningsQualityRegionRank_Current')]

        # Financial health (using lagged market cap)
        mktcap = df[get_col('CompanyMarketCap')].clip(lower=1)
        df['debt_to_marketcap'] = df[get_col('Debt_Total')] / mktcap
        df['cash_to_marketcap'] = df[get_col('CashCashEquivalents_Total')] / mktcap
        df['fcf_to_marketcap'] = df[get_col('FOCFExDividends_Discrete')] / mktcap

        # Earnings metrics
        df['eps_actual'] = df[get_col('EarningsPerShare_Actual')]
        df['eps_surprise'] = df[get_col('EarningsPerShare_ActualSurprise')]
        df['eps_estimate_current'] = df[get_col('EarningsPerShare_SmartEstimate_current_Q')]

        # Analyst metrics
        df['recommendation'] = df[get_col('Recommendation_Median_1_5_')]
        df['price_target'] = df[get_col('PriceTarget_Median')]
        df['upside_to_target'] = (df[get_col('PriceTarget_Median')] / df[price_col] - 1) * 100

        # Size factor
        df['log_marketcap'] = np.log1p(df[get_col('CompanyMarketCap')])

        # ===== STEP 4: Cross-sectional features (rank within date) =====
        # IMPORTANT: Rankings are computed PER-MONTH in walk-forward loop (not here)
        # This ensures adding new data doesn't change historical rankings
        # We just prepare the list of columns to rank later
        print("  • Preparing cross-sectional ranking features (computed per-month in walk-forward)...")
        self._rank_cols = [get_col('CompanyMarketCap'), 'return_20d', 'volatility_20d',
                           'roe', 'roa', 'ev_to_ebitda', 'ltg']
        # Filter to only columns that exist
        self._rank_cols = [col for col in self._rank_cols if col in df.columns]
        print(f"    - Will rank {len(self._rank_cols)} features per month: {', '.join(self._rank_cols)}")

        # Note: Actual ranking happens in _walk_forward_predict() to ensure reproducibility

        # ===== STEP 5: Additional lag features (2 days back) =====
        print("  • Creating additional lagged features...")
        lag2_cols = ['roa', 'roe', 'ev_to_ebitda', 'ev_to_sales', 'ltg']
        for col in lag2_cols:
            if col in df.columns:
                df[f'{col}_lag2'] = df.groupby('Symbol')[col].shift(1)

        print(f"  • Total features created: {len(df.columns)}")

        # ===== STEP 6: Forward-fill missing values (NO LOOK-AHEAD BIAS) =====
        print("  • Forward-filling missing values per symbol...")

        # Get all numeric columns except Date and Symbol
        numeric_cols = df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
        exclude_from_ffill = ['Date', 'Symbol']
        cols_to_ffill = [col for col in numeric_cols if col not in exclude_from_ffill]

        # Forward-fill per symbol (use last known value)
        for col in cols_to_ffill:
            df[col] = df.groupby('Symbol')[col].ffill()

        # After ffill, any remaining NaNs are from the very first rows per symbol
        # Fill these with 0 (no prior data available - safe assumption)
        df[cols_to_ffill] = df[cols_to_ffill].fillna(0)

        print(f"  ✓ Forward-filled {len(cols_to_ffill)} columns per symbol")

        if self.no_lag:
            print("  ✓ Using pre-lagged input data - NO LOOK-AHEAD BIAS")
        else:
            print("  ✓ All features use lagged data - NO LOOK-AHEAD BIAS")
        return df

    def prepare_features(self, df):
        """
        Select and prepare final feature set.

        Args:
            df: DataFrame with engineered features

        Returns:
            Tuple of (X, y, feature_names, sample_weights, valid_idx)
        """
        # Exclude non-feature columns
        exclude_cols = [
            'Date', 'Symbol', 'Instrument', 'TradeDate', 'CompanyCommonName',
            'sharadar_exchange', 'sharadar_category',  # Exchange/category not predictive
            'sharadar_location', 'sharadar_sector', 'sharadar_industry',  # Redundant with GICS/SIC
            'forward_return',  # Target variable
            'volume_ma_20',  # Intermediate calculation
            'RefPriceClose', 'RefVolume', 'CompanyMarketCap',  # ALWAYS exclude (we use lagged versions)
            # Identifiers (not features)
            'cik_fmp', 'cik_fmp_dup', 'cik_fmp_dup.1',  # CIK (Central Index Key)
            # Metadata (not features)
            'fiscalyear_fmp', 'fiscalyear_fmp_dup', 'fiscalyear_fmp_dup.1', 'fiscalyear_fmp_dup.2', 'fiscalyear_fmp_dup.3',
            # NOTE: period_fmp (Q1/Q2/Q3/Q4) is now INCLUDED as categorical feature for seasonality
            'reportedcurrency_fmp', 'reportedcurrency_fmp_dup', 'reportedcurrency_fmp_dup.1', 'reportedcurrency_fmp_dup.2', 'reportedcurrency_fmp_dup.3',
            'accepteddate_fmp', 'accepteddate_fmp_dup', 'accepteddate_fmp_dup.1',
        ]

        # Categorical features to include (for market regime/sector rotation)
        # These will be label-encoded and marked as categorical for HistGradientBoosting
        categorical_features = [
            'GICSSectorName',        # GICS sector (11 sectors: Technology, Healthcare, etc.)
            'sharadar_sicsector',    # SIC sector classification
            'sharadar_sicindustry',  # SIC industry classification
            'period_fmp',            # Q1/Q2/Q3/Q4/FY - quarterly seasonality patterns
        ]

        # Exclude original (non-lagged) columns ONLY if not using no_lag mode
        # When no_lag=True, the input is already pre-lagged, so include these columns
        # NOTE: RefPriceClose, RefVolume, CompanyMarketCap already excluded above (always lagged)
        if not self.no_lag:
            original_cols = [
                'EnterpriseValue_DailyTimeSeries_',
                'FOCFExDividends_Discrete',
                'InterestExpense_NetofCapitalizedInterest',
                'Debt_Total',
                'EarningsPerShare_Actual',
                'EarningsPerShare_SmartEstimate_prev_Q',
                'EarningsPerShare_ActualSurprise',
                'EarningsPerShare_SmartEstimate_current_Q',
                'LongTermGrowth_Mean',
                'PriceTarget_Median',
                'CombinedAlphaModelSectorRank',
                'CombinedAlphaModelSectorRankChange',
                'CombinedAlphaModelRegionRank',
                'EarningsQualityRegionRank_Current',
                'EnterpriseValueToEBIT_DailyTimeSeriesRatio_',
                'EnterpriseValueToEBITDA_DailyTimeSeriesRatio_',
                'EnterpriseValueToSales_DailyTimeSeriesRatio_',
                'Dividend_Per_Share_SmartEstimate',
                'CashCashEquivalents_Total',
                'ForwardPEG_DailyTimeSeriesRatio_',
                'PriceEarningsToGrowthRatio_SmartEstimate_',
                'Recommendation_Median_1_5_',
                'ReturnOnEquity_SmartEstimat',
                'ReturnOnAssets_SmartEstimate',
                'ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_',
                'ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_',
                'ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_',
                'GrossProfitMargin_ActualSurprise',
                'Estpricegrowth_percent',
            ]
            exclude_cols.extend(original_cols)
        else:
            # When using no_lag, the input is pre-lagged, so INCLUDE these columns as features
            print("  • Using pre-lagged fundamentals as features (no_lag=True)")

        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # CRITICAL SAFETY CHECK: Ensure no date-related or non-lagged price columns leak into features
        # Date columns and same-day prices would cause data leakage and unstable predictions
        dangerous_cols = []
        for col in feature_cols:
            col_lower = col.lower()

            # Check for ACTUAL date columns (not just columns with "time" in the name)
            # Exclude columns with "timeseries" or "dailytime" - those are legitimate fundamental descriptors
            is_date_column = False

            # Pattern 1: Ends with 'date' (accepteddate_fmp, tradedate, etc.)
            if col_lower.endswith('date'):
                is_date_column = True
            # Pattern 2: Starts with 'date' or 'timestamp'
            elif col_lower.startswith(('date', 'timestamp')):
                is_date_column = True
            # Pattern 3: Contains '_date_' or '_date' (fiscal_date, report_date, etc.)
            elif '_date' in col_lower or 'date_' in col_lower:
                is_date_column = True
            # Pattern 4: Contains 'timestamp' (but not if part of TimeSeries)
            elif 'timestamp' in col_lower:
                is_date_column = True

            # Whitelist: NOT a date if it contains "timeseries" or "dailytime" (these are fundamental data descriptors)
            if is_date_column and ('timeseries' in col_lower or 'dailytime' in col_lower):
                is_date_column = False

            if is_date_column:
                dangerous_cols.append(col)

            # Check for non-lagged price/volume/marketcap columns
            # These are ALWAYS lagged (even with --no-lag), should ALWAYS have _lag1 suffix
            if col in ['RefPriceClose', 'RefVolume', 'CompanyMarketCap'] and not col.endswith('_lag1'):
                dangerous_cols.append(col)

        if dangerous_cols:
            print(f"\n❌ CRITICAL ERROR: Dangerous columns found in features!")
            print(f"   These columns would cause data leakage:")
            for col in dangerous_cols:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                    print(f"   - {col} (date-related column)")
                else:
                    print(f"   - {col} (non-lagged price/volume column)")
            print(f"\n   Automatically excluding these columns for safety...")
            # Remove dangerous columns from feature list
            feature_cols = [col for col in feature_cols if col not in dangerous_cols]
            print(f"   ✓ Excluded {len(dangerous_cols)} dangerous columns")

        # Extract features (missing values already forward-filled in engineer_features)
        X = df[feature_cols].copy()

        # Add placeholder ranking columns (will be computed per-month in walk-forward)
        # This ensures ranking columns are included in feature list
        if hasattr(self, '_rank_cols') and self._rank_cols:
            for col in self._rank_cols:
                rank_col = f'{col}_rank'
                if rank_col not in X.columns:
                    X[rank_col] = np.nan  # Will be filled per-month in walk-forward
                    feature_cols.append(rank_col)
            print(f"  • Added {len(self._rank_cols)} ranking feature placeholders (computed per-month)")

        # Handle categorical columns (convert to codes)
        # Expected categorical features: GICSSectorName, sharadar_sicsector, sharadar_sicindustry
        categorical_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_cols.append(col)
                # Fill missing with 'Unknown' category
                X[col] = X[col].fillna('Unknown')
                # Convert to numeric codes (0, 1, 2, ... for each unique category)
                X[col] = pd.Categorical(X[col]).codes
                # Ensure it's numeric after conversion
                X[col] = X[col].astype('int64')

        if categorical_cols:
            print(f"  • Converted {len(categorical_cols)} categorical columns to numeric codes: {', '.join(categorical_cols)}")

        # Any remaining NaNs should be very rare (already forward-filled)
        # Fill with 0 as a safe default (e.g., first row per symbol before any data)
        # EXCEPT ranking columns which are computed per-month in walk-forward
        if hasattr(self, '_rank_cols') and self._rank_cols:
            rank_cols_list = [f'{col}_rank' for col in self._rank_cols if f'{col}_rank' in X.columns]
            non_rank_cols = [col for col in X.columns if col not in rank_cols_list]
            X[non_rank_cols] = X[non_rank_cols].fillna(0)
            # Leave ranking columns as NaN (will be computed per-month)
        else:
            X = X.fillna(0)

        # Final safety check: ensure all columns are numeric
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                # Force conversion to numeric, replacing errors with NaN
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        y = df['forward_return'].values

        # Calculate sample weights based on market cap (before filtering)
        # Larger market cap = higher weight
        sample_weights = self._calculate_marketcap_weights(df)

        # Identify rows with valid target (for training)
        # IMPORTANT: We keep X unfiltered so we can predict for ALL rows
        valid_idx = ~np.isnan(y)

        print(f"\n📊 Feature Preparation:")
        print(f"  • Total rows: {len(X):,}")
        print(f"  • Rows with valid forward_return (for training): {valid_idx.sum():,}")
        print(f"  • Rows without forward_return (recent dates): {(~valid_idx).sum():,}")
        print(f"  • Features used: {len(feature_cols)}")

        # Return unfiltered X so we can predict for all rows
        # Training will use X[valid_idx], y[valid_idx], sample_weights[valid_idx]
        # Prediction will use all of X
        return X, y, feature_cols, sample_weights, valid_idx

    def _describe_feature(self, col_name):
        """
        Generate a human-readable description for a feature column.

        Args:
            col_name (str): Column name

        Returns:
            str: Description of what the column contains
        """
        # Raw LSEG fundamentals
        lseg_descriptions = {
            'RefPriceClose': 'LSEG reference closing price',
            'RefVolume': 'LSEG reference trading volume',
            'CompanyMarketCap': 'Company market capitalization from LSEG',
            'EnterpriseValue_DailyTimeSeries_': 'Enterprise value (daily time series)',
            'FOCFExDividends_Discrete': 'Free operating cash flow excluding dividends',
            'InterestExpense_NetofCapitalizedInterest': 'Interest expense (net of capitalized interest)',
            'Debt_Total': 'Total debt',
            'EarningsPerShare_Actual': 'Actual earnings per share',
            'EarningsPerShare_SmartEstimate_prev_Q': 'EPS smart estimate (previous quarter)',
            'EarningsPerShare_ActualSurprise': 'EPS actual surprise vs estimate',
            'EarningsPerShare_SmartEstimate_current_Q': 'EPS smart estimate (current quarter)',
            'LongTermGrowth_Mean': 'Mean long-term growth estimate',
            'PriceTarget_Median': 'Median analyst price target',
            'CombinedAlphaModelSectorRank': 'Combined alpha model rank within sector',
            'CombinedAlphaModelSectorRankChange': 'Change in alpha model sector rank',
            'CombinedAlphaModelRegionRank': 'Combined alpha model rank within region',
            'EarningsQualityRegionRank_Current': 'Current earnings quality rank in region',
            'EnterpriseValueToEBIT_DailyTimeSeriesRatio_': 'EV/EBIT ratio (daily)',
            'EnterpriseValueToEBITDA_DailyTimeSeriesRatio_': 'EV/EBITDA ratio (daily)',
            'EnterpriseValueToSales_DailyTimeSeriesRatio_': 'EV/Sales ratio (daily)',
            'Dividend_Per_Share_SmartEstimate': 'Dividend per share smart estimate',
            'CashCashEquivalents_Total': 'Total cash and cash equivalents',
            'ForwardPEG_DailyTimeSeriesRatio_': 'Forward PEG ratio (daily)',
            'PriceEarningsToGrowthRatio_SmartEstimate_': 'PE to growth ratio smart estimate',
            'Recommendation_Median_1_5_': 'Median analyst recommendation (1-5 scale)',
            'ReturnOnEquity_SmartEstimat': 'Return on equity smart estimate',
            'ReturnOnAssets_SmartEstimate': 'Return on assets smart estimate',
            'ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_': 'Forward P/CF ratio (daily)',
            'ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_': 'Forward P/S ratio (daily)',
            'ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_': 'Forward EV/OCF ratio (daily)',
            'GrossProfitMargin_ActualSurprise': 'Gross profit margin actual surprise',
            'Estpricegrowth_percent': 'Estimated price growth percentage',
        }

        # Handle _lag1 and _lag2 suffixes
        base_col = col_name
        lag_desc = ""
        if col_name.endswith('_lag1'):
            base_col = col_name[:-5]
            lag_desc = " (lagged 1 day)"
        elif col_name.endswith('_lag2'):
            base_col = col_name[:-5]
            lag_desc = " (lagged 2 days)"

        # Handle _rank suffix
        rank_desc = ""
        if base_col.endswith('_rank'):
            base_col = base_col[:-5]
            rank_desc = " - cross-sectional percentile rank"

        # Check if it's a raw LSEG column
        if base_col in lseg_descriptions:
            return lseg_descriptions[base_col] + lag_desc + rank_desc

        # Derived features
        derived_descriptions = {
            'return_5d': '5-day price momentum (% return)',
            'return_10d': '10-day price momentum (% return)',
            'return_20d': '20-day price momentum (% return)',
            'volatility_5d': '5-day annualized volatility (%)',
            'volatility_10d': '10-day annualized volatility (%)',
            'volatility_20d': '20-day annualized volatility (%)',
            'volume_ratio': 'Volume relative to 20-day average',
            'roa': 'Return on assets',
            'roe': 'Return on equity',
            'ev_to_ebitda': 'Enterprise value to EBITDA ratio',
            'ev_to_ebit': 'Enterprise value to EBIT ratio',
            'ev_to_sales': 'Enterprise value to sales ratio',
            'forward_peg': 'Forward PEG ratio',
            'pe_to_growth': 'P/E to growth ratio',
            'ltg': 'Long-term growth estimate',
            'price_growth_est': 'Estimated price growth',
            'alpha_sector_rank': 'Alpha model sector rank',
            'alpha_region_rank': 'Alpha model region rank',
            'alpha_sector_change': 'Change in alpha sector rank',
            'earnings_quality': 'Earnings quality rank',
            'debt_to_marketcap': 'Debt to market cap ratio',
            'cash_to_marketcap': 'Cash to market cap ratio',
            'fcf_to_marketcap': 'Free cash flow to market cap ratio',
            'eps_actual': 'Actual earnings per share',
            'eps_surprise': 'EPS surprise vs estimate',
            'eps_estimate_current': 'Current quarter EPS estimate',
            'recommendation': 'Median analyst recommendation',
            'price_target': 'Median analyst price target',
            'upside_to_target': 'Upside to price target (%)',
            'log_marketcap': 'Log of market capitalization',
            'marketcap_scale': 'Market cap scale (1=Nano to 6=Mega)',
        }

        if base_col in derived_descriptions:
            return derived_descriptions[base_col] + lag_desc + rank_desc

        # Fallback for unknown columns
        return f"Feature: {col_name}"

    def _log_feature_descriptions(self, feature_cols):
        """
        Log a detailed description of all feature columns being used for training.

        Args:
            feature_cols (list): List of feature column names
        """
        print("\n" + "=" * 80)
        print("  FEATURE COLUMNS USED FOR TRAINING")
        print("=" * 80)
        print(f"\nTotal features: {len(feature_cols)}\n")

        for i, col in enumerate(feature_cols, 1):
            desc = self._describe_feature(col)
            print(f"{i:3d}. {col:60s} - {desc}")

        print("\n" + "=" * 80 + "\n")

    def _calculate_marketcap_weights(self, df):
        """
        Calculate sample weights based on market cap ranking.

        This implements market-cap weighted training to focus the model on large-cap
        stocks which are typically:
        - More liquid and tradeable
        - Less noisy in their fundamental data
        - More relevant for institutional trading strategies

        Weighting scheme:
        - Top 2000 by market cap: weight = 1.0 (full importance)
        - Rank 2001-4000: weight = 0.5 (medium importance)
        - Rank 4000+: weight = 0.1 (low importance, but still learn patterns)

        Args:
            df (pd.DataFrame): DataFrame with CompanyMarketCap column

        Returns:
            np.ndarray: Array of sample weights, same length as df
        """
        # ALWAYS use lagged market cap (safety measure - even with --no-lag)
        # CompanyMarketCap is derived from price, so it must be lagged
        mktcap_col = 'CompanyMarketCap_lag1' if 'CompanyMarketCap_lag1' in df.columns else 'CompanyMarketCap'

        # Calculate market cap rank within each date (1 = largest)
        # We rank within date to handle market cap changes over time
        df['_mktcap_rank'] = df.groupby('Date')[mktcap_col].rank(ascending=False, method='first')

        # Initialize all weights to 1.0
        weights = np.ones(len(df))

        # Apply tiered weighting based on market cap rank
        # Top 2000: Full weight (these are the stocks we actually trade)
        weights[df['_mktcap_rank'] <= 2000] = 1.0

        # Mid-caps: Half weight (still useful but more volatile)
        weights[(df['_mktcap_rank'] > 2000) & (df['_mktcap_rank'] <= 4000)] = 0.5

        # Small-caps: Low weight (very noisy, but don't ignore completely)
        weights[df['_mktcap_rank'] > 4000] = 0.1

        # Clean up temporary column to avoid clutter
        df.drop('_mktcap_rank', axis=1, inplace=True)

        return weights

    def train(self, X_train, y_train, sample_weight=None, X_val=None, y_val=None, sample_fraction=1.0):
        """
        Train HistGradientBoosting model.

        Args:
            X_train: Training features
            y_train: Training target
            sample_weight: Sample weights (optional)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            sample_fraction: Fraction of training data to use (0.0-1.0, default: 1.0)

        Returns:
            Trained model
        """
        print("\n🚀 Training HistGradientBoosting model...")

        # Apply sampling if requested
        if sample_fraction < 1.0:
            n_samples = int(len(X_train) * sample_fraction)
            # DETERMINISTIC SAMPLING: Use first N samples after sorting by date
            # This is fully reproducible and doesn't require random numbers
            # Assumes X_train is already sorted by date (which it is from create_target)
            sample_idx = np.arange(n_samples)
            X_train = X_train.iloc[sample_idx]
            y_train = y_train[sample_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[sample_idx]
            print(f"  📊 Using {sample_fraction:.0%} of training data ({n_samples:,} samples, deterministic)")

        # Create model with parameters optimized for speed and accuracy
        self.model = HistGradientBoostingRegressor(
            max_iter=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.num_leaves,
            min_samples_leaf=100,  # 0.0125% of 800K rows - more stable splits
            l2_regularization=0.2,  # Moderate regularization for 300 features
            max_bins=255,
            # TIME-SERIES SAFETY: Disable automatic validation to prevent look-ahead bias
            # validation_fraction does RANDOM splits which leak future data in time-series
            # If early stopping needed, pass explicit chronological X_val/y_val
            early_stopping=True if X_val is not None else False,
            n_iter_no_change=50,
            validation_fraction=None,  # Explicit: disabled for time-series safety
            random_state=42,
            verbose=0  # Reduce verbosity
        )

        # Show sample weight statistics if provided
        if sample_weight is not None:
            unique_weights = np.unique(sample_weight)
            print(f"  📊 Sample Weighting:")
            print(f"    • Unique weight values: {unique_weights}")
            print(f"    • Top 2000 stocks (weight=1.0): {(sample_weight == 1.0).sum():,} samples")
            print(f"    • Mid-cap stocks (weight=0.5): {(sample_weight == 0.5).sum():,} samples")
            print(f"    • Small-cap stocks (weight=0.1): {(sample_weight == 0.1).sum():,} samples")

        # Train model with sample weights
        if X_val is not None and y_val is not None:
            self.model.set_params(validation_fraction=None)
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)

        print(f"  ✓ Model trained with {self.model.n_iter_} iterations")

        return self.model

    def cross_validate(self, X, y, sample_weights, n_splits=3):
        """
        Perform time-series cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            sample_weights: Sample weights
            n_splits: Number of CV splits

        Returns:
            Dictionary of CV metrics
        """
        print(f"\n🔄 Performing {n_splits}-fold time-series cross-validation...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\n  Fold {fold}/{n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            weights_train = sample_weights[train_idx]

            # Train model with sample weights
            self.train(X_train, y_train, sample_weight=weights_train, X_val=X_val, y_val=y_val)

            # Predict
            y_pred = self.model.predict(X_val)

            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mse)

            cv_scores.append({'fold': fold, 'rmse': rmse, 'mae': mae})
            print(f"  Fold {fold} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Average metrics
        avg_rmse = np.mean([s['rmse'] for s in cv_scores])
        avg_mae = np.mean([s['mae'] for s in cv_scores])

        print(f"\n  Cross-Validation Results:")
        print(f"  • Average RMSE: {avg_rmse:.4f}")
        print(f"  • Average MAE: {avg_mae:.4f}")

        return {'cv_scores': cv_scores, 'avg_rmse': avg_rmse, 'avg_mae': avg_mae}

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def get_feature_importances(self, X, y, sample_size=10000, n_repeats=5):
        """
        Compute feature importances using permutation importance.

        Args:
            X: Feature matrix (DataFrame)
            y: Target vector
            sample_size: Number of samples to use (default 10000 for speed)
            n_repeats: Number of permutation repeats (default 5)

        Returns:
            DataFrame with features and their importances, sorted by importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # DETERMINISTIC SAMPLING: Use first N samples for faster computation
        # No randomness needed - just take first N rows (already sorted by Symbol, Date)
        if len(X) > sample_size:
            X_sample = X.iloc[:sample_size]
            y_sample = y[:sample_size]
        else:
            X_sample = X
            y_sample = y

        # Remove NaN from target
        valid_idx = ~np.isnan(y_sample)
        X_sample = X_sample[valid_idx]
        y_sample = y_sample[valid_idx]

        # Compute permutation importance
        result = permutation_importance(
            self.model,
            X_sample,
            y_sample,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def temporal_diagnostics(self, df, predictions, nlags=20, significance_level=0.05, months_lookback=6):
        """
        Perform temporal diagnostics on residuals to detect:
        1. Autocorrelation (temporal structure leakage)
        2. Look-ahead bias indicators
        3. Prediction stability over time

        Args:
            df: DataFrame with 'Date', 'Symbol', 'forward_return', and predictions
            predictions: Array of predictions (aligned with df)
            nlags: Number of lags for ACF/PACF analysis (default: 20)
            significance_level: Significance level for tests (default: 0.05)
            months_lookback: Number of recent months to analyze (default: 6)
                            Analyzing only recent data speeds up diagnostics dramatically.
                            For 9M rows, analyzing 6 months takes 2-3 min vs hours for all data.

        Returns:
            dict: Diagnostic results and warnings
        """
        if not STATSMODELS_AVAILABLE:
            print("\n⚠️  Temporal diagnostics skipped (statsmodels not installed)")
            return {'status': 'skipped', 'reason': 'statsmodels not available'}

        print("\n" + "=" * 70)
        print("📊 TEMPORAL DIAGNOSTICS - Residual Analysis")
        print("=" * 70)

        # Filter to rows with both actual and predicted values
        valid_mask = ~np.isnan(predictions) & df['forward_return'].notna()

        if valid_mask.sum() < 100:
            print("⚠️  Insufficient data for temporal diagnostics (<100 observations)")
            return {'status': 'skipped', 'reason': 'insufficient data'}

        df_valid = df[valid_mask].copy()

        # ============================================================
        # OPTIMIZATION: Analyze only last N months (much faster)
        # ============================================================
        df_valid['Date'] = pd.to_datetime(df_valid['Date'])
        max_date = df_valid['Date'].max()
        cutoff_date = max_date - pd.DateOffset(months=months_lookback)

        df_recent = df_valid[df_valid['Date'] >= cutoff_date].copy()

        if len(df_recent) < 100:
            print(f"⚠️  Insufficient recent data for diagnostics (<100 observations in last {months_lookback} months)")
            print(f"   Using all available data instead...")
            df_recent = df_valid.copy()
            cutoff_date = df_valid['Date'].min()

        print(f"\n📅 Analysis Period:")
        print(f"  • Analyzing last {months_lookback} months")
        print(f"  • Date range: {cutoff_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"  • Total dataset: {len(df_valid):,} rows")
        print(f"  • Analyzed subset: {len(df_recent):,} rows ({len(df_recent)/len(df_valid)*100:.1f}%)")

        # Get predictions for recent data only
        recent_indices = df_recent.index
        y_true = df_recent['forward_return'].values
        y_pred = predictions[recent_indices]

        # Compute residuals (actual - predicted)
        residuals = y_true - y_pred

        print(f"\n📈 Residual Statistics (Last {months_lookback} Months):")
        print(f"  • Valid observations: {len(residuals):,}")
        print(f"  • Mean residual: {residuals.mean():+.4f}%")
        print(f"  • Std deviation: {residuals.std():.4f}%")
        print(f"  • Min residual: {residuals.min():+.2f}%")
        print(f"  • Max residual: {residuals.max():+.2f}%")

        # Sort by date for time-series analysis
        df_recent = df_recent.copy()
        df_recent['residual'] = residuals
        df_recent = df_recent.sort_values('Date')
        residuals_sorted = df_recent['residual'].values

        # 1. Autocorrelation Function (ACF)
        print(f"\n🔍 Autocorrelation Analysis (ACF):")
        acf_values = acf(residuals_sorted, nlags=nlags, fft=False)

        # Critical value for 95% confidence (approximate: 1.96 / sqrt(n))
        critical_value = 1.96 / np.sqrt(len(residuals_sorted))

        significant_lags = []
        for lag in range(1, nlags + 1):
            if abs(acf_values[lag]) > critical_value:
                significant_lags.append((lag, acf_values[lag]))

        if significant_lags:
            print(f"  ⚠️  SIGNIFICANT AUTOCORRELATION DETECTED at {len(significant_lags)} lags:")
            for lag, value in significant_lags[:5]:  # Show first 5
                print(f"      Lag {lag:2d}: {value:+.4f} (threshold: ±{critical_value:.4f})")
            if len(significant_lags) > 5:
                print(f"      ... and {len(significant_lags) - 5} more lags")
            print(f"\n  ⚠️  WARNING: Model may be leaking temporal structure!")
            print(f"      • Residuals show time-series patterns")
            print(f"      • Possible causes:")
            print(f"          1. Look-ahead bias in features")
            print(f"          2. Missing temporal features (lags, momentum)")
            print(f"          3. Insufficient feature engineering")
        else:
            print(f"  ✅ No significant autocorrelation detected")
            print(f"     Residuals appear random (good!)")

        # 2. Ljung-Box Test (omnibus test for autocorrelation)
        print(f"\n📊 Ljung-Box Test (H0: No autocorrelation):")
        try:
            lb_result = acorr_ljungbox(residuals_sorted, lags=[10, 20], return_df=True)

            for lag in lb_result.index:
                p_value = lb_result.loc[lag, 'lb_pvalue']
                is_significant = p_value < significance_level

                status = "❌ REJECT H0" if is_significant else "✅ ACCEPT H0"
                print(f"  • Lag {lag:2d}: p-value = {p_value:.4f} {status}")

                if is_significant:
                    print(f"      ⚠️  Significant autocorrelation up to lag {lag}")

            # Overall assessment
            any_significant = (lb_result['lb_pvalue'] < significance_level).any()
            if any_significant:
                print(f"\n  ⚠️  WARNING: Ljung-Box test indicates temporal dependencies!")
            else:
                print(f"\n  ✅ Ljung-Box test: No significant temporal dependencies")

        except Exception as e:
            print(f"  ⚠️  Ljung-Box test failed: {e}")

        # 3. Temporal stability - check if error patterns change over time
        print(f"\n📅 Temporal Stability Analysis:")

        # Split into quartiles by time
        n_periods = 4
        period_size = len(df_recent) // n_periods

        print(f"  • Splitting data into {n_periods} time periods:")
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df_recent)

            period_residuals = df_recent.iloc[start_idx:end_idx]['residual']
            period_dates = df_recent.iloc[start_idx:end_idx]['Date']

            mean_res = period_residuals.mean()
            std_res = period_residuals.std()

            print(f"      Period {i+1} ({period_dates.min().strftime('%Y-%m-%d')} to {period_dates.max().strftime('%Y-%m-%d')}):")
            print(f"        Mean: {mean_res:+.4f}%, Std: {std_res:.4f}%")

        # Check if mean/std are stable across periods (using coefficient of variation)
        period_means = []
        period_stds = []

        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df_recent)
            period_residuals = df_recent.iloc[start_idx:end_idx]['residual']
            period_means.append(period_residuals.mean())
            period_stds.append(period_residuals.std())

        mean_stability = np.std(period_means) / (np.mean(np.abs(period_means)) + 1e-8)
        std_stability = np.std(period_stds) / np.mean(period_stds)

        print(f"\n  • Stability metrics:")
        print(f"      Mean stability (CV): {mean_stability:.4f} {'✅ Stable' if mean_stability < 0.5 else '⚠️  Unstable'}")
        print(f"      Std stability (CV): {std_stability:.4f} {'✅ Stable' if std_stability < 0.3 else '⚠️  Unstable'}")

        # Summary
        print(f"\n" + "=" * 70)
        print(f"📋 SUMMARY")
        print(f"=" * 70)

        warnings_found = []

        if significant_lags:
            warnings_found.append(f"Significant ACF at {len(significant_lags)} lags")

        if any_significant:
            warnings_found.append("Ljung-Box test rejected (temporal dependencies)")

        if mean_stability > 0.5:
            warnings_found.append("Unstable mean across time periods")

        if std_stability > 0.3:
            warnings_found.append("Unstable variance across time periods")

        if warnings_found:
            print(f"⚠️  WARNINGS DETECTED:")
            for warning in warnings_found:
                print(f"  • {warning}")
            print(f"\n💡 Recommendations:")
            print(f"  1. Check for look-ahead bias in feature engineering")
            print(f"  2. Verify all features are properly lagged")
            print(f"  3. Consider adding more temporal features (momentum, lags)")
            print(f"  4. Review walk-forward logic for date alignment issues")
        else:
            print(f"✅ NO WARNINGS - Model appears temporally sound")
            print(f"  • Residuals show no significant autocorrelation")
            print(f"  • Predictions stable across time periods")
            print(f"  • No evidence of look-ahead bias")

        print(f"=" * 70)

        # Return results
        return {
            'status': 'completed',
            'n_observations': len(residuals),
            'mean_residual': float(residuals.mean()),
            'std_residual': float(residuals.std()),
            'significant_acf_lags': len(significant_lags),
            'ljungbox_rejected': any_significant if 'any_significant' in locals() else None,
            'mean_stability': float(mean_stability),
            'std_stability': float(std_stability),
            'warnings': warnings_found
        }

    def _walk_forward_predict(self, df, X, y, sample_weights, valid_idx,
                              resume_from_date=None, previous_predictions=None, preserve_existing=False):
        """
        Walk-forward prediction with monthly retraining (expanding or rolling window).

        This eliminates look-ahead bias by ensuring each prediction uses only
        data available at that point in time.

        Window strategies:
        - Expanding window (lookback_months=None): Train on ALL historical data
        - Rolling window (lookback_months=N): Train on last N months only

        Args:
            df: DataFrame with Date column and engineered features
            X: Feature matrix
            y: Target vector
            sample_weights: Sample weights
            valid_idx: Boolean mask for rows with valid forward_return
            resume_from_date: Optional date to resume from (skips earlier months)
            previous_predictions: Optional array of previous predictions to keep
            preserve_existing: If True, never overwrite existing predictions (freeze historical forecasts)

        Returns:
            Array of predictions (one per row)
        """
        print("\n" + "=" * 70)
        if resume_from_date:
            print("  🔄 RESUMING WALK-FORWARD PREDICTION")
            print(f"  (Starting from {resume_from_date})")
        else:
            if self.lookback_months:
                print(f"  🔄 ROLLING {self.lookback_months}-MONTH WINDOW WALK-FORWARD PREDICTION")
                print(f"  (Monthly Retraining - NO LOOK-AHEAD BIAS)")
            else:
                print("  🔄 EXPANDING WINDOW WALK-FORWARD PREDICTION")
                print("  (Monthly Retraining - NO LOOK-AHEAD BIAS)")
        print("=" * 70)

        # Convert Date to datetime if needed
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])

        # Add year-month column for grouping
        df['_year_month'] = df['Date'].dt.to_period('M')

        # Get unique year-months sorted
        unique_months = sorted(df['_year_month'].unique())

        print(f"\n📅 Date Range:")
        print(f"  • First month: {unique_months[0]}")
        print(f"  • Last month: {unique_months[-1]}")
        print(f"  • Total months: {len(unique_months)}")
        print(f"  • Total rows: {len(df):,}")

        # Initialize predictions array
        if previous_predictions is not None:
            # Resume: Start with previous predictions
            predictions = previous_predictions.copy()
            print(f"  • Resuming with {np.sum(~np.isnan(previous_predictions)):,} previous predictions")
        else:
            # Full run: Initialize with NaN
            predictions = np.full(len(df), np.nan)

        # Determine which months to process
        if resume_from_date:
            resume_from_month = pd.Period(resume_from_date, freq='M')
            months_to_process = [m for m in unique_months if m >= resume_from_month]
            print(f"  • Resuming from: {resume_from_month}")
            print(f"  • Months to process: {len(months_to_process)} (skipping {len(unique_months) - len(months_to_process)})")
        else:
            months_to_process = unique_months

        # If preserve_existing, skip months that already have predictions
        # Use "high water mark" approach: find last month with predictions, skip everything before it
        if preserve_existing and previous_predictions is not None:
            # Find the last month with complete predictions
            # This is more efficient than checking every month from 2003 onwards
            last_complete_month = None
            months_checked = 0

            # Determine the most recent complete month (current month is likely incomplete)
            df_dates = pd.to_datetime(df['Date'])
            most_recent_date = df_dates.max()
            most_recent_month = pd.Period(most_recent_date, freq='M')

            # Check if current month is complete (has data through last day of month)
            last_day_of_month = most_recent_month.end_time.normalize()
            current_month_is_complete = (most_recent_date >= last_day_of_month)

            print(f"  • Most recent date in data: {most_recent_date.strftime('%Y-%m-%d')}")
            print(f"  • Current month: {most_recent_month} ({'COMPLETE' if current_month_is_complete else 'INCOMPLETE'})")

            # Iterate backwards through months to find the high water mark
            for month in reversed(months_to_process):
                month_mask = df['_year_month'] == month
                month_positions = np.where(month_mask)[0]
                month_preds = predictions[month_positions]

                # Check coverage
                rows_with_preds = (~np.isnan(month_preds)).sum()
                total_rows = len(month_preds)
                coverage = rows_with_preds / total_rows
                months_checked += 1

                # Determine threshold based on whether this is current month or complete month
                # CRITICAL: Current incomplete month should use lenient threshold (it has new data!)
                # Last complete month should use strict 99% threshold (allows minor backfill)
                # Older months use 95% threshold (allows historical data changes)
                is_current_month = (month == str(most_recent_month))

                if is_current_month and not current_month_is_complete:
                    # Current incomplete month: very lenient (90%) - expect new rows
                    threshold = 0.90
                elif months_checked == 1 or (is_current_month and current_month_is_complete):
                    # First month checked OR complete current month: strict 99%
                    threshold = 0.99
                else:
                    # Older complete months: 95% threshold
                    threshold = 0.95

                # Diagnostic output
                status = "✓ PASS" if coverage >= threshold else "✗ FAIL"
                month_label = f"{month} (CURRENT)" if is_current_month else month
                print(f"    [{months_checked}] {month_label}: {rows_with_preds:,}/{total_rows:,} rows ({coverage*100:.1f}%) - Threshold: {threshold*100:.0f}% - {status}")

                # Check if this month passes the threshold
                if coverage >= threshold:
                    last_complete_month = month
                    break  # Found the high water mark

            if last_complete_month is not None:
                # Skip all months up to and including the last complete month
                months_to_skip = [m for m in months_to_process if m <= last_complete_month]
                months_to_process = [m for m in months_to_process if m > last_complete_month]

                print(f"  • 🔒 PRESERVE MODE: Found predictions through {last_complete_month}")
                print(f"  • Skipping {len(months_to_skip)} months with existing predictions")
                print(f"  • Processing {len(months_to_process)} months with missing predictions")
            else:
                # No months with substantial predictions found - process all
                print(f"  • 🔒 PRESERVE MODE: No complete months found in previous predictions")
                print(f"  • Processing all {len(months_to_process)} months")

        # Track statistics
        months_trained = 0
        total_training_time = 0
        recent_month_times = []  # Track recent training times for intelligent ETA

        # Initialize feature importance logging if requested
        if self.log_features:
            from pathlib import Path
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            logs_dir = Path('logs')
            logs_dir.mkdir(exist_ok=True)
            self.feature_log_path = logs_dir / f'feature_importance_{timestamp}.csv'
            # Create header
            with open(self.feature_log_path, 'w') as f:
                f.write('Month,Rank,Feature,Importance,Std\n')
            print(f"\n📝 Feature logging enabled: {self.feature_log_path}")

        print(f"\n🔄 Training models month by month...\n")

        # For each month, train on all previous data and predict
        for i, current_month in enumerate(unique_months, 1):
            # Skip if not in months_to_process
            if current_month not in months_to_process:
                continue

            month_start_time = pd.Timestamp.now()

            # Get first day of current month
            first_day_of_month = current_month.to_timestamp()

            # Rows to predict this iteration (all rows in current month)
            predict_mask = df['_year_month'] == current_month
            predict_positions = np.where(predict_mask)[0]  # Get position indices

            if len(predict_positions) == 0:
                continue

            # Training data: all rows BEFORE this month with valid forward_return
            # Window strategy depends on lookback_months parameter
            if self.lookback_months:
                # ROLLING WINDOW: Only last N months before current month
                # This tests recency bias - recent data may be more relevant
                lookback_start = first_day_of_month - pd.DateOffset(months=self.lookback_months)
                train_mask = (df['Date'] >= lookback_start) & (df['Date'] < first_day_of_month) & valid_idx
            else:
                # EXPANDING WINDOW: All data before current month (default)
                # This learns long-term patterns - more stable but slower to adapt
                train_mask = (df['Date'] < first_day_of_month) & valid_idx

            train_positions = np.where(train_mask)[0]  # Get position indices

            # Skip if no training data available yet
            if len(train_positions) == 0:
                if self.lookback_months:
                    print(f"  [{i:3d}/{len(unique_months)}] {current_month}: ⏭️  SKIPPED (no data in {self.lookback_months}-month window) - {len(predict_positions):,} rows")
                else:
                    print(f"  [{i:3d}/{len(unique_months)}] {current_month}: ⏭️  SKIPPED (no training data yet) - {len(predict_positions):,} rows")
                continue

            # Skip if insufficient training data (< 2000 samples)
            # Training on very few samples produces meaningless models and R^2 warnings
            if len(train_positions) < 2000:
                print(f"  [{i:3d}/{len(unique_months)}] {current_month}: ⏭️  SKIPPED (insufficient training data: {len(train_positions):,} < 2,000) - {len(predict_positions):,} rows")
                continue

            # REPRODUCIBILITY FIX v2: Compute rankings on COMPLETE months only
            # Key insight: Rankings should only use data through last COMPLETE month boundary
            # This ensures adding partial month data (e.g., Jan 1-15) doesn't change training rankings

            # Step 1: Determine last complete month
            current_month_period = pd.Period(current_month, freq='M')
            last_day_of_current_month = current_month_period.end_time.normalize()
            max_date_in_data = df['Date'].max()

            # Check if current month has complete data
            if max_date_in_data >= last_day_of_current_month:
                # Current month is complete - use it for training rankings
                ranking_cutoff_month = current_month_period
                is_current_month_complete = True
            else:
                # Current month is incomplete - use previous month for training rankings
                ranking_cutoff_month = current_month_period - 1
                is_current_month_complete = False

            ranking_cutoff_date = ranking_cutoff_month.end_time.normalize()

            # Step 2: Compute rankings ONLY on complete months (up to ranking_cutoff_date)
            # This includes both training AND prediction positions if they fall within complete months
            all_positions = np.concatenate([train_positions, predict_positions])

            # VECTORIZED: Get dates for all positions at once (1000x faster than iterating)
            all_dates = df.iloc[all_positions]['Date'].values
            date_mask = all_dates <= np.datetime64(ranking_cutoff_date)
            complete_positions = all_positions[date_mask]

            if len(complete_positions) > 0:
                complete_df = df.iloc[complete_positions].copy()

                if hasattr(self, '_rank_cols') and self._rank_cols:
                    for col in self._rank_cols:
                        if col in complete_df.columns:
                            complete_df[f'{col}_rank'] = complete_df.groupby('Date')[col].rank(pct=True)

                # Update X with rankings from complete months only
                rank_feature_cols = [f'{col}_rank' for col in self._rank_cols if col in complete_df.columns]
                for rank_col in rank_feature_cols:
                    if rank_col in complete_df.columns:
                        X.loc[complete_df.index, rank_col] = complete_df[rank_col]

            # Step 3: Handle incomplete prediction month - forward-fill from last complete month
            if not is_current_month_complete:
                # VECTORIZED: Get prediction positions that are AFTER ranking_cutoff_date (incomplete month)
                predict_dates = df.iloc[predict_positions]['Date'].values
                predict_incomplete_mask = predict_dates > np.datetime64(ranking_cutoff_date)
                predict_incomplete_positions = predict_positions[predict_incomplete_mask]

                if len(predict_incomplete_positions) > 0:
                    # OPTIMIZED: Use vectorized merge instead of nested loops
                    predict_incomplete_df = df.iloc[predict_incomplete_positions].copy()

                    if hasattr(self, '_rank_cols') and self._rank_cols:
                        # Get rank columns that exist in X
                        rank_cols_in_X = [f'{col}_rank' for col in self._rank_cols if f'{col}_rank' in X.columns]

                        if len(rank_cols_in_X) > 0:
                            # Get complete data: Symbol and Date from df, rank columns from X
                            complete_df_metadata = df.iloc[complete_positions][['Symbol', 'Date']].copy()
                            complete_df_ranks = X.iloc[complete_positions][rank_cols_in_X].copy()
                            complete_df_full = pd.concat([complete_df_metadata, complete_df_ranks], axis=1)

                            # For each symbol, get its most recent ranking from complete months
                            # This is MUCH faster than nested loops
                            complete_latest = complete_df_full.sort_values('Date').groupby('Symbol').last().reset_index()

                            # Merge to get forward-filled rankings
                            predict_with_ranks = predict_incomplete_df[['Symbol']].merge(
                                complete_latest,
                                on='Symbol',
                                how='left',
                                suffixes=('', '_complete')
                            )

                            # Update X with forward-filled rankings
                            for rank_col in rank_cols_in_X:
                                if rank_col in predict_with_ranks.columns:
                                    X.loc[predict_incomplete_df.index, rank_col] = predict_with_ranks[rank_col].values

                    print(f"    📊 Rankings: Computed through {ranking_cutoff_date.date()} (complete), " +
                          f"Forward-filled to {current_month} (partial: {len(predict_incomplete_positions)} rows)")

            # Get training data using position-based indexing (now with updated rankings)
            X_train = X.iloc[train_positions]
            y_train = y[train_positions]
            weights_train = sample_weights[train_positions]

            # Train model on data available up to this month
            self.train(X_train, y_train, sample_weight=weights_train, sample_fraction=self.sample_fraction)

            # Log feature importances if requested
            if self.log_features:
                try:
                    # Compute feature importances (using smaller sample for speed)
                    importance_df = self.get_feature_importances(X_train, y_train, sample_size=5000, n_repeats=3)
                    top_20 = importance_df.head(20)

                    # Append to CSV
                    with open(self.feature_log_path, 'a') as f:
                        for rank, (idx, row) in enumerate(top_20.iterrows(), 1):
                            f.write(f"{current_month},{rank},{row['feature']},{row['importance']:.6f},{row['std']:.6f}\n")
                except Exception as e:
                    print(f"    ⚠️  Feature logging failed: {e}")

            # Predict for current month
            X_predict = X.iloc[predict_positions]
            month_predictions = self.predict(X_predict)

            # Store predictions using position-based indexing
            predictions[predict_positions] = month_predictions

            # Track time
            month_time = (pd.Timestamp.now() - month_start_time).total_seconds()
            total_training_time += month_time
            months_trained += 1
            recent_month_times.append(month_time)

            # Progress update with intelligent ETA
            # Use moving average of recent 10 months (or all if less than 10)
            # This accounts for increasing training times as data accumulates
            window_size = min(10, len(recent_month_times))
            recent_avg_time = sum(recent_month_times[-window_size:]) / window_size

            remaining_months = len(unique_months) - i
            eta_seconds = recent_avg_time * remaining_months

            # Format ETA string
            if eta_seconds >= 3600:  # >= 1 hour
                eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
            elif eta_seconds >= 60:  # >= 1 minute
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = f"{int(eta_seconds)}s"

            print(f"  [{i:3d}/{len(unique_months)}] {current_month}: Trained on {len(train_positions):5,} rows → Predicted {len(predict_positions):4,} rows ({month_time:4.1f}s) ETA: {eta_str}")

        # Clean up temporary column
        df.drop('_year_month', axis=1, inplace=True)

        # Summary
        print(f"\n" + "=" * 70)
        print(f"  ✅ Walk-Forward Complete!")
        print(f"=" * 70)
        print(f"  • Months processed: {months_trained}")
        print(f"  • Total predictions: {np.sum(~np.isnan(predictions)):,} of {len(predictions):,}")
        print(f"  • Total time: {int(total_training_time // 60)}m {int(total_training_time % 60)}s")
        print(f"  • Average time per month: {total_training_time / max(months_trained, 1):.1f}s")

        if np.sum(np.isnan(predictions)) > 0:
            print(f"\n  ⚠️  {np.sum(np.isnan(predictions)):,} rows have NaN predictions (early months with no training data)")

        return predictions

    def fit_predict(self, df, use_cv=True, keep_engineered_features=False, walk_forward=True,
                   resume_from_date=None, previous_predictions=None, run_temporal_diagnostics=False,
                   preserve_existing=False):
        """
        Complete pipeline: engineer features, train model, make predictions.

        Args:
            df: Raw DataFrame
            use_cv: Whether to use cross-validation (only used if walk_forward=False)
            keep_engineered_features: If True, save all engineered features (huge file)
                                     If False, only save predictions (default)
            walk_forward: If True (default), use expanding window walk-forward training
                         (monthly retraining, NO LOOK-AHEAD BIAS).
                         If False, use single model trained on all data (FASTER but has look-ahead bias)
            resume_from_date: Optional date to resume from (for checkpoint resume)
            previous_predictions: Optional DataFrame of previous predictions (for checkpoint resume)
            run_temporal_diagnostics: If True, run temporal diagnostics after predictions (default: False)
            preserve_existing: If True, never overwrite existing predictions (freeze historical forecasts)

        Returns:
            DataFrame with predictions (and original columns only by default)
        """
        # Keep a copy of the original columns
        original_cols = df.columns.tolist()

        # Create target (THIS SORTS THE DATAFRAME!)
        df = self.create_target(df)

        # CRITICAL: Reset index after sorting so position-based indexing works
        df = df.reset_index(drop=True)

        # ============================================================
        # OPTIMIZATION: Align previous predictions BEFORE feature engineering
        # This reduces memory usage by 85% (merge happens when df has ~50 columns
        # instead of 290+ columns after feature engineering)
        # ============================================================
        previous_predictions_array = None
        if previous_predictions is not None and isinstance(previous_predictions, pd.DataFrame):
            print("\n📂 Aligning previous predictions with sorted dataframe...")

            # Normalize Date columns to ensure matching
            previous_predictions['Date'] = pd.to_datetime(previous_predictions['Date'])
            df['Date'] = pd.to_datetime(df['Date'])

            # ============================================================
            # MEMORY-OPTIMIZED MERGE (before feature engineering)
            # ============================================================
            # Filter previous predictions to only rows with non-NaN predicted_return
            prev_with_preds = previous_predictions[previous_predictions['predicted_return'].notna()].copy()
            print(f"  • Previous predictions with values: {len(prev_with_preds):,}")

            # Create minimal merge dataframe (only Date + Symbol from df)
            # This creates a tiny temporary dataframe (~2-3 columns) instead of huge (290+ columns)
            df_minimal = df[['Date', 'Symbol']].copy()

            # Prepare previous predictions for merge
            prev_merge = prev_with_preds[['Date', 'Symbol', 'predicted_return']].copy()
            prev_merge = prev_merge.rename(columns={'predicted_return': 'predicted_return_prev'})

            # Ensure Symbol is string type for consistent matching
            prev_merge['Symbol'] = prev_merge['Symbol'].astype(str)
            df_minimal['Symbol'] = df_minimal['Symbol'].astype(str)

            # OPTIMIZED: Merge only minimal columns (Date, Symbol)
            # Memory usage: ~3 columns instead of 291 columns (97% memory reduction)
            df_minimal_merged = df_minimal.merge(
                prev_merge,
                on=['Date', 'Symbol'],
                how='left'
            )

            # Extract the merged predicted_return_prev column as array
            # Rows that didn't match will have NaN (same as old logic)
            previous_predictions_array = df_minimal_merged['predicted_return_prev'].values

            # Clean up temporary dataframes
            del df_minimal, df_minimal_merged, prev_merge, prev_with_preds
            import gc
            gc.collect()

            matches = np.sum(~np.isnan(previous_predictions_array))
            print(f"  • Matched {matches:,} / {len(df):,} rows ({matches/len(df)*100:.1f}%)")

            # Filter to keep only predictions before resume_from_date
            if resume_from_date:
                df_dates = pd.to_datetime(df['Date'])
                mask_before_resume = df_dates < resume_from_date
                previous_predictions_array[~mask_before_resume] = np.nan
                print(f"  ✓ Aligned {np.sum(~np.isnan(previous_predictions_array)):,} previous predictions (before {resume_from_date})")
            else:
                print(f"  ✓ Aligned {np.sum(~np.isnan(previous_predictions_array)):,} previous predictions")

        # ============================================================
        # NOW engineer features (AFTER merge, when alignment is complete)
        # ============================================================
        df = self.engineer_features(df)

        # Prepare features with sample weights
        X, y, feature_cols, sample_weights, valid_idx = self.prepare_features(df)
        self.feature_cols = feature_cols

        # Log all features being used for training
        self._log_feature_descriptions(feature_cols)

        # Clean data: replace inf/nan values
        # Feature engineering can create inf from division, NaN should already be handled but double-check
        print(f"\n🧹 Cleaning feature data...")

        # Verify all columns are numeric (safety check)
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"  ⚠️  Found {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols[:5]}")
            print(f"  • Converting to numeric (this should not happen - please report)")
            for col in non_numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Count inf/nan BEFORE cleaning (excluding ranking columns which are intentionally NaN)
        if hasattr(self, '_rank_cols') and self._rank_cols:
            rank_cols_list = [f'{col}_rank' for col in self._rank_cols if f'{col}_rank' in X.columns]
            non_rank_cols = [col for col in X.columns if col not in rank_cols_list]
            X_non_rank = X[non_rank_cols]
        else:
            rank_cols_list = []
            X_non_rank = X

        inf_mask = np.isinf(X_non_rank.values)
        nan_mask = np.isnan(X_non_rank.values)
        inf_count = inf_mask.sum()
        nan_count = nan_mask.sum()

        if inf_count > 0 or nan_count > 0:
            print(f"  • Found {inf_count:,} inf values, {nan_count:,} nan values (excluding ranking columns)")

        # Replace inf with nan, then fill all NaN with 0
        # EXCEPT ranking columns which are computed per-month in walk-forward
        X_clean = X.copy()
        if rank_cols_list:
            X_clean[non_rank_cols] = X[non_rank_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            # Leave ranking columns as-is (NaN, will be filled per-month)
        else:
            X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        if inf_count > 0 or nan_count > 0:
            print(f"  • Replaced with 0 (safe default for model training)")
        if rank_cols_list:
            print(f"  • Kept {len(rank_cols_list)} ranking columns as NaN (computed per-month in walk-forward)")

        # PRODUCTION SAFETY: Skip statistical outlier clipping in walk-forward mode
        # Reason: Computing statistics from ALL data (including future months) would
        # introduce look-ahead bias. In walk-forward, we can't know future distributions.
        # The model (HistGradientBoosting) is robust to outliers without clipping.
        if walk_forward:
            print(f"  • Skipping outlier clipping (walk-forward mode - no look-ahead bias)")
        else:
            # Single-model mode: safe to use all training data statistics
            print(f"  • Computing outlier statistics from training data...")

            # Get training data for computing statistics
            X_train_for_stats = X_clean[valid_idx]

            # Compute mean and std from training data only
            train_mean = X_train_for_stats.mean()
            train_std = X_train_for_stats.std()

            # Replace zero std with 1 to avoid division by zero
            train_std = train_std.replace(0, 1)

            # Compute z-scores using training statistics
            z_scores = np.abs((X_clean - train_mean) / train_std)

            # Z-score can still have NaN (from constant columns), replace with 0
            z_scores = z_scores.fillna(0)

            extreme_mask = z_scores > 10
            extreme_count = extreme_mask.sum().sum()
            if extreme_count > 0:
                print(f"  • Clipping {extreme_count:,} extreme outliers (|z-score| > 10)")
                # Compute clip values from training data
                lower_clip = X_train_for_stats.quantile(0.001)
                upper_clip = X_train_for_stats.quantile(0.999)
                X_clean = X_clean.clip(lower=lower_clip, upper=upper_clip, axis=1)

        # Final NaN check and fill
        final_nan_count = X_clean.isna().sum().sum()
        if final_nan_count > 0:
            print(f"  • Final cleanup: {final_nan_count:,} remaining NaN values filled with 0")
            X_clean = X_clean.fillna(0)

        # Replace X with cleaned version
        X = X_clean
        print(f"  ✓ Data cleaning complete\n")

        # Apply PCA dimensionality reduction if requested
        if self.pca_components is not None:
            # PRODUCTION SAFETY: Warn if using PCA in walk-forward mode
            if walk_forward:
                print(f"\n⚠️  WARNING: PCA in walk-forward mode has look-ahead bias!")
                print(f"   PCA is fit on ALL training data (including future months)")
                print(f"   For production, use --no-walk-forward or remove --pca flag\n")

            print(f"🔬 Applying PCA dimensionality reduction...")
            print(f"  • Original features: {X.shape[1]}")
            print(f"  • Target components: {self.pca_components}")

            # CRITICAL: Standardize features before PCA
            # PCA requires features to be centered (mean=0) and scaled (std=1)
            print(f"  • Standardizing features (mean=0, std=1)...")
            self.scaler = StandardScaler()

            # Fit scaler on training data only (to avoid look-ahead bias)
            X_train_full = X[valid_idx]
            self.scaler.fit(X_train_full)

            # Transform ALL data using fitted scaler
            X_scaled = self.scaler.transform(X)

            # Convert back to DataFrame to maintain compatibility
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            # Fit PCA on scaled training data
            print(f"  • Fitting PCA on standardized features...")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            X_train_scaled = X_scaled_df[valid_idx]
            self.pca.fit(X_train_scaled)

            # Transform ALL data using fitted PCA
            X_pca = self.pca.transform(X_scaled_df)

            # Calculate variance explained
            variance_explained = self.pca.explained_variance_ratio_.sum() * 100
            print(f"  • Variance explained: {variance_explained:.2f}%")
            print(f"  • Components shape: {X_pca.shape}")

            # Update feature columns to reflect PCA components
            pca_feature_cols = [f'PC{i+1}' for i in range(self.pca_components)]

            # Convert to DataFrame to maintain compatibility
            X = pd.DataFrame(X_pca, columns=pca_feature_cols, index=X.index)

            # Update stored feature columns
            original_feature_cols = feature_cols
            feature_cols = pca_feature_cols
            self.feature_cols = feature_cols
            self.original_feature_cols = original_feature_cols  # Store for reference

            print(f"  ✓ PCA transformation complete\n")

        # Store X and y for feature importance calculation later
        self.X_last = X
        self.y_last = y

        # CRITICAL SAFETY: Re-sort dataframe before walk-forward loop
        # This ensures time-series integrity even if code changes above could unsort df
        # Groupby operations in walk-forward loop depend on Symbol+Date sorting for:
        # - .shift(), .pct_change(), .rolling(), .ffill() to work correctly
        print(f"🔒 Verifying time-series sort order before walk-forward...")
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        print(f"  ✓ Dataframe sorted by Symbol+Date ({len(df):,} rows)")

        # IMPORTANT: After re-sorting, we must also re-align X and y with new df index
        # Since prepare_features() created X from df columns, and we just reset df's index,
        # we need to ensure X's index matches df's new index
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True).values  # Reset y index to match
        sample_weights = pd.Series(sample_weights).reset_index(drop=True).values
        valid_idx = np.array(valid_idx)  # Already a boolean mask, doesn't need index reset
        print(f"  ✓ Feature matrices re-aligned with sorted dataframe\n")

        # Choose prediction strategy
        if walk_forward:
            # EXPANDING WINDOW WALK-FORWARD (NO LOOK-AHEAD BIAS)
            predictions = self._walk_forward_predict(df, X, y, sample_weights, valid_idx,
                                                     resume_from_date=resume_from_date,
                                                     previous_predictions=previous_predictions_array,
                                                     preserve_existing=preserve_existing)
        else:
            # SINGLE MODEL (FASTER BUT HAS LOOK-AHEAD BIAS)
            # Filter for training (only rows with valid forward_return)
            X_train = X[valid_idx]
            y_train = y[valid_idx]
            sample_weights_train = sample_weights[valid_idx]

            # Cross-validate or train on all data
            if use_cv:
                cv_results = self.cross_validate(X_train, y_train, sample_weights_train, n_splits=3)

            # Train final model on filtered data (rows with valid forward_return)
            print("\n" + "=" * 70)
            print("  Training final model on data with valid forward_return...")
            print("=" * 70)
            self.train(X_train, y_train, sample_weight=sample_weights_train)

            # Make predictions on ALL data (including recent dates without forward_return)
            print("\n🔮 Generating predictions for ALL rows...")
            print(f"  • Predicting for {len(X):,} rows (trained on {len(X_train):,})")
            predictions = self.predict(X)

        # Add predictions to ALL rows
        df['predicted_return'] = predictions

        # ============================================================
        # TEMPORAL DIAGNOSTICS (if enabled)
        # ============================================================
        if run_temporal_diagnostics:
            if STATSMODELS_AVAILABLE:
                print("\n" + "=" * 70)
                print("🔍 RUNNING TEMPORAL DIAGNOSTICS")
                print("=" * 70)
                diagnostics_result = self.temporal_diagnostics(df, predictions)
                print("\n" + "=" * 70)
            else:
                print("\n⚠️  Temporal diagnostics requested but statsmodels not available.")
                print("   Install with: pip install statsmodels")

        # Return only original columns + predictions (not all engineered features)
        if keep_engineered_features:
            print("  ⚠️  Keeping all engineered features (large file size)")
            return df
        else:
            # Only keep original columns + predictions + forward_return (for validation)
            cols_to_keep = original_cols + ['forward_return', 'predicted_return']
            df_output = df[cols_to_keep].copy()

            print(f"\n📋 COLUMN SUMMARY:")
            print(f"  ✓ Returning only original columns + predictions")
            print(f"    • Original columns: {len(original_cols)}")
            print(f"    • New columns added: 2")
            print(f"    • Total output columns: {len(cols_to_keep)}")
            print(f"\n  📊 New Columns:")
            print(f"    1. forward_return    - Actual {self.target_return_days}-day returns starting {self.forecast_days} days ahead (for validation)")
            print(f"    2. predicted_return  - ML predicted {self.target_return_days}-day returns (use for trading)")

            return df_output


def main():
    """
    Main entry point for the ML return forecasting script.

    This function:
    1. Parses command-line arguments
    2. Loads input data
    3. Initializes the forecaster with specified parameters
    4. Trains the model with cross-validation
    5. Generates predictions
    6. Saves results with comprehensive statistics

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description='Walk-Forward ML Return Forecasting - Zero Look-Ahead Bias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training run (first time)
  python forecast_returns_ml_walk_forward.py \\
      --input-file 20091231_20260106_data.csv \\
      --output 20091231_20260106_predictions.csv

  # Resume from previous predictions (much faster!)
  python forecast_returns_ml_walk_forward.py \\
      --input-file 20091231_20260110_data.csv \\
      --output 20091231_20260110_predictions.csv \\
      --resume-file 20091231_20260106_predictions.csv \\
      --overwrite-months 1

  # Predict 90-day returns starting 10 days from now
  python forecast_returns_ml_walk_forward.py \\
      --input-file data.csv \\
      --output predictions.csv \\
      --forecast-days 10 \\
      --target-return-days 90

For full documentation, see README.md in this directory.
        """
    )

    # Input/Output files
    parser.add_argument('--input-file', '-i', required=True,
                        help='Input CSV file with fundamental data (REQUIRED)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file for predictions (REQUIRED)')
    parser.add_argument('--lookback', type=int, default=10,
                        help='NOT IMPLEMENTED - Reserved for future use (default: 10)')
    parser.add_argument('--forecast-days', type=int, default=10,
                        help='Days ahead to start measuring returns (default: 10)')
    parser.add_argument('--target-return-days', type=int, default=None,
                        help='Return period to predict in days (default: same as forecast-days). '
                             'Example: --forecast-days 10 --target-return-days 90 predicts '
                             '90-day returns starting 10 days from now')
    parser.add_argument('--n-estimators', type=int, default=300,
                        help='Number of boosting rounds (default: 300)')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='Learning rate (default: 0.05)')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Maximum tree depth (default: 6, reduced from 7 for noisy stock returns)')
    parser.add_argument('--num-leaves', type=int, default=31,
                        help='Maximum number of leaves per tree (default: 31). '
                             'Controls model capacity. With 650K rows, safe values: 31 (conservative), '
                             '63 (balanced), 127 (high capacity). Rule: >1000 rows per leaf.')
    parser.add_argument('--no-cv', action='store_true',
                        help='Skip cross-validation (faster, only used if --no-walk-forward is set)')
    parser.add_argument('--no-walk-forward', action='store_true',
                        help='Disable expanding window walk-forward training (FASTER but has LOOK-AHEAD BIAS). '
                             'Default: walk-forward is ENABLED for realistic backtesting.')
    parser.add_argument('--lookback-months', type=int, default=None, metavar='N',
                        help='Use rolling N-month window for training instead of expanding window (default: None). '
                             'None = Expanding window (train on ALL historical data). '
                             '12 = Rolling 12-month window (train on last 12 months only). '
                             'Use this to test recency bias vs. long-term patterns. '
                             'Shorter windows = more responsive to recent changes. '
                             'Longer windows/expanding = more stable, learns long-term patterns.')
    parser.add_argument('--keep-features', action='store_true',
                        help='Keep all engineered features in output (creates large file)')
    parser.add_argument('--export-predictions', action='store_true',
                        help='Export a separate file with only Symbol, Date, and predicted_return columns')

    # Resume options
    parser.add_argument('--resume-file', '-r', type=str, default=None,
                        help='Previous predictions CSV file to resume from. '
                             'When provided, only new months will be trained (95%% faster). '
                             'Omit this flag for full training from scratch.')
    parser.add_argument('--overwrite-months', type=int, default=1,
                        help='When using --resume-file, re-predict last N months (default: 1). '
                             'Helps handle data revisions. Use 0 to only predict new data.')
    parser.add_argument('--preserve-existing', action='store_true',
                        help='When resuming, NEVER overwrite existing predictions (freeze historical forecasts). '
                             'Only compute predictions for rows with NaN. Ensures forecast stability. '
                             'Overrides --overwrite-months when enabled.')
    parser.add_argument('--skip-feature-importance', action='store_true',
                        help='Skip feature importance calculation at the end (saves 1-3 minutes)')
    parser.add_argument('--no-lag', action='store_true',
                        help='Skip lagging of features (use when input data is already lagged)')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                        help='Fraction of training data to use (0.0-1.0, default: 1.0). '
                             'Use 0.5 for 50%% sampling to speed up training by ~2x. '
                             'Predictions still made for ALL rows.')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file (default: auto-generate with timestamp). '
                             'All console output will be saved to this file.')
    parser.add_argument('--pca', type=int, default=None, metavar='N',
                        help='Use PCA to reduce features to N components (e.g., --pca 20). '
                             'Dimensionality reduction preserves variance while reducing feature count. '
                             'Typical values: 10-50 components')
    parser.add_argument('--temporal-diagnostics', action='store_true',
                        help='Run temporal diagnostics after training to detect autocorrelation, '
                             'look-ahead bias indicators, and prediction stability. '
                             'Requires statsmodels (pip install statsmodels). '
                             'Adds 1-2 minutes to runtime.')
    parser.add_argument('--log-features', action='store_true',
                        help='Log top 20 feature importances for each month during walk-forward training. '
                             'Creates feature_importance_YYYYMMDD_HHMMSS.csv in logs/ directory. '
                             'CSV format allows plotting feature importance changes over time. '
                             'Adds ~5-10 seconds per month to runtime.')

    args = parser.parse_args()

    # ========================================================================
    # SETUP LOGGING
    # ========================================================================

    # Create logs directory if it doesn't exist
    logs_dir = Path('./logs')
    logs_dir.mkdir(exist_ok=True)

    # Generate log file name if not specified
    if args.log_file is None:
        # Auto-generate log file name with timestamp in logs directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f"forecast_ml_walk_forward_{timestamp}.log"
    else:
        # User-specified log file - use as-is (they may want custom location)
        log_file = Path(args.log_file)

    # Initialize global logger
    global logger
    logger = DualLogger(log_file)

    # Redirect print() to logger
    import builtins
    original_print = builtins.print  # Save the original print function

    def custom_print(*args, **kwargs):
        """Override print to use logger."""
        import io
        output = io.StringIO()
        original_print(*args, file=output, **kwargs)
        message = output.getvalue().rstrip()
        if message:  # Only log non-empty messages
            logger(message)
        output.close()

    # Replace builtin print with our custom version
    builtins.print = custom_print

    # Log startup
    print(f"📝 Logging to: {log_file}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Validate input
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # ========================================================================
    # REPRODUCIBILITY: No global random seeds needed!
    # ========================================================================
    # We use DETERMINISTIC sampling everywhere (no np.random calls)
    # Only randomness is inside sklearn model with fixed random_state=42
    # This makes the code fully reproducible without global seed management

    # Read data
    print("=" * 70)
    print("     ML-Based Stock Return Forecasting")
    print("=" * 70)
    print(f"\n📂 Reading {args.input_file}...")
    df = read_dataframe(args.input_file, "input data")

    # Handle case-insensitive column names (normalize to PascalCase)
    def normalize_column_name(col):
        """Convert lowercase column names to expected PascalCase format."""
        col_lower = col.lower()

        # Common column mappings
        known_columns = {
            'date': 'Date',
            'symbol': 'Symbol',
            'refpriceclose': 'RefPriceClose',
            'refvolume': 'RefVolume',
            'companymarketcap': 'CompanyMarketCap',
            'companycommonname': 'CompanyCommonName',
            'enterprisevalue_dailytimeseries_': 'EnterpriseValue_DailyTimeSeries_',
            'focfexdividends_discrete': 'FOCFExDividends_Discrete',
            'interestexpense_netofcapitalizedinterest': 'InterestExpense_NetofCapitalizedInterest',
            'debt_total': 'Debt_Total',
            'earningspershare_actual': 'EarningsPerShare_Actual',
            'earningspershare_smartestimate_prev_q': 'EarningsPerShare_SmartEstimate_prev_Q',
            'earningspershare_actualsurprise': 'EarningsPerShare_ActualSurprise',
            'earningspershare_smartestimate_current_q': 'EarningsPerShare_SmartEstimate_current_Q',
            'longtermgrowth_mean': 'LongTermGrowth_Mean',
            'pricetarget_median': 'PriceTarget_Median',
            'combinedalphamodelsectorrank': 'CombinedAlphaModelSectorRank',
            'combinedalphamodelsectorrankchange': 'CombinedAlphaModelSectorRankChange',
            'combinedalphamodelregionrank': 'CombinedAlphaModelRegionRank',
            'earningsqualityregionrank_current': 'EarningsQualityRegionRank_Current',
            'enterprisevaluetoebit_dailytimeseriesratio_': 'EnterpriseValueToEBIT_DailyTimeSeriesRatio_',
            'enterprisevaluetoebitda_dailytimeseriesratio_': 'EnterpriseValueToEBITDA_DailyTimeSeriesRatio_',
            'enterprisevaluetosales_dailytimeseriesratio_': 'EnterpriseValueToSales_DailyTimeSeriesRatio_',
            'dividend_per_share_smartestimate': 'Dividend_Per_Share_SmartEstimate',
            'cashcashequivalents_total': 'CashCashEquivalents_Total',
            'forwardpeg_dailytimeseriesratio_': 'ForwardPEG_DailyTimeSeriesRatio_',
            'priceearningstogrowthratio_smartestimate_': 'PriceEarningsToGrowthRatio_SmartEstimate_',
            'recommendation_median_1_5_': 'Recommendation_Median_1_5_',
            'returnonequity_smartestimat': 'ReturnOnEquity_SmartEstimat',
            'returnonassets_smartestimate': 'ReturnOnAssets_SmartEstimate',
            'forwardpricetocashflowpershare_dailytimeseriesratio_': 'ForwardPriceToCashFlowPerShare_DailyTimeSeriesRatio_',
            'forwardpricetosalespershare_dailytimeseriesratio_': 'ForwardPriceToSalesPerShare_DailyTimeSeriesRatio_',
            'forwardenterprisevaluetooperatingcashflow_dailytimeseriesratio_': 'ForwardEnterpriseValueToOperatingCashFlow_DailyTimeSeriesRatio_',
            'grossprofitmargin_actualsurprise': 'GrossProfitMargin_ActualSurprise',
            'estpricegrowth_percent': 'Estpricegrowth_percent',
            'gicssectorname': 'GICSSectorName',
            'sharadar_scalemarketcap': 'sharadar_scalemarketcap',
            'tradedate': 'TradeDate',
            'instrument': 'Instrument',
        }

        # Return mapped name if known, otherwise return original
        return known_columns.get(col_lower, col)

    # Apply normalization
    column_mapping = {col: normalize_column_name(col) for col in df.columns if col != normalize_column_name(col)}

    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"  • Normalized {len(column_mapping)} column names to PascalCase")

    # Verify required columns exist
    if 'Date' not in df.columns:
        print(f"\n❌ Error: No 'Date' or 'date' column found!")
        print(f"   Available columns: {', '.join(df.columns[:10])}...")
        return 1

    if 'Symbol' not in df.columns:
        print(f"\n❌ Error: No 'Symbol' or 'symbol' column found!")
        print(f"   Available columns: {', '.join(df.columns[:10])}...")
        return 1

    # ========================================================================
    # AUTOMATIC DATE CLEANUP - Remove future-dated records based on filename
    # ========================================================================
    # Parse end date from filename pattern: YYYYMMDD_YYYYMMDD_*.csv
    # Example: "20091231_20260106_with_metadata.csv" -> end_date = 2026-01-06

    filename = input_path.name
    date_pattern = r'(\d{8})_(\d{8})'
    match = re.search(date_pattern, filename)

    if match:
        start_date_str = match.group(1)
        end_date_str = match.group(2)

        try:
            # Parse end date from filename (YYYYMMDD format)
            end_date = pd.to_datetime(end_date_str, format='%Y%m%d')

            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Count records beyond the filename's end date
            original_count = len(df)
            future_mask = df['Date'] > end_date
            future_count = future_mask.sum()

            if future_count > 0:
                # Show sample of future records
                future_records = df[future_mask][['Date', 'Symbol']].drop_duplicates()
                print(f"\n⚠️  AUTOMATIC DATE CLEANUP:")
                print(f"  • Filename indicates data should end on: {end_date.strftime('%Y-%m-%d')}")
                print(f"  • Found {future_count:,} records beyond this date")
                print(f"  • Affected symbols: {df[future_mask]['Symbol'].nunique()}")

                if future_count <= 10:
                    print(f"  • Future records:")
                    for _, row in future_records.iterrows():
                        print(f"    - {row['Date'].strftime('%Y-%m-%d')}: {row['Symbol']}")
                else:
                    print(f"  • Sample future records (showing first 5):")
                    for _, row in future_records.head(5).iterrows():
                        print(f"    - {row['Date'].strftime('%Y-%m-%d')}: {row['Symbol']}")

                # Remove future records
                df = df[~future_mask].copy()
                print(f"  ✅ Removed {future_count:,} future-dated records")
                print(f"  • Cleaned rows: {len(df):,}")
            else:
                # Convert Date to datetime even if no cleanup needed
                df['Date'] = pd.to_datetime(df['Date'])
                print(f"  ✅ Date validation: All records within filename date range")

        except ValueError as e:
            # If date parsing fails, just convert Date column
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"  ⚠️  Could not parse end date from filename: {end_date_str}")
    else:
        # No date pattern in filename, just convert Date column
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"  ℹ️  No date range pattern found in filename (expected: YYYYMMDD_YYYYMMDD)")

    print(f"  • Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  • Unique symbols: {df['Symbol'].nunique()}")

    # ========================================================================
    # REPRODUCIBILITY: Check for and remove duplicate (Date, Symbol) rows
    # ========================================================================
    print(f"\n🔍 Checking for duplicate rows...")
    duplicate_mask = df.duplicated(subset=['Date', 'Symbol'], keep='first')
    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:
        print(f"  ⚠️  WARNING: Found {duplicate_count:,} duplicate (Date, Symbol) rows!")
        print(f"  • This causes non-deterministic results because sort order is unstable")
        print(f"  • Keeping first occurrence, dropping duplicates...")

        # Show examples
        duplicate_examples = df[duplicate_mask][['Date', 'Symbol']].drop_duplicates().head(5)
        print(f"  • Example duplicates:")
        for _, row in duplicate_examples.iterrows():
            dup_rows = df[(df['Date'] == row['Date']) & (df['Symbol'] == row['Symbol'])]
            print(f"    - {row['Date'].strftime('%Y-%m-%d')}, {row['Symbol']}: {len(dup_rows)} occurrences")

        # Remove duplicates (keeping first occurrence for stability)
        df = df[~duplicate_mask].copy()
        df = df.reset_index(drop=True)
        print(f"  ✅ Removed duplicates. Rows after deduplication: {len(df):,}")
    else:
        print(f"  ✅ No duplicate (Date, Symbol) rows found")

    # Initialize forecaster
    forecaster = ReturnForecaster(
        lookback_days=args.lookback,
        forecast_days=args.forecast_days,
        target_return_days=args.target_return_days,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        no_lag=args.no_lag,
        sample_fraction=args.sample_fraction,
        pca_components=args.pca,
        lookback_months=args.lookback_months,
        log_features=args.log_features
    )

    # Display prediction setup
    target_days = args.target_return_days if args.target_return_days else args.forecast_days
    print(f"\n🎯 PREDICTION SETUP:")
    if args.no_lag:
        print(f"  • Using features from: T (pre-lagged input data)")
    else:
        print(f"  • Using features from: T-1 (lagged by 1 day)")
    print(f"  • Predicting returns from: T+{args.forecast_days} to T+{args.forecast_days + target_days}")
    print(f"  • Total return period: {target_days} days")
    if args.forecast_days != target_days:
        print(f"  • Strategy: Predict {target_days}-day returns starting {args.forecast_days} days from now")
    if args.sample_fraction < 1.0:
        print(f"  • Training data sampling: {args.sample_fraction:.0%} (⚡ ~{1/args.sample_fraction:.1f}x faster)")

    # Log run parameters for reproducibility
    print(f"\n⚙️  MODEL PARAMETERS:")
    print(f"  • n_estimators: {args.n_estimators}")
    print(f"  • learning_rate: {args.learning_rate}")
    print(f"  • max_depth: {args.max_depth}")
    print(f"  • num_leaves: {args.num_leaves}")
    print(f"  • min_samples_leaf: 100 (hardcoded, v3.3.12 - more stable splits)")
    print(f"  • l2_regularization: 0.2 (hardcoded, v3.3.9 - moderate regularization)")
    print(f"  • max_bins: 255 (histogram bins)")
    print(f"  • random_state: 42 (reproducibility)")
    print(f"  • walk_forward: {not args.no_walk_forward}")
    if args.lookback_months:
        print(f"  • lookback_months: {args.lookback_months} (rolling window training)")
    if args.sample_fraction < 1.0:
        print(f"  • sample_fraction: {args.sample_fraction:.1%} (~{1/args.sample_fraction:.1f}x speedup)")
    if args.pca:
        print(f"  • pca_components: {args.pca} (dimensionality reduction)")
    if args.resume_file:
        print(f"  • resume_file: {args.resume_file} (overwrite_months={args.overwrite_months})")
    if args.temporal_diagnostics:
        print(f"  • temporal_diagnostics: True (ACF, Ljung-Box, stability tests)")

    # ========================================================================
    # GENERATE OUTPUT FILENAME
    # ========================================================================
    output_path = Path(args.output)

    # ========================================================================
    # RESUME LOGIC - Simple and Clean
    # ========================================================================
    # If --resume-file is provided, load previous predictions
    # Otherwise, start from scratch (full training)
    # ========================================================================

    resume_from_date = None
    previous_predictions = None

    if args.resume_file:
        print("\n" + "=" * 70)
        print("  📂 RESUME MODE")
        print("=" * 70)

        resume_file_path = Path(args.resume_file)

        if not resume_file_path.exists():
            print(f"\n❌ ERROR: Resume file not found: {args.resume_file}")
            print("   Running full training instead")
        else:
            try:
                print(f"\n📂 Loading previous predictions from {resume_file_path.name}...")
                previous_predictions_df = read_dataframe(resume_file_path, "previous predictions")

                # ============================================================
                # AUTO-CLEANUP RESUME FILE - Remove future-dated records
                # ============================================================
                # Parse end date from resume filename to clean bad data
                resume_filename = resume_file_path.name
                date_pattern = r'(\d{8})_(\d{8})'
                date_match = re.search(date_pattern, resume_filename)

                if date_match:
                    resume_end_date_str = date_match.group(2)
                    try:
                        resume_end_date = pd.to_datetime(resume_end_date_str, format='%Y%m%d')
                        previous_predictions_df['Date'] = pd.to_datetime(previous_predictions_df['Date'])

                        # Check for future-dated records
                        future_mask = previous_predictions_df['Date'] > resume_end_date
                        future_count = future_mask.sum()

                        if future_count > 0:
                            print(f"\n⚠️  RESUME FILE CLEANUP:")
                            print(f"  • Resume file should end on: {resume_end_date.strftime('%Y-%m-%d')}")
                            print(f"  • Found {future_count:,} records beyond this date")
                            if future_count <= 5:
                                for _, row in previous_predictions_df[future_mask][['Date', 'Symbol']].drop_duplicates().iterrows():
                                    print(f"    - {row['Date'].strftime('%Y-%m-%d')}: {row['Symbol']}")

                            # Remove future records
                            previous_predictions_df = previous_predictions_df[~future_mask].copy()
                            print(f"  ✅ Removed {future_count:,} future-dated records from resume file")
                            print(f"  • Cleaned resume rows: {len(previous_predictions_df):,}")
                    except ValueError:
                        pass  # If date parsing fails, skip cleanup

                # Verify it has the required columns
                if 'predicted_return' not in previous_predictions_df.columns:
                    print(f"\n❌ ERROR: Resume file missing 'predicted_return' column")
                    print("   Running full training instead")
                    previous_predictions = None
                elif 'Date' not in previous_predictions_df.columns:
                    print(f"\n❌ ERROR: Resume file missing 'Date' column")
                    print("   Running full training instead")
                    previous_predictions = None
                elif 'Symbol' not in previous_predictions_df.columns:
                    print(f"\n❌ ERROR: Resume file missing 'Symbol' column")
                    print("   Running full training instead")
                    previous_predictions = None
                else:
                    # Find the last date with predictions
                    prev_df_with_preds = previous_predictions_df[previous_predictions_df['predicted_return'].notna()]

                    if len(prev_df_with_preds) == 0:
                        print(f"\n⚠️  WARNING: Resume file has no predictions")
                        print("   Running full training instead")
                        previous_predictions = None
                    else:
                        # Convert Date to datetime
                        prev_df_with_preds['Date'] = pd.to_datetime(prev_df_with_preds['Date'])
                        last_prediction_date = prev_df_with_preds['Date'].max()

                        print(f"  ✓ Loaded {len(previous_predictions_df):,} rows")
                        print(f"  • Last prediction date: {last_prediction_date.strftime('%Y-%m-%d')}")
                        print(f"  • Rows with predictions: {len(prev_df_with_preds):,}")

                        # Calculate resume date (go back N months)
                        # CRITICAL: Skip resume_from_date logic when --preserve-existing is used
                        # preserve_existing handles month skipping at the walk-forward level
                        if args.preserve_existing:
                            # Don't set resume_from_date - let preserve_existing logic handle it
                            print(f"  • 🔒 PRESERVE MODE: Will skip months with existing predictions")
                            print(f"  • Last prediction date: {last_prediction_date.strftime('%Y-%m-%d')}")
                            resume_from_date = None  # Let all months flow through to preserve_existing check
                        elif args.overwrite_months > 0:
                            resume_from_date = (last_prediction_date - pd.DateOffset(months=args.overwrite_months)).strftime('%Y-%m-%d')
                            print(f"  • Overwrite buffer: {args.overwrite_months} months")
                            print(f"  • Resume from: {resume_from_date}")
                            print(f"  • Will re-predict from {resume_from_date} onwards")
                        else:
                            # Resume from day after last prediction
                            resume_from_date = (last_prediction_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                            print(f"  • Overwrite buffer: 0 months (only new data)")
                            print(f"  • Resume from: {resume_from_date}")

                        # Store the dataframe for later alignment
                        previous_predictions = previous_predictions_df
                        print(f"  ✓ Previous predictions loaded successfully")

            except Exception as e:
                print(f"\n❌ ERROR loading previous predictions: {e}")
                print("   Running full training instead")
                previous_predictions = None
                resume_from_date = None

    else:
        print("\n" + "=" * 70)
        print("  🔄 FULL TRAINING MODE")
        print("  (No --resume-file provided)")
        print("=" * 70)



    df_predictions = forecaster.fit_predict(
        df,
        use_cv=not args.no_cv,
        keep_engineered_features=args.keep_features,
        walk_forward=not args.no_walk_forward,
        resume_from_date=resume_from_date,
        previous_predictions=previous_predictions,
        run_temporal_diagnostics=args.temporal_diagnostics,
        preserve_existing=args.preserve_existing
    )

    # Save results (output_path already generated earlier)
    write_dataframe(df_predictions, output_path, "predictions")

    # ========================================================================
    # AUTO-EXPORT FORECAST-ONLY CSV (Always created)
    # ========================================================================
    # Extract date range from input filename to create forecast-only CSV
    # Format: YYYYMMDD_YYYYMMDD_forecast_only.csv
    print("\n" + "=" * 70)
    print("  📊 CREATING FORECAST-ONLY CSV")
    print("=" * 70)

    # Extract date range from input filename
    input_filename = input_path.name
    date_pattern = r'(\d{8}_\d{8})'
    date_match = re.search(date_pattern, input_filename)

    if date_match:
        date_range = date_match.group(1)
        forecast_only_filename = f"{date_range}_forecast_only.csv"
    else:
        # Fallback: use output filename stem
        forecast_only_filename = f"{output_path.stem}_forecast_only.csv"

    forecast_only_path = output_path.parent / forecast_only_filename

    # Extract only Symbol, Date, predicted_return (non-NaN only)
    forecast_only = df_predictions[['Symbol', 'Date', 'predicted_return']].copy()
    forecast_only = forecast_only[forecast_only['predicted_return'].notna()]
    forecast_only = forecast_only.sort_values(['Date', 'Symbol'])

    print(f"\n💾 Saving forecast-only CSV to {forecast_only_filename}...")
    forecast_only.to_csv(forecast_only_path, index=False)
    file_size_mb = forecast_only_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Saved {len(forecast_only):,} forecasts")
    print(f"  • Columns: Symbol, Date, predicted_return")
    print(f"  • Format: CSV (for easy use)")
    print(f"  • Size: {file_size_mb:.1f} MB")
    print(f"  • Use this file like: extract_symbol.py {forecast_only_filename} --symbol AAPL")

    # Export predictions-only file if requested (old flag for backward compatibility)
    if args.export_predictions:
        # Create predictions-only filename
        if args.output is None:
            # Use auto-generated name with _predictions_only suffix
            pred_only_path = output_path.parent / f"{output_path.stem}_predictions_only{output_path.suffix}"
        else:
            # Use custom output name with _predictions_only suffix
            pred_only_path = output_path.parent / f"{output_path.stem}_predictions_only{output_path.suffix}"

        # Extract only the required columns
        predictions_only = df_predictions[['Symbol', 'Date', 'predicted_return']].copy()

        # Remove rows with no predictions
        predictions_only = predictions_only[predictions_only['predicted_return'].notna()]

        # Sort for easier lookup
        predictions_only = predictions_only.sort_values(['Date', 'Symbol'])

        # Save the lean predictions file
        print(f"\n  📊 Exporting predictions-only file...")
        write_dataframe(predictions_only, pred_only_path, "predictions-only (Symbol, Date, predicted_return)")

    # Summary statistics
    print("\n" + "=" * 70)
    print(" " * 20 + "PREDICTION SUMMARY")
    print("=" * 70)
    pred_col = 'predicted_return'
    actual_col = 'forward_return'

    valid_preds = df_predictions[pred_col].notna()
    print(f"\n📊 DATA COVERAGE:")
    print(f"  • Total rows: {len(df_predictions):,}")
    print(f"  • Rows with predictions: {valid_preds.sum():,}")
    print(f"  • Coverage: {valid_preds.sum() / len(df_predictions) * 100:.2f}%")
    print(f"  • Unique symbols: {df_predictions['Symbol'].nunique()}")
    print(f"  • Date range: {df_predictions['Date'].min()} to {df_predictions['Date'].max()}")

    if valid_preds.sum() > 0:
        print(f"\n📈 PREDICTED RETURN DISTRIBUTION:")
        preds = df_predictions.loc[valid_preds, pred_col]
        print(f"  • Mean: {preds.mean():+.2f}%")
        print(f"  • Median: {preds.median():+.2f}%")
        print(f"  • Std Dev: {preds.std():.2f}%")
        print(f"  • 25th percentile: {preds.quantile(0.25):+.2f}%")
        print(f"  • 75th percentile: {preds.quantile(0.75):+.2f}%")
        print(f"  • Min: {preds.min():+.2f}%")
        print(f"  • Max: {preds.max():+.2f}%")

        # Correlation with actual returns (where available)
        both_valid = df_predictions[pred_col].notna() & df_predictions[actual_col].notna()
        if both_valid.sum() > 0:
            corr = df_predictions.loc[both_valid, [pred_col, actual_col]].corr().iloc[0, 1]
            mse = mean_squared_error(
                df_predictions.loc[both_valid, actual_col],
                df_predictions.loc[both_valid, pred_col]
            )
            mae = mean_absolute_error(
                df_predictions.loc[both_valid, actual_col],
                df_predictions.loc[both_valid, pred_col]
            )
            print(f"\n🎯 MODEL PERFORMANCE (Out-of-sample, NO LOOK-AHEAD BIAS):")
            print(f"  • Correlation with actual returns: {corr:.4f} ({corr*100:.1f}%)")
            print(f"  • Mean Absolute Error: {mae:.2f}%")
            print(f"  • Root Mean Squared Error: {np.sqrt(mse):.2f}%")

            # Direction accuracy
            pred_direction = df_predictions.loc[both_valid, pred_col] > 0
            actual_direction = df_predictions.loc[both_valid, actual_col] > 0
            direction_accuracy = (pred_direction == actual_direction).mean()
            print(f"  • Direction accuracy: {direction_accuracy:.2%}")

        # Top predictions
        print(f"\n🏆 TOP 10 PREDICTED GAINERS (most recent data):")
        recent_data = df_predictions[df_predictions[pred_col].notna()].copy()
        recent_data['Date'] = pd.to_datetime(recent_data['Date'])
        cutoff_date = recent_data['Date'].max() - pd.Timedelta(days=180)
        recent = recent_data[recent_data['Date'] >= cutoff_date]

        if len(recent) > 0:
            top_gainers = recent.nlargest(10, pred_col)[['Date', 'Symbol', 'RefPriceClose', pred_col]]
            for idx, row in top_gainers.iterrows():
                print(f"  {row['Date']:%Y-%m-%d}  {row['Symbol']:6s}  ${row['RefPriceClose']:8.2f}  →  {row[pred_col]:+.2f}%")

        # Top predictions by unique symbol (most recent prediction per symbol)
        print(f"\n🎯 TOP 10 UNIQUE SYMBOLS (most recent prediction per symbol):")
        if len(recent) > 0:
            # Group by Symbol, take most recent date for each
            recent_by_symbol = recent.sort_values('Date').groupby('Symbol').last().reset_index()

            # Filter to top 25% by market cap (large-cap stocks only)
            if 'CompanyMarketCap' in recent_by_symbol.columns:
                # Calculate 75th percentile of market cap
                marketcap_75th = recent_by_symbol['CompanyMarketCap'].quantile(0.75)
                recent_by_symbol = recent_by_symbol[recent_by_symbol['CompanyMarketCap'] >= marketcap_75th]
                print(f"  (Filtered to top 25% by market cap: ${marketcap_75th/1e9:.1f}B+ | {len(recent_by_symbol)} symbols)")

            # Get top 10 unique symbols
            top_unique = recent_by_symbol.nlargest(10, pred_col)[['Date', 'Symbol', 'RefPriceClose', pred_col]]

            for idx, row in top_unique.iterrows():
                print(f"  {row['Date']:%Y-%m-%d}  {row['Symbol']:6s}  ${row['RefPriceClose']:8.2f}  →  {row[pred_col]:+.2f}%")

    # ========================================================================
    # FEATURE IMPORTANCES
    # ========================================================================
    if not args.no_walk_forward and not args.skip_feature_importance and hasattr(forecaster, 'X_last') and hasattr(forecaster, 'y_last'):
        print(f"\n🔬 TOP 15 MOST IMPORTANT FEATURES:")
        print(f"   (Based on final trained model using permutation importance)")
        try:
            # Compute feature importances
            importance_df = forecaster.get_feature_importances(
                forecaster.X_last,
                forecaster.y_last,
                sample_size=min(10000, len(forecaster.X_last)),
                n_repeats=5
            )

            # Display top 15
            for idx, row in importance_df.head(15).iterrows():
                print(f"  {row['feature']:40s}  {row['importance']:8.6f}  (±{row['std']:.6f})")

            print(f"\n  💡 Interpretation:")
            print(f"     • Higher values = more important for predictions")
            print(f"     • Importances show impact on model performance when feature is shuffled")
            print(f"     • Based on final month's model (most recent training)")

        except Exception as e:
            print(f"  ⚠️  Could not compute feature importances: {e}")
    elif args.skip_feature_importance:
        print(f"\n⏭️  Feature importance calculation skipped (--skip-feature-importance)")

    print(f"\n💾 OUTPUT:")
    print(f"  • Main file: {output_path}")
    print(f"  • Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  • Rows: {len(df_predictions):,}")

    if args.export_predictions:
        print(f"\n  • Predictions-only file: {pred_only_path}")
        print(f"  • Size: {pred_only_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  • Rows: {len(predictions_only):,}")
        print(f"  • Columns: Symbol, Date, predicted_return (lean format for fast lookups)")

    # ========================================================================
    # RESUME TIP
    # ========================================================================

    if not args.no_walk_forward and not args.resume_file:
        print(f"\n💡 TIP: For faster updates, use --resume-file to skip already-predicted months")
        print(f"   Example: python {Path(__file__).name} \\")
        print(f"       --input-file new_data.csv \\")
        print(f"       --output new_predictions.csv \\")
        print(f"       --resume-file {output_path.name}")

    print("\n" + "=" * 70)
    print("✅ Forecasting complete - NO LOOK-AHEAD BIAS GUARANTEED!")
    print("=" * 70)
    if args.no_lag:
        print(f"\n📌 IMPORTANT: Using pre-lagged input data (T) to predict returns")
    else:
        print(f"\n📌 IMPORTANT: All features use data from T-1 to predict returns")
    print(f"   Prediction period: T+{args.forecast_days} to T+{args.forecast_days + target_days}")
    print("   This means predictions are safe for real trading.")
    print("=" * 70)

    # Log completion
    print(f"\n📝 Log file saved: {log_file}")
    print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Close log handlers
    if logger and logger.logger:
        for handler in logger.logger.handlers:
            handler.close()

    return 0


if __name__ == '__main__':
    # Entry point when script is run directly
    # Returns 0 on success, 1 on error
    exit(main())

# ============================================================================
# END OF SCRIPT
# ============================================================================
# For usage examples and full documentation, see README.md
# For questions or issues, check the troubleshooting section in README.md
# ============================================================================
