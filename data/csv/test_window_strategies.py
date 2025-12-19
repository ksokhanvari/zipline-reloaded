#!/usr/bin/env python3
"""
Walk-Forward Window Strategy Testing Framework

This script compares different walk-forward training approaches:
1. Expanding Window (uses all historical data)
2. Rolling Window (uses fixed lookback period)

Tests multiple window sizes and generates comprehensive accuracy metrics
to determine the optimal approach for your data.

Usage:
    # Test with default settings (expanding + rolling 6,12,24,36 months)
    python test_window_strategies.py data.csv

    # Test specific window sizes
    python test_window_strategies.py data.csv --windows 6 12 24

    # Custom forecast parameters
    python test_window_strategies.py data.csv --forecast-days 10 --target-return-days 90

    # Quick test (skip some months for faster execution)
    python test_window_strategies.py data.csv --sample-every 3

Output:
    - {input}_window_comparison.csv - Detailed monthly metrics
    - {input}_window_comparison_summary.txt - Overall statistics
    - {input}_window_comparison_plots.png - Visualization (if matplotlib available)
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
import time

# Suppress warnings
warnings.filterwarnings('ignore')


class WindowTester:
    """
    Framework for testing different walk-forward window strategies.

    Compares expanding window vs rolling windows of various sizes.
    """

    def __init__(self, forecast_days=10, target_return_days=None,
                 n_estimators=300, learning_rate=0.05, max_depth=7):
        """
        Initialize the tester with model parameters.

        Args:
            forecast_days: Days ahead to start measuring returns
            target_return_days: Return period to predict (default: same as forecast_days)
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate for gradient boosting
            max_depth: Maximum tree depth
        """
        self.forecast_days = forecast_days
        self.target_return_days = target_return_days if target_return_days else forecast_days
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def create_target(self, df):
        """Create forward return target (same as forecast_returns_ml.py)."""
        print(f"\n📊 Creating target: {self.target_return_days}-day returns, {self.forecast_days} days ahead...")

        df = df.sort_values(['Symbol', 'Date']).copy()

        # Calculate forward return
        df['future_price'] = df.groupby('Symbol')['RefPriceClose'].shift(-self.forecast_days - self.target_return_days)
        df['start_price'] = df.groupby('Symbol')['RefPriceClose'].shift(-self.forecast_days)

        df['forward_return'] = ((df['future_price'] - df['start_price']) / df['start_price']) * 100

        df = df.drop(['future_price', 'start_price'], axis=1)

        valid_count = df['forward_return'].notna().sum()
        print(f"  ✓ Created target with {valid_count:,} valid values ({valid_count/len(df)*100:.1f}%)")

        return df

    def engineer_features(self, df):
        """Create features (simplified version - key features only)."""
        print("\n⚙️  Engineering features...")

        df = df.sort_values(['Symbol', 'Date']).copy()

        # 1. Lag fundamentals by 1 day (NO LOOK-AHEAD BIAS)
        fundamental_cols = [col for col in df.columns if col not in
                          ['Date', 'Symbol', 'RefPriceClose', 'RefVolume', 'forward_return']]

        for col in fundamental_cols:
            if df[col].dtype in ['float64', 'int64']:
                df[f'{col}_lag1'] = df.groupby('Symbol')[col].shift(1)

        # 2. Price momentum (5, 10, 20 day)
        for window in [5, 10, 20]:
            df[f'momentum_{window}d'] = df.groupby('Symbol')['RefPriceClose'].pct_change(window) * 100
            df[f'momentum_{window}d'] = df.groupby('Symbol')[f'momentum_{window}d'].shift(1)

        # 3. Volatility (5, 10, 20 day)
        df['returns_1d'] = df.groupby('Symbol')['RefPriceClose'].pct_change()
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df.groupby('Symbol')['returns_1d'].rolling(window).std() * 100
            df[f'volatility_{window}d'] = df.groupby('Symbol')[f'volatility_{window}d'].shift(1)

        df = df.drop('returns_1d', axis=1)

        # 4. Volume MA
        df['volume_ma_20d'] = df.groupby('Symbol')['RefVolume'].rolling(20).mean()
        df['volume_ma_20d'] = df.groupby('Symbol')['volume_ma_20d'].shift(1)

        feature_cols = [col for col in df.columns if col.endswith('_lag1') or
                       'momentum' in col or 'volatility' in col or 'volume_ma' in col]

        print(f"  ✓ Engineered {len(feature_cols)} features")

        return df, feature_cols

    def prepare_features(self, df, feature_cols):
        """Prepare feature matrix and target vector."""
        # Select features
        X = df[feature_cols].copy()
        y = df['forward_return'].values

        # Create sample weights based on market cap
        if 'CompanyMarketCap' in df.columns:
            market_cap = df['CompanyMarketCap'].values
            market_cap_rank = df.groupby('Date')['CompanyMarketCap'].rank(ascending=False, method='first')

            sample_weights = np.ones(len(df))
            sample_weights[market_cap_rank <= 2000] = 1.0
            sample_weights[(market_cap_rank > 2000) & (market_cap_rank <= 3500)] = 0.5
            sample_weights[market_cap_rank > 3500] = 0.1
        else:
            sample_weights = np.ones(len(df))

        # Valid indices (rows with target)
        valid_idx = ~np.isnan(y)

        return X, y, sample_weights, valid_idx

    def train_model(self, X_train, y_train, sample_weights_train):
        """Train a single model."""
        model = HistGradientBoostingRegressor(
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )

        model.fit(X_train, y_train, sample_weight=sample_weights_train)

        return model

    def walk_forward_predict(self, df, X, y, sample_weights, valid_idx,
                            window_type='expanding', window_months=None,
                            sample_every=1):
        """
        Perform walk-forward prediction with specified window strategy.

        Args:
            df: DataFrame with Date column
            X: Feature matrix
            y: Target vector
            sample_weights: Sample weights
            valid_idx: Boolean mask for valid targets
            window_type: 'expanding' or 'rolling'
            window_months: Number of months for rolling window (ignored for expanding)
            sample_every: Only process every Nth month (for faster testing)

        Returns:
            predictions: Array of predictions
            monthly_stats: DataFrame with monthly statistics
        """
        # Convert Date to datetime
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])

        df['_year_month'] = df['Date'].dt.to_period('M')
        unique_months = sorted(df['_year_month'].unique())

        # Initialize
        predictions = np.full(len(df), np.nan)
        monthly_stats = []

        # For each month
        for i, current_month in enumerate(unique_months, 1):
            # Sample months for faster testing
            if (i - 1) % sample_every != 0:
                continue

            first_day_of_month = current_month.to_timestamp()

            # Rows to predict
            predict_mask = df['_year_month'] == current_month
            predict_positions = np.where(predict_mask)[0]

            if len(predict_positions) == 0:
                continue

            # Training mask based on window type
            if window_type == 'expanding':
                # Use all data before this month
                train_mask = (df['Date'] < first_day_of_month) & valid_idx
            elif window_type == 'rolling':
                # Use only last N months
                start_date = first_day_of_month - pd.DateOffset(months=window_months)
                train_mask = (df['Date'] >= start_date) & (df['Date'] < first_day_of_month) & valid_idx
            else:
                raise ValueError(f"Unknown window_type: {window_type}")

            train_positions = np.where(train_mask)[0]

            # Skip if insufficient training data
            if len(train_positions) < 100:
                continue

            # Get training data
            X_train = X.iloc[train_positions]
            y_train = y[train_positions]
            weights_train = sample_weights[train_positions]

            # Train model
            try:
                model = self.train_model(X_train, y_train, weights_train)
            except Exception as e:
                print(f"  ⚠️  {current_month}: Training failed - {e}")
                continue

            # Predict for current month
            X_predict = X.iloc[predict_positions]
            month_predictions = model.predict(X_predict)

            # Store predictions
            predictions[predict_positions] = month_predictions

            # Calculate statistics for this month (if we have actuals)
            actual_values = y[predict_positions]
            valid_pred_mask = ~np.isnan(actual_values) & ~np.isnan(month_predictions)

            if valid_pred_mask.sum() > 10:  # Need at least 10 valid pairs
                stats = self._calculate_month_stats(
                    month_predictions[valid_pred_mask],
                    actual_values[valid_pred_mask],
                    current_month,
                    len(train_positions),
                    len(predict_positions)
                )
                monthly_stats.append(stats)

        df.drop('_year_month', axis=1, inplace=True)

        return predictions, pd.DataFrame(monthly_stats)

    def _calculate_month_stats(self, predictions, actuals, month, n_train, n_predict):
        """Calculate accuracy statistics for a single month."""
        # Correlation
        corr = np.corrcoef(predictions, actuals)[0, 1]

        # Spearman (rank) correlation
        ic, _ = spearmanr(predictions, actuals)

        # RMSE and MAE
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))

        # Hit rate (sign accuracy)
        hit_rate = np.mean(np.sign(predictions) == np.sign(actuals))

        # Top quintile performance
        pred_quintiles = pd.qcut(predictions, 5, labels=False, duplicates='drop')
        if pred_quintiles.max() == 4:  # Successfully created 5 quintiles
            top_quintile_mask = pred_quintiles == 4
            top_quintile_return = actuals[top_quintile_mask].mean()
            bottom_quintile_mask = pred_quintiles == 0
            bottom_quintile_return = actuals[bottom_quintile_mask].mean()
            spread = top_quintile_return - bottom_quintile_return
        else:
            top_quintile_return = np.nan
            bottom_quintile_return = np.nan
            spread = np.nan

        return {
            'month': str(month),
            'n_train': n_train,
            'n_predict': n_predict,
            'correlation': corr,
            'ic': ic,
            'rmse': rmse,
            'mae': mae,
            'hit_rate': hit_rate,
            'top_quintile_return': top_quintile_return,
            'bottom_quintile_return': bottom_quintile_return,
            'long_short_spread': spread
        }

    def run_comparison(self, df, window_configs, sample_every=1):
        """
        Run walk-forward test for multiple window configurations.

        Args:
            df: Input DataFrame
            window_configs: List of (name, type, months) tuples
                           e.g., [('Expanding', 'expanding', None),
                                  ('Rolling_12M', 'rolling', 12)]
            sample_every: Only process every Nth month for faster testing

        Returns:
            results: Dict mapping config name to (predictions, monthly_stats)
        """
        # Create target
        df = self.create_target(df)

        # Engineer features
        df, feature_cols = self.engineer_features(df)

        # Prepare features
        X, y, sample_weights, valid_idx = self.prepare_features(df, feature_cols)

        print(f"\n📊 Dataset Summary:")
        print(f"  • Total rows: {len(df):,}")
        print(f"  • Rows with valid target: {valid_idx.sum():,}")
        print(f"  • Features: {len(feature_cols)}")
        print(f"  • Date range: {df['Date'].min()} to {df['Date'].max()}")

        # Run each window configuration
        results = {}

        for config_name, window_type, window_months in window_configs:
            print(f"\n" + "=" * 70)
            print(f"  Testing: {config_name}")
            print("=" * 70)

            start_time = time.time()

            predictions, monthly_stats = self.walk_forward_predict(
                df, X, y, sample_weights, valid_idx,
                window_type=window_type,
                window_months=window_months,
                sample_every=sample_every
            )

            elapsed = time.time() - start_time

            n_predictions = np.sum(~np.isnan(predictions))
            print(f"\n  ✅ {config_name} Complete!")
            print(f"  • Total predictions: {n_predictions:,}")
            print(f"  • Months evaluated: {len(monthly_stats)}")
            print(f"  • Time elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s")

            results[config_name] = {
                'predictions': predictions,
                'monthly_stats': monthly_stats,
                'elapsed_time': elapsed
            }

        return results, df


def generate_summary_report(results, output_path):
    """Generate text summary comparing all window strategies."""

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  WALK-FORWARD WINDOW STRATEGY COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        # Overall statistics for each strategy
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n\n")

        summary_data = []

        for config_name, result in results.items():
            stats = result['monthly_stats']

            if len(stats) == 0:
                continue

            summary = {
                'Strategy': config_name,
                'Months': len(stats),
                'Avg Correlation': stats['correlation'].mean(),
                'Std Correlation': stats['correlation'].std(),
                'Avg IC': stats['ic'].mean(),
                'Std IC': stats['ic'].std(),
                'Avg RMSE': stats['rmse'].mean(),
                'Avg MAE': stats['mae'].mean(),
                'Avg Hit Rate': stats['hit_rate'].mean(),
                'Avg Top Quintile': stats['top_quintile_return'].mean(),
                'Avg Bottom Quintile': stats['bottom_quintile_return'].mean(),
                'Avg Spread': stats['long_short_spread'].mean(),
                'Time (min)': result['elapsed_time'] / 60
            }

            summary_data.append(summary)

            # Write detailed stats
            f.write(f"{config_name}:\n")
            f.write(f"  Months evaluated: {summary['Months']}\n")
            f.write(f"  Time: {summary['Time (min)']:.1f} minutes\n")
            f.write(f"\n")
            f.write(f"  Correlation:     {summary['Avg Correlation']:>7.4f} ± {summary['Std Correlation']:.4f}\n")
            f.write(f"  IC (Spearman):   {summary['Avg IC']:>7.4f} ± {summary['Std IC']:.4f}\n")
            f.write(f"  RMSE:            {summary['Avg RMSE']:>7.2f}\n")
            f.write(f"  MAE:             {summary['Avg MAE']:>7.2f}\n")
            f.write(f"  Hit Rate:        {summary['Avg Hit Rate']:>7.2%}\n")
            f.write(f"\n")
            f.write(f"  Top Quintile Ret:    {summary['Avg Top Quintile']:>7.2f}%\n")
            f.write(f"  Bottom Quintile Ret: {summary['Avg Bottom Quintile']:>7.2f}%\n")
            f.write(f"  Long/Short Spread:   {summary['Avg Spread']:>7.2f}%\n")
            f.write(f"\n\n")

        # Comparison table
        f.write("\n" + "=" * 70 + "\n")
        f.write("COMPARISON TABLE (RANKED BY CORRELATION)\n")
        f.write("=" * 70 + "\n\n")

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Avg Correlation', ascending=False)

        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        # Interpretation
        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 70 + "\n\n")

        f.write("Correlation: Higher is better (measures linear relationship)\n")
        f.write("  • > 0.10: Strong predictive power\n")
        f.write("  • 0.05-0.10: Moderate predictive power\n")
        f.write("  • < 0.05: Weak predictive power\n\n")

        f.write("IC (Information Coefficient): Higher is better (rank correlation)\n")
        f.write("  • > 0.05: Excellent (institutional grade)\n")
        f.write("  • 0.03-0.05: Good\n")
        f.write("  • < 0.03: Weak\n\n")

        f.write("Hit Rate: % correct direction predictions\n")
        f.write("  • > 55%: Strong\n")
        f.write("  • 50-55%: Moderate\n")
        f.write("  • < 50%: Weak (worse than coin flip)\n\n")

        f.write("Long/Short Spread: Difference between top and bottom quintile returns\n")
        f.write("  • Larger spread = better stock selection ability\n")
        f.write("  • This is the basis for long/short strategies\n\n")

    print(f"\n📄 Summary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare walk-forward window strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default configurations (expanding + rolling 6,12,24,36 months)
  python test_window_strategies.py data.csv

  # Test specific rolling windows
  python test_window_strategies.py data.csv --windows 12 24

  # Quick test (sample every 3 months for speed)
  python test_window_strategies.py data.csv --sample-every 3

  # Custom forecast parameters
  python test_window_strategies.py data.csv --forecast-days 10 --target-return-days 90
        """
    )

    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('--windows', nargs='+', type=int, default=[6, 12, 24, 36],
                       help='Rolling window sizes in months (default: 6 12 24 36)')
    parser.add_argument('--forecast-days', type=int, default=10,
                       help='Days ahead to start measuring returns (default: 10)')
    parser.add_argument('--target-return-days', type=int, default=None,
                       help='Return period to predict (default: same as forecast-days)')
    parser.add_argument('--sample-every', type=int, default=1,
                       help='Process every Nth month (default: 1 = all months). Use 2-3 for faster testing.')
    parser.add_argument('--output', '-o', help='Output filename prefix (default: auto-generate)')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    # Load data
    print("=" * 70)
    print("  WALK-FORWARD WINDOW STRATEGY TESTING")
    print("=" * 70)
    print(f"\n📂 Loading {args.input}...")

    df = pd.read_csv(args.input)
    print(f"  ✓ Loaded {len(df):,} rows")

    # Initialize tester
    tester = WindowTester(
        forecast_days=args.forecast_days,
        target_return_days=args.target_return_days,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7
    )

    # Configure window strategies to test
    window_configs = [
        ('Expanding', 'expanding', None),
    ]

    for months in args.windows:
        window_configs.append((f'Rolling_{months}M', 'rolling', months))

    print(f"\n🔬 Testing {len(window_configs)} window strategies:")
    for name, wtype, months in window_configs:
        if wtype == 'expanding':
            print(f"  • {name}: Uses all historical data")
        else:
            print(f"  • {name}: Uses last {months} months only")

    if args.sample_every > 1:
        print(f"\n⚡ FAST MODE: Processing every {args.sample_every} months (for quick testing)")

    # Run comparison
    results, df = tester.run_comparison(df, window_configs, sample_every=args.sample_every)

    # Generate output filenames
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = input_path.stem + "_window_comparison"

    output_dir = input_path.parent

    # Save detailed monthly statistics
    all_monthly_stats = []
    for config_name, result in results.items():
        stats = result['monthly_stats'].copy()
        stats['strategy'] = config_name
        all_monthly_stats.append(stats)

    combined_stats = pd.concat(all_monthly_stats, ignore_index=True)
    stats_path = output_dir / f"{output_prefix}.csv"
    combined_stats.to_csv(stats_path, index=False)
    print(f"\n💾 Detailed monthly statistics saved to: {stats_path}")

    # Generate summary report
    summary_path = output_dir / f"{output_prefix}_summary.txt"
    generate_summary_report(results, summary_path)

    # Try to generate plots
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Walk-Forward Window Strategy Comparison', fontsize=16, fontweight='bold')

        # Plot 1: Correlation over time
        ax = axes[0, 0]
        for config_name, result in results.items():
            stats = result['monthly_stats']
            ax.plot(range(len(stats)), stats['correlation'], label=config_name, alpha=0.7)
        ax.set_title('Correlation Over Time')
        ax.set_xlabel('Month Index')
        ax.set_ylabel('Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Plot 2: IC over time
        ax = axes[0, 1]
        for config_name, result in results.items():
            stats = result['monthly_stats']
            ax.plot(range(len(stats)), stats['ic'], label=config_name, alpha=0.7)
        ax.set_title('Information Coefficient (IC) Over Time')
        ax.set_xlabel('Month Index')
        ax.set_ylabel('IC (Spearman)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Plot 3: Long/Short Spread
        ax = axes[1, 0]
        for config_name, result in results.items():
            stats = result['monthly_stats']
            ax.plot(range(len(stats)), stats['long_short_spread'], label=config_name, alpha=0.7)
        ax.set_title('Long/Short Spread Over Time')
        ax.set_xlabel('Month Index')
        ax.set_ylabel('Spread (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Plot 4: Average performance comparison
        ax = axes[1, 1]
        strategies = []
        corrs = []
        ics = []
        for config_name, result in results.items():
            stats = result['monthly_stats']
            strategies.append(config_name)
            corrs.append(stats['correlation'].mean())
            ics.append(stats['ic'].mean())

        x = np.arange(len(strategies))
        width = 0.35
        ax.bar(x - width/2, corrs, width, label='Correlation', alpha=0.8)
        ax.bar(x + width/2, ics, width, label='IC', alpha=0.8)
        ax.set_title('Average Performance Comparison')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        plot_path = output_dir / f"{output_prefix}_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_path}")

    except ImportError:
        print("\n⚠️  matplotlib not available - skipping plots")
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")

    print("\n" + "=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  1. {stats_path.name} - Detailed monthly metrics")
    print(f"  2. {summary_path.name} - Summary report with recommendations")
    if 'plot_path' in locals():
        print(f"  3. {plot_path.name} - Visual comparison")

    return 0


if __name__ == '__main__':
    exit(main())
