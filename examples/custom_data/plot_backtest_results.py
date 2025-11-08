#!/usr/bin/env python
"""
Plot Backtest Results

Visualizes the performance of the fundamental-based strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
print("Loading backtest results...")
perf = pd.read_csv('backtest_results.csv', index_col=0, parse_dates=True)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# 1. Portfolio Value Over Time
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(perf.index, perf['portfolio_value'], linewidth=2, label='Portfolio Value')
ax1.axhline(y=perf['portfolio_value'].iloc[0], color='gray', linestyle='--',
            alpha=0.5, label='Initial Capital')
ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# ============================================================================
# 2. Cumulative Returns
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
cumulative_returns = (perf['portfolio_value'] / perf['portfolio_value'].iloc[0] - 1) * 100
ax2.plot(perf.index, cumulative_returns, linewidth=2, color='green')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Return (%)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

# ============================================================================
# 3. Drawdown
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
daily_returns = perf['portfolio_value'].pct_change()
cumulative = (1 + daily_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = ((cumulative - running_max) / running_max) * 100
ax3.fill_between(perf.index, drawdown, 0, color='red', alpha=0.3)
ax3.plot(perf.index, drawdown, linewidth=1, color='darkred')
ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Drawdown (%)', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

# ============================================================================
# 4. Daily Returns Distribution
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])
daily_returns_clean = daily_returns.dropna() * 100
ax4.hist(daily_returns_clean, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax4.axvline(x=daily_returns_clean.mean(), color='green', linestyle='--',
            alpha=0.5, label=f'Mean: {daily_returns_clean.mean():.2f}%')
ax4.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Daily Return (%)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 5. Monthly Returns Heatmap
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

# Calculate monthly returns
monthly_returns = perf['portfolio_value'].resample('M').last().pct_change() * 100
monthly_returns = monthly_returns.dropna()

if len(monthly_returns) > 0:
    # Create pivot table for heatmap
    monthly_data = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })

    pivot = monthly_data.pivot_table(values='Return', index='Month', columns='Year')

    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                linewidths=1, linecolor='black', cbar_kws={'label': 'Return (%)'}, ax=ax5)
    ax5.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Year', fontsize=12)
    ax5.set_ylabel('Month', fontsize=12)
else:
    ax5.text(0.5, 0.5, 'Insufficient data for monthly heatmap',
             ha='center', va='center', fontsize=12)
    ax5.axis('off')

# ============================================================================
# Save and Show
# ============================================================================
plt.savefig('backtest_performance.png', dpi=150, bbox_inches='tight')
print("✓ Performance chart saved to: backtest_performance.png")

plt.show()

# ============================================================================
# Print Statistics
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE STATISTICS")
print("="*70)

total_return = cumulative_returns.iloc[-1]
sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
max_dd = drawdown.min()
volatility = daily_returns.std() * np.sqrt(252) * 100

print(f"Total Return:        {total_return:.2f}%")
print(f"Annualized Return:   {(((perf['portfolio_value'].iloc[-1] / perf['portfolio_value'].iloc[0]) ** (252/len(perf))) - 1) * 100:.2f}%")
print(f"Annualized Volatility: {volatility:.2f}%")
print(f"Sharpe Ratio:        {sharpe:.2f}")
print(f"Max Drawdown:        {max_dd:.2f}%")
print(f"Best Day:            {daily_returns.max()*100:.2f}%")
print(f"Worst Day:           {daily_returns.min()*100:.2f}%")
print(f"Win Rate:            {(daily_returns > 0).sum() / len(daily_returns.dropna()) * 100:.1f}%")
print("="*70 + "\n")
