# FlightLog - Best Practices and Patterns

Complete guide to using FlightLog elegantly in your trading strategies.

## Table of Contents
- [Quick Start](#quick-start)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)
- [Log Levels](#log-levels)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Start FlightLog Server

**Terminal 1:**
```bash
# Inside Docker container
python scripts/flightlog.py --host 0.0.0.0 --level INFO
```

### 2. Connect in Your Strategy

```python
import logging
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog

# Setup logging
logging.basicConfig(level=logging.INFO)

# Connect to FlightLog (use localhost inside Docker)
enable_flightlog(host='localhost', port=9020)

# Enable progress bars
enable_progress_logging(algo_name='My-Strategy', update_interval=5)
```

### 3. Log Important Events

```python
def initialize(context):
    log_to_flightlog('Strategy initialized', level='INFO')

def rebalance(context, data):
    log_to_flightlog('Portfolio rebalanced', level='INFO')
```

## Best Practices

### ✅ DO

**1. Log Strategic Events**
```python
# Good: Log important decisions
log_to_flightlog('Entered new position in AAPL', level='INFO')
log_to_flightlog('Risk limit reached - reducing exposure', level='WARNING')
log_to_flightlog('Rebalancing portfolio (5 positions)', level='INFO')
```

**2. Use Appropriate Log Levels**
```python
# INFO: Normal operations
log_to_flightlog('Daily rebalance complete', level='INFO')

# WARNING: Notable but not critical
log_to_flightlog('Low liquidity detected for TSLA', level='WARNING')

# ERROR: Something went wrong
log_to_flightlog('Failed to place order for AAPL', level='ERROR')
```

**3. Include Context in Messages**
```python
# Good: Informative
portfolio_value = context.portfolio.portfolio_value
positions = len(context.portfolio.positions)
log_to_flightlog(
    f'Rebalance complete: {positions} positions, ${portfolio_value:,.0f}',
    level='INFO'
)

# Bad: Too vague
log_to_flightlog('Done', level='INFO')
```

**4. Log Summary Metrics**
```python
def analyze(context, perf):
    total_return = perf['algorithm_period_return'].iloc[-1] * 100
    sharpe = perf['sharpe'].iloc[-1] if 'sharpe' in perf.columns else None

    log_to_flightlog('=' * 50, level='INFO')
    log_to_flightlog(f'Total Return: {total_return:+.2f}%', level='INFO')
    if sharpe:
        log_to_flightlog(f'Sharpe Ratio: {sharpe:.2f}', level='INFO')
    log_to_flightlog('=' * 50, level='INFO')
```

### ❌ DON'T

**1. Don't Log Too Frequently**
```python
# Bad: Logs on every bar (too noisy!)
def handle_data(context, data):
    log_to_flightlog('Processing bar...', level='INFO')  # ❌
```

**2. Don't Duplicate Progress Information**
```python
# Bad: Progress logging already shows this
def handle_data(context, data):
    log_to_flightlog(
        f'Portfolio value: ${context.portfolio.portfolio_value}',
        level='INFO'
    )  # ❌ Already in progress bar
```

**3. Don't Log Sensitive Information**
```python
# Bad: Never log credentials
log_to_flightlog(f'API Key: {api_key}', level='INFO')  # ❌ SECURITY RISK
```

**4. Don't Use Wrong Log Levels**
```python
# Bad: Everything is not CRITICAL
log_to_flightlog('Placed order', level='CRITICAL')  # ❌ Wrong level
```

## Common Patterns

### Pattern 1: Initialization Logging

```python
def initialize(context):
    log_to_flightlog('Initializing strategy...', level='INFO')

    # Setup parameters
    context.lookback = 20
    context.top_n = 5

    # Log configuration
    log_to_flightlog(
        f'Config: lookback={context.lookback}d, top_n={context.top_n}',
        level='INFO'
    )

    # Setup universe
    context.stocks = [symbol('AAPL'), symbol('MSFT'), symbol('GOOGL')]

    log_to_flightlog(
        f'Universe: {len(context.stocks)} stocks',
        level='INFO'
    )
```

### Pattern 2: Conditional Logging

```python
def rebalance(context, data):
    # Only log when actually rebalancing
    if context.days_since_rebalance < REBALANCE_INTERVAL:
        return  # Don't log when not rebalancing

    log_to_flightlog(
        f'Rebalancing (interval={REBALANCE_INTERVAL} days)',
        level='INFO'
    )

    # ... rebalancing logic ...
```

### Pattern 3: Error Handling with Logging

```python
def safe_calculate_momentum(stock, data):
    try:
        prices = data.history(stock, 'price', 20, '1d')
        momentum = (prices[-1] / prices[0] - 1) * 100
        return momentum
    except Exception as e:
        log_to_flightlog(
            f'Error calculating momentum for {stock.symbol}: {str(e)}',
            level='WARNING'
        )
        return None
```

### Pattern 4: Progress Milestones

```python
def rebalance(context, data):
    context.rebalance_count += 1

    # Log milestones
    if context.rebalance_count % 10 == 0:
        log_to_flightlog(
            f'Milestone: {context.rebalance_count} rebalances completed',
            level='INFO'
        )
```

### Pattern 5: Risk Monitoring

```python
def check_risk_limits(context, data):
    portfolio_value = context.portfolio.portfolio_value
    cash = context.portfolio.cash
    leverage = context.account.leverage

    # Log warnings when approaching limits
    if leverage > 1.5:
        log_to_flightlog(
            f'High leverage detected: {leverage:.2f}x',
            level='WARNING'
        )

    if cash / portfolio_value < 0.05:
        log_to_flightlog(
            f'Low cash warning: {cash/portfolio_value*100:.1f}% of portfolio',
            level='WARNING'
        )
```

## Log Levels

### DEBUG
Detailed diagnostic information for debugging.

```python
log_to_flightlog('Calculating indicators for AAPL', level='DEBUG')
log_to_flightlog(f'Momentum score: {score:.4f}', level='DEBUG')
```

**When to use:** Development, troubleshooting, verbose mode

### INFO
General informational messages about normal operations.

```python
log_to_flightlog('Portfolio rebalanced successfully', level='INFO')
log_to_flightlog('Strategy initialized with 10 stocks', level='INFO')
```

**When to use:** Normal operations, milestones, summaries

### WARNING
Warning messages for unusual but recoverable situations.

```python
log_to_flightlog('No valid signals found - skipping rebalance', level='WARNING')
log_to_flightlog('Stock TSLA has low liquidity', level='WARNING')
```

**When to use:** Unusual conditions, potential issues, degraded performance

### ERROR
Error messages for serious problems that need attention.

```python
log_to_flightlog('Failed to place order for AAPL', level='ERROR')
log_to_flightlog('Data missing for critical calculations', level='ERROR')
```

**When to use:** Failures, exceptions, data issues

### CRITICAL
Very serious errors that may cause the strategy to fail.

```python
log_to_flightlog('Risk limit exceeded - halting trading', level='CRITICAL')
log_to_flightlog('Data feed disconnected', level='CRITICAL')
```

**When to use:** Critical failures, safety violations, emergency stops

## Performance Tips

### 1. Use Appropriate Update Intervals

```python
# Short backtests (< 1 year)
enable_progress_logging(update_interval=1)  # Daily

# Medium backtests (1-5 years)
enable_progress_logging(update_interval=5)  # Weekly

# Long backtests (5-10 years)
enable_progress_logging(update_interval=20)  # Monthly

# Very long backtests (10+ years)
enable_progress_logging(update_interval=60)  # Quarterly
```

### 2. Conditional Logging

```python
# Only log on important events
if context.portfolio.positions_value > 0:
    log_to_flightlog('First position entered', level='INFO')
```

### 3. Batch Summary Logging

```python
# Instead of logging each order...
# orders = []
# for stock in stocks:
#     log_to_flightlog(f'Ordering {stock}')  # ❌ Too many logs

# Log summary
log_to_flightlog(f'Placed {len(orders)} orders', level='INFO')  # ✅ Better
```

## Troubleshooting

### FlightLog Not Connecting

**Problem:** `enable_flightlog()` returns `False`

**Solution:**
1. Check FlightLog is running: `docker compose ps flightlog`
2. Verify hostname: Use `localhost` inside Docker, not `flightlog` service name when in same container
3. Check port: Default is 9020
4. Test connection:
   ```python
   import socket
   sock = socket.socket()
   result = sock.connect_ex(('localhost', 9020))
   print("Open" if result == 0 else "Closed")
   ```

### Messages Not Appearing

**Problem:** Messages sent but not appearing in FlightLog

**Solution:**
1. Check log level: `logging.basicConfig(level=logging.INFO)`
2. Verify FlightLog level: Start with `--level DEBUG` to see everything
3. Check logger propagation:
   ```python
   logger = logging.getLogger('algorithm')
   logger.setLevel(logging.INFO)
   ```

### Duplicate Messages

**Problem:** Same message appears twice

**Solution:**
- You might have multiple handlers. Check:
  ```python
  import logging
  root = logging.getLogger()
  print(f"Handlers: {root.handlers}")
  ```

### Performance Issues

**Problem:** Backtest runs slowly with FlightLog

**Solution:**
1. Reduce logging frequency (don't log in `handle_data`)
2. Increase `update_interval` for progress logging
3. Use INFO level instead of DEBUG in production
4. Batch messages instead of individual logs

## Example: Complete Production Strategy

```python
import logging
from zipline import run_algorithm
from zipline.api import *
from zipline.utils.progress import enable_progress_logging
from zipline.utils.flightlog_client import enable_flightlog, log_to_flightlog
import pandas as pd

# Setup
logging.basicConfig(level=logging.INFO)
enable_flightlog(host='localhost', port=9020)
enable_progress_logging(algo_name='Production-Strategy', update_interval=5)

def initialize(context):
    """Initialize with clean logging."""
    log_to_flightlog('=== STRATEGY INITIALIZATION ===', level='INFO')

    context.stocks = [symbol('AAPL'), symbol('MSFT'), symbol('GOOGL')]
    context.rebalance_days = 5
    context.days_since_rebalance = 0

    log_to_flightlog(
        f'Config: {len(context.stocks)} stocks, rebalance every {context.rebalance_days} days',
        level='INFO'
    )

    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())

    log_to_flightlog('Initialization complete', level='INFO')

def rebalance(context, data):
    """Rebalance with strategic logging."""
    context.days_since_rebalance += 1

    if context.days_since_rebalance < context.rebalance_days:
        return

    context.days_since_rebalance = 0

    try:
        # Log rebalance start
        log_to_flightlog('Starting rebalance...', level='INFO')

        # Equal weight
        weight = 1.0 / len(context.stocks)

        # Place orders
        for stock in context.stocks:
            if data.can_trade(stock):
                order_target_percent(stock, weight)

        # Log success
        portfolio_value = context.portfolio.portfolio_value
        positions = len(context.portfolio.positions)
        log_to_flightlog(
            f'Rebalance complete: {positions} positions, ${portfolio_value:,.0f}',
            level='INFO'
        )

    except Exception as e:
        log_to_flightlog(f'Rebalance failed: {str(e)}', level='ERROR')

def analyze(context, perf):
    """Final analysis with summary logging."""
    total_return = perf['algorithm_period_return'].iloc[-1] * 100
    final_value = perf['portfolio_value'].iloc[-1]

    log_to_flightlog('=' * 60, level='INFO')
    log_to_flightlog('BACKTEST COMPLETE', level='INFO')
    log_to_flightlog(f'Total Return: {total_return:+.2f}%', level='INFO')
    log_to_flightlog(f'Final Value: ${final_value:,.0f}', level='INFO')
    log_to_flightlog('=' * 60, level='INFO')

# Run
result = run_algorithm(
    start=pd.Timestamp('2023-01-01'),
    end=pd.Timestamp('2023-12-31'),
    initialize=initialize,
    analyze=analyze,
    capital_base=100000,
    bundle='sharadar'
)
```

## Summary

**Golden Rules:**
1. 🎯 **Log strategically** - only important events
2. 🎨 **Use proper levels** - INFO for normal, WARNING/ERROR for issues
3. 📊 **Include context** - make messages informative
4. ⚡ **Avoid spam** - don't log in `handle_data()`
5. 🔒 **Stay secure** - never log credentials
6. 📈 **Set intervals** - adjust `update_interval` for backtest length
7. ✅ **Test first** - verify FlightLog connection before long runs

FlightLog + Progress Logging = Professional monitoring! 🚀
