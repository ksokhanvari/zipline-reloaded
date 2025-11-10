"""
Real-time progress logging for Zipline backtests.

Displays live backtest progress with portfolio metrics similar to QuantRocket.
"""

import logging
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np


class BacktestProgressLogger:
    """
    Real-time progress logger for Zipline backtests.

    Displays progress bar, current date, and key portfolio metrics during
    backtest execution.
    """

    def __init__(
        self,
        algo_name: str = "Strategy",
        update_interval: int = 1,
        show_metrics: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize progress logger.

        Parameters
        ----------
        algo_name : str, optional
            Name of the algorithm/strategy to display in logs.
            Default is "Strategy".
        update_interval : int, optional
            How often to log progress (in trading days).
            1 = every day, 5 = every 5 days, etc.
            Default is 1 (daily updates).
        show_metrics : bool, optional
            Whether to show portfolio metrics. Default is True.
        logger : logging.Logger, optional
            Logger to use. If None, uses 'zipline.progress' logger.
        """
        self.algo_name = algo_name
        self.update_interval = update_interval
        self.show_metrics = show_metrics
        self.enabled = True

        # Use provided logger or create default
        if logger is None:
            self.logger = logging.getLogger('zipline.progress')

            # Clear any existing handlers to prevent duplicates
            self.logger.handlers.clear()

            # Add our handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

            # IMPORTANT: Prevent logs from propagating to root logger
            # This avoids duplicate messages when logging.basicConfig() is called multiple times
            self.logger.propagate = False

            # BUT: Also add any SocketHandlers from root logger to progress logger
            # so that FlightLog still receives progress logs
            root_logger = logging.getLogger()
            for root_handler in root_logger.handlers:
                # Check if this is a SocketHandler (without importing logging.handlers locally)
                if root_handler.__class__.__name__ == 'SocketHandler':
                    # Check if we don't already have this handler
                    has_socket = any(
                        h.__class__.__name__ == 'SocketHandler' and
                        getattr(h, 'host', None) == getattr(root_handler, 'host', None) and
                        getattr(h, 'port', None) == getattr(root_handler, 'port', None)
                        for h in self.logger.handlers
                    )
                    if not has_socket:
                        # Add the socket handler to progress logger too
                        self.logger.addHandler(root_handler)
        else:
            self.logger = logger

        # State tracking
        self.start_date = None
        self.end_date = None
        self.total_days = 0
        self.days_completed = 0
        self.start_portfolio_value = None
        self.peak_portfolio_value = None
        self.max_drawdown = 0  # Track maximum drawdown seen
        self.returns_history = []  # Track returns for Sharpe calculation
        self.header_shown = False

    def initialize(self, start_date: pd.Timestamp, end_date: pd.Timestamp, calendar):
        """
        Initialize progress tracking with backtest date range.

        Parameters
        ----------
        start_date : pd.Timestamp
            Backtest start date
        end_date : pd.Timestamp
            Backtest end date
        calendar : TradingCalendar
            Trading calendar for counting trading days
        """
        if not self.enabled:
            return

        self.start_date = start_date
        self.end_date = end_date

        # Count total trading days
        try:
            sessions = calendar.sessions_in_range(start_date, end_date)
            self.total_days = len(sessions)
        except:
            # Fallback if calendar doesn't work
            self.total_days = (end_date - start_date).days

        # Log initialization
        self.logger.info(
            f"[{self.algo_name}] Backtest initialized: "
            f"{start_date.date()} to {end_date.date()} "
            f"({self.total_days} trading days)"
        )

        # Show header row
        self._show_header()

    def _show_header(self):
        """Show column headers for metrics."""
        if not self.enabled or not self.show_metrics or self.header_shown:
            return

        # Header with proper alignment
        header = (
            f"[{self.algo_name}] "
            f"{'Progress':<12}  "
            f"{'Pct':<5}  "
            f"{'Date':<12}  "
            f"{'Cum Returns':>14}  "
            f"{'Sharpe':>12}  "
            f"{'Max DD':>10}  "
            f"{'Cum PNL':>18}"
        )
        self.logger.info(header)
        self.header_shown = True

    def update(self, dt: pd.Timestamp, daily: dict):
        """
        Update progress with current backtest state.

        Parameters
        ----------
        dt : pd.Timestamp
            Current simulation date
        daily : dict
            Daily performance dictionary from zipline
        """
        if not self.enabled:
            return

        self.days_completed += 1

        # ALWAYS track metrics (even if we don't log them)
        # This ensures max drawdown is accurate
        self._track_metrics(daily)

        # Only log at specified intervals
        if self.days_completed % self.update_interval != 0 and self.days_completed != self.total_days:
            return

        # Calculate progress
        progress_pct = (self.days_completed / self.total_days * 100) if self.total_days > 0 else 0
        progress_bar = self._make_progress_bar(progress_pct)

        # Format metrics for display
        metrics_str = self._format_metrics(daily) if self.show_metrics else ""

        # Format the progress line
        log_line = (
            f"[{self.algo_name}] "
            f"{progress_bar:<12}  "
            f"{int(progress_pct):>3}%  "
            f"{dt.date()}  "
            f"{metrics_str}"
        )

        self.logger.info(log_line)

    def _make_progress_bar(self, percent: float, width: int = 10) -> str:
        """Create a text progress bar."""
        filled = int(width * percent / 100)
        bar = '█' * filled + '-' * (width - filled)
        return bar

    def _track_metrics(self, daily: dict):
        """
        Track portfolio metrics on EVERY day (not just logging days).

        This ensures max drawdown is accurate even when update_interval > 1.

        Parameters
        ----------
        daily : dict
            Daily performance dictionary from zipline
        """
        # Portfolio value
        portfolio_value = daily.get("portfolio_value", 0)

        # Initialize starting value on first update
        if self.start_portfolio_value is None:
            self.start_portfolio_value = portfolio_value
            self.peak_portfolio_value = portfolio_value

        # Track returns for Sharpe calculation
        daily_return = daily.get("returns", 0)
        self.returns_history.append(daily_return)

        # Update peak portfolio value
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        # Track maximum drawdown
        if self.peak_portfolio_value > 0:
            current_drawdown = ((portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value) * 100
            # Update max drawdown if this is worse
            if current_drawdown < self.max_drawdown:
                self.max_drawdown = current_drawdown

    def _format_metrics(self, daily: dict) -> str:
        """
        Format portfolio metrics for display.

        Metrics displayed:
        - Cumulative Returns: percentage gain/loss from start
        - Sharpe Ratio: annualized Sharpe ratio
        - Max Drawdown: maximum decline from peak (percentage)
        - Cumulative PNL: dollar profit/loss
        """
        # Portfolio value
        portfolio_value = daily.get("portfolio_value", 0)

        # Calculate cumulative return
        if self.start_portfolio_value > 0:
            cum_return = (portfolio_value / self.start_portfolio_value - 1) * 100
        else:
            cum_return = 0.0

        # Calculate Sharpe ratio
        if len(self.returns_history) > 2:
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)

            if std_return > 0:
                # Annualized Sharpe (252 trading days)
                sharpe = (mean_return / std_return) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Calculate cumulative PNL
        cum_pnl = portfolio_value - self.start_portfolio_value

        # Format metrics with proper alignment
        metrics_str = (
            f"{cum_return:>13.0f}%  "
            f"{sharpe:>12.2f}  "
            f"{self.max_drawdown:>9.0f}%  "
            f"{self._format_currency(cum_pnl):>18}"
        )

        return metrics_str

    def _format_currency(self, value: float) -> str:
        """Format currency values with commas and abbreviations."""
        if abs(value) >= 1_000_000_000:
            return f"${value/1_000_000_000:,.2f}B"
        elif abs(value) >= 1_000_000:
            return f"${value/1_000_000:,.2f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:,.1f}K"
        else:
            return f"${value:,.0f}"

    def finalize(self, perf: pd.DataFrame):
        """
        Show final summary after backtest completion.

        Parameters
        ----------
        perf : pd.DataFrame
            Complete performance dataframe from zipline
        """
        if not self.enabled:
            return

        try:
            total_return = perf['algorithm_period_return'].iloc[-1] * 100

            # Get Sharpe from performance or calculate
            if 'sharpe' in perf.columns:
                sharpe = perf['sharpe'].iloc[-1]
            else:
                sharpe = 0.0

            # Get max drawdown
            if 'max_drawdown' in perf.columns:
                max_dd = perf['max_drawdown'].iloc[-1] * 100
            else:
                max_dd = abs(self.max_drawdown)

            # Get final portfolio value
            final_value = perf['portfolio_value'].iloc[-1]

            # Show summary
            self.logger.info(f"\n[{self.algo_name}] {'='*50}")
            self.logger.info(f"[{self.algo_name}] Backtest Complete!")
            self.logger.info(f"[{self.algo_name}] {'='*50}")
            self.logger.info(f"[{self.algo_name}] Trading Days:     {self.total_days}")
            self.logger.info(f"[{self.algo_name}] Total Return:     {total_return:+.2f}%")
            self.logger.info(f"[{self.algo_name}] Sharpe Ratio:     {sharpe:.2f}")
            self.logger.info(f"[{self.algo_name}] Max Drawdown:     {max_dd:.1f}%")
            self.logger.info(f"[{self.algo_name}] Final Value:      ${final_value:,.0f}")
            self.logger.info(f"[{self.algo_name}] {'='*50}\n")
        except Exception as e:
            self.logger.warning(f"[{self.algo_name}] Could not show final summary: {e}")

    def disable(self):
        """Disable progress logging."""
        self.enabled = False


# Global instance for convenience
_global_progress_logger = None


def enable_progress_logging(
    algo_name: str = "Strategy",
    update_interval: int = 1,
    show_metrics: bool = True,
) -> BacktestProgressLogger:
    """
    Enable global progress logging for backtests.

    This is the simplest way to add progress logging - just call this function
    before running your algorithm.

    Parameters
    ----------
    algo_name : str, optional
        Name of your strategy to display in logs.
        Default is "Strategy".
    update_interval : int, optional
        How often to log progress in trading days.
        - 1 = daily (good for short backtests < 1 year)
        - 5 = weekly (good for 1-5 year backtests)
        - 20 = monthly (good for 5-10 year backtests)
        - 60 = quarterly (good for 10+ year backtests)
        Default is 1.
    show_metrics : bool, optional
        Whether to show portfolio metrics. Default is True.

    Returns
    -------
    BacktestProgressLogger
        The global progress logger instance

    Examples
    --------
    >>> from zipline import run_algorithm
    >>> from zipline.utils.progress import enable_progress_logging
    >>>
    >>> # Enable progress logging
    >>> enable_progress_logging(algo_name='MyStrategy', update_interval=5)
    >>>
    >>> # Run backtest - progress will display automatically
    >>> result = run_algorithm(...)
    """
    global _global_progress_logger

    _global_progress_logger = BacktestProgressLogger(
        algo_name=algo_name,
        update_interval=update_interval,
        show_metrics=show_metrics,
    )

    return _global_progress_logger


def disable_progress_logging():
    """Disable global progress logging."""
    global _global_progress_logger
    if _global_progress_logger is not None:
        _global_progress_logger.disable()
        _global_progress_logger = None


def get_progress_logger() -> Optional[BacktestProgressLogger]:
    """Get the global progress logger instance, if enabled."""
    return _global_progress_logger
