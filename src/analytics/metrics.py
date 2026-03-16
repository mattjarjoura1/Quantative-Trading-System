"""Performance metrics calculator for backtesting analytics."""

from dataclasses import dataclass

import numpy as np

_MS_PER_YEAR = 365.25 * 24 * 3600 * 1000


@dataclass(frozen=True)
class BacktestMetrics:
    """Performance metrics computed from a backtest equity curve.

    Attributes:
        total_pnl: Final equity minus initial equity (dollar amount).
        total_return_pct: (final / initial) - 1 as a percentage.
        cagr: Compound annual growth rate. Uses actual calendar duration.
        annualised_sharpe: Sharpe ratio annualised from data frequency. ddof=1, rf=0.
            NaN when std of returns is zero.
        max_drawdown_pct: Worst peak-to-trough decline as a positive percentage.
            0.0 when equity is monotonically increasing.
        max_drawdown_duration_ms: Longest time spent below a previous peak in ms.
            0 when no drawdown occurs.
        num_trades: Total number of fills in the trade log.
        win_rate: Fraction of profitable trades (0.0 to 1.0). NaN when num_trades is 0.
        profit_factor: Gross profit / gross loss. Inf when no losing trades. NaN when no trades.
    """

    total_pnl: float
    total_return_pct: float
    cagr: float
    annualised_sharpe: float
    max_drawdown_pct: float
    max_drawdown_duration_ms: int
    num_trades: int
    win_rate: float
    profit_factor: float


class MetricsCalculator:
    """Computes performance metrics from an equity curve.

    All methods are static — no state. Takes numpy arrays, returns numbers.
    """

    @staticmethod
    def compute(
        equity: np.ndarray,
        timestamps: np.ndarray,
        num_trades: int = 0,
        trade_pnls: list[float] | None = None,
    ) -> BacktestMetrics:
        """Compute all metrics and return as BacktestMetrics.

        Args:
            equity: 1D array of portfolio equity values (from tracker.equity_curve).
            timestamps: 1D array of millisecond timestamps (same length as equity).
            num_trades: Total number of executions in the trade log.
            trade_pnls: Realised PnL per position-close from PortfolioTracker.trade_pnls.

        Returns:
            BacktestMetrics with all fields populated.
        """
        pnls = trade_pnls or []

        if len(equity) == 0:
            return BacktestMetrics(
                total_pnl=0.0,
                total_return_pct=0.0,
                cagr=0.0,
                annualised_sharpe=float("nan"),
                max_drawdown_pct=0.0,
                max_drawdown_duration_ms=0,
                num_trades=num_trades,
                win_rate=MetricsCalculator._win_rate(pnls),
                profit_factor=MetricsCalculator._profit_factor(pnls),
            )

        total_pnl = float(equity[-1] - equity[0])
        total_return_pct = float(equity[-1] / equity[0]) - 1.0
        cagr = MetricsCalculator._cagr(equity, timestamps)
        sharpe = MetricsCalculator._sharpe(equity, timestamps)
        max_dd, max_dd_dur = MetricsCalculator._drawdown(equity, timestamps)

        return BacktestMetrics(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            cagr=cagr,
            annualised_sharpe=sharpe,
            max_drawdown_pct=max_dd,
            max_drawdown_duration_ms=max_dd_dur,
            num_trades=num_trades,
            win_rate=MetricsCalculator._win_rate(pnls),
            profit_factor=MetricsCalculator._profit_factor(pnls),
        )

    @staticmethod
    def _cagr(equity: np.ndarray, timestamps: np.ndarray) -> float:
        """Compute compound annual growth rate.

        Args:
            equity: Equity values array.
            timestamps: Timestamp array in milliseconds.

        Returns:
            CAGR as a decimal (e.g. 0.10 for 10%). 0.0 if duration is zero.
        """
        duration_ms = float(timestamps[-1] - timestamps[0])
        if duration_ms <= 0:
            return 0.0
        years = duration_ms / _MS_PER_YEAR
        if years < 1.0 / 365.25:  # less than one day — CAGR is meaningless
            return 0.0
        return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)

    @staticmethod
    def _sharpe(equity: np.ndarray, timestamps: np.ndarray) -> float:
        """Compute annualised Sharpe ratio (rf=0, ddof=1).

        Args:
            equity: Equity values array.
            timestamps: Timestamp array in milliseconds.

        Returns:
            Annualised Sharpe ratio. NaN when std of returns is zero or
            fewer than 2 data points.
        """
        if len(equity) < 3:  # need at least 2 returns for ddof=1
            return float("nan")
        returns = np.diff(equity) / equity[:-1]
        std = float(np.std(returns, ddof=1))
        if std < 1e-12:
            return float("nan")
        mean_interval_ms = float(np.mean(np.diff(timestamps)))
        if mean_interval_ms <= 0:
            return float("nan")
        periods_per_year = _MS_PER_YEAR / mean_interval_ms
        return float(np.mean(returns) / std * np.sqrt(periods_per_year))

    @staticmethod
    def _drawdown(equity: np.ndarray, timestamps: np.ndarray) -> tuple[float, int]:
        """Compute max drawdown percentage and max drawdown duration in ms.

        Args:
            equity: Equity values array.
            timestamps: Timestamp array in milliseconds.

        Returns:
            Tuple of (max_drawdown_pct as positive float, max_drawdown_duration_ms).
        """
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        max_dd = float(abs(min(drawdowns)))

        # Max drawdown duration: longest run where equity < peak
        in_dd = equity < peaks
        max_dur = 0
        start_idx = None

        for i, is_below in enumerate(in_dd):
            if is_below and start_idx is None:
                start_idx = i
            elif not is_below and start_idx is not None:
                # Duration from peak (start_idx - 1) to last below-peak tick (i - 1)
                dur = int(timestamps[i - 1] - timestamps[start_idx - 1])
                max_dur = max(max_dur, dur)
                start_idx = None

        # If still in drawdown at end, extend to final timestamp
        if start_idx is not None:
            dur = int(timestamps[-1] - timestamps[start_idx - 1])
            max_dur = max(max_dur, dur)

        return max_dd, max_dur

    @staticmethod
    def _win_rate(trade_pnls: list[float]) -> float:
        """Compute fraction of position-closes with positive realised PnL.

        Args:
            trade_pnls: Realised PnL per position-close (from PortfolioTracker).

        Returns:
            Win rate in [0, 1]. NaN when no closed positions.
        """
        if not trade_pnls:
            return float("nan")
        return sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)

    @staticmethod
    def _profit_factor(trade_pnls: list[float]) -> float:
        """Compute gross profit divided by gross loss across all position-closes.

        Args:
            trade_pnls: Realised PnL per position-close (from PortfolioTracker).

        Returns:
            Profit factor. Inf when no losing trades. NaN when no closed positions.
            0.0 when no winning trades.
        """
        if not trade_pnls:
            return float("nan")
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0))
        if gross_loss < 1e-12:
            return float("inf") if gross_profit > 0 else float("nan")
        return gross_profit / gross_loss
