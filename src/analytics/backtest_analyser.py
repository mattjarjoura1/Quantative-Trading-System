"""Orchestrates post-run backtesting analytics."""

from dataclasses import dataclass

import numpy as np

from src.analytics.cost_model import BaseCostModel, FlatPerTrade
from src.analytics.metrics import BacktestMetrics, MetricsCalculator
from src.analytics.portfolio_tracker import PortfolioTracker
from src.types import TradeRecord

# Sort priorities within the same millisecond: fills before ticks
_FILL_PRIORITY = 0
_TICK_PRIORITY = 1


@dataclass(frozen=True)
class BacktestResult:
    """Container for all analytics outputs.

    Attributes:
        equity_curve: List of (timestamp_ms, equity) tuples.
        metrics: Computed performance metrics.
        tracker: The PortfolioTracker with final state (positions, cash, realised_pnl).
    """

    equity_curve: list[tuple[int, float]]
    metrics: BacktestMetrics
    tracker: PortfolioTracker


class BacktestAnalyser:
    """Drives post-run backtesting analytics.

    Merges trade_log and market_history into a single chronological
    timeline, walks it to build an equity curve via PortfolioTracker,
    then computes performance metrics.

    Args:
        trade_log: Chronological list of TradeRecord from BacktestOrchestrator.
        market_history: {symbol: list[tick]} from BacktestOrchestrator.
        initial_capital: Starting cash for the portfolio.
        cost_model: Transaction cost calculator. Defaults to zero cost.
    """

    def __init__(
        self,
        trade_log: list[TradeRecord],
        market_history: dict[str, list],
        initial_capital: float = 100_000.0,
        cost_model: BaseCostModel | None = None,
    ) -> None:
        self._trade_log = trade_log
        self._market_history = market_history
        self._initial_capital = initial_capital
        self._cost_model = cost_model or FlatPerTrade(0.0)

    def run(self) -> BacktestResult:
        """Execute the full analysis pipeline.

        1. Merge all events into unified timeline sorted by timestamp.
           Fills sort before ticks at the same timestamp so MtM reflects
           post-fill position state.
        2. Walk the timeline:
           - Fill event → tracker.on_fill(trade)
           - Tick event → update last_prices, tracker.mark_to_market(ts, last_prices)
        3. Extract equity curve as numpy array.
        4. Compute metrics via MetricsCalculator.
        5. Return BacktestResult.

        Returns:
            BacktestResult containing equity curve, metrics, and tracker reference.
        """
        tracker = PortfolioTracker(self._initial_capital, self._cost_model)
        events: list[tuple[int, int, object]] = []

        for trade in self._trade_log:
            events.append((trade.filled_at_ms, _FILL_PRIORITY, trade))

        for ticks in self._market_history.values():
            for tick in ticks:
                events.append((tick.timestamp_ms, _TICK_PRIORITY, tick))

        events.sort(key=lambda e: (e[0], e[1]))
    
        last_prices: dict[str, float] = {}

        for timestamp_ms, priority, payload in events:
            if priority == _FILL_PRIORITY:
                tracker.on_fill(payload)
            else:
                last_prices[payload.symbol] = payload.mtm_price()
                tracker.mark_to_market(timestamp_ms, last_prices)

        curve = tracker.equity_curve
        if curve:
            timestamps = np.array([t for t, _ in curve], dtype=np.float64)
            equity = np.array([e for _, e in curve], dtype=np.float64)
        else:
            timestamps = np.array([], dtype=np.float64)
            equity = np.array([], dtype=np.float64)

        metrics = MetricsCalculator.compute(
            equity,
            timestamps,
            num_trades=len(self._trade_log),
            trade_pnls=tracker.trade_pnls,
        )

        return BacktestResult(
            equity_curve=curve,
            metrics=metrics,
            tracker=tracker,
        )
