"""Orchestrator for historical backtesting runs."""

import asyncio
import threading

from src.bus.buffer_view import BufferView
from src.data.backtest_data_source import BacktestDataSource
from src.orchestrator.base_orchestrator import BaseOrchestrator
from src.types import TradeRecord


class BacktestOrchestrator(BaseOrchestrator):
    """Runs the pipeline to completion on historical data.

    Extends BaseOrchestrator with:
    - Type guard: source must be a BacktestDataSource (paced replay).
    - Market data recording into market_history for post-run analysis.
    - run() that blocks until the data source is exhausted.

    After run() returns, two datasets are available:
    - self._execution.trade_log  — TradeRecord list (what trades happened)
    - self.market_history        — symbol → list of market data entries

    Args:
        config: Configuration dict (same format as BaseOrchestrator).

    Raises:
        TypeError: If source is not a BacktestDataSource.
    """

    def __init__(self, config: dict) -> None:
        """Build pipeline and register as a market data listener.

        Args:
            config: Configuration dict as described in BaseOrchestrator.

        Raises:
            TypeError: If the configured source is not a BacktestDataSource.
        """
        super().__init__(config)

        if not isinstance(self._source, BacktestDataSource):
            raise TypeError(
                f"{type(self._source).__name__} is not a BacktestDataSource. "
                f"Backtesting requires a paced data source."
            )

        md_ch = self._bus.channel("market_data")
        self._md_event = md_ch.register_listener("backtest_recorder")
        self._md_views: dict[str, BufferView] = {}
        self.market_history: dict[str, list] = {}

    def run(self) -> list[TradeRecord]:
        """Run the full pipeline to completion and return the trade log.

        Starts consumers and the market data recorder on threads, runs the
        async data source until data is exhausted, shuts everything down,
        and returns the trade log from SimulationExecution.

        Returns:
            List of TradeRecord objects in execution order.
        """
        threads = self._start_consumers()
        recorder_thread = threading.Thread(target=self._record_market_data)
        recorder_thread.start()

        asyncio.run(self._source.run())

        self._shutdown(threads)
        self._md_event.set()
        recorder_thread.join()

        return self._execution.trade_log

    def _record_market_data(self) -> None:
        """Drain market data into market_history for post-run analysis.

        Runs on its own thread. Wakes when market_data publishes, drains all
        new ticks per symbol into market_history, and exits once the source
        has finished.
        """
        while True:
            self._md_event.wait()
            self._md_event.clear()
            dirty = self._bus.channel("market_data").get_dirty("backtest_recorder")
            for symbol in dirty:
                if symbol not in self._md_views:
                    self._md_views[symbol] = BufferView(
                        self._bus.channel("market_data").get_buffer(symbol),
                        from_start=True,
                    )
                if symbol not in self.market_history:
                    self.market_history[symbol] = []
                self.market_history[symbol].extend(self._md_views[symbol].drain())
            if not self._source._running:
                break
