"""Abstract base class for paced backtest data sources."""

import asyncio
from abc import abstractmethod
from typing import Generic, TypeVar

from src.data.base_data_source import BaseDataSource

T = TypeVar("T")


class BacktestDataSource(BaseDataSource[T], Generic[T]):
    """Paced data source ABC for backtesting.

    Overrides BaseDataSource.run() to wait until all downstream consumers
    have processed the previous tick before emitting the next one. This
    guarantees the ring buffer never accumulates a backlog, which is
    critical for two reasons:

    - No ticks are lost to buffer overflow, preserving backtest accuracy.
    - SimulationExecution can fill signals at the next tick's price, giving
      an honest slippage estimate. Without pacing, "next tick" in the buffer
      could be many ticks ahead in market time.

    Concrete subclasses implement only fetch() — pacing is handled here.
    Live data sources inherit from BaseDataSource directly and do not pace.
    This class hierarchy also acts as a type guard: the backtest orchestrator
    can assert isinstance(source, BacktestDataSource) before wiring.
    """

    async def run(self) -> None:
        """Paced run loop. Waits for all consumers to process before each tick.

        Replaces BaseDataSource.run(). Yields to the event loop while
        consumers are busy so they have CPU time to clear their events.
        """
        while self._running:
            while not self._channel.all_listeners_clear():
                await asyncio.sleep(0)

            entry = await self.fetch()
            if entry is not None:
                self._channel.publish(entry.symbol, entry)

    @abstractmethod
    async def fetch(self) -> T | None:
        """Return the next item to publish, or None if not yet available.

        Returns:
            Next data item, or None. Returning None skips the publish but
            does not stop the loop — call self.stop() to terminate.
        """
