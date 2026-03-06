"""Abstract base class for all data sources."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.bus.message_bus import MessageBus
from src.types import OrderBookEntry

T = TypeVar("T")


class BaseDataSource(ABC, Generic[T]):
    """Producer ABC. Drives the pipeline by fetching and publishing market data.

    The ABC owns the run/stop lifecycle and channel publishing. The concrete
    class implements fetch() to pull data from whatever external source it
    wraps (websocket, file, API, DataFrame, etc.).

    Args:
        bus: The shared message bus.
        publish_channel: Name of the channel to publish market data to.
    """

    PRODUCES: type  # must be set by every concrete subclass

    def __init__(self, bus: MessageBus, publish_channel: str) -> None:
        """Resolve the publish channel and arm the run loop.

        Args:
            bus: The shared message bus.
            publish_channel: Name of the channel to publish to.
        """
        self._channel = bus.channel(publish_channel)
        self._running = True

    async def run(self) -> None:
        """Fetch-publish loop. Runs until stop() is called.

        Calls fetch() on every iteration. Non-None results are published to
        the channel. None results are silently skipped — the source has no
        data right now but isn't done.
        """
        while self._running:
            entry: T = await self.fetch()
            if entry is not None:
                self._channel.publish(entry.symbol, entry)

    def stop(self) -> None:
        """Signal the run loop to exit on its next iteration."""
        self._running = False

    @abstractmethod
    async def fetch(self) -> T | None:
        """Return the next order book entry, or None if nothing is available yet.

        Returns:
            The next OrderBookEntry from the underlying data source, or None.
        """