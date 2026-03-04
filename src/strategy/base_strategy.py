"""Abstract base class for all strategies."""

from abc import ABC, abstractmethod

from src.bus.message_bus import MessageBus
from src.types import Signal


class BaseStrategy(ABC):
    """Consumer ABC. Reads market data from the bus, computes, and publishes signals.

    The ABC owns the listener lifecycle: event wait, clear, get dirty, publish.
    The concrete class owns computation and decides what "ready" means.
    """

    def __init__(self, bus: MessageBus, listener_id: str) -> None:
        """Register as a bus listener and arm the run loop.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
        """
        self._bus = bus
        self._listener_id = listener_id
        self._event = bus.register_listener(listener_id)
        self._running = True

    def run(self) -> None:
        """Event-driven loop. Sleeps until data arrives, then calls on_data.

        Publishes the returned Signal to the bus if on_data returns one.
        """
        while self._running:
            self._event.wait()
            self._event.clear()
            dirty = self._bus.get_dirty(self._listener_id)
            signal = self.on_data(dirty)
            if signal:
                self._bus.publish_signal(signal.symbol, signal)

    def stop(self) -> None:
        """Signal the run loop to exit and unblock any sleeping event.wait()."""
        self._running = False
        self._event.set()

    @abstractmethod
    def on_data(self, dirty: set[str]) -> Signal | None:
        """Process new market data and optionally emit a signal.

        Args:
            dirty: Set of symbols that have new data since last wake.

        Returns:
            A Signal to publish, or None if no action required.
        """
