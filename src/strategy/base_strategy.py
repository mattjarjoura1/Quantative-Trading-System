"""Abstract base class for all strategies."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.bus.message_bus import MessageBus
from src.types import Signal

T = TypeVar("T")


class BaseStrategy(ABC, Generic[T]):
    """Consumer/producer ABC. Reads market data from one channel, publishes signals to another.

    The ABC owns the listener lifecycle: register, event wait, clear, get dirty,
    publish. The concrete class owns computation and decides what "ready" means.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for market data.
        publish_channel: Channel name to publish signals to.
    """

    CONSUMES: type  # must be set by every concrete subclass

    def __init__(
        self,
        bus: MessageBus,
        listener_id: str,
        listen_channel: str,
        publish_channel: str,
    ) -> None:
        """Resolve channels, register as a listener, and arm the run loop.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for market data.
            publish_channel: Channel name to publish signals to.
        """
        self._listener_id = listener_id
        self._listen_ch = bus.channel(listen_channel)
        self._publish_ch = bus.channel(publish_channel)
        self._event = self._listen_ch.register_listener(listener_id)
        self._running = True

    def run(self) -> None:
        """Event-driven loop. Sleeps until data arrives, then calls on_data.

        Publishes the returned Signal to the publish channel if on_data
        returns one.
        """
        while self._running:
            self._event.wait()
            self._event.clear()
            dirty = self._listen_ch.get_dirty(self._listener_id)
            signals = self.on_data(dirty)
            for signal in signals:
                self._publish_ch.publish(signal.symbol, signal)

    def stop(self) -> None:
        """Signal the run loop to exit and unblock any sleeping event.wait()."""
        self._running = False
        self._event.set()

    @abstractmethod
    def on_data(self, dirty: set[str]) -> list[Signal]:
        """Process new market data and emit zero or more signals.

        Args:
            dirty: Set of symbols that have new data since last wake.

        Returns:
            A list of Signals to publish. Empty list means no action.
        """
