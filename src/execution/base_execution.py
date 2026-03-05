"""Abstract base class for all execution handlers."""

from abc import ABC, abstractmethod

from src.bus.message_bus import MessageBus


class BaseExecution(ABC):
    """Terminal consumer ABC. Reads approved signals and acts on them.

    Never publishes back to the bus. Concrete implementations range from a
    simple trade logger (backtesting) to a live exchange connector.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for approved signals.
    """

    def __init__(
        self,
        bus: MessageBus,
        listener_id: str,
        listen_channel: str,
    ) -> None:
        """Resolve the listen channel, register as a listener, and arm the run loop.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for approved signals.
        """
        self._listener_id = listener_id
        self._listen_ch = bus.channel(listen_channel)
        self._event = self._listen_ch.register_listener(listener_id)
        self._running = True

    def run(self) -> None:
        """Event-driven loop. Sleeps until a signal arrives, then calls execute."""
        while self._running:
            self._event.wait()
            self._event.clear()
            dirty = self._listen_ch.get_dirty(self._listener_id)
            self.execute(dirty)

    def stop(self) -> None:
        """Signal the run loop to exit and unblock any sleeping event.wait()."""
        self._running = False
        self._event.set()

    @abstractmethod
    def execute(self, dirty: set[str]) -> None:
        """Act on approved signals.

        Args:
            dirty: Set of symbols that have new signals since last wake.
        """
