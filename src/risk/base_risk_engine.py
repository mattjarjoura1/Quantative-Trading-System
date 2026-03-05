"""Abstract base class for all risk engines."""

from abc import ABC, abstractmethod

from src.bus.message_bus import MessageBus
from src.types import Signal


class BaseRiskEngine(ABC):
    """Consumer/producer ABC. Reads signals from one channel, publishes approved signals to another.

    Sits between strategy and execution. Listens on the strategy's output
    channel, evaluates each signal for risk, and either passes it through,
    modifies it, or vetoes it by returning None.

    Because listen and publish channels are independent, publishing to the
    approved_signals channel never wakes this engine's own listener — the
    infinite loop that would arise with a flat listener pool is structurally
    impossible.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for incoming signals.
        publish_channel: Channel name to publish approved signals to.
    """

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
            listen_channel: Channel name to listen on for incoming signals.
            publish_channel: Channel name to publish approved signals to.
        """
        self._listener_id = listener_id
        self._listen_ch = bus.channel(listen_channel)
        self._publish_ch = bus.channel(publish_channel)
        self._event = self._listen_ch.register_listener(listener_id)
        self._running = True

    def run(self) -> None:
        """Event-driven loop. Sleeps until a signal arrives, then calls evaluate.

        Publishes all returned Signals to the publish channel.
        """
        while self._running:
            self._event.wait()
            self._event.clear()
            dirty = self._listen_ch.get_dirty(self._listener_id)
            for signal in self.evaluate(dirty):
                self._publish_ch.publish(signal.symbol, signal)

    def stop(self) -> None:
        """Signal the run loop to exit and unblock any sleeping event.wait()."""
        self._running = False
        self._event.set()

    @abstractmethod
    def evaluate(self, dirty: set[str]) -> list[Signal]:
        """Evaluate incoming signals and decide whether to pass them through.

        Args:
            dirty: Set of symbols that have new signals since last wake.

        Returns:
            A list of approved Signals to forward. Empty list vetoes all.
        """
