"""Passthrough risk engine — forwards every signal without modification."""

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.risk.base_risk_engine import BaseRiskEngine
from src.types import Signal


class PassthroughRisk(BaseRiskEngine):
    """Forwards every incoming signal unchanged.

    Placeholder until real risk logic is designed (position limits,
    drawdown checks, exposure caps). Uses lazy BufferView creation with
    from_start=True so no signal is missed on first encounter of a symbol.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for incoming signals.
        publish_channel: Channel name to publish approved signals to.
    """

    def __init__(
        self,
        bus: MessageBus,
        listener_id: str = "passthrough_risk",
        listen_channel: str = "strategy_signals",
        publish_channel: str = "approved_signals",
    ) -> None:
        """Register as a listener and initialise per-symbol buffer views.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for incoming signals.
            publish_channel: Channel name to publish approved signals to.
        """
        super().__init__(bus, listener_id, listen_channel, publish_channel)
        self._views: dict[str, BufferView] = {}

    def evaluate(self, dirty: set[str]) -> list[Signal]:
        """Forward all incoming signals without modification.

        Creates a BufferView lazily on first encounter of each symbol using
        from_start=True to avoid missing the signal that triggered the wake.

        Args:
            dirty: Set of symbols that have new signals since last wake.

        Returns:
            All signals from dirty symbols, unchanged.
        """
        signals = []
        for symbol in dirty:
            if symbol not in self._views:
                self._views[symbol] = BufferView(
                    self._listen_ch.get_buffer(symbol), from_start=True
                )
            signals.extend(self._views[symbol].drain())
        return signals
