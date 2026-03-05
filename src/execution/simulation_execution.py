"""Simulated execution handler for backtesting."""

import time

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.execution.base_execution import BaseExecution
from src.types import TradeRecord, Signal


class SimulationExecution(BaseExecution):
    """Records approved signals as trade records without executing real orders.

    Accepts any symbol by creating BufferViews lazily on first encounter,
    using from_start=True to avoid missing the signal that triggered the wake.
    Fill price defaults to signal.price (perfect fill assumption).

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
        """Register as a listener and initialise the trade log.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for approved signals.
        """
        super().__init__(bus, listener_id, listen_channel)
        self._views: dict[str, BufferView[Signal]] = {}
        self.trade_log: list[TradeRecord] = []

    def execute(self, dirty: set[str]) -> None:
        """Drain approved signals for each dirty symbol and record them.

        Args:
            dirty: Set of symbols that have new approved signals since last wake.
        """
        now_ms = int(time.time() * 1000)
        for symbol in dirty:
            if symbol not in self._views:
                self._views[symbol] = BufferView(
                    self._listen_ch.get_buffer(symbol), from_start=True
                )
            
            for signal in self._views[symbol].drain():
                self.trade_log.append(TradeRecord(
                    signal=signal,
                    fill_price=signal.price, # TODO: get the last updated price in the buffer
                    filled_at_ms=now_ms,
                ))
