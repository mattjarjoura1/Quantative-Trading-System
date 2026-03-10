"""Simulated execution handler for backtesting."""

import time

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.execution.base_execution import BaseExecution
from src.types import Signal, TradeRecord


class SimulationExecution(BaseExecution):
    """Records approved signals as trade records without executing real orders.

    Fill price is read from the market data channel at the time of execution
    via each contract's fill_price(side) method. Because BacktestDataSource
    paces one tick at a time, latest() on the market buffer returns the tick
    that triggered the signal — not a future price.

    Falls back to signal.price if no market data has arrived yet for a symbol.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for approved signals.
        market_channel: Channel name to read current market prices from.
    """

    def __init__(
        self,
        bus: MessageBus,
        listener_id: str,
        listen_channel: str,
        market_channel: str,
    ) -> None:
        """Register as a listener and initialise per-channel views.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for approved signals.
            market_channel: Channel name to read current market prices from.
        """
        super().__init__(bus, listener_id, listen_channel)
        self._market_ch = bus.channel(market_channel)
        self._signal_views: dict[str, BufferView[Signal]] = {}
        self._market_views: dict[str, BufferView] = {}
        self._positions: dict[str, float] = {}
        self.trade_log: list[TradeRecord] = []

    def execute(self, dirty: set[str]) -> None:
        """Drain approved signals, compute position deltas, and record fills.

        For each signal, computes the delta needed to reach the signal's
        target_position from the current tracked position. Skips if delta
        is negligible. Fill price is read from the market channel using the
        delta's direction.

        Args:
            dirty: Set of symbols that have new approved signals since last wake.
        """
        now_ms = int(time.time() * 1000)
        for symbol in dirty:
            if symbol not in self._signal_views:
                self._signal_views[symbol] = BufferView(
                    self._listen_ch.get_buffer(symbol), from_start=True
                )
            if symbol not in self._market_views:
                self._market_views[symbol] = BufferView(
                    self._market_ch.get_buffer(symbol)
                )

            market_entry = self._market_views[symbol].latest()
            for signal in self._signal_views[symbol].drain():
                current = self._positions.get(symbol, 0.0)
                delta = signal.target_position - current
                if abs(delta) < 1e-12:
                    continue
                side = "BUY" if delta > 0 else "SELL"
                fill = (
                    market_entry.fill_price(side)
                    if market_entry is not None
                    else signal.price
                )
                self.trade_log.append(TradeRecord(
                    signal=signal,
                    delta_quantity=delta,
                    fill_price=fill,
                    filled_at_ms=now_ms,
                ))
                self._positions[symbol] = self._positions.get(symbol, 0.0) + delta
