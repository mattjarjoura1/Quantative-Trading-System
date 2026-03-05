"""Random strategy for pipeline testing."""

import time

import numpy as np

from src.bus import MessageBus, BufferView
from src.strategy import BaseStrategy
from src.types import OrderBookEntry, Signal

# Probability of generating a signal on any given tick
RANDOM_THRESHOLD = 0.3
# Maximum position size in units
MAX_TRADE = 2


class RandomStrategyOBE(BaseStrategy[OrderBookEntry]):
    CONSUMES = OrderBookEntry
    """Emits random buy/sell signals for end-to-end pipeline testing.

    Not a real strategy — used to verify that DataSource → Strategy → Execution
    wiring works correctly end-to-end. Remove once real strategies are in place.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for market data.
        publish_channel: Channel name to publish signals to.
        symbols: Asset symbols to watch and generate signals for.
    """

    def __init__(
        self,
        bus: MessageBus,
        listener_id: str,
        listen_channel: str,
        publish_channel: str,
        symbols: list[str] = ["btcusdt"],
    ) -> None:
        """Set up buffer views for each symbol.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for market data.
            publish_channel: Channel name to publish signals to.
            symbols: Asset symbols to watch and generate signals for.
        """
        super().__init__(bus, listener_id, listen_channel, publish_channel)
        self._views: dict[str, BufferView] = {
            symbol: BufferView(self._listen_ch.get_buffer(symbol))
            for symbol in symbols
        }

    def on_data(self, dirty: set[str]) -> list[Signal]:
        """Drain new entries for each dirty symbol and emit a random signal.

        Args:
            dirty: Set of symbols with new data since last wake.

        Returns:
            A list containing one random Signal per watched dirty symbol,
            or an empty list if no watched symbols have new data.
        """
        signals = []

        for symbol in dirty:
            if symbol not in self._views:
                continue

            data: list[OrderBookEntry] = self._views[symbol].drain()

            if not data:
                continue

            entry = data[-1]
            signals.append(Signal(
                timestamp_ms=entry.timestamp_ms,
                symbol=entry.symbol,
                side=np.random.choice(["BUY", "SELL"]),
                quantity=MAX_TRADE * np.random.random(),
                price=entry.asks[0][0],
                metadata={},
            ))

        return signals
