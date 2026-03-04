"""Registry that connects data sources to strategy consumers via ring buffers."""

import threading

from src.bus.ring_buffer import RingBuffer
from src.types import OrderBookEntry, Signal


class MessageBus:
    """Holds ring buffers and coordinates notification between publishers and consumers.

    Two channel groups exist: market data (keyed by symbol, carrying
    OrderBookEntry) and signals (keyed by symbol, carrying Signal). Buffers
    are created lazily on first access.

    Each consumer registers once via `register_listener` and receives a single
    `threading.Event`. On every publish, the bus adds the published symbol to
    every listener's dirty set and sets their event. The consumer calls
    `event.wait()`, then `get_dirty()` to learn which symbols changed, then
    decides what to do. One wait per cycle regardless of how many symbols are
    being watched.

    Expected usage pattern:
        Setup (once, before any publishing):
            event = bus.register_listener("my_strategy")
            buf = bus.get_market_data_buffer("btcusdt")
            view = BufferView(buf)

        Run loop (on consumer thread):
            while running:
                event.wait()
                event.clear()
                dirty = bus.get_dirty("my_strategy")
                if "btcusdt" in dirty:
                    data = view.last_n(14)
                    ...

    Args:
        market_data_capacity: Capacity of each per-symbol market data buffer.
        signal_capacity: Capacity of each per-symbol signal buffer.
    """

    def __init__(self, market_data_capacity: int, signal_capacity: int) -> None:
        """Initialise the bus with buffer capacities.

        Args:
            market_data_capacity: Capacity for each market data RingBuffer.
            signal_capacity: Capacity for each signal RingBuffer.
        """
        self._market_data_capacity = market_data_capacity
        self._signal_capacity = signal_capacity
        self._market_data: dict[str, RingBuffer[OrderBookEntry]] = {}
        self._signals: dict[str, RingBuffer[Signal]] = {}
        self._listeners: dict[str, threading.Event] = {}
        self._dirty: dict[str, set[str]] = {}

    def register_listener(self, listener_id: str) -> threading.Event:
        """Register a consumer and return its notification event.

        Expected to be called at setup time before any publishing begins.
        The returned event is set whenever any symbol publishes to either
        channel. Call `get_dirty` after waking to see which symbols changed.

        Args:
            listener_id: Unique identifier for this consumer (e.g. "rsi_strategy").

        Returns:
            A threading.Event that will be set on every publish.
        """
        event = threading.Event()
        self._listeners[listener_id] = event
        self._dirty[listener_id] = set()
        return event

    def publish_market_data(self, symbol: str, entry: OrderBookEntry) -> None:
        """Write a market data entry and notify all registered listeners.

        Creates a buffer for the symbol on first publish. Non-blocking —
        `event.set()` returns instantly regardless of consumer state.

        Args:
            symbol: Asset identifier (e.g. "btcusdt").
            entry: The order book snapshot to publish.
        """
        if symbol not in self._market_data:
            self._market_data[symbol] = RingBuffer(self._market_data_capacity)
        self._market_data[symbol].append(entry)
        for listener_id, event in self._listeners.items():
            self._dirty[listener_id].add(symbol)
            event.set()

    def publish_signal(self, symbol: str, signal: Signal) -> None:
        """Write a signal and notify all registered listeners.

        Creates a buffer for the symbol on first publish.

        Args:
            symbol: Asset identifier the signal applies to.
            signal: The signal to publish.
        """
        if symbol not in self._signals:
            self._signals[symbol] = RingBuffer(self._signal_capacity)
        self._signals[symbol].append(signal)
        for listener_id, event in self._listeners.items():
            self._dirty[listener_id].add(symbol)
            event.set()

    def get_market_data_buffer(self, symbol: str) -> RingBuffer[OrderBookEntry]:
        """Return the market data buffer for a symbol, creating it if needed.

        Consumers can call this before any data has arrived. The buffer will
        fill up as the data source publishes.

        Args:
            symbol: Asset identifier.

        Returns:
            The RingBuffer for this symbol's market data.
        """
        if symbol not in self._market_data:
            self._market_data[symbol] = RingBuffer(self._market_data_capacity)
        return self._market_data[symbol]

    def get_signal_buffer(self, symbol: str) -> RingBuffer[Signal]:
        """Return the signal buffer for a symbol, creating it if needed.

        Args:
            symbol: Asset identifier.

        Returns:
            The RingBuffer for this symbol's signals.
        """
        if symbol not in self._signals:
            self._signals[symbol] = RingBuffer(self._signal_capacity)
        return self._signals[symbol]

    def get_dirty(self, listener_id: str) -> set[str]:
        """Return and clear the set of symbols that published since last check.

        Atomic read-and-clear via reference swap: the internal set is replaced
        with a new empty set in a single assignment (atomic under CPython's
        GIL). The consumer receives the populated set; any symbol added between
        the two lines goes to the consumer, not lost.

        Args:
            listener_id: The consumer id passed to `register_listener`.

        Returns:
            Set of symbol strings that published since the last `get_dirty` call.

        Raises:
            KeyError: If listener_id was never registered. This is a
                programming error — register before consuming.
        """
        old = self._dirty[listener_id]
        self._dirty[listener_id] = set()
        return old

    @property
    def symbols(self) -> set[str]:
        """All symbols known to the bus across both channel groups."""
        return set(self._market_data) | set(self._signals)
