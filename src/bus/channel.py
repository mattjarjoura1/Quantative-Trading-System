"""Single named data flow: ring buffers keyed by symbol with its own listener pool."""

import threading
from typing import Generic, TypeVar

from src.bus.ring_buffer import RingBuffer

T = TypeVar("T")


class Channel(Generic[T]):
    """A single publish/subscribe unit.

    Holds one RingBuffer per symbol, one threading.Event per listener, and one
    dirty set per listener. Publishing to this channel notifies only this
    channel's listeners — completely isolated from any other Channel instance.

    Buffers are created lazily on first access so consumers can subscribe
    before any data arrives.

    Args:
        capacity: Capacity of each per-symbol RingBuffer.
    """

    def __init__(self, capacity: int) -> None:
        """Initialise with buffer capacity.

        Args:
            capacity: Number of items each per-symbol buffer can hold.
        """
        self._capacity = capacity
        self._buffers: dict[str, RingBuffer[T]] = {}
        self._listeners: dict[str, threading.Event] = {}
        self._dirty: dict[str, set[str]] = {}

    def publish(self, symbol: str, item: T) -> None:
        """Append item to the symbol's buffer and notify all listeners.

        Creates the buffer on first publish for this symbol. Non-blocking —
        event.set() returns instantly regardless of consumer state.

        Args:
            symbol: Asset identifier (e.g. "btcusdt").
            item: Data item to publish.
        """
        if symbol not in self._buffers:
            self._buffers[symbol] = RingBuffer(self._capacity)
        self._buffers[symbol].append(item)
        for listener_id, event in self._listeners.items():
            self._dirty[listener_id].add(symbol)
            event.set()

    def register_listener(self, listener_id: str) -> threading.Event:
        """Register a consumer and return its notification event.

        Expected to be called at setup time before any publishing begins.
        The returned event is set on every publish to this channel.

        Args:
            listener_id: Unique identifier for this consumer.

        Returns:
            A threading.Event that will be set on every publish.
        """
        event = threading.Event()
        self._listeners[listener_id] = event
        self._dirty[listener_id] = set()
        return event

    def get_dirty(self, listener_id: str) -> set[str]:
        """Return and clear the set of symbols published to since last check.

        Atomic read-and-clear via reference swap (safe under CPython GIL).

        Args:
            listener_id: The id passed to register_listener.

        Returns:
            Set of symbol strings published since the last get_dirty call.

        Raises:
            KeyError: If listener_id was never registered.
        """
        old = self._dirty[listener_id]
        self._dirty[listener_id] = set()
        return old

    def get_buffer(self, symbol: str) -> RingBuffer[T]:
        """Return the buffer for a symbol, creating it if it doesn't exist.

        Args:
            symbol: Asset identifier.

        Returns:
            The RingBuffer for this symbol.
        """
        if symbol not in self._buffers:
            self._buffers[symbol] = RingBuffer(self._capacity)
        return self._buffers[symbol]
