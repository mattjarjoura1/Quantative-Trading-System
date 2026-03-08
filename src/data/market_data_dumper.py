"""Listens on a market data channel and writes ticks to a JSONL file."""

import json

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus


class MarketDataDumper:
    """Listens on a market data channel and writes ticks to a JSONL file.

    One JSON object per line. Each line is independently parseable.
    The dumper is agnostic to the data type — it calls to_dict() on
    whatever it drains from the buffer.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for market data.
        filepath: Path to the output JSONL file.
        symbols: List of symbols to record.
    """

    def __init__(
        self,
        bus: MessageBus,
        listener_id: str,
        listen_channel: str,
        filepath: str,
        symbols: list[str],
    ) -> None:
        """Register as a listener and open the output file.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for market data.
            filepath: Path to the output JSONL file.
            symbols: List of symbols to record.
        """
        self._listener_id = listener_id
        self._channel = bus.channel(listen_channel)
        self._event = self._channel.register_listener(listener_id)
        self._symbols = set(symbols)
        self._views: dict[str, BufferView] = {}
        self._file = open(filepath, "w")
        self._running = True

    def run(self) -> None:
        """Event loop. Wakes on publish, drains all dirty symbols, writes lines.

        Closes the output file when the loop exits.
        """
        while self._running:
            self._event.wait()
            self._event.clear()
            dirty = self._channel.get_dirty(self._listener_id)
            for symbol in dirty:
                if symbol not in self._symbols:
                    continue
                if symbol not in self._views:
                    self._views[symbol] = BufferView(
                        self._channel.get_buffer(symbol), from_start=True
                    )
                for tick in self._views[symbol].drain():
                    self._file.write(json.dumps(tick.to_dict()) + "\n")
            self._file.flush()
        self._file.close()

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False
        self._event.set()
