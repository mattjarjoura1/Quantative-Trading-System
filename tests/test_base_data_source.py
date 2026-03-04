"""Tests for Layer 4a — BaseDataSource ABC."""

import threading

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.data.base_data_source import BaseDataSource
from src.types import OrderBookEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    return MessageBus(market_data_capacity=64, signal_capacity=64)


def make_entry(symbol: str = "AAPL", timestamp_ms: int = 1_000_000) -> OrderBookEntry:
    return OrderBookEntry(
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        bids=((100.0, 1.0),),
        asks=((101.0, 1.0),),
    )


class ListSource(BaseDataSource):
    """Emits entries from a list then stops itself."""

    def __init__(self, bus: MessageBus, entries: list[OrderBookEntry]) -> None:
        super().__init__(bus)
        self._entries = list(entries)
        self._idx = 0

    def fetch(self) -> OrderBookEntry | None:
        if self._idx >= len(self._entries):
            self.stop()
            return None
        entry = self._entries[self._idx]
        self._idx += 1
        return entry


class NoneSource(BaseDataSource):
    """Always returns None — simulates a source with no data yet."""

    def fetch(self) -> OrderBookEntry | None:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseDataSource:
    def test_run_publishes_all_entries(self):
        bus = make_bus()
        entries = [make_entry("AAPL", ts) for ts in [1, 2, 3]]
        ListSource(bus, entries).run()

        view = BufferView(bus.get_market_data_buffer("AAPL"))
        result = view.last_n(10)
        assert len(result) == 3
        assert [e.timestamp_ms for e in result] == [1, 2, 3]

    def test_none_fetch_does_not_break_loop(self):
        """None from fetch is skipped; the loop continues and processes the next entry."""
        bus = make_bus()

        class OnceNoneThenEntry(BaseDataSource):
            def __init__(self, b):
                super().__init__(b)
                self._calls = 0

            def fetch(self):
                self._calls += 1
                if self._calls == 1:
                    return None
                self.stop()
                return make_entry("AAPL", self._calls)

        OnceNoneThenEntry(bus).run()
        view = BufferView(bus.get_market_data_buffer("AAPL"))
        assert len(view.last_n(10)) == 1

    def test_stop_exits_run(self):
        """stop() called from another thread causes run() to exit cleanly."""
        bus = make_bus()
        source = NoneSource(bus)
        t = threading.Thread(target=source.run)
        t.start()
        source.stop()
        t.join(timeout=1.0)
        assert not t.is_alive()

    def test_entries_appear_in_correct_buffer(self):
        """Each entry lands in the buffer keyed by its own symbol."""
        bus = make_bus()
        entries = [
            make_entry("AAPL", 1),
            make_entry("GOOG", 2),
            make_entry("AAPL", 3),
        ]
        ListSource(bus, entries).run()

        aapl = BufferView(bus.get_market_data_buffer("AAPL"))
        goog = BufferView(bus.get_market_data_buffer("GOOG"))
        assert len(aapl.last_n(10)) == 2
        assert len(goog.last_n(10)) == 1

    def test_empty_list_loops_until_stopped(self):
        """A source with no data loops on None returns until stop() is called."""
        bus = make_bus()
        source = NoneSource(bus)
        t = threading.Thread(target=source.run)
        t.start()
        source.stop()
        t.join(timeout=1.0)
        assert not t.is_alive()
        # Nothing was published
        buf = bus.get_market_data_buffer("AAPL")
        assert buf.count == 0
