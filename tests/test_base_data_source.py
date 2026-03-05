"""Tests for Layer 4a — BaseDataSource ABC."""

import asyncio

import pytest

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.data.base_data_source import BaseDataSource
from src.types import OrderBookEntry

PUBLISH_CH = "market_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(PUBLISH_CH, capacity=64)
    return bus


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
        super().__init__(bus, PUBLISH_CH)
        self._entries = list(entries)
        self._idx = 0

    async def fetch(self) -> OrderBookEntry | None:
        if self._idx >= len(self._entries):
            self.stop()
            return None
        entry = self._entries[self._idx]
        self._idx += 1
        return entry


class NoneSource(BaseDataSource):
    """Always returns None — simulates a source waiting for data.

    Yields to the event loop on each fetch to mimic real async I/O.
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(bus, PUBLISH_CH)

    async def fetch(self) -> OrderBookEntry | None:
        await asyncio.sleep(0)  # simulate async I/O, allow stop() to be seen
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseDataSource:
    async def test_run_publishes_all_entries(self):
        bus = make_bus()
        entries = [make_entry("AAPL", ts) for ts in [1, 2, 3]]
        await ListSource(bus, entries).run()

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"))
        result = view.last_n(10)
        assert len(result) == 3
        assert [e.timestamp_ms for e in result] == [1, 2, 3]

    async def test_none_fetch_does_not_break_loop(self):
        """None from fetch is skipped; the loop continues and processes the next entry."""
        bus = make_bus()

        class OnceNoneThenEntry(BaseDataSource):
            def __init__(self, b):
                super().__init__(b, PUBLISH_CH)
                self._calls = 0

            async def fetch(self):
                self._calls += 1
                if self._calls == 1:
                    return None
                self.stop()
                return make_entry("AAPL", self._calls)

        await OnceNoneThenEntry(bus).run()
        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"))
        assert len(view.last_n(10)) == 1

    async def test_stop_exits_run(self):
        """stop() causes run() to exit on its next iteration."""
        bus = make_bus()
        source = NoneSource(bus)

        task = asyncio.create_task(source.run())
        await asyncio.sleep(0)  # yield to let the task start
        source.stop()
        await asyncio.wait_for(task, timeout=1.0)

    async def test_entries_appear_in_correct_buffer(self):
        """Each entry lands in the buffer keyed by its own symbol."""
        bus = make_bus()
        entries = [
            make_entry("AAPL", 1),
            make_entry("GOOG", 2),
            make_entry("AAPL", 3),
        ]
        await ListSource(bus, entries).run()

        ch = bus.channel(PUBLISH_CH)
        aapl = BufferView(ch.get_buffer("AAPL"))
        goog = BufferView(ch.get_buffer("GOOG"))
        assert len(aapl.last_n(10)) == 2
        assert len(goog.last_n(10)) == 1

    async def test_empty_list_loops_until_stopped(self):
        """A source with no data loops on None returns until stop() is called."""
        bus = make_bus()
        source = NoneSource(bus)

        task = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        source.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert bus.channel(PUBLISH_CH).get_buffer("AAPL").count == 0

    async def test_publishes_only_to_named_channel(self):
        """Data published to publish_channel does not appear on other channels."""
        bus = make_bus()
        bus.create_channel("other", capacity=64)
        other_event = bus.channel("other").register_listener("watcher")

        await ListSource(bus, [make_entry("AAPL", 1)]).run()

        assert not other_event.is_set()
