"""Tests for BacktestDataSource ABC."""

import asyncio

from src.bus.message_bus import MessageBus
from src.bus.buffer_view import BufferView
from src.data.backtest_data_source import BacktestDataSource
from src.types import PriceTick

PUBLISH_CH = "market_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(PUBLISH_CH, capacity=64)
    return bus


def make_tick(symbol: str, price: float) -> PriceTick:
    return PriceTick(symbol=symbol, timestamp_ms=int(price * 1000), price=price)


class SequenceSource(BacktestDataSource[PriceTick]):
    """Emits ticks from a list then stops."""

    PRODUCES = PriceTick

    def __init__(self, bus: MessageBus, ticks: list[PriceTick]) -> None:
        super().__init__(bus, PUBLISH_CH)
        self._ticks = ticks
        self._index = 0

    async def fetch(self) -> PriceTick | None:
        if self._index >= len(self._ticks):
            self.stop()
            return None
        tick = self._ticks[self._index]
        self._index += 1
        return tick


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBacktestDataSource:
    async def test_publishes_all_ticks_no_listeners(self):
        """All ticks published when no listeners registered (always clear)."""
        bus = make_bus()
        ticks = [make_tick("AAPL", float(i)) for i in range(1, 4)]
        await SequenceSource(bus, ticks).run()

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"), from_start=True)
        assert view.drain() == ticks

    async def test_exits_when_fetch_exhausted(self):
        """run() exits cleanly once fetch() calls stop()."""
        bus = make_bus()
        ticks = [make_tick("AAPL", 1.0)]
        task = asyncio.create_task(SequenceSource(bus, ticks).run())
        await asyncio.wait_for(task, timeout=1.0)

    async def test_empty_source_exits_immediately(self):
        """A source with no ticks stops on the first fetch call."""
        bus = make_bus()
        task = asyncio.create_task(SequenceSource(bus, []).run())
        await asyncio.wait_for(task, timeout=1.0)

    async def test_is_backtest_data_source_instance(self):
        """Concrete subclass passes the type-guard isinstance check."""
        bus = make_bus()
        source = SequenceSource(bus, [])
        assert isinstance(source, BacktestDataSource)

    async def test_pacing_waits_for_listener_between_ticks(self):
        """Source does not publish the next tick until the listener clears."""
        bus = make_bus()
        ticks = [make_tick("AAPL", float(i)) for i in range(1, 4)]
        source = SequenceSource(bus, ticks)

        event = bus.channel(PUBLISH_CH).register_listener("consumer")
        buf = bus.channel(PUBLISH_CH).get_buffer("AAPL")

        task = asyncio.create_task(source.run())

        for expected_count in range(1, 4):
            # Yield twice: once for source to publish, once for it to block
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert buf.count == expected_count
            event.clear()  # unblock source for the next tick

        await asyncio.wait_for(task, timeout=1.0)

    async def test_multiple_symbols_paced_independently(self):
        """Ticks for different symbols are each paced through the listener."""
        bus = make_bus()
        ticks = [
            make_tick("AAPL", 1.0),
            make_tick("GOOG", 2.0),
            make_tick("AAPL", 3.0),
        ]
        source = SequenceSource(bus, ticks)

        event = bus.channel(PUBLISH_CH).register_listener("consumer")

        task = asyncio.create_task(source.run())

        total_published = 0
        for _ in range(3):
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            total_published += 1
            event.clear()

        await asyncio.wait_for(task, timeout=1.0)

        aapl = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"), from_start=True)
        goog = BufferView(bus.channel(PUBLISH_CH).get_buffer("GOOG"), from_start=True)
        assert len(aapl.drain()) == 2
        assert len(goog.drain()) == 1
