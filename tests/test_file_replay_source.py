"""Tests for FileReplaySource."""

import json
import pytest

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.data.backtest_data_source import BacktestDataSource
from src.data.file_replay_source import FileReplaySource
from src.registry import DATA_TYPES
from src.types import OrderBookEntry, PriceTick

PUBLISH_CH = "market_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(PUBLISH_CH, capacity=64)
    return bus


def make_ticks(n: int) -> list[PriceTick]:
    return [
        PriceTick(symbol="BTC-USD", timestamp_ms=i + 1, price=float(i + 1))
        for i in range(n)
    ]


def write_jsonl(filepath, items) -> None:
    with open(filepath, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFileReplaySourceProperties:
    def test_produces_matches_data_cls(self, tmp_path):
        """PRODUCES instance attribute is set to the supplied data_cls."""
        filepath = str(tmp_path / "ticks.jsonl")
        write_jsonl(filepath, make_ticks(1))
        source = FileReplaySource(make_bus(), PUBLISH_CH, filepath, PriceTick)
        assert source.PRODUCES is PriceTick

    def test_produces_matches_order_book_entry(self, tmp_path):
        """PRODUCES is set correctly for OrderBookEntry too."""
        entry = OrderBookEntry(
            symbol="btcusdt",
            timestamp_ms=1_000_000,
            bids=((100.0, 1.0),),
            asks=((101.0, 1.0),),
        )
        filepath = str(tmp_path / "entries.jsonl")
        write_jsonl(filepath, [entry])
        source = FileReplaySource(make_bus(), PUBLISH_CH, filepath, OrderBookEntry)
        assert source.PRODUCES is OrderBookEntry

    def test_is_backtest_data_source(self, tmp_path):
        """FileReplaySource is a BacktestDataSource (type guard passes)."""
        filepath = str(tmp_path / "ticks.jsonl")
        write_jsonl(filepath, make_ticks(1))
        source = FileReplaySource(make_bus(), PUBLISH_CH, filepath, PriceTick)
        assert isinstance(source, BacktestDataSource)


class TestFileReplaySourceReplay:
    async def test_replays_all_lines(self, tmp_path):
        """All ticks from the file are published to the channel."""
        bus = make_bus()
        ticks = make_ticks(3)
        filepath = str(tmp_path / "ticks.jsonl")
        write_jsonl(filepath, ticks)

        source = FileReplaySource(bus, PUBLISH_CH, filepath, PriceTick)
        await source.run()

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("BTC-USD"), from_start=True)
        assert view.drain() == ticks

    async def test_correct_dataclass_type(self, tmp_path):
        """Each published item is an instance of the specified data_cls."""
        bus = make_bus()
        ticks = make_ticks(2)
        filepath = str(tmp_path / "ticks.jsonl")
        write_jsonl(filepath, ticks)

        source = FileReplaySource(bus, PUBLISH_CH, filepath, PriceTick)
        await source.run()

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("BTC-USD"), from_start=True)
        for item in view.drain():
            assert isinstance(item, PriceTick)

    async def test_exhaustion_stops_source(self, tmp_path):
        """After all lines are consumed, _running is False."""
        bus = make_bus()
        filepath = str(tmp_path / "ticks.jsonl")
        write_jsonl(filepath, make_ticks(2))

        source = FileReplaySource(bus, PUBLISH_CH, filepath, PriceTick)
        await source.run()

        assert not source._running

    async def test_empty_file_produces_no_ticks(self, tmp_path):
        """Empty JSONL file results in zero published ticks."""
        bus = make_bus()
        filepath = str(tmp_path / "empty.jsonl")
        filepath_obj = tmp_path / "empty.jsonl"
        filepath_obj.write_text("")

        source = FileReplaySource(bus, PUBLISH_CH, str(filepath_obj), PriceTick)
        await source.run()

        ch = bus.channel(PUBLISH_CH)
        assert "BTC-USD" not in ch._buffers

    async def test_order_book_entry_round_trip(self, tmp_path):
        """OrderBookEntry ticks are correctly deserialised from JSONL."""
        bus = make_bus()
        entry = OrderBookEntry(
            symbol="btcusdt",
            timestamp_ms=1_000_000,
            bids=((100.0, 1.0), (99.0, 0.5)),
            asks=((101.0, 1.0), (102.0, 0.5)),
        )
        filepath = str(tmp_path / "entries.jsonl")
        write_jsonl(filepath, [entry])

        source = FileReplaySource(bus, PUBLISH_CH, filepath, OrderBookEntry)
        await source.run()

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("btcusdt"), from_start=True)
        result = view.drain()
        assert len(result) == 1
        assert result[0] == entry
        assert isinstance(result[0].bids, tuple)
        assert isinstance(result[0].bids[0], tuple)


class TestDataTypesRegistry:
    def test_contains_price_tick(self):
        assert DATA_TYPES["PriceTick"] is PriceTick

    def test_contains_order_book_entry(self):
        assert DATA_TYPES["OrderBookEntry"] is OrderBookEntry

    def test_file_replay_in_sources(self):
        from src.registry import SOURCES
        assert "file_replay" in SOURCES
        assert SOURCES["file_replay"] is FileReplaySource
