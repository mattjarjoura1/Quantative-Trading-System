"""Tests for MarketDataDumper."""

import json
import threading
import time

from src.bus.message_bus import MessageBus
from src.data.market_data_dumper import MarketDataDumper
from src.types import PriceTick

MARKET_CH = "market_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(MARKET_CH, capacity=64)
    return bus


def make_tick(symbol: str = "AAPL", price: float = 100.0, ts: int = 1_000_000) -> PriceTick:
    return PriceTick(symbol=symbol, timestamp_ms=ts, price=price)


def run_dumper(
    bus: MessageBus,
    filepath: str,
    symbols: list[str],
    ticks: list[PriceTick],
) -> MarketDataDumper:
    """Publish ticks, run dumper on a thread, stop and join. Returns the dumper."""
    dumper = MarketDataDumper(bus, "dumper", MARKET_CH, filepath, symbols)

    for tick in ticks:
        bus.channel(MARKET_CH).publish(tick.symbol, tick)

    t = threading.Thread(target=dumper.run)
    t.start()
    time.sleep(0.05)
    dumper.stop()
    t.join(timeout=1.0)
    return dumper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarketDataDumperWrites:
    def test_writes_valid_json_lines(self, tmp_path):
        """Every line in the output file is valid JSON."""
        filepath = str(tmp_path / "out.jsonl")
        bus = make_bus()
        run_dumper(bus, filepath, ["AAPL"], [make_tick("AAPL")])

        with open(filepath) as f:
            lines = f.readlines()

        assert len(lines) == 1
        json.loads(lines[0])  # raises if invalid

    def test_one_line_per_tick(self, tmp_path):
        """Line count equals the number of published ticks."""
        filepath = str(tmp_path / "out.jsonl")
        bus = make_bus()
        ticks = [make_tick("AAPL", price=float(i + 1), ts=i + 1) for i in range(5)]
        run_dumper(bus, filepath, ["AAPL"], ticks)

        with open(filepath) as f:
            lines = [l for l in f.readlines() if l.strip()]

        assert len(lines) == 5

    def test_all_ticks_recorded(self, tmp_path):
        """Content of each recorded line matches the published tick."""
        filepath = str(tmp_path / "out.jsonl")
        bus = make_bus()
        ticks = [
            make_tick("AAPL", price=10.0, ts=1_000_001),
            make_tick("AAPL", price=20.0, ts=1_000_002),
            make_tick("AAPL", price=30.0, ts=1_000_003),
        ]
        run_dumper(bus, filepath, ["AAPL"], ticks)

        with open(filepath) as f:
            records = [json.loads(line) for line in f]

        prices = [r["price"] for r in records]
        assert prices == [10.0, 20.0, 30.0]

    def test_handles_multiple_symbols(self, tmp_path):
        """Ticks from multiple symbols are all recorded."""
        filepath = str(tmp_path / "out.jsonl")
        bus = make_bus()
        ticks = [
            make_tick("AAPL", price=100.0, ts=1),
            make_tick("GOOG", price=200.0, ts=2),
        ]
        run_dumper(bus, filepath, ["AAPL", "GOOG"], ticks)

        with open(filepath) as f:
            records = [json.loads(line) for line in f]

        symbols = {r["symbol"] for r in records}
        assert symbols == {"AAPL", "GOOG"}
        assert len(records) == 2

    def test_ignores_unregistered_symbols(self, tmp_path):
        """Ticks for symbols not in the symbols list are not written."""
        filepath = str(tmp_path / "out.jsonl")
        bus = make_bus()
        ticks = [
            make_tick("AAPL", price=100.0, ts=1),
            make_tick("GOOG", price=200.0, ts=2),  # not in symbols list
        ]
        run_dumper(bus, filepath, ["AAPL"], ticks)

        with open(filepath) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 1
        assert records[0]["symbol"] == "AAPL"


class TestMarketDataDumperLifecycle:
    def test_file_closed_after_stop(self, tmp_path):
        """File handle is closed after stop() and thread join."""
        filepath = str(tmp_path / "out.jsonl")
        bus = make_bus()
        dumper = run_dumper(bus, filepath, ["AAPL"], [make_tick("AAPL")])

        assert dumper._file.closed

    def test_file_parseable_line_by_line(self, tmp_path):
        """Each line is independently parseable (no trailing commas, no array wrapper)."""
        filepath = str(tmp_path / "out.jsonl")
        bus = make_bus()
        ticks = [make_tick("AAPL", price=float(i + 1), ts=i + 1) for i in range(3)]
        run_dumper(bus, filepath, ["AAPL"], ticks)

        with open(filepath) as f:
            for line in f:
                obj = json.loads(line)
                assert isinstance(obj, dict)
