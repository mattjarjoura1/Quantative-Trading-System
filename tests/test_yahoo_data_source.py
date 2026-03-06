"""Tests for YahooDataSource."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.bus.message_bus import MessageBus
from src.bus.buffer_view import BufferView
from src.data.backtest_data_source import BacktestDataSource
from src.data.yahoo_data_source import YahooDataSource
from src.types import PriceTick

PUBLISH_CH = "market_data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(PUBLISH_CH, capacity=64)
    return bus


def make_close_df(symbols: list[str], prices: dict[str, list[float]]) -> pd.DataFrame:
    """Build a fake yfinance Close DataFrame with a DatetimeIndex."""
    dates = pd.date_range("2020-01-01", periods=len(next(iter(prices.values()))), freq="D")
    return pd.DataFrame(prices, index=dates)


# ---------------------------------------------------------------------------
# Construction / validation tests
# ---------------------------------------------------------------------------

class TestYahooDataSourceConstruction:
    def test_empty_symbols_raises(self):
        bus = make_bus()
        with patch("yfinance.download"):
            with pytest.raises(ValueError, match="symbols must be a non-empty list"):
                YahooDataSource(bus, symbols=[])

    def test_end_before_start_raises(self):
        bus = make_bus()
        with patch("yfinance.download"):
            with pytest.raises(ValueError, match="must be after start"):
                YahooDataSource(bus, start="2021-01-01", end="2020-01-01")

    def test_invalid_date_format_raises(self):
        bus = make_bus()
        with patch("yfinance.download"):
            with pytest.raises(ValueError, match="Invalid date format"):
                YahooDataSource(bus, start="01-01-2020")

    def test_invalid_interval_raises(self):
        bus = make_bus()
        with patch("yfinance.download"):
            with pytest.raises(ValueError, match="not supported"):
                YahooDataSource(bus, interval="2d")

    def test_produces_price_tick(self):
        assert YahooDataSource.PRODUCES is PriceTick

    def test_is_backtest_data_source(self):
        bus = make_bus()
        with patch.object(YahooDataSource, "_download", return_value=[]):
            source = YahooDataSource(bus)
        assert isinstance(source, BacktestDataSource)


# ---------------------------------------------------------------------------
# Replay behaviour tests
# ---------------------------------------------------------------------------

class TestYahooDataSourceReplay:
    async def test_publishes_all_ticks(self):
        """All downloaded ticks are published to the channel."""
        bus = make_bus()
        expected = [
            PriceTick(symbol="AAPL", timestamp_ms=1_000_000, price=100.0),
            PriceTick(symbol="AAPL", timestamp_ms=2_000_000, price=101.0),
            PriceTick(symbol="AAPL", timestamp_ms=3_000_000, price=102.0),
        ]
        with patch.object(YahooDataSource, "_download", return_value=expected):
            source = YahooDataSource(bus, symbols=["AAPL"])

        await source.run()

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"), from_start=True)
        assert view.drain() == expected

    async def test_exits_when_exhausted(self):
        """run() exits cleanly after all ticks are emitted."""
        import asyncio
        bus = make_bus()
        ticks = [PriceTick(symbol="AAPL", timestamp_ms=1_000_000, price=50.0)]
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            source = YahooDataSource(bus, symbols=["AAPL"])

        task = asyncio.create_task(source.run())
        await asyncio.wait_for(task, timeout=1.0)

    async def test_ticks_in_chronological_order(self):
        """Published ticks are ordered by timestamp_ms ascending."""
        bus = make_bus()
        # _download sorts, so provide in reverse to confirm sorting
        unsorted = [
            PriceTick(symbol="AAPL", timestamp_ms=3_000_000, price=103.0),
            PriceTick(symbol="AAPL", timestamp_ms=1_000_000, price=101.0),
            PriceTick(symbol="AAPL", timestamp_ms=2_000_000, price=102.0),
        ]
        sorted_ticks = sorted(unsorted, key=lambda t: t.timestamp_ms)
        with patch.object(YahooDataSource, "_download", return_value=sorted_ticks):
            source = YahooDataSource(bus, symbols=["AAPL"])

        await source.run()

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"), from_start=True)
        result = view.drain()
        assert result == sorted_ticks

    async def test_multiple_symbols_published_to_correct_buffers(self):
        """Ticks for each symbol land in the correct per-symbol buffer."""
        bus = make_bus()
        ticks = [
            PriceTick(symbol="AAPL", timestamp_ms=1_000_000, price=100.0),
            PriceTick(symbol="GOOG", timestamp_ms=1_000_000, price=200.0),
            PriceTick(symbol="AAPL", timestamp_ms=2_000_000, price=101.0),
        ]
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            source = YahooDataSource(bus, symbols=["AAPL", "GOOG"])

        await source.run()

        ch = bus.channel(PUBLISH_CH)
        aapl = BufferView(ch.get_buffer("AAPL"), from_start=True)
        goog = BufferView(ch.get_buffer("GOOG"), from_start=True)
        assert len(aapl.drain()) == 2
        assert len(goog.drain()) == 1


# ---------------------------------------------------------------------------
# Download / parsing tests
# ---------------------------------------------------------------------------

class TestYahooDataSourceDownload:
    def test_nan_prices_skipped(self):
        """Rows with NaN close prices are not included in the tick list."""
        import math
        bus = make_bus()
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        close_series = pd.Series([100.0, float("nan"), 102.0], index=dates, name="AAPL")

        with patch.object(YahooDataSource, "_download", return_value=[]):
            source = YahooDataSource(bus, symbols=["AAPL"])

        # Mock yf.download to return a dict-like object so ["Close"] gives our Series
        with patch("src.data.yahoo_data_source.yf.download", return_value={"Close": close_series}):
            result = source._download(["AAPL"], "2020-01-01", "2020-01-04", "1d")

        assert len(result) == 2
        assert all(not math.isnan(t.price) for t in result)

    def test_single_symbol_series_normalised_to_dataframe(self):
        """yfinance Series output for a single ticker is handled correctly."""
        bus = make_bus()
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        close_series = pd.Series([100.0, 101.0], index=dates, name="BTC-USD")

        with patch.object(YahooDataSource, "_download", return_value=[]):
            source = YahooDataSource(bus, symbols=["BTC-USD"])

        with patch("src.data.yahoo_data_source.yf.download", return_value={"Close": close_series}):
            result = source._download(["BTC-USD"], "2020-01-01", "2020-01-03", "1d")

        assert len(result) == 2
        assert result[0].symbol == "BTC-USD"
        assert result[0].price == 100.0
