"""Tests for Layer 4b — BaseStrategy ABC."""

import threading

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.strategy.base_strategy import BaseStrategy
from src.types import OrderBookEntry, Signal


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


def make_signal(symbol: str = "AAPL") -> Signal:
    return Signal(
        timestamp_ms=1_000_000,
        symbol=symbol,
        side="BUY",
        quantity=1.0,
        price=101.0,
        metadata={},
    )


class FixedSignalStrategy(BaseStrategy):
    """Returns a preset signal once then stops."""

    def __init__(self, bus: MessageBus, listener_id: str, signal: Signal) -> None:
        super().__init__(bus, listener_id)
        self._signal = signal

    def on_data(self, dirty: set[str]) -> Signal | None:
        self.stop()
        return self._signal


class NullStrategy(BaseStrategy):
    """Always returns None then stops."""

    def on_data(self, dirty: set[str]) -> Signal | None:
        self.stop()
        return None


class DirtyCapture(BaseStrategy):
    """Captures the dirty set passed to on_data."""

    def __init__(self, bus: MessageBus, listener_id: str) -> None:
        super().__init__(bus, listener_id)
        self.captured: set[str] = set()

    def on_data(self, dirty: set[str]) -> Signal | None:
        self.captured = set(dirty)
        self.stop()
        return None


class CountingStrategy(BaseStrategy):
    """Counts on_data calls; stops and signals when target reached."""

    def __init__(self, bus: MessageBus, listener_id: str, target: int) -> None:
        super().__init__(bus, listener_id)
        self._target = target
        self.call_count = 0
        self.processed = threading.Event()

    def on_data(self, dirty: set[str]) -> Signal | None:
        self.call_count += 1
        self.processed.set()
        if self.call_count >= self._target:
            self.stop()
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseStrategy:
    def test_signal_published_to_bus(self):
        """on_data returning a Signal causes it to appear in the signal buffer."""
        bus = make_bus()
        sig = make_signal("AAPL")
        strategy = FixedSignalStrategy(bus, "strat", sig)

        t = threading.Thread(target=strategy.run)
        t.start()
        bus.publish_market_data("AAPL", make_entry())
        t.join(timeout=1.0)

        view = BufferView(bus.get_signal_buffer("AAPL"))
        result = view.last_n(10)
        assert len(result) == 1
        assert result[0] == sig

    def test_none_return_no_signal_published(self):
        """on_data returning None publishes nothing to the signal buffer."""
        bus = make_bus()
        strategy = NullStrategy(bus, "strat")

        t = threading.Thread(target=strategy.run)
        t.start()
        bus.publish_market_data("AAPL", make_entry())
        t.join(timeout=1.0)

        assert bus.get_signal_buffer("AAPL").count == 0

    def test_stop_unblocks_event_wait(self):
        """stop() wakes a strategy sleeping on event.wait() and exits run()."""
        bus = make_bus()

        class BlockingStrategy(BaseStrategy):
            def on_data(self, dirty: set[str]) -> Signal | None:
                return None

        strategy = BlockingStrategy(bus, "strat")
        t = threading.Thread(target=strategy.run)
        t.start()
        strategy.stop()
        t.join(timeout=1.0)
        assert not t.is_alive()

    def test_dirty_set_passed_to_on_data(self):
        """The dirty set received by on_data matches the symbols published."""
        bus = make_bus()
        strategy = DirtyCapture(bus, "strat")

        t = threading.Thread(target=strategy.run)
        t.start()
        bus.publish_market_data("AAPL", make_entry("AAPL"))
        t.join(timeout=1.0)

        assert "AAPL" in strategy.captured

    def test_on_data_called_each_time_event_fires(self):
        """on_data is called once per event fire, not batched or skipped."""
        bus = make_bus()
        strategy = CountingStrategy(bus, "strat", target=3)

        t = threading.Thread(target=strategy.run)
        t.start()

        for _ in range(3):
            strategy.processed.clear()
            bus.publish_market_data("AAPL", make_entry())
            strategy.processed.wait(timeout=1.0)

        t.join(timeout=1.0)
        assert strategy.call_count == 3
