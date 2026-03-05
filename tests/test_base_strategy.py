"""Tests for Layer 4b — BaseStrategy ABC."""

import threading

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.strategy.base_strategy import BaseStrategy
from src.strategy.random_strategy_obe import RandomStrategyOBE
from src.types import OrderBookEntry, PriceTick, Signal

LISTEN_CH = "market_data"
PUBLISH_CH = "strategy_signals"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(LISTEN_CH, capacity=64)
    bus.create_channel(PUBLISH_CH, capacity=64)
    return bus


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


class FixedSignalStrategy(BaseStrategy[OrderBookEntry]):
    """Returns a preset signal once then stops."""

    CONSUMES = OrderBookEntry

    def __init__(self, bus: MessageBus, listener_id: str, signal: Signal) -> None:
        super().__init__(bus, listener_id, LISTEN_CH, PUBLISH_CH)
        self._signal = signal

    def on_data(self, dirty: set[str]) -> list[Signal]:
        self.stop()
        return [self._signal]


class NullStrategy(BaseStrategy[OrderBookEntry]):
    """Always returns an empty list then stops."""

    CONSUMES = OrderBookEntry

    def __init__(self, bus: MessageBus, listener_id: str) -> None:
        super().__init__(bus, listener_id, LISTEN_CH, PUBLISH_CH)

    def on_data(self, dirty: set[str]) -> list[Signal]:
        self.stop()
        return []


class DirtyCapture(BaseStrategy[OrderBookEntry]):
    """Captures the dirty set passed to on_data."""

    CONSUMES = OrderBookEntry

    def __init__(self, bus: MessageBus, listener_id: str) -> None:
        super().__init__(bus, listener_id, LISTEN_CH, PUBLISH_CH)
        self.captured: set[str] = set()

    def on_data(self, dirty: set[str]) -> list[Signal]:
        self.captured = set(dirty)
        self.stop()
        return []


class CountingStrategy(BaseStrategy[OrderBookEntry]):
    """Counts on_data calls; signals when each call completes."""

    CONSUMES = OrderBookEntry

    def __init__(self, bus: MessageBus, listener_id: str, target: int) -> None:
        super().__init__(bus, listener_id, LISTEN_CH, PUBLISH_CH)
        self._target = target
        self.call_count = 0
        self.processed = threading.Event()

    def on_data(self, dirty: set[str]) -> list[Signal]:
        self.call_count += 1
        self.processed.set()
        if self.call_count >= self._target:
            self.stop()
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseStrategy:
    def test_signal_published_to_publish_channel(self):
        """on_data returning a Signal causes it to appear on the publish channel."""
        bus = make_bus()
        sig = make_signal("AAPL")
        strategy = FixedSignalStrategy(bus, "strat", sig)

        t = threading.Thread(target=strategy.run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", make_entry())
        t.join(timeout=1.0)

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"))
        result = view.last_n(10)
        assert len(result) == 1
        assert result[0] == sig

    def test_none_return_no_signal_published(self):
        """on_data returning None publishes nothing to the publish channel."""
        bus = make_bus()
        strategy = NullStrategy(bus, "strat")

        t = threading.Thread(target=strategy.run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", make_entry())
        t.join(timeout=1.0)

        assert bus.channel(PUBLISH_CH).get_buffer("AAPL").count == 0

    def test_stop_unblocks_event_wait(self):
        """stop() wakes a strategy sleeping on event.wait() and exits run()."""
        bus = make_bus()

        class BlockingStrategy(BaseStrategy):
            def __init__(self, b):
                super().__init__(b, "strat", LISTEN_CH, PUBLISH_CH)

            def on_data(self, dirty: set[str]) -> list[Signal]:
                return []

        strategy = BlockingStrategy(bus)
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
        bus.channel(LISTEN_CH).publish("AAPL", make_entry("AAPL"))
        t.join(timeout=1.0)

        assert "AAPL" in strategy.captured

    def test_on_data_called_each_time_event_fires(self):
        """on_data is called once per event fire."""
        bus = make_bus()
        strategy = CountingStrategy(bus, "strat", target=3)

        t = threading.Thread(target=strategy.run)
        t.start()

        for _ in range(3):
            strategy.processed.clear()
            bus.channel(LISTEN_CH).publish("AAPL", make_entry())
            strategy.processed.wait(timeout=1.0)

        t.join(timeout=1.0)
        assert strategy.call_count == 3

    def test_signal_does_not_trigger_own_listener(self):
        """Signal published to publish_channel does not cause a second on_data call.

        If channels were not isolated, the signal publish would wake the strategy's
        listen_channel listener again, resulting in on_data being called twice.
        """
        bus = make_bus()
        call_count = 0

        class SignalAndCount(BaseStrategy):
            def __init__(self, b):
                super().__init__(b, "strat", LISTEN_CH, PUBLISH_CH)

            def on_data(self, dirty: set[str]) -> list[Signal]:
                nonlocal call_count
                call_count += 1
                self.stop()
                return [make_signal()]  # publishes to PUBLISH_CH

        t = threading.Thread(target=SignalAndCount(bus).run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", make_entry())
        t.join(timeout=1.0)

        assert call_count == 1


class TestCompatibility:
    def test_random_strategy_consumes_order_book_entry(self):
        assert RandomStrategyOBE.CONSUMES is OrderBookEntry

    def test_concrete_subclass_has_consumes(self):
        assert FixedSignalStrategy.CONSUMES is OrderBookEntry

    def test_type_mismatch_detectable(self):
        """Simulate the orchestrator compatibility check."""
        from src.data.binance_data_source import BinanceDataSource

        class PriceTickStrategy(BaseStrategy[PriceTick]):
            CONSUMES = PriceTick
            def on_data(self, dirty: set[str]): return []

        assert BinanceDataSource.PRODUCES != PriceTickStrategy.CONSUMES

    def test_type_match_passes(self):
        from src.data.binance_data_source import BinanceDataSource

        class OrderBookStrategy(BaseStrategy[OrderBookEntry]):
            CONSUMES = OrderBookEntry
            def on_data(self, dirty: set[str]): return []

        assert BinanceDataSource.PRODUCES == OrderBookStrategy.CONSUMES
