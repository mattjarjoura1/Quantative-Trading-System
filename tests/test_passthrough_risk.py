"""Tests for PassthroughRisk."""

import threading

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.risk.passthrough_risk import PassthroughRisk
from src.types import Signal

LISTEN_CH = "strategy_signals"
PUBLISH_CH = "approved_signals"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(LISTEN_CH, capacity=64)
    bus.create_channel(PUBLISH_CH, capacity=64)
    return bus


def make_signal(symbol: str = "AAPL", price: float = 101.0) -> Signal:
    return Signal(
        timestamp_ms=1_000_000,
        symbol=symbol,
        target_position=1.0,
        price=price,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPassthroughRisk:
    def test_signal_forwarded_unchanged(self):
        """A signal published to listen_channel appears unchanged on publish_channel."""
        bus = make_bus()
        sig = make_signal("AAPL")
        risk = PassthroughRisk(bus)

        t = threading.Thread(target=risk.run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", sig)
        # Give it time to process, then stop
        import time; time.sleep(0.05)
        risk.stop()
        t.join(timeout=1.0)

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"), from_start=True)
        result = view.last_n(10)
        assert len(result) == 1
        assert result[0] == sig

    def test_multiple_signals_same_symbol_all_forwarded(self):
        """All signals for a symbol that arrived in one dirty cycle are forwarded."""
        bus = make_bus()
        signals = [make_signal("AAPL", price=float(100 + i)) for i in range(3)]
        risk = PassthroughRisk(bus)

        # Pre-publish before starting so all arrive in a single dirty wake
        for sig in signals:
            bus.channel(LISTEN_CH).publish("AAPL", sig)

        t = threading.Thread(target=risk.run)
        t.start()
        import time; time.sleep(0.05)
        risk.stop()
        t.join(timeout=1.0)

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"), from_start=True)
        result = view.last_n(10)
        assert len(result) == 3
        assert result == signals

    def test_multiple_symbols_all_forwarded(self):
        """Signals for different symbols are all forwarded in one evaluate cycle."""
        bus = make_bus()
        sig_a = make_signal("AAPL")
        sig_b = make_signal("GOOG")

        # Register first, then publish so the event is set for this listener
        risk = PassthroughRisk(bus)
        bus.channel(LISTEN_CH).publish("AAPL", sig_a)
        bus.channel(LISTEN_CH).publish("GOOG", sig_b)

        t = threading.Thread(target=risk.run)
        t.start()
        import time; time.sleep(0.05)
        risk.stop()
        t.join(timeout=1.0)

        aapl = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"), from_start=True)
        goog = BufferView(bus.channel(PUBLISH_CH).get_buffer("GOOG"), from_start=True)
        assert len(aapl.last_n(10)) == 1
        assert len(goog.last_n(10)) == 1
        assert aapl.last_n(10)[0] == sig_a
        assert goog.last_n(10)[0] == sig_b

    def test_empty_dirty_returns_no_signals(self):
        """evaluate() with an empty dirty set returns an empty list."""
        bus = make_bus()
        risk = PassthroughRisk(bus)
        result = risk.evaluate(set())
        assert result == []

    def test_approved_signal_does_not_retrigger_listener(self):
        """Publishing to publish_channel does not cause evaluate to be called again."""
        bus = make_bus()
        call_count = 0
        original_evaluate = PassthroughRisk.evaluate

        class CountingPassthrough(PassthroughRisk):
            def evaluate(self, dirty: set[str]) -> list[Signal]:
                nonlocal call_count
                call_count += 1
                self.stop()
                return super().evaluate(dirty)

        risk = CountingPassthrough(bus)
        bus.channel(LISTEN_CH).publish("AAPL", make_signal())
        t = threading.Thread(target=risk.run)
        t.start()
        t.join(timeout=1.0)

        assert call_count == 1
