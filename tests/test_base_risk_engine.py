"""Tests for Layer 4c — BaseRiskEngine ABC."""

import threading

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.risk.base_risk_engine import BaseRiskEngine
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


def make_signal(symbol: str = "AAPL") -> Signal:
    return Signal(
        timestamp_ms=1_000_000,
        symbol=symbol,
        side="BUY",
        quantity=1.0,
        price=101.0,
        metadata={},
    )


class PassthroughRisk(BaseRiskEngine):
    """Reads the first signal from the listen channel and forwards it."""

    def __init__(self, bus: MessageBus, listener_id: str) -> None:
        super().__init__(bus, listener_id, LISTEN_CH, PUBLISH_CH)
        from src.bus.buffer_view import BufferView
        self._view = BufferView(self._listen_ch.get_buffer("AAPL"))

    def evaluate(self, dirty: set[str]) -> Signal | None:
        self.stop()
        return self._view.latest()


class VetoRisk(BaseRiskEngine):
    """Always vetoes — returns None."""

    def __init__(self, bus: MessageBus, listener_id: str) -> None:
        super().__init__(bus, listener_id, LISTEN_CH, PUBLISH_CH)

    def evaluate(self, dirty: set[str]) -> Signal | None:
        self.stop()
        return None


class DirtyCapture(BaseRiskEngine):
    """Captures the dirty set passed to evaluate."""

    def __init__(self, bus: MessageBus, listener_id: str) -> None:
        super().__init__(bus, listener_id, LISTEN_CH, PUBLISH_CH)
        self.captured: set[str] = set()

    def evaluate(self, dirty: set[str]) -> Signal | None:
        self.captured = set(dirty)
        self.stop()
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseRiskEngine:
    def test_passthrough_signal_appears_on_publish_channel(self):
        """evaluate returning a Signal causes it to appear on the publish channel."""
        bus = make_bus()
        sig = make_signal("AAPL")
        risk = PassthroughRisk(bus, "risk")

        t = threading.Thread(target=risk.run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", sig)
        t.join(timeout=1.0)

        view = BufferView(bus.channel(PUBLISH_CH).get_buffer("AAPL"))
        result = view.last_n(10)
        assert len(result) == 1
        assert result[0] == sig

    def test_veto_no_signal_published(self):
        """evaluate returning None publishes nothing to the publish channel."""
        bus = make_bus()
        risk = VetoRisk(bus, "risk")

        t = threading.Thread(target=risk.run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", make_signal())
        t.join(timeout=1.0)

        assert bus.channel(PUBLISH_CH).get_buffer("AAPL").count == 0

    def test_stop_unblocks_event_wait(self):
        """stop() wakes a risk engine sleeping on event.wait() and exits run()."""
        bus = make_bus()

        class BlockingRisk(BaseRiskEngine):
            def __init__(self, b):
                super().__init__(b, "risk", LISTEN_CH, PUBLISH_CH)

            def evaluate(self, dirty: set[str]) -> Signal | None:
                return None

        risk = BlockingRisk(bus)
        t = threading.Thread(target=risk.run)
        t.start()
        risk.stop()
        t.join(timeout=1.0)
        assert not t.is_alive()

    def test_dirty_set_passed_to_evaluate(self):
        """The dirty set received by evaluate matches the symbols published."""
        bus = make_bus()
        risk = DirtyCapture(bus, "risk")

        t = threading.Thread(target=risk.run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", make_signal("AAPL"))
        t.join(timeout=1.0)

        assert "AAPL" in risk.captured

    def test_approved_signal_does_not_trigger_own_listener(self):
        """Publishing to publish_channel does not cause evaluate to be called again."""
        bus = make_bus()
        call_count = 0

        class CountingPassthrough(BaseRiskEngine):
            def __init__(self, b):
                super().__init__(b, "risk", LISTEN_CH, PUBLISH_CH)
                self._view = BufferView(self._listen_ch.get_buffer("AAPL"))

            def evaluate(self, dirty: set[str]) -> Signal | None:
                nonlocal call_count
                call_count += 1
                self.stop()
                return self._view.latest()

        t = threading.Thread(target=CountingPassthrough(bus).run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", make_signal())
        t.join(timeout=1.0)

        assert call_count == 1
