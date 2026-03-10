"""Tests for Layer 4d — BaseExecution ABC."""

import threading

from src.bus.buffer_view import BufferView
from src.bus.message_bus import MessageBus
from src.execution.base_execution import BaseExecution
from src.types import Signal

LISTEN_CH = "approved_signals"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(LISTEN_CH, capacity=64)
    return bus


def make_signal(symbol: str = "AAPL") -> Signal:
    return Signal(
        timestamp_ms=1_000_000,
        symbol=symbol,
        target_position=1.0,
        price=101.0,
        metadata={},
    )


class RecordingExecution(BaseExecution):
    """Records every signal it sees via pre-subscribed BufferViews."""

    def __init__(self, bus: MessageBus, listener_id: str, symbols: list[str]) -> None:
        super().__init__(bus, listener_id, LISTEN_CH)
        self.recorded: list[Signal] = []
        # Views created before data arrives so drain() sees everything written after init.
        self._views: dict[str, BufferView] = {
            sym: BufferView(self._listen_ch.get_buffer(sym)) for sym in symbols
        }

    def execute(self, dirty: set[str]) -> None:
        for symbol in dirty:
            if symbol in self._views:
                self.recorded.extend(self._views[symbol].drain())
        self.stop()


class DirtyCapture(BaseExecution):
    """Captures the dirty set passed to execute."""

    def __init__(self, bus: MessageBus, listener_id: str) -> None:
        super().__init__(bus, listener_id, LISTEN_CH)
        self.captured: set[str] = set()

    def execute(self, dirty: set[str]) -> None:
        self.captured = set(dirty)
        self.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseExecution:
    def test_signals_recorded_after_run(self):
        """Signals published to the listen channel are received by execute."""
        bus = make_bus()
        sig = make_signal("AAPL")
        execution = RecordingExecution(bus, "exec", ["AAPL"])

        t = threading.Thread(target=execution.run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", sig)
        t.join(timeout=1.0)

        assert len(execution.recorded) == 1
        assert execution.recorded[0] == sig

    def test_stop_unblocks_event_wait(self):
        """stop() wakes execution sleeping on event.wait() and exits run()."""
        bus = make_bus()

        class BlockingExecution(BaseExecution):
            def __init__(self, b):
                super().__init__(b, "exec", LISTEN_CH)

            def execute(self, dirty: set[str]) -> None:
                pass

        execution = BlockingExecution(bus)
        t = threading.Thread(target=execution.run)
        t.start()
        execution.stop()
        t.join(timeout=1.0)
        assert not t.is_alive()

    def test_dirty_set_passed_to_execute(self):
        """The dirty set received by execute matches the symbols published."""
        bus = make_bus()
        execution = DirtyCapture(bus, "exec")

        t = threading.Thread(target=execution.run)
        t.start()
        bus.channel(LISTEN_CH).publish("AAPL", make_signal("AAPL"))
        t.join(timeout=1.0)

        assert "AAPL" in execution.captured

    def test_multiple_symbols_all_processed(self):
        """Signals across multiple symbols are all received in one wake cycle."""
        bus = make_bus()
        signals = [make_signal("AAPL"), make_signal("GOOG")]
        # Pre-subscribe before any data arrives so drain() sees everything.
        execution = RecordingExecution(bus, "exec", ["AAPL", "GOOG"])

        # Pre-publish both signals; event is set before thread starts so it
        # wakes immediately and processes both in a single execute() call.
        bus.channel(LISTEN_CH).publish("AAPL", signals[0])
        bus.channel(LISTEN_CH).publish("GOOG", signals[1])

        t = threading.Thread(target=execution.run)
        t.start()
        t.join(timeout=1.0)

        assert len(execution.recorded) == 2
        assert set(s.symbol for s in execution.recorded) == {"AAPL", "GOOG"}
