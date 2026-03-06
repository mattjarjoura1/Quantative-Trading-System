"""Tests for SimulationExecution."""

import threading

from src.bus.message_bus import MessageBus
from src.bus.buffer_view import BufferView
from src.execution.simulation_execution import SimulationExecution
from src.types import OrderBookEntry, PriceTick, Signal, TradeRecord

SIGNAL_CH = "approved_signals"
MARKET_CH = "market_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bus() -> MessageBus:
    bus = MessageBus()
    bus.create_channel(SIGNAL_CH, capacity=64)
    bus.create_channel(MARKET_CH, capacity=64)
    return bus


def make_signal(symbol: str = "AAPL", side: str = "BUY", price: float = 101.0) -> Signal:
    return Signal(
        timestamp_ms=1_000_000,
        symbol=symbol,
        side=side,
        quantity=1.0,
        price=price,
        metadata={},
    )


def make_entry(
    symbol: str = "AAPL",
    bid: float = 100.0,
    ask: float = 101.0,
) -> OrderBookEntry:
    return OrderBookEntry(
        symbol=symbol,
        timestamp_ms=1_000_000,
        bids=((bid, 1.0),),
        asks=((ask, 1.0),),
    )


def make_tick(symbol: str = "AAPL", price: float = 50.0) -> PriceTick:
    return PriceTick(symbol=symbol, timestamp_ms=1_000_000, price=price)


def run_with_signals(
    bus: MessageBus,
    signals: list[Signal],
    market_entry=None,
) -> SimulationExecution:
    """Publish market data and signals, run execution synchronously, return it."""
    if market_entry is not None:
        bus.channel(MARKET_CH).publish(market_entry.symbol, market_entry)

    execution = SimulationExecution(bus, "sim_exec", SIGNAL_CH, MARKET_CH)

    # Pre-publish signals so execution wakes immediately when thread starts
    for sig in signals:
        bus.channel(SIGNAL_CH).publish(sig.symbol, sig)

    t = threading.Thread(target=execution.run)
    t.start()
    import time; time.sleep(0.05)
    execution.stop()
    t.join(timeout=1.0)
    return execution


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSimulationExecutionFillPrice:
    def test_order_book_buy_fills_at_best_ask(self):
        """BUY signal with OrderBookEntry market data fills at best ask."""
        bus = make_bus()
        entry = make_entry(bid=99.0, ask=101.0)
        execution = run_with_signals(bus, [make_signal(side="BUY")], market_entry=entry)

        assert len(execution.trade_log) == 1
        assert execution.trade_log[0].fill_price == 101.0

    def test_order_book_sell_fills_at_best_bid(self):
        """SELL signal with OrderBookEntry market data fills at best bid."""
        bus = make_bus()
        entry = make_entry(bid=99.0, ask=101.0)
        execution = run_with_signals(bus, [make_signal(side="SELL")], market_entry=entry)

        assert len(execution.trade_log) == 1
        assert execution.trade_log[0].fill_price == 99.0

    def test_price_tick_fills_at_tick_price_regardless_of_side(self):
        """PriceTick market data fills at the same price for BUY and SELL."""
        bus = make_bus()
        tick = make_tick(price=55.0)
        bus.channel(MARKET_CH).publish(tick.symbol, tick)

        execution = SimulationExecution(bus, "sim_exec", SIGNAL_CH, MARKET_CH)
        for side in ("BUY", "SELL"):
            bus.channel(SIGNAL_CH).publish("AAPL", make_signal(side=side))

        t = threading.Thread(target=execution.run)
        t.start()
        import time; time.sleep(0.05)
        execution.stop()
        t.join(timeout=1.0)

        assert len(execution.trade_log) == 2
        assert all(r.fill_price == 55.0 for r in execution.trade_log)

    def test_fallback_to_signal_price_when_no_market_data(self):
        """fill_price falls back to signal.price when no market data exists."""
        bus = make_bus()
        sig = make_signal(side="BUY", price=42.0)
        execution = run_with_signals(bus, [sig], market_entry=None)

        assert len(execution.trade_log) == 1
        assert execution.trade_log[0].fill_price == 42.0

    def test_trade_record_contains_original_signal(self):
        """The recorded TradeRecord wraps the original Signal unchanged."""
        bus = make_bus()
        sig = make_signal()
        entry = make_entry()
        execution = run_with_signals(bus, [sig], market_entry=entry)

        assert execution.trade_log[0].signal == sig

    def test_market_view_uses_latest_not_drain(self):
        """Multiple signals against the same market entry all fill at that price.

        If the market view used drain() it would advance past the entry and
        subsequent signals would fall back to signal.price. latest() never
        advances the cursor so every signal in a dirty cycle sees the same
        market price.
        """
        bus = make_bus()
        entry = make_entry(ask=101.0)
        bus.channel(MARKET_CH).publish(entry.symbol, entry)

        signals = [make_signal(side="BUY", price=999.0) for _ in range(3)]
        execution = SimulationExecution(bus, "sim_exec", SIGNAL_CH, MARKET_CH)
        for sig in signals:
            bus.channel(SIGNAL_CH).publish(sig.symbol, sig)

        t = threading.Thread(target=execution.run)
        t.start()
        import time; time.sleep(0.05)
        execution.stop()
        t.join(timeout=1.0)

        assert len(execution.trade_log) == 3
        # All filled at market ask, not signal.price (999.0)
        assert all(r.fill_price == 101.0 for r in execution.trade_log)

    def test_multiple_symbols_each_use_own_market_data(self):
        """Each symbol's fill price comes from its own market buffer."""
        bus = make_bus()
        bus.channel(MARKET_CH).publish("AAPL", make_entry("AAPL", ask=101.0))
        bus.channel(MARKET_CH).publish("GOOG", make_entry("GOOG", ask=200.0))

        execution = SimulationExecution(bus, "sim_exec", SIGNAL_CH, MARKET_CH)
        bus.channel(SIGNAL_CH).publish("AAPL", make_signal("AAPL", side="BUY"))
        bus.channel(SIGNAL_CH).publish("GOOG", make_signal("GOOG", side="BUY"))

        t = threading.Thread(target=execution.run)
        t.start()
        import time; time.sleep(0.05)
        execution.stop()
        t.join(timeout=1.0)

        fills = {r.signal.symbol: r.fill_price for r in execution.trade_log}
        assert fills["AAPL"] == 101.0
        assert fills["GOOG"] == 200.0
