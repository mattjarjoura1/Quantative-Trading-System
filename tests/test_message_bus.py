"""Tests for Layer 2: MessageBus."""

import threading

import pytest

from src.bus.message_bus import MessageBus
from src.types import OrderBookEntry, Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BIDS = ((100.0, 1.0), (99.0, 2.0))
ASKS = ((101.0, 1.5), (102.0, 0.5))


def make_bus(market_cap: int = 64, signal_cap: int = 32) -> MessageBus:
    """Return a fresh MessageBus with sensible default capacities."""
    return MessageBus(market_data_capacity=market_cap, signal_capacity=signal_cap)


def make_entry(symbol: str = "btcusdt", ts: int = 1_000) -> OrderBookEntry:
    """Return a minimal valid OrderBookEntry."""
    return OrderBookEntry(symbol=symbol, timestamp_ms=ts, bids=BIDS, asks=ASKS)


def make_signal(symbol: str = "btcusdt", ts: int = 1_000) -> Signal:
    """Return a minimal valid Signal."""
    return Signal(timestamp_ms=ts, symbol=symbol, side="BUY", quantity=0.01, price=101.0)


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_empty_on_init(self):
        bus = make_bus()
        assert bus.symbols == set()

    def test_capacities_stored(self):
        bus = MessageBus(market_data_capacity=128, signal_capacity=16)
        assert bus._market_data_capacity == 128
        assert bus._signal_capacity == 16


# ===========================================================================
# register_listener
# ===========================================================================


class TestRegisterListener:
    def test_returns_event(self):
        bus = make_bus()
        event = bus.register_listener("strat_a")
        assert isinstance(event, threading.Event)

    def test_event_initially_unset(self):
        bus = make_bus()
        event = bus.register_listener("strat_a")
        assert not event.is_set()

    def test_two_listeners_get_independent_events(self):
        bus = make_bus()
        e1 = bus.register_listener("strat_a")
        e2 = bus.register_listener("strat_b")
        assert e1 is not e2

    def test_re_register_same_id_replaces(self):
        bus = make_bus()
        e1 = bus.register_listener("strat_a")
        e2 = bus.register_listener("strat_a")
        assert e1 is not e2


# ===========================================================================
# publish_market_data
# ===========================================================================


class TestPublishMarketData:
    def test_new_symbol_creates_buffer(self):
        bus = make_bus()
        bus.publish_market_data("btcusdt", make_entry())
        buf = bus.get_market_data_buffer("btcusdt")
        assert buf.count == 1

    def test_existing_symbol_appends(self):
        bus = make_bus()
        bus.publish_market_data("btcusdt", make_entry(ts=1))
        bus.publish_market_data("btcusdt", make_entry(ts=2))
        buf = bus.get_market_data_buffer("btcusdt")
        assert buf.count == 2

    def test_data_round_trips(self):
        bus = make_bus()
        entry = make_entry()
        bus.publish_market_data("btcusdt", entry)
        buf = bus.get_market_data_buffer("btcusdt")
        assert buf[0] is entry

    def test_sets_listener_event(self):
        bus = make_bus()
        event = bus.register_listener("strat_a")
        assert not event.is_set()
        bus.publish_market_data("btcusdt", make_entry())
        assert event.is_set()

    def test_adds_symbol_to_dirty(self):
        bus = make_bus()
        bus.register_listener("strat_a")
        bus.publish_market_data("btcusdt", make_entry())
        dirty = bus.get_dirty("strat_a")
        assert "btcusdt" in dirty

    def test_multiple_listeners_both_notified(self):
        bus = make_bus()
        e1 = bus.register_listener("strat_a")
        e2 = bus.register_listener("strat_b")
        bus.publish_market_data("btcusdt", make_entry())
        assert e1.is_set()
        assert e2.is_set()
        assert "btcusdt" in bus.get_dirty("strat_a")
        assert "btcusdt" in bus.get_dirty("strat_b")

    def test_no_listeners_does_not_error(self):
        bus = make_bus()
        bus.publish_market_data("btcusdt", make_entry())  # must not raise


# ===========================================================================
# publish_signal
# ===========================================================================


class TestPublishSignal:
    def test_new_symbol_creates_buffer(self):
        bus = make_bus()
        bus.publish_signal("btcusdt", make_signal())
        buf = bus.get_signal_buffer("btcusdt")
        assert buf.count == 1

    def test_data_round_trips(self):
        bus = make_bus()
        sig = make_signal()
        bus.publish_signal("btcusdt", sig)
        buf = bus.get_signal_buffer("btcusdt")
        assert buf[0] is sig

    def test_sets_listener_event(self):
        bus = make_bus()
        event = bus.register_listener("risk_engine")
        bus.publish_signal("btcusdt", make_signal())
        assert event.is_set()

    def test_adds_symbol_to_dirty(self):
        bus = make_bus()
        bus.register_listener("risk_engine")
        bus.publish_signal("btcusdt", make_signal())
        assert "btcusdt" in bus.get_dirty("risk_engine")

    def test_signal_side_independent_of_market_data(self):
        bus = make_bus()
        bus.publish_signal("btcusdt", make_signal())
        # Signal publish must not create a market data buffer
        assert "btcusdt" not in bus._market_data


# ===========================================================================
# get_dirty
# ===========================================================================


class TestGetDirty:
    def test_dirty_clears_after_read(self):
        bus = make_bus()
        bus.register_listener("strat_a")
        bus.publish_market_data("btcusdt", make_entry())
        bus.get_dirty("strat_a")
        assert bus.get_dirty("strat_a") == set()

    def test_accumulates_multiple_symbols(self):
        bus = make_bus()
        bus.register_listener("strat_a")
        bus.publish_market_data("btcusdt", make_entry("btcusdt"))
        bus.publish_market_data("ethusdt", make_entry("ethusdt"))
        dirty = bus.get_dirty("strat_a")
        assert dirty == {"btcusdt", "ethusdt"}

    def test_symbol_independence(self):
        bus = make_bus()
        bus.register_listener("strat_a")
        bus.publish_market_data("btcusdt", make_entry("btcusdt"))
        dirty = bus.get_dirty("strat_a")
        assert "ethusdt" not in dirty

    def test_listeners_independent(self):
        bus = make_bus()
        bus.register_listener("strat_a")
        bus.register_listener("strat_b")
        bus.publish_market_data("btcusdt", make_entry())
        # Reading strat_a's dirty set must not affect strat_b's
        bus.get_dirty("strat_a")
        assert "btcusdt" in bus.get_dirty("strat_b")

    def test_unknown_listener_raises(self):
        bus = make_bus()
        with pytest.raises(KeyError):
            bus.get_dirty("ghost")

    def test_atomic_swap_returns_populated_set(self):
        bus = make_bus()
        bus.register_listener("strat_a")
        bus.publish_market_data("btcusdt", make_entry())
        dirty = bus.get_dirty("strat_a")
        assert dirty == {"btcusdt"}
        # Internal state is now a fresh empty set
        assert bus._dirty["strat_a"] == set()
        assert bus._dirty["strat_a"] is not dirty  # different objects

    def test_dirty_empty_before_any_publish(self):
        bus = make_bus()
        bus.register_listener("strat_a")
        assert bus.get_dirty("strat_a") == set()

    def test_subset_check_pattern(self):
        """Verify the dirty >= required pattern used by pairs strategies."""
        bus = make_bus()
        bus.register_listener("pairs")
        required = {"btcusdt", "ethusdt"}
        bus.publish_market_data("btcusdt", make_entry("btcusdt"))
        bus.publish_market_data("ethusdt", make_entry("ethusdt"))
        dirty = bus.get_dirty("pairs")
        assert dirty >= required


# ===========================================================================
# get_market_data_buffer / get_signal_buffer
# ===========================================================================


class TestGetBuffers:
    def test_get_market_data_buffer_before_publish_returns_empty(self):
        bus = make_bus()
        buf = bus.get_market_data_buffer("btcusdt")
        assert buf.count == 0

    def test_get_market_data_buffer_same_reference(self):
        bus = make_bus()
        buf1 = bus.get_market_data_buffer("btcusdt")
        buf2 = bus.get_market_data_buffer("btcusdt")
        assert buf1 is buf2

    def test_get_signal_buffer_before_publish_returns_empty(self):
        bus = make_bus()
        buf = bus.get_signal_buffer("btcusdt")
        assert buf.count == 0

    def test_get_signal_buffer_same_reference(self):
        bus = make_bus()
        buf1 = bus.get_signal_buffer("btcusdt")
        buf2 = bus.get_signal_buffer("btcusdt")
        assert buf1 is buf2

    def test_publish_fills_pre_subscribed_buffer(self):
        bus = make_bus()
        buf = bus.get_market_data_buffer("btcusdt")  # subscribe before any data
        entry = make_entry()
        bus.publish_market_data("btcusdt", entry)
        assert buf.count == 1
        assert buf[0] is entry

    def test_market_data_and_signal_buffers_independent(self):
        bus = make_bus()
        md_buf = bus.get_market_data_buffer("btcusdt")
        sig_buf = bus.get_signal_buffer("btcusdt")
        assert md_buf is not sig_buf

    def test_buffers_respect_capacity(self):
        bus = MessageBus(market_data_capacity=4, signal_capacity=2)
        assert bus.get_market_data_buffer("btcusdt").capacity == 4
        assert bus.get_signal_buffer("btcusdt").capacity == 2


# ===========================================================================
# symbols property
# ===========================================================================


class TestSymbols:
    def test_empty_on_init(self):
        assert make_bus().symbols == set()

    def test_symbols_from_market_data_publish(self):
        bus = make_bus()
        bus.publish_market_data("btcusdt", make_entry())
        assert "btcusdt" in bus.symbols

    def test_symbols_from_market_data_get(self):
        bus = make_bus()
        bus.get_market_data_buffer("ethusdt")
        assert "ethusdt" in bus.symbols

    def test_symbols_from_signals(self):
        bus = make_bus()
        bus.publish_signal("btcusdt", make_signal())
        assert "btcusdt" in bus.symbols

    def test_symbols_union_across_channels(self):
        bus = make_bus()
        bus.publish_market_data("btcusdt", make_entry("btcusdt"))
        bus.publish_signal("ethusdt", make_signal("ethusdt"))
        assert bus.symbols == {"btcusdt", "ethusdt"}

    def test_same_symbol_both_channels_appears_once(self):
        bus = make_bus()
        bus.publish_market_data("btcusdt", make_entry())
        bus.publish_signal("btcusdt", make_signal())
        assert bus.symbols == {"btcusdt"}


# ===========================================================================
# Repeated publish behaviour
# ===========================================================================


class TestRepeatedPublish:
    def test_event_stays_set_without_clear(self):
        bus = make_bus()
        event = bus.register_listener("strat_a")
        for i in range(5):
            bus.publish_market_data("btcusdt", make_entry(ts=i + 1))
        assert event.is_set()

    def test_buffer_accumulates_all_entries(self):
        bus = make_bus()
        for i in range(5):
            bus.publish_market_data("btcusdt", make_entry(ts=i + 1))
        assert bus.get_market_data_buffer("btcusdt").count == 5

    def test_dirty_set_accumulates_symbol_once(self):
        bus = make_bus()
        bus.register_listener("strat_a")
        for i in range(5):
            bus.publish_market_data("btcusdt", make_entry(ts=i + 1))
        dirty = bus.get_dirty("strat_a")
        # Symbol appears once in a set regardless of publish count
        assert dirty == {"btcusdt"}
