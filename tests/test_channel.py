"""Tests for Channel — the core publish/subscribe unit."""

from src.bus.channel import Channel


def make_channel(capacity: int = 16) -> Channel:
    return Channel(capacity=capacity)


class TestPublish:
    def test_publish_new_symbol_creates_buffer(self):
        ch: Channel[int] = make_channel()
        ch.publish("AAPL", 1)
        assert ch.get_buffer("AAPL").count == 1

    def test_publish_existing_symbol_appends(self):
        ch: Channel[int] = make_channel()
        ch.publish("AAPL", 1)
        ch.publish("AAPL", 2)
        assert ch.get_buffer("AAPL").count == 2

    def test_publish_no_listeners_does_not_error(self):
        ch: Channel[int] = make_channel()
        ch.publish("AAPL", 42)  # no exception

    def test_data_roundtrips_through_publish_get_buffer(self):
        ch: Channel[str] = make_channel()
        ch.publish("AAPL", "hello")
        buf = ch.get_buffer("AAPL")
        assert buf[buf.write_idx - 1] == "hello"

    def test_multiple_symbols_independent_buffers(self):
        ch: Channel[int] = make_channel()
        ch.publish("AAPL", 1)
        ch.publish("GOOG", 99)
        assert ch.get_buffer("AAPL").count == 1
        assert ch.get_buffer("GOOG").count == 1
        assert ch.get_buffer("AAPL")[0] == 1
        assert ch.get_buffer("GOOG")[0] == 99


class TestGetBuffer:
    def test_get_buffer_before_publish_returns_empty(self):
        ch: Channel[int] = make_channel()
        buf = ch.get_buffer("AAPL")
        assert buf.count == 0

    def test_get_buffer_same_symbol_same_reference(self):
        ch: Channel[int] = make_channel()
        assert ch.get_buffer("AAPL") is ch.get_buffer("AAPL")

    def test_get_buffer_after_publish_same_reference(self):
        ch: Channel[int] = make_channel()
        buf_before = ch.get_buffer("AAPL")
        ch.publish("AAPL", 1)
        buf_after = ch.get_buffer("AAPL")
        assert buf_before is buf_after


class TestListeners:
    def test_register_listener_returns_unset_event(self):
        ch: Channel[int] = make_channel()
        event = ch.register_listener("strat")
        assert not event.is_set()

    def test_publish_sets_listener_event(self):
        ch: Channel[int] = make_channel()
        event = ch.register_listener("strat")
        ch.publish("AAPL", 1)
        assert event.is_set()

    def test_publish_adds_symbol_to_dirty_set(self):
        ch: Channel[int] = make_channel()
        ch.register_listener("strat")
        ch.publish("AAPL", 1)
        dirty = ch.get_dirty("strat")
        assert "AAPL" in dirty

    def test_get_dirty_clears_set(self):
        ch: Channel[int] = make_channel()
        ch.register_listener("strat")
        ch.publish("AAPL", 1)
        ch.get_dirty("strat")
        assert ch.get_dirty("strat") == set()

    def test_multiple_listeners_independent_events(self):
        ch: Channel[int] = make_channel()
        ev_a = ch.register_listener("a")
        ev_b = ch.register_listener("b")
        ch.publish("AAPL", 1)
        assert ev_a.is_set()
        assert ev_b.is_set()

    def test_multiple_listeners_independent_dirty_sets(self):
        ch: Channel[int] = make_channel()
        ch.register_listener("a")
        ch.register_listener("b")
        ch.publish("AAPL", 1)
        ch.get_dirty("a")  # clears a's set
        assert ch.get_dirty("b") == {"AAPL"}  # b's set untouched

    def test_rapid_publishes_symbol_appears_once_in_dirty(self):
        ch: Channel[int] = make_channel()
        ch.register_listener("strat")
        for i in range(10):
            ch.publish("AAPL", i)
        dirty = ch.get_dirty("strat")
        assert dirty == {"AAPL"}

    def test_publishes_to_multiple_symbols_all_in_dirty(self):
        ch: Channel[int] = make_channel()
        ch.register_listener("strat")
        ch.publish("AAPL", 1)
        ch.publish("GOOG", 2)
        ch.publish("MSFT", 3)
        dirty = ch.get_dirty("strat")
        assert dirty == {"AAPL", "GOOG", "MSFT"}


class TestAllListenersClear:
    def test_true_when_no_listeners(self):
        """No listeners registered — trivially clear."""
        ch: Channel[int] = make_channel()
        assert ch.all_listeners_clear() is True

    def test_true_when_all_events_cleared(self):
        """All listeners processed and cleared their events."""
        ch: Channel[int] = make_channel()
        event = ch.register_listener("strat")
        ch.publish("AAPL", 1)
        event.clear()
        assert ch.all_listeners_clear() is True

    def test_false_when_any_event_is_set(self):
        """At least one listener has an unprocessed event."""
        ch: Channel[int] = make_channel()
        ch.register_listener("strat")
        ch.publish("AAPL", 1)  # sets event
        assert ch.all_listeners_clear() is False

    def test_false_when_one_of_two_listeners_still_set(self):
        """Only one listener cleared — still not all clear."""
        ch: Channel[int] = make_channel()
        event_a = ch.register_listener("a")
        ch.register_listener("b")
        ch.publish("AAPL", 1)
        event_a.clear()  # a clears, b still set
        assert ch.all_listeners_clear() is False

    def test_true_after_all_events_cleared(self):
        """Becomes True once every listener clears."""
        ch: Channel[int] = make_channel()
        event_a = ch.register_listener("a")
        event_b = ch.register_listener("b")
        ch.publish("AAPL", 1)
        event_a.clear()
        event_b.clear()
        assert ch.all_listeners_clear() is True

