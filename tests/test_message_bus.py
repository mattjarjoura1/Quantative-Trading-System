"""Tests for MessageBus — the Channel registry."""

import pytest

from src.bus.message_bus import MessageBus


def make_bus() -> MessageBus:
    return MessageBus()


class TestCreateChannel:
    def test_create_channel_returns_channel(self):
        bus = make_bus()
        ch = bus.create_channel("market_data", capacity=64)
        assert ch is not None

    def test_create_channel_stored_and_retrievable(self):
        bus = make_bus()
        ch = bus.create_channel("market_data", capacity=64)
        assert bus.channel("market_data") is ch

    def test_create_channel_duplicate_name_raises(self):
        bus = make_bus()
        bus.create_channel("market_data", capacity=64)
        with pytest.raises(ValueError):
            bus.create_channel("market_data", capacity=32)

    def test_create_multiple_channels_independent(self):
        bus = make_bus()
        ch_a = bus.create_channel("market_data", capacity=64)
        ch_b = bus.create_channel("signals", capacity=32)
        assert ch_a is not ch_b


class TestChannel:
    def test_channel_returns_same_instance(self):
        bus = make_bus()
        bus.create_channel("market_data", capacity=64)
        assert bus.channel("market_data") is bus.channel("market_data")

    def test_channel_nonexistent_raises_key_error(self):
        bus = make_bus()
        with pytest.raises(KeyError):
            bus.channel("does_not_exist")


class TestChannelNames:
    def test_channel_names_empty_initially(self):
        bus = make_bus()
        assert bus.channel_names == set()

    def test_channel_names_contains_created_channels(self):
        bus = make_bus()
        bus.create_channel("market_data", capacity=64)
        bus.create_channel("signals", capacity=32)
        assert bus.channel_names == {"market_data", "signals"}


class TestIsolation:
    def test_publish_on_one_channel_does_not_wake_listeners_on_another(self):
        """The critical test: cross-channel isolation prevents the infinite loop."""
        bus = make_bus()
        ch_a = bus.create_channel("market_data", capacity=64)
        ch_b = bus.create_channel("signals", capacity=64)

        event_b = ch_b.register_listener("risk_engine")

        ch_a.publish("AAPL", "tick")  # publish to market_data

        assert not event_b.is_set()  # signals listener must NOT have woken
