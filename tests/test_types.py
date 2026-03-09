"""Tests for Layer 0 data contracts: OrderBookEntry, PriceTick, and Signal."""

import json
import pytest
from dataclasses import FrozenInstanceError

from src.types import OrderBookEntry, PriceTick, Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_BIDS = ((100.0, 1.0), (99.0, 2.0))
VALID_ASKS = ((101.0, 1.5), (102.0, 0.5))


def make_entry(**overrides) -> OrderBookEntry:
    """Build a valid OrderBookEntry with optional field overrides."""
    kwargs = dict(
        symbol="btcusdt",
        timestamp_ms=1_700_000_000_000,
        bids=VALID_BIDS,
        asks=VALID_ASKS,
    )
    kwargs.update(overrides)
    return OrderBookEntry(**kwargs)


def make_signal(**overrides) -> Signal:
    """Build a valid Signal with optional field overrides."""
    kwargs = dict(
        timestamp_ms=1_700_000_000_000,
        symbol="btcusdt",
        side="BUY",
        quantity=0.01,
        price=101.0,
    )
    kwargs.update(overrides)
    return Signal(**kwargs)


# ===========================================================================
# OrderBookEntry tests
# ===========================================================================


class TestOrderBookEntryConstruction:
    def test_valid_construction(self):
        entry = make_entry()
        assert entry.symbol == "btcusdt"
        assert entry.timestamp_ms == 1_700_000_000_000
        assert entry.bids == VALID_BIDS
        assert entry.asks == VALID_ASKS

    def test_single_level_book(self):
        entry = make_entry(bids=((50000.0, 1.0),), asks=((50001.0, 1.0),))
        assert len(entry.bids) == 1
        assert len(entry.asks) == 1

    def test_zero_quantity_is_valid(self):
        # Quantity of 0 is allowed (e.g. cancelled level still in snapshot)
        entry = make_entry(bids=((100.0, 0.0),), asks=((101.0, 0.0),))
        assert entry.bids[0][1] == 0.0


class TestOrderBookEntryValidation:
    def test_empty_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            make_entry(symbol="")

    def test_zero_timestamp(self):
        with pytest.raises(ValueError, match="timestamp_ms"):
            make_entry(timestamp_ms=0)

    def test_negative_timestamp(self):
        with pytest.raises(ValueError, match="timestamp_ms"):
            make_entry(timestamp_ms=-1)

    def test_empty_bids(self):
        with pytest.raises(ValueError, match="bids"):
            make_entry(bids=())

    def test_empty_asks(self):
        with pytest.raises(ValueError, match="asks"):
            make_entry(asks=())

    def test_bid_wrong_length_one(self):
        with pytest.raises(ValueError, match="bid entry"):
            make_entry(bids=((100.0,),))  # type: ignore[arg-type]

    def test_bid_wrong_length_three(self):
        with pytest.raises(ValueError, match="bid entry"):
            make_entry(bids=((100.0, 1.0, 99.0),))  # type: ignore[arg-type]

    def test_ask_wrong_length_one(self):
        with pytest.raises(ValueError, match="ask entry"):
            make_entry(asks=((101.0,),))  # type: ignore[arg-type]

    def test_ask_wrong_length_three(self):
        with pytest.raises(ValueError, match="ask entry"):
            make_entry(asks=((101.0, 1.0, 99.0),))  # type: ignore[arg-type]

    def test_zero_bid_price(self):
        with pytest.raises(ValueError, match="bid price"):
            make_entry(bids=((0.0, 1.0),))

    def test_negative_bid_price(self):
        with pytest.raises(ValueError, match="bid price"):
            make_entry(bids=((-1.0, 1.0),))

    def test_negative_bid_quantity(self):
        with pytest.raises(ValueError, match="bid quantity"):
            make_entry(bids=((100.0, -0.5),))

    def test_zero_ask_price(self):
        with pytest.raises(ValueError, match="ask price"):
            make_entry(asks=((0.0, 1.0),))

    def test_negative_ask_price(self):
        with pytest.raises(ValueError, match="ask price"):
            make_entry(asks=((-1.0, 1.0),))

    def test_negative_ask_quantity(self):
        with pytest.raises(ValueError, match="ask quantity"):
            make_entry(asks=((101.0, -0.5),))

    def test_bids_not_sorted_descending(self):
        # Bids in ascending order — invalid
        with pytest.raises(ValueError, match="bids must be sorted descending"):
            make_entry(bids=((99.0, 1.0), (100.0, 2.0)))

    def test_asks_not_sorted_ascending(self):
        # Asks in descending order — invalid
        with pytest.raises(ValueError, match="asks must be sorted ascending"):
            make_entry(asks=((102.0, 0.5), (101.0, 1.5)))


class TestOrderBookEntryImmutability:
    def test_frozen(self):
        entry = make_entry()
        with pytest.raises(FrozenInstanceError):
            entry.symbol = "ethusdt"  # type: ignore[misc]


class TestOrderBookEntryValidateFlag:
    def test_validate_false_skips_all_checks(self):
        # Every field here is invalid — should not raise when _validate=False
        entry = OrderBookEntry(
            symbol="",
            timestamp_ms=-999,
            bids=(),
            asks=(),
            _validate=False,
        )
        assert entry.symbol == ""
        assert entry.timestamp_ms == -999

    def test_validate_false_with_unsorted_bids(self):
        entry = OrderBookEntry(
            symbol="btcusdt",
            timestamp_ms=1_000,
            bids=((99.0, 1.0), (100.0, 2.0)),  # ascending — normally invalid
            asks=VALID_ASKS,
            _validate=False,
        )
        assert entry.bids[0][0] == 99.0


class TestOrderBookEntryHashability:
    def test_hash_succeeds(self):
        entry = make_entry()
        assert isinstance(hash(entry), int)

    def test_usable_in_set(self):
        e1 = make_entry()
        e2 = make_entry()
        s = {e1, e2}
        assert len(s) == 1

    def test_usable_as_dict_key(self):
        entry = make_entry()
        d = {entry: "value"}
        assert d[entry] == "value"

    def test_validate_flag_does_not_affect_hash(self):
        e_valid = make_entry()
        e_no_validate = OrderBookEntry(
            symbol=e_valid.symbol,
            timestamp_ms=e_valid.timestamp_ms,
            bids=e_valid.bids,
            asks=e_valid.asks,
            _validate=False,
        )
        assert hash(e_valid) == hash(e_no_validate)
        assert e_valid == e_no_validate


# ===========================================================================
# Signal tests
# ===========================================================================


class TestSignalConstruction:
    def test_valid_construction(self):
        sig = make_signal()
        assert sig.symbol == "btcusdt"
        assert sig.side == "BUY"
        assert sig.quantity == 0.01
        assert sig.price == 101.0

    def test_all_valid_sides(self):
        for side in ("BUY", "SELL", "HOLD"):
            sig = make_signal(side=side)
            assert sig.side == side

    def test_zero_quantity_is_valid(self):
        sig = make_signal(quantity=0.0)
        assert sig.quantity == 0.0

    def test_default_metadata_is_empty_dict(self):
        sig = make_signal()
        assert sig.metadata == {}

    def test_metadata_is_stored(self):
        sig = make_signal(metadata={"rsi": 72.3, "spread": -0.02})
        assert sig.metadata["rsi"] == 72.3


class TestSignalValidation:
    def test_zero_timestamp(self):
        with pytest.raises(ValueError, match="timestamp_ms"):
            make_signal(timestamp_ms=0)

    def test_negative_timestamp(self):
        with pytest.raises(ValueError, match="timestamp_ms"):
            make_signal(timestamp_ms=-1)

    def test_empty_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            make_signal(symbol="")

    def test_invalid_side_lowercase(self):
        with pytest.raises(ValueError, match="side"):
            make_signal(side="buy")

    def test_invalid_side_with_space(self):
        with pytest.raises(ValueError, match="side"):
            make_signal(side="BUY ")

    def test_invalid_side_empty(self):
        with pytest.raises(ValueError, match="side"):
            make_signal(side="")

    def test_invalid_side_arbitrary(self):
        with pytest.raises(ValueError, match="side"):
            make_signal(side="LONG")

    def test_negative_quantity(self):
        with pytest.raises(ValueError, match="quantity"):
            make_signal(quantity=-0.01)

    def test_zero_price(self):
        with pytest.raises(ValueError, match="price"):
            make_signal(price=0.0)

    def test_negative_price(self):
        with pytest.raises(ValueError, match="price"):
            make_signal(price=-100.0)


class TestSignalImmutability:
    def test_frozen(self):
        sig = make_signal()
        with pytest.raises(FrozenInstanceError):
            sig.side = "SELL"  # type: ignore[misc]


class TestSignalValidateFlag:
    def test_validate_false_skips_all_checks(self):
        sig = Signal(
            timestamp_ms=-1,
            symbol="",
            side="INVALID",
            quantity=-999.0,
            price=-1.0,
            _validate=False,
        )
        assert sig.side == "INVALID"
        assert sig.quantity == -999.0


class TestSignalHashability:
    def test_hash_succeeds(self):
        sig = make_signal()
        assert isinstance(hash(sig), int)

    def test_hashable_with_metadata(self):
        sig = make_signal(metadata={"rsi": 72.3})
        assert isinstance(hash(sig), int)

    def test_usable_in_set(self):
        s1 = make_signal()
        s2 = make_signal()
        s = {s1, s2}
        assert len(s) == 1

    def test_usable_as_dict_key(self):
        sig = make_signal()
        d = {sig: "executed"}
        assert d[sig] == "executed"


class TestSignalMetadataEquality:
    def test_metadata_ignored_in_equality(self):
        s1 = make_signal(metadata={"rsi": 72.3})
        s2 = make_signal(metadata={"rsi": 30.0, "extra": True})
        assert s1 == s2

    def test_metadata_ignored_in_hash(self):
        s1 = make_signal(metadata={"rsi": 72.3})
        s2 = make_signal(metadata={})
        assert hash(s1) == hash(s2)

    def test_validate_flag_ignored_in_equality(self):
        s1 = make_signal()
        s2 = Signal(
            timestamp_ms=s1.timestamp_ms,
            symbol=s1.symbol,
            side=s1.side,
            quantity=s1.quantity,
            price=s1.price,
            _validate=False,
        )
        assert s1 == s2
        assert hash(s1) == hash(s2)


# ---------------------------------------------------------------------------
# PriceTick helpers
# ---------------------------------------------------------------------------

def make_tick(**overrides) -> PriceTick:
    defaults = dict(symbol="GOOG", timestamp_ms=1_000_000, price=150.0)
    return PriceTick(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# PriceTick tests
# ---------------------------------------------------------------------------

class TestPriceTickConstruction:
    def test_valid_construction(self):
        tick = make_tick()
        assert tick.symbol == "GOOG"
        assert tick.timestamp_ms == 1_000_000
        assert tick.price == 150.0

    def test_different_symbols(self):
        assert make_tick(symbol="BTC-USD").symbol == "BTC-USD"
        assert make_tick(symbol="AAPL").symbol == "AAPL"

    def test_various_prices(self):
        assert make_tick(price=0.001).price == 0.001
        assert make_tick(price=1_000_000.0).price == 1_000_000.0


class TestPriceTickValidation:
    def test_empty_symbol_raises(self):
        with pytest.raises(ValueError, match="symbol"):
            make_tick(symbol="")

    def test_zero_timestamp_raises(self):
        with pytest.raises(ValueError, match="timestamp_ms"):
            make_tick(timestamp_ms=0)

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError, match="timestamp_ms"):
            make_tick(timestamp_ms=-1)

    def test_zero_price_raises(self):
        with pytest.raises(ValueError, match="price"):
            make_tick(price=0.0)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="price"):
            make_tick(price=-1.0)


class TestPriceTickImmutability:
    def test_frozen(self):
        tick = make_tick()
        with pytest.raises(FrozenInstanceError):
            tick.price = 999.0  # type: ignore[misc]


class TestPriceTickValidateFlag:
    def test_validate_false_skips_all_checks(self):
        tick = PriceTick(symbol="", timestamp_ms=-1, price=-99.0, _validate=False)
        assert tick.symbol == ""
        assert tick.price == -99.0

    def test_validate_flag_ignored_in_equality(self):
        t1 = make_tick()
        t2 = PriceTick(symbol=t1.symbol, timestamp_ms=t1.timestamp_ms,
                       price=t1.price, _validate=False)
        assert t1 == t2
        assert hash(t1) == hash(t2)


class TestPriceTickHashability:
    def test_hash_succeeds(self):
        assert isinstance(hash(make_tick()), int)

    def test_usable_in_set(self):
        t1 = make_tick()
        t2 = make_tick()
        assert len({t1, t2}) == 1

    def test_usable_as_dict_key(self):
        tick = make_tick()
        d = {tick: "close"}
        assert d[tick] == "close"


# ===========================================================================
# fill_price method tests
# ===========================================================================

class TestOrderBookEntryFillPrice:
    def test_buy_fills_at_best_ask(self):
        entry = make_entry()
        assert entry.fill_price("BUY") == VALID_ASKS[0][0]  # 101.0

    def test_sell_fills_at_best_bid(self):
        entry = make_entry()
        assert entry.fill_price("SELL") == VALID_BIDS[0][0]  # 100.0

    def test_buy_uses_first_ask_not_second(self):
        entry = make_entry(asks=((101.0, 1.0), (102.0, 0.5)))
        assert entry.fill_price("BUY") == 101.0

    def test_sell_uses_first_bid_not_second(self):
        entry = make_entry(bids=((100.0, 1.0), (99.0, 2.0)))
        assert entry.fill_price("SELL") == 100.0


class TestPriceTickFillPrice:
    def test_buy_returns_price(self):
        tick = make_tick(price=55.0)
        assert tick.fill_price("BUY") == 55.0

    def test_sell_returns_same_price(self):
        tick = make_tick(price=55.0)
        assert tick.fill_price("SELL") == 55.0

    def test_side_is_ignored(self):
        tick = make_tick(price=123.45)
        assert tick.fill_price("BUY") == tick.fill_price("SELL")


# ===========================================================================
# mtm_price method tests
# ===========================================================================


class TestOrderBookEntryMtmPrice:
    def test_returns_mid_price(self):
        # bid=100, ask=102 → mid=101
        entry = make_entry(bids=((100.0, 1.0),), asks=((102.0, 1.0),))
        assert entry.mtm_price() == 101.0

    def test_asymmetric_spread(self):
        # bid=99, ask=101 → mid=100
        entry = make_entry(bids=((99.0, 1.0),), asks=((101.0, 1.0),))
        assert entry.mtm_price() == 100.0

    def test_uses_best_bid_and_ask_only(self):
        # Second levels should not affect the result
        entry = make_entry(
            bids=((100.0, 1.0), (90.0, 2.0)),
            asks=((102.0, 1.0), (110.0, 2.0)),
        )
        assert entry.mtm_price() == 101.0

    def test_zero_spread(self):
        entry = make_entry(bids=((100.0, 1.0),), asks=((100.0, 1.0),))
        assert entry.mtm_price() == 100.0


class TestPriceTickMtmPrice:
    def test_returns_price(self):
        tick = make_tick(price=150.0)
        assert tick.mtm_price() == 150.0

    def test_matches_fill_price(self):
        tick = make_tick(price=55.75)
        assert tick.mtm_price() == tick.fill_price("BUY")
        assert tick.mtm_price() == tick.fill_price("SELL")


# ===========================================================================
# Serialisation tests
# ===========================================================================


class TestPriceTickSerialisation:
    def test_to_dict_contains_expected_keys(self):
        d = make_tick().to_dict()
        assert set(d.keys()) == {"symbol", "timestamp_ms", "price"}

    def test_to_dict_values(self):
        tick = make_tick(symbol="AAPL", timestamp_ms=2_000_000, price=200.0)
        d = tick.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["timestamp_ms"] == 2_000_000
        assert d["price"] == 200.0

    def test_validate_excluded_from_dict(self):
        assert "_validate" not in make_tick().to_dict()

    def test_round_trip_json(self):
        original = make_tick()
        restored = PriceTick.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored == original

    def test_from_dict_validates_by_default(self):
        d = make_tick().to_dict()
        d["price"] = -1.0
        with pytest.raises(ValueError, match="price"):
            PriceTick.from_dict(d)


class TestOrderBookEntrySerialisation:
    def test_to_dict_contains_expected_keys(self):
        d = make_entry().to_dict()
        assert set(d.keys()) == {"symbol", "timestamp_ms", "bids", "asks"}

    def test_validate_excluded_from_dict(self):
        assert "_validate" not in make_entry().to_dict()

    def test_bids_asks_are_lists(self):
        d = make_entry().to_dict()
        assert isinstance(d["bids"], list)
        assert isinstance(d["bids"][0], list)
        assert isinstance(d["asks"], list)
        assert isinstance(d["asks"][0], list)

    def test_round_trip_json(self):
        original = make_entry()
        restored = OrderBookEntry.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored == original

    def test_from_dict_restores_tuple_structure(self):
        restored = OrderBookEntry.from_dict(make_entry().to_dict())
        assert isinstance(restored.bids, tuple)
        assert isinstance(restored.bids[0], tuple)
        assert isinstance(restored.asks, tuple)
        assert isinstance(restored.asks[0], tuple)

    def test_from_dict_validates_by_default(self):
        d = make_entry().to_dict()
        d["symbol"] = ""
        with pytest.raises(ValueError, match="symbol"):
            OrderBookEntry.from_dict(d)
