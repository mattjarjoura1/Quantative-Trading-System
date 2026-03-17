"""Tests for RSIStrategyNumba."""

import json
import tempfile

import pytest

from src.bus.message_bus import MessageBus
from src.orchestrator.backtest_orchestrator import BacktestOrchestrator
from src.strategy.rsi_strategy_numba import RSIStrategyNumba
from src.types import OrderBookEntry, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOL = "btcusdt"
LISTEN_CH = "market_data"
PUBLISH_CH = "strategy_signals"


def _make_bus() -> MessageBus:
    """Return a MessageBus with the two channels used by the strategy."""
    bus = MessageBus()
    bus.create_channel(LISTEN_CH, capacity=256)
    bus.create_channel(PUBLISH_CH, capacity=256)
    return bus


def _make_entry(bid: float, ask: float, ts: int = 1_000_000) -> OrderBookEntry:
    """Return a single-level OrderBookEntry for SYMBOL."""
    return OrderBookEntry(
        symbol=SYMBOL,
        timestamp_ms=ts,
        bids=((bid, 1.0),),
        asks=((ask, 1.0),),
    )


def _make_strategy(bus: MessageBus, **overrides) -> RSIStrategyNumba:
    """Construct RSIStrategyNumba with sensible defaults, applying any overrides."""
    defaults = dict(
        listener_id="strat",
        symbols=[SYMBOL],
        rsi_period=3,
        overbought=70.0,
        oversold=30.0,
        levels=1,
        vwmp=False,
    )
    defaults.update(overrides)
    return RSIStrategyNumba(
        bus=bus,
        listen_channel=LISTEN_CH,
        publish_channel=PUBLISH_CH,
        **defaults,
    )


def _publish_and_call(
    strategy: RSIStrategyNumba,
    bus: MessageBus,
    entries: list[OrderBookEntry],
) -> list[Signal]:
    """Publish entries to the channel then call on_data directly."""
    ch = bus.channel(LISTEN_CH)
    for entry in entries:
        ch.publish(SYMBOL, entry)
    return strategy.on_data({SYMBOL})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_construction(self):
        strategy = _make_strategy(_make_bus())
        assert strategy.CONSUMES is OrderBookEntry

    def test_consumes_is_order_book_entry(self):
        assert RSIStrategyNumba.CONSUMES is OrderBookEntry

    def test_rsi_period_too_small(self):
        with pytest.raises(ValueError, match="rsi_period"):
            _make_strategy(_make_bus(), rsi_period=1)

    def test_rsi_period_zero_raises(self):
        with pytest.raises(ValueError, match="rsi_period"):
            _make_strategy(_make_bus(), rsi_period=0)

    def test_overbought_not_greater_than_oversold(self):
        with pytest.raises(ValueError, match="overbought"):
            _make_strategy(_make_bus(), overbought=30.0, oversold=30.0)

    def test_overbought_below_oversold(self):
        with pytest.raises(ValueError, match="overbought"):
            _make_strategy(_make_bus(), overbought=20.0, oversold=30.0)

    def test_overbought_out_of_range_high(self):
        with pytest.raises(ValueError, match="overbought"):
            _make_strategy(_make_bus(), overbought=110.0)

    def test_overbought_out_of_range_low(self):
        with pytest.raises(ValueError, match="overbought"):
            _make_strategy(_make_bus(), overbought=-1.0, oversold=-5.0)

    def test_oversold_out_of_range_high(self):
        # overbought left at default (70) to isolate the oversold check.
        with pytest.raises(ValueError, match="oversold"):
            _make_strategy(_make_bus(), oversold=110.0)

    def test_oversold_out_of_range_low(self):
        with pytest.raises(ValueError, match="oversold"):
            _make_strategy(_make_bus(), oversold=-1.0)

    def test_levels_too_small(self):
        with pytest.raises(ValueError, match="levels"):
            _make_strategy(_make_bus(), levels=0)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_no_signals_during_warmup(self):
        # rsi_period=3: first RSI at idx=3. Ticks at idx=0,1,2 → NaN → no signals.
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3)
        for i in range(3):
            signals = _publish_and_call(strategy, bus, [_make_entry(100.0 + i, 101.0 + i, ts=i + 1)])
            assert signals == [], f"expected no signal during warmup tick {i}"

    def test_first_signal_emitted_after_warmup(self):
        # Tick at idx=3 should produce the first signal (ascending prices → RSI=100).
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3, overbought=70.0)
        for i in range(3):
            _publish_and_call(strategy, bus, [_make_entry(100.0 + i, 101.0 + i, ts=i + 1)])
        signals = _publish_and_call(strategy, bus, [_make_entry(103.0, 104.0, ts=4)])
        assert len(signals) == 1


# ---------------------------------------------------------------------------
# Signal emission
# ---------------------------------------------------------------------------


class TestSignalEmission:
    def _drive_to_steady_state(
        self, strategy: RSIStrategyNumba, bus: MessageBus, n: int, start_bid: float = 100.0
    ) -> None:
        """Feed n ascending ticks to move past warmup without collecting signals."""
        for i in range(n):
            _publish_and_call(
                strategy, bus,
                [_make_entry(start_bid + i, start_bid + i + 1.0, ts=i + 1)],
            )

    def test_short_signal_when_overbought(self):
        # Ascending prices → RSI = 100 > overbought=70 → target_position = -1.0
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3, overbought=70.0)
        self._drive_to_steady_state(strategy, bus, n=3)
        signals = _publish_and_call(strategy, bus, [_make_entry(103.0, 104.0, ts=4)])
        assert len(signals) == 1
        assert signals[0].target_position == -1.0

    def test_long_signal_when_oversold(self):
        # Descending prices → RSI = 0 < oversold=30 → target_position = 1.0
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3, oversold=30.0)
        for i in range(3):
            _publish_and_call(
                strategy, bus,
                [_make_entry(100.0 - i, 101.0 - i, ts=i + 1)],
            )
        signals = _publish_and_call(strategy, bus, [_make_entry(97.0, 98.0, ts=4)])
        assert len(signals) == 1
        assert signals[0].target_position == 1.0

    def test_flat_signal_when_neutral(self):
        # Flat prices → RSI = 100 (loss_sum = 0 guard). If overbought=100 exactly
        # RSI is not strictly greater → neutral. Use overbought > 100 is invalid,
        # so instead use a mixed sequence that lands RSI in the neutral band.
        #
        # period=3, prices: 100, 102, 100, 102 → at idx=3:
        # window = deltas at idx=2 (-2) and idx=3 (+2) → gain=2, loss=2 → RSI=50.
        # overbought=70, oversold=30 → neutral → target_position=0.0
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3, overbought=70.0, oversold=30.0)
        prices = [100.0, 102.0, 100.0, 102.0]
        signals = None
        for i, p in enumerate(prices):
            signals = _publish_and_call(strategy, bus, [_make_entry(p, p + 1.0, ts=i + 1)])
        assert signals is not None
        assert len(signals) == 1
        assert signals[0].target_position == 0.0

    def test_signal_uses_mtm_price(self):
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3, overbought=70.0)
        for i in range(3):
            _publish_and_call(strategy, bus, [_make_entry(100.0 + i, 101.0 + i, ts=i + 1)])
        last_entry = _make_entry(103.0, 105.0, ts=4)
        signals = _publish_and_call(strategy, bus, [last_entry])
        assert len(signals) == 1
        assert signals[0].price == pytest.approx(last_entry.mtm_price())

    def test_signal_uses_last_entry_timestamp(self):
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3, overbought=70.0)
        for i in range(3):
            _publish_and_call(strategy, bus, [_make_entry(100.0 + i, 101.0 + i, ts=i + 1)])
        last_entry = _make_entry(103.0, 104.0, ts=99_999)
        signals = _publish_and_call(strategy, bus, [last_entry])
        assert len(signals) == 1
        assert signals[0].timestamp_ms == 99_999

    def test_unknown_symbol_skipped(self):
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3)
        # Dirty set contains a symbol the strategy wasn't initialised for.
        signals = strategy.on_data({"unknown_sym"})
        assert signals == []

    def test_empty_dirty_set_returns_empty(self):
        bus = _make_bus()
        strategy = _make_strategy(bus, rsi_period=3)
        assert strategy.on_data(set()) == []


# ---------------------------------------------------------------------------
# End-to-end via BacktestOrchestrator + FileReplaySource
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _make_config(self, filepath: str) -> dict:
        return {
            "bus": {
                "market_data_capacity": 64,
                "signal_capacity": 64,
                "approved_capacity": 64,
            },
            "source": {
                "type": "file_replay",
                "params": {"filepath": filepath, "data_cls": "OrderBookEntry"},
            },
            "strategy": {
                "type": "rsi_numba",
                "params": {
                    "listener_id": "strat",
                    "symbols": [SYMBOL],
                    "rsi_period": 3,
                    "overbought": 70.0,
                    "oversold": 30.0,
                    "levels": 1,
                    "vwmp": False,
                },
            },
            "risk": {"type": "passthrough", "params": {}},
            "execution": {"type": "simulation", "params": {"listener_id": "exec"}},
        }

    def _ascending_entries(self, n: int) -> list[dict]:
        """Return n ascending single-level OrderBookEntry dicts as JSONL lines."""
        return [
            OrderBookEntry(
                symbol=SYMBOL,
                timestamp_ms=1_000_000 * (i + 1),
                bids=((100.0 + i, 1.0),),
                asks=((101.0 + i, 1.0),),
            ).to_dict()
            for i in range(n)
        ]

    def test_backtest_run_produces_trades(self):
        # 6 ascending ticks: warmup ends at tick 3, then 3 SHORT signals.
        # SimulationExecution converts the first signal to a trade (delta != 0).
        entries = self._ascending_entries(6)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
            filepath = f.name

        orch = BacktestOrchestrator(self._make_config(filepath))
        trade_log = orch.run()
        assert len(trade_log) > 0
