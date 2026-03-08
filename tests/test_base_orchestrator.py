"""Tests for BaseOrchestrator."""

import json
import pytest
from unittest.mock import patch

from src.data.yahoo_data_source import YahooDataSource
from src.orchestrator.base_orchestrator import BaseOrchestrator
from src.types import OrderBookEntry, PriceTick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(source_type: str = "yahoo", strategy_type: str = "random_pt") -> dict:
    return {
        "bus": {
            "market_data_capacity": 64,
            "signal_capacity": 64,
            "approved_capacity": 64,
        },
        "source": {
            "type": source_type,
            "params": {
                "symbols": ["BTC-USD"],
                "start": "2020-01-01",
                "end": "2021-01-01",
            },
        },
        "strategy": {
            "type": strategy_type,
            "params": {"listener_id": "strat", "symbols": ["BTC-USD"]},
        },
        "risk": {"type": "passthrough", "params": {}},
        "execution": {"type": "simulation", "params": {"listener_id": "exec"}},
    }


@pytest.fixture
def orch() -> BaseOrchestrator:
    with patch.object(YahooDataSource, "_download", return_value=[]):
        return BaseOrchestrator(_config())


# ---------------------------------------------------------------------------
# Type safety
# ---------------------------------------------------------------------------

class TestBaseOrchestratorTypeSafety:
    def test_type_mismatch_raises(self):
        """yahoo PRODUCES PriceTick, random_obe CONSUMES OrderBookEntry → TypeError."""
        with pytest.raises(TypeError):
            BaseOrchestrator(_config(strategy_type="random_obe"))

    def test_error_message_names_both_classes(self):
        with pytest.raises(TypeError, match="YahooDataSource"):
            BaseOrchestrator(_config(strategy_type="random_obe"))

    def test_compatible_types_construct_without_error(self):
        with patch.object(YahooDataSource, "_download", return_value=[]):
            BaseOrchestrator(_config())  # should not raise

    def test_unknown_source_type_raises_key_error(self):
        cfg = _config()
        cfg["source"]["type"] = "not_a_source"
        with pytest.raises(KeyError):
            BaseOrchestrator(cfg)

    def test_unknown_strategy_type_raises_key_error(self):
        cfg = _config()
        cfg["strategy"]["type"] = "not_a_strategy"
        with pytest.raises(KeyError):
            BaseOrchestrator(cfg)


# ---------------------------------------------------------------------------
# Channel wiring
# ---------------------------------------------------------------------------

class TestBaseOrchestratorWiring:
    def test_source_wired_to_market_data_channel(self, orch):
        assert orch._source._channel is orch._bus.channel("market_data")

    def test_strategy_listens_on_market_data(self, orch):
        assert orch._strategy._listen_ch is orch._bus.channel("market_data")

    def test_strategy_publishes_to_strategy_signals(self, orch):
        assert orch._strategy._publish_ch is orch._bus.channel("strategy_signals")

    def test_risk_listens_on_strategy_signals(self, orch):
        assert orch._risk._listen_ch is orch._bus.channel("strategy_signals")

    def test_risk_publishes_to_approved_signals(self, orch):
        assert orch._risk._publish_ch is orch._bus.channel("approved_signals")

    def test_execution_listens_on_approved_signals(self, orch):
        assert orch._execution._listen_ch is orch._bus.channel("approved_signals")

    def test_execution_reads_market_data_for_fill_price(self, orch):
        assert orch._execution._market_ch is orch._bus.channel("market_data")


# ---------------------------------------------------------------------------
# FileReplaySource / data_cls resolution
# ---------------------------------------------------------------------------

def _replay_config(filepath: str, data_cls: str, strategy_type: str) -> dict:
    return {
        "bus": {
            "market_data_capacity": 64,
            "signal_capacity": 64,
            "approved_capacity": 64,
        },
        "source": {
            "type": "file_replay",
            "params": {"filepath": filepath, "data_cls": data_cls},
        },
        "strategy": {
            "type": strategy_type,
            "params": {"listener_id": "strat", "symbols": ["btcusdt"]},
        },
        "risk": {"type": "passthrough", "params": {}},
        "execution": {"type": "simulation", "params": {"listener_id": "exec"}},
    }


def _write_jsonl(filepath: str, items) -> None:
    with open(filepath, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")


class TestBaseOrchestratorDataClsResolution:
    def test_data_cls_string_resolved_to_price_tick(self, tmp_path):
        """'PriceTick' string in config is resolved to the PriceTick class."""
        tick = PriceTick(symbol="btcusdt", timestamp_ms=1, price=1.0)
        filepath = str(tmp_path / "ticks.jsonl")
        _write_jsonl(filepath, [tick])

        orch = BaseOrchestrator(_replay_config(filepath, "PriceTick", "random_pt"))
        assert orch._source.PRODUCES is PriceTick

    def test_data_cls_string_resolved_to_order_book_entry(self, tmp_path):
        """'OrderBookEntry' string in config is resolved to the OrderBookEntry class."""
        entry = OrderBookEntry(
            symbol="btcusdt", timestamp_ms=1,
            bids=((100.0, 1.0),), asks=((101.0, 1.0),),
        )
        filepath = str(tmp_path / "entries.jsonl")
        _write_jsonl(filepath, [entry])

        orch = BaseOrchestrator(_replay_config(filepath, "OrderBookEntry", "random_obe"))
        assert orch._source.PRODUCES is OrderBookEntry

    def test_type_mismatch_raises_for_file_replay(self, tmp_path):
        """FileReplaySource(PriceTick) paired with random_obe raises TypeError."""
        tick = PriceTick(symbol="btcusdt", timestamp_ms=1, price=1.0)
        filepath = str(tmp_path / "ticks.jsonl")
        _write_jsonl(filepath, [tick])

        with pytest.raises(TypeError):
            BaseOrchestrator(_replay_config(filepath, "PriceTick", "random_obe"))

    def test_unknown_data_cls_raises_key_error(self, tmp_path):
        """Unrecognised data_cls string raises KeyError."""
        filepath = str(tmp_path / "ticks.jsonl")
        (tmp_path / "ticks.jsonl").write_text("")

        cfg = _replay_config(filepath, "NotAType", "random_pt")
        with pytest.raises(KeyError):
            BaseOrchestrator(cfg)
