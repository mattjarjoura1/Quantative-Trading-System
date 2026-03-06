"""Tests for BaseOrchestrator."""

import pytest
from unittest.mock import patch

from src.data.yahoo_data_source import YahooDataSource
from src.orchestrator.base_orchestrator import BaseOrchestrator


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
