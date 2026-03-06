"""Tests for LiveOrchestrator."""

import threading
from unittest.mock import patch

import pytest

from src.data.yahoo_data_source import YahooDataSource
from src.orchestrator.live_orchestrator import LiveOrchestrator
from src.types import PriceTick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config() -> dict:
    return {
        "bus": {
            "market_data_capacity": 64,
            "signal_capacity": 64,
            "approved_capacity": 64,
        },
        "source": {
            "type": "yahoo",
            "params": {
                "symbols": ["BTC-USD"],
                "start": "2020-01-01",
                "end": "2021-01-01",
            },
        },
        "strategy": {
            "type": "random_pt",
            "params": {"listener_id": "strat", "symbols": ["BTC-USD"]},
        },
        "risk": {"type": "passthrough", "params": {}},
        "execution": {"type": "simulation", "params": {"listener_id": "exec"}},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLiveOrchestrator:
    def test_starts_and_stops_cleanly_with_no_data(self):
        """With an empty source, run() exits without hanging."""
        with patch.object(YahooDataSource, "_download", return_value=[]):
            orch = LiveOrchestrator(_config())

        t = threading.Thread(target=orch.run)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "run() did not exit within timeout"

    def test_run_processes_ticks_before_stopping(self):
        """run() processes all ticks before returning."""
        ticks = [
            PriceTick(symbol="BTC-USD", timestamp_ms=1_000_000, price=100.0),
            PriceTick(symbol="BTC-USD", timestamp_ms=2_000_000, price=101.0),
        ]
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            orch = LiveOrchestrator(_config())

        t = threading.Thread(target=orch.run)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert len(orch._execution.trade_log) > 0

    def test_accepts_any_data_source_type(self):
        """LiveOrchestrator does not check for BacktestDataSource."""
        with patch.object(YahooDataSource, "_download", return_value=[]):
            orch = LiveOrchestrator(_config())
        assert orch._source is not None
