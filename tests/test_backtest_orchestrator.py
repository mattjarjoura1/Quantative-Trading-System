"""Tests for BacktestOrchestrator."""

import pytest
from unittest.mock import patch

from src.data.yahoo_data_source import YahooDataSource
from src.orchestrator.backtest_orchestrator import BacktestOrchestrator
from src.types import PriceTick


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
            "params": {"listener_id": "strat", "symbols": ["BTC-USD"], "random_threshold": 1.0},
        },
        "risk": {"type": "passthrough", "params": {}},
        "execution": {"type": "simulation", "params": {"listener_id": "exec"}},
    }


def _make_ticks(n: int = 3) -> list[PriceTick]:
    return [
        PriceTick(symbol="BTC-USD", timestamp_ms=1_000_000 * (i + 1), price=100.0 + i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------

class TestBacktestOrchestratorTypeGuard:
    def test_non_backtest_source_raises(self):
        """BinanceDataSource is not a BacktestDataSource → TypeError."""
        cfg = _config(source_type="binance", strategy_type="random_obe")
        cfg["source"]["params"] = {"symbols": ["btcusdt"]}
        with pytest.raises(TypeError, match="not a BacktestDataSource"):
            BacktestOrchestrator(cfg)

    def test_error_names_the_source_class(self):
        cfg = _config(source_type="binance", strategy_type="random_obe")
        cfg["source"]["params"] = {"symbols": ["btcusdt"]}
        with pytest.raises(TypeError, match="BinanceDataSource"):
            BacktestOrchestrator(cfg)

    def test_yahoo_source_accepted(self):
        with patch.object(YahooDataSource, "_download", return_value=[]):
            BacktestOrchestrator(_config())  # should not raise


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------

class TestBacktestOrchestratorRun:
    def test_run_returns_trade_log(self):
        """Each tick produces at least one signal → trade log is non-empty."""
        ticks = _make_ticks(3)
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            orch = BacktestOrchestrator(_config())
        trade_log = orch.run()
        assert len(trade_log) > 0

    def test_run_populates_market_history(self):
        """All published ticks appear in market_history."""
        ticks = _make_ticks(3)
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            orch = BacktestOrchestrator(_config())
        orch.run()
        assert "BTC-USD" in orch.market_history
        assert len(orch.market_history["BTC-USD"]) == 3

    def test_market_history_matches_published_ticks(self):
        """market_history entries equal the ticks the source published."""
        ticks = _make_ticks(3)
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            orch = BacktestOrchestrator(_config())
        orch.run()
        assert orch.market_history["BTC-USD"] == ticks

    def test_run_exits_cleanly_after_data_exhausted(self):
        """run() returns without hanging when the source runs out of data."""
        import threading
        ticks = _make_ticks(2)
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            orch = BacktestOrchestrator(_config())

        result = []
        t = threading.Thread(target=lambda: result.append(orch.run()))
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "run() did not exit within timeout"

    def test_empty_source_produces_no_trades(self):
        """No ticks → no signals → empty trade log."""
        with patch.object(YahooDataSource, "_download", return_value=[]):
            orch = BacktestOrchestrator(_config())
        trade_log = orch.run()
        assert trade_log == []

    def test_empty_source_produces_empty_market_history(self):
        """No ticks → market_history is empty."""
        with patch.object(YahooDataSource, "_download", return_value=[]):
            orch = BacktestOrchestrator(_config())
        orch.run()
        assert orch.market_history == {}
