"""Tests for RecordingOrchestrator."""

import json
import threading
from unittest.mock import patch

from src.data.yahoo_data_source import YahooDataSource
from src.orchestrator.recording_orchestrator import RecordingOrchestrator
from src.types import PriceTick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(filepath: str) -> dict:
    return {
        "bus": {"market_data_capacity": 64},
        "source": {
            "type": "yahoo",
            "params": {
                "symbols": ["BTC-USD"],
                "start": "2020-01-01",
                "end": "2021-01-01",
            },
        },
        "recording": {
            "listener_id": "dumper",
            "filepath": filepath,
            "symbols": ["BTC-USD"],
        },
    }


def _make_ticks(n: int) -> list[PriceTick]:
    return [
        PriceTick(symbol="BTC-USD", timestamp_ms=i + 1, price=float(i + 1))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecordingOrchestratorWiring:
    def test_source_wired_to_market_data_channel(self, tmp_path):
        """Source publishes to the market_data channel."""
        filepath = str(tmp_path / "out.jsonl")
        with patch.object(YahooDataSource, "_download", return_value=[]):
            orch = RecordingOrchestrator(_config(filepath))

        assert orch._source._channel is orch._bus.channel("market_data")

    def test_dumper_registered_as_listener(self, tmp_path):
        """Dumper is registered as a listener on the market_data channel."""
        filepath = str(tmp_path / "out.jsonl")
        with patch.object(YahooDataSource, "_download", return_value=[]):
            orch = RecordingOrchestrator(_config(filepath))

        ch = orch._bus.channel("market_data")
        assert "dumper" in ch._listeners

    def test_unknown_source_type_raises(self, tmp_path):
        """Unrecognised source type raises KeyError."""
        import pytest
        filepath = str(tmp_path / "out.jsonl")
        cfg = _config(filepath)
        cfg["source"]["type"] = "not_a_real_source"
        with pytest.raises(KeyError):
            RecordingOrchestrator(cfg)


class TestRecordingOrchestratorRun:
    def test_starts_and_stops_cleanly_with_no_data(self, tmp_path):
        """With an empty source, run() exits without hanging."""
        filepath = str(tmp_path / "out.jsonl")
        with patch.object(YahooDataSource, "_download", return_value=[]):
            orch = RecordingOrchestrator(_config(filepath))

        t = threading.Thread(target=orch.run)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "run() did not exit within timeout"

    def test_output_file_created_and_populated(self, tmp_path):
        """Output file is created and contains one line per published tick."""
        filepath = str(tmp_path / "out.jsonl")
        ticks = _make_ticks(3)
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            orch = RecordingOrchestrator(_config(filepath))

        t = threading.Thread(target=orch.run)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive()

        with open(filepath) as f:
            lines = [l for l in f if l.strip()]

        assert len(lines) == 3

    def test_recorded_ticks_are_valid_jsonl(self, tmp_path):
        """Every line in the output file is valid JSON with expected keys."""
        filepath = str(tmp_path / "out.jsonl")
        ticks = _make_ticks(2)
        with patch.object(YahooDataSource, "_download", return_value=ticks):
            orch = RecordingOrchestrator(_config(filepath))

        t = threading.Thread(target=orch.run)
        t.start()
        t.join(timeout=5.0)

        with open(filepath) as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) == 2
        for record in records:
            assert "symbol" in record
            assert "timestamp_ms" in record
            assert "price" in record
