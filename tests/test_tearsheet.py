"""Tests for render_tearsheet."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed

import math

import matplotlib.pyplot as plt
import pytest

from src.analytics.backtest_analyser import BacktestAnalyser, BacktestResult
from src.analytics.cost_model import FlatPerTrade
from src.analytics.metrics import BacktestMetrics
from src.analytics.portfolio_tracker import PortfolioTracker
from src.analytics.tearsheet import render_tearsheet
from src.types import PriceTick, Signal, TradeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MS = 1_000
_DAY = 24 * 3600 * _MS


def _tick(symbol: str, ts: int, price: float) -> PriceTick:
    return PriceTick(symbol=symbol, timestamp_ms=ts, price=price)


def _trade(symbol: str, delta: float, price: float, ts: int) -> TradeRecord:
    sig = Signal(
        timestamp_ms=ts,
        symbol=symbol,
        target_position=delta,
        price=price,
        _validate=False,
    )
    return TradeRecord(signal=sig, delta_quantity=delta, fill_price=price, filled_at_ms=ts)


def _result_with_trades() -> BacktestResult:
    """BacktestResult with a handful of trades and market ticks."""
    trade_log = [
        _trade("AAPL", 1.0, 100.0, 1 * _DAY),
        _trade("AAPL", -1.0, 110.0, 2 * _DAY),
        _trade("AAPL", 1.0, 105.0, 3 * _DAY),
        _trade("AAPL", -1.0, 95.0, 4 * _DAY),
    ]
    market_history = {
        "AAPL": [
            _tick("AAPL", 1 * _DAY, 100.0),
            _tick("AAPL", 2 * _DAY, 110.0),
            _tick("AAPL", 3 * _DAY, 105.0),
            _tick("AAPL", 4 * _DAY, 95.0),
        ]
    }
    analyser = BacktestAnalyser(trade_log, market_history, initial_capital=10_000.0)
    return analyser.run()


def _result_empty() -> BacktestResult:
    """BacktestResult with no trades and no market data."""
    analyser = BacktestAnalyser([], {}, initial_capital=10_000.0)
    return analyser.run()


def _result_nan_metrics() -> BacktestResult:
    """BacktestResult that produces NaN metrics (single equity point, no trades)."""
    market_history = {"AAPL": [_tick("AAPL", _DAY, 100.0)]}
    analyser = BacktestAnalyser([], market_history, initial_capital=10_000.0)
    return analyser.run()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderTearsheetReturnsCorrectShape:
    def test_returns_figure(self) -> None:
        result = _result_with_trades()
        fig = render_tearsheet(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_four_axes(self) -> None:
        result = _result_with_trades()
        fig = render_tearsheet(result)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_figure_size(self) -> None:
        result = _result_with_trades()
        fig = render_tearsheet(result)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12.0)
        assert h == pytest.approx(8.0)
        plt.close(fig)


class TestRenderTearsheetEdgeCases:
    def test_empty_equity_curve_does_not_crash(self) -> None:
        result = _result_empty()
        fig = render_tearsheet(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_trade_pnls_does_not_crash(self) -> None:
        result = _result_nan_metrics()
        fig = render_tearsheet(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_metrics_does_not_crash(self) -> None:
        # NaN win_rate and profit_factor arise when there are no trades
        result = _result_empty()
        assert math.isnan(result.metrics.win_rate)
        assert math.isnan(result.metrics.profit_factor)
        fig = render_tearsheet(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
