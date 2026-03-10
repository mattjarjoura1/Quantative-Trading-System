"""Tests for BacktestAnalyser and BacktestResult."""

import pytest

from src.analytics.backtest_analyser import BacktestAnalyser, BacktestResult
from src.analytics.cost_model import FlatPerTrade
from src.analytics.metrics import BacktestMetrics
from src.analytics.portfolio_tracker import PortfolioTracker
from src.types import PriceTick, Signal, TradeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tick(symbol: str, ts: int, price: float) -> PriceTick:
    return PriceTick(symbol=symbol, timestamp_ms=ts, price=price)


def _trade(symbol: str, delta: float, price: float, ts: int) -> TradeRecord:
    """Build a TradeRecord. Positive delta = bought, negative = sold."""
    sig = Signal(
        timestamp_ms=ts,
        symbol=symbol,
        target_position=delta,
        price=price,
        _validate=False,
    )
    return TradeRecord(signal=sig, delta_quantity=delta, fill_price=price, filled_at_ms=ts)


def _analyser(trades=None, history=None, capital=100_000.0, cost=None) -> BacktestAnalyser:
    return BacktestAnalyser(
        trade_log=trades or [],
        market_history=history or {},
        initial_capital=capital,
        cost_model=cost,
    )


# ===========================================================================
# Basic result structure
# ===========================================================================


class TestResultStructure:
    def test_returns_backtest_result(self):
        result = _analyser().run()
        assert isinstance(result, BacktestResult)

    def test_result_contains_tracker(self):
        result = _analyser().run()
        assert isinstance(result.tracker, PortfolioTracker)

    def test_result_contains_metrics(self):
        result = _analyser().run()
        assert isinstance(result.metrics, BacktestMetrics)

    def test_result_is_frozen(self):
        from dataclasses import FrozenInstanceError
        result = _analyser().run()
        with pytest.raises(FrozenInstanceError):
            result.equity_curve = []  # type: ignore[misc]


# ===========================================================================
# Empty inputs
# ===========================================================================


class TestEmptyInputs:
    def test_no_trades_no_history_empty_curve(self):
        result = _analyser().run()
        assert result.equity_curve == []

    def test_no_trades_with_history_flat_curve(self):
        history = {"SYM": [_tick("SYM", 1_000, 100.0), _tick("SYM", 2_000, 100.0)]}
        result = _analyser(history=history).run()
        assert len(result.equity_curve) == 2
        for _, equity in result.equity_curve:
            assert equity == pytest.approx(100_000.0)

    def test_no_history_no_trades_zero_num_trades(self):
        result = _analyser().run()
        assert result.metrics.num_trades == 0


# ===========================================================================
# Equity curve content
# ===========================================================================


class TestEquityCurve:
    def test_all_market_ticks_produce_curve_entries(self):
        history = {"SYM": [_tick("SYM", i * 1_000, 100.0) for i in range(1, 6)]}
        result = _analyser(history=history).run()
        assert len(result.equity_curve) == 5

    def test_equity_rises_after_buy_and_price_increase(self):
        trade = _trade("SYM", 1.0, 100.0, ts=1_000)
        history = {
            "SYM": [
                _tick("SYM", 1_000, 100.0),
                _tick("SYM", 2_000, 110.0),
            ]
        }
        result = _analyser(trades=[trade], history=history).run()
        _, e1 = result.equity_curve[0]
        _, e2 = result.equity_curve[1]
        assert e2 > e1

    def test_equity_falls_after_buy_and_price_decrease(self):
        trade = _trade("SYM", 1.0, 100.0, ts=1_000)
        history = {
            "SYM": [
                _tick("SYM", 1_000, 100.0),
                _tick("SYM", 2_000, 90.0),
            ]
        }
        result = _analyser(trades=[trade], history=history).run()
        _, e1 = result.equity_curve[0]
        _, e2 = result.equity_curve[1]
        assert e2 < e1


# ===========================================================================
# Fill/tick ordering at same timestamp
# ===========================================================================


class TestEventOrdering:
    def test_fill_sorts_before_tick_at_same_timestamp(self):
        """Fill at ts=1000 and tick at ts=1000: fill processed first, then MtM."""
        trade = _trade("SYM", 1.0, 100.0, ts=1_000)
        history = {"SYM": [_tick("SYM", 1_000, 110.0)]}
        result = _analyser(trades=[trade], history=history).run()
        # After fill: cash = 99_900, position = 1 @ 100
        # MtM at price 110: equity = 99_900 + 1 * 110 = 100_010
        assert len(result.equity_curve) == 1
        _, equity = result.equity_curve[0]
        assert equity == pytest.approx(100_010.0)


# ===========================================================================
# Cost model threading
# ===========================================================================


class TestCostModel:
    def test_default_zero_cost(self):
        trade = _trade("SYM", 1.0, 100.0, ts=1_000)
        history = {"SYM": [_tick("SYM", 1_000, 100.0)]}
        result = _analyser(trades=[trade], history=history).run()
        _, equity = result.equity_curve[0]
        assert equity == pytest.approx(100_000.0)

    def test_flat_cost_deducted(self):
        trade = _trade("SYM", 1.0, 100.0, ts=1_000)
        history = {"SYM": [_tick("SYM", 1_000, 100.0)]}
        result = _analyser(trades=[trade], history=history, cost=FlatPerTrade(5.0)).run()
        _, equity = result.equity_curve[0]
        assert equity == pytest.approx(99_995.0)


# ===========================================================================
# Tracker final state
# ===========================================================================


class TestTrackerFinalState:
    def test_tracker_reflects_executed_trades(self):
        trades = [
            _trade("SYM", 10.0, 100.0, ts=1_000),
            _trade("SYM", -10.0, 110.0, ts=2_000),
        ]
        history = {"SYM": [_tick("SYM", 1_000, 100.0), _tick("SYM", 2_000, 110.0)]}
        result = _analyser(trades=trades, history=history).run()
        assert result.tracker.realised_pnl == pytest.approx(100.0)
        assert "SYM" not in result.tracker.positions

    def test_metrics_num_trades_matches_trade_log(self):
        trades = [_trade("SYM", 1.0, 100.0, ts=i * 1_000) for i in range(1, 4)]
        result = _analyser(trades=trades).run()
        assert result.metrics.num_trades == 3
