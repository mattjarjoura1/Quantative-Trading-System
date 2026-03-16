"""Tests for MetricsCalculator and BacktestMetrics."""

import math

import numpy as np
import pytest

from src.analytics.metrics import BacktestMetrics, MetricsCalculator

_MS_PER_YEAR = 365.25 * 24 * 3600 * 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _curve(values: list[float], spacing_ms: int = 1_000) -> tuple[np.ndarray, np.ndarray]:
    """Build (equity, timestamps) arrays from a list of equity values."""
    n = len(values)
    ts = np.arange(n, dtype=np.float64) * spacing_ms
    eq = np.array(values, dtype=np.float64)
    return eq, ts


def _compute(
    values: list[float],
    spacing_ms: int = 1_000,
    num_trades: int = 0,
    trade_pnls: list[float] | None = None,
) -> BacktestMetrics:
    eq, ts = _curve(values, spacing_ms)
    return MetricsCalculator.compute(eq, ts, num_trades=num_trades, trade_pnls=trade_pnls or [])


# ===========================================================================
# Total PnL and Return
# ===========================================================================


class TestTotalPnlAndReturn:
    def test_flat_curve_zero_pnl(self):
        m = _compute([100.0, 100.0, 100.0])
        assert m.total_pnl == pytest.approx(0.0)
        assert m.total_return_pct == pytest.approx(0.0)

    def test_gaining_curve(self):
        m = _compute([100.0, 110.0])
        assert m.total_pnl == pytest.approx(10.0)
        assert m.total_return_pct == pytest.approx(0.10)

    def test_losing_curve(self):
        m = _compute([100.0, 90.0])
        assert m.total_pnl == pytest.approx(-10.0)
        assert m.total_return_pct == pytest.approx(-0.10)


# ===========================================================================
# CAGR
# ===========================================================================


class TestCAGR:
    def test_zero_duration_returns_zero(self):
        eq = np.array([100.0, 110.0])
        ts = np.array([1000.0, 1000.0])  # same timestamp
        m = MetricsCalculator.compute(eq, ts)
        assert m.cagr == 0.0

    def test_one_year_doubling(self):
        eq = np.array([100.0, 200.0])
        ts = np.array([0.0, _MS_PER_YEAR])
        m = MetricsCalculator.compute(eq, ts)
        assert m.cagr == pytest.approx(1.0)  # 100% return in 1 year

    def test_sub_year_computes(self):
        # Should not raise, just produce a value
        eq = np.array([100.0, 105.0])
        ts = np.array([0.0, _MS_PER_YEAR / 2])
        m = MetricsCalculator.compute(eq, ts)
        # CAGR > return because compounding < 1 year
        assert m.cagr > 0.0


# ===========================================================================
# Sharpe
# ===========================================================================


class TestSharpe:
    def test_flat_curve_sharpe_is_nan(self):
        m = _compute([100.0, 100.0, 100.0])
        assert math.isnan(m.annualised_sharpe)

    def test_monotone_increase_positive_sharpe(self):
        m = _compute([100.0, 101.0, 102.0, 103.0])
        assert m.annualised_sharpe > 0.0

    def test_single_point_is_nan(self):
        eq = np.array([100.0])
        ts = np.array([0.0])
        m = MetricsCalculator.compute(eq, ts)
        assert math.isnan(m.annualised_sharpe)


# ===========================================================================
# Max Drawdown
# ===========================================================================


class TestMaxDrawdown:
    def test_monotone_increasing_zero_drawdown(self):
        m = _compute([100.0, 110.0, 120.0])
        assert m.max_drawdown_pct == pytest.approx(0.0)

    def test_known_drawdown(self):
        # Peak=100, trough=80 → drawdown = 20%
        m = _compute([100.0, 90.0, 80.0, 95.0, 100.0])
        assert m.max_drawdown_pct == pytest.approx(0.20)

    def test_partial_recovery(self):
        # Peak=100, trough=80, recover to 90 only → drawdown = 20%
        m = _compute([100.0, 80.0, 90.0])
        assert m.max_drawdown_pct == pytest.approx(0.20)

    def test_never_new_peak_after_drop(self):
        m = _compute([100.0, 50.0])
        assert m.max_drawdown_pct == pytest.approx(0.50)


# ===========================================================================
# Max Drawdown Duration
# ===========================================================================


class TestMaxDrawdownDuration:
    def test_no_drawdown_duration_zero(self):
        m = _compute([100.0, 110.0, 120.0], spacing_ms=1_000)
        assert m.max_drawdown_duration_ms == 0

    def test_known_duration_recovered(self):
        # [100, 90, 80, 100] with 1000ms spacing, ts = [0, 1000, 2000, 3000]
        # Peak at ts[0]=0, below-peak at ts[1] and ts[2], recovery at ts[3]
        # duration = ts[2] - ts[0] = 2000ms
        m = _compute([100.0, 90.0, 80.0, 100.0], spacing_ms=1_000)
        assert m.max_drawdown_duration_ms == 2_000

    def test_unrecovered_drawdown_extends_to_end(self):
        # [100, 90, 80] ts = [0, 1000, 2000] — never recovers
        # Peak at ts[0]=0, below-peak through ts[-1]=2000
        # duration = ts[-1] - ts[0] = 2000ms
        m = _compute([100.0, 90.0, 80.0], spacing_ms=1_000)
        assert m.max_drawdown_duration_ms == 2_000


# ===========================================================================
# Trade-level metrics
# ===========================================================================


class TestNumTrades:
    def test_no_trades(self):
        m = _compute([100.0, 100.0], num_trades=0)
        assert m.num_trades == 0

    def test_counts_trades(self):
        m = _compute([100.0, 100.0], num_trades=3)
        assert m.num_trades == 3


class TestWinRate:
    def test_no_trades_is_nan(self):
        m = _compute([100.0])
        assert math.isnan(m.win_rate)

    def test_single_win(self):
        m = _compute([100.0, 110.0], trade_pnls=[10.0])
        assert m.win_rate == pytest.approx(1.0)

    def test_single_loss(self):
        m = _compute([100.0, 90.0], trade_pnls=[-10.0])
        assert m.win_rate == pytest.approx(0.0)

    def test_mixed_trades(self):
        # 2 wins, 1 loss
        m = _compute([100.0, 110.0], trade_pnls=[10.0, 10.0, -10.0])
        assert m.win_rate == pytest.approx(2 / 3)

    def test_sell_win(self):
        m = _compute([100.0], trade_pnls=[10.0])
        assert m.win_rate == pytest.approx(1.0)


class TestProfitFactor:
    def test_no_trades_is_nan(self):
        m = _compute([100.0])
        assert math.isnan(m.profit_factor)

    def test_all_wins_is_inf(self):
        m = _compute([100.0], trade_pnls=[10.0])
        assert math.isinf(m.profit_factor)

    def test_all_losses_is_zero(self):
        m = _compute([100.0], trade_pnls=[-10.0])
        assert m.profit_factor == pytest.approx(0.0)

    def test_known_ratio(self):
        # 2 wins of 10 each, 1 loss of 5 → PF = 20 / 5 = 4.0
        m = _compute([100.0], trade_pnls=[10.0, 10.0, -5.0])
        assert m.profit_factor == pytest.approx(4.0)


# ===========================================================================
# Empty equity curve
# ===========================================================================


class TestEmptyCurve:
    def test_empty_curve_zero_pnl(self):
        m = MetricsCalculator.compute(np.array([]), np.array([]))
        assert m.total_pnl == 0.0
        assert m.total_return_pct == 0.0
        assert m.cagr == 0.0
        assert m.max_drawdown_pct == 0.0
        assert m.max_drawdown_duration_ms == 0
