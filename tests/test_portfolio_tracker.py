"""Tests for PortfolioTracker — the core analytics state machine."""

import pytest

from src.analytics.cost_model import FlatPerTrade
from src.analytics.portfolio_tracker import PortfolioTracker
from src.types import Signal, TradeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ZERO_COST = FlatPerTrade(0.0)
FLAT_1 = FlatPerTrade(1.0)


def _signal(side: str, quantity: float, price: float, symbol: str = "SYM") -> Signal:
    return Signal(
        timestamp_ms=1_000,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        _validate=False,
    )


def _trade(side: str, quantity: float, fill_price: float, symbol: str = "SYM") -> TradeRecord:
    return TradeRecord(
        signal=_signal(side, quantity, fill_price, symbol),
        fill_price=fill_price,
        filled_at_ms=1_000,
    )


def _tracker(capital: float = 100_000.0, cost_model=None) -> PortfolioTracker:
    return PortfolioTracker(capital, cost_model or ZERO_COST)


# ===========================================================================
# Round-trip trades
# ===========================================================================


class TestRoundTrip:
    def test_buy_then_sell_same_qty_zero_cost_restores_capital(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 10.0, 50.0))
        t.on_fill(_trade("SELL", 10.0, 50.0))
        assert t.cash == pytest.approx(100_000.0)
        assert "SYM" not in t.positions

    def test_buy_then_sell_with_flat_cost_deducts_twice(self):
        t = _tracker(cost_model=FLAT_1)
        t.on_fill(_trade("BUY", 10.0, 50.0))
        t.on_fill(_trade("SELL", 10.0, 50.0))
        assert t.cash == pytest.approx(100_000.0 - 2.0)

    def test_buy_then_sell_at_higher_price_positive_pnl(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 10.0, 100.0))
        t.on_fill(_trade("SELL", 10.0, 110.0))
        assert t.realised_pnl == pytest.approx(100.0)
        assert t.cash == pytest.approx(100_000.0 + 100.0)

    def test_buy_then_sell_at_lower_price_negative_pnl(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 10.0, 100.0))
        t.on_fill(_trade("SELL", 10.0, 90.0))
        assert t.realised_pnl == pytest.approx(-100.0)
        assert t.cash == pytest.approx(100_000.0 - 100.0)

    def test_position_removed_after_full_close(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 5.0, 200.0))
        t.on_fill(_trade("SELL", 5.0, 200.0))
        assert "SYM" not in t.positions


# ===========================================================================
# Scale-in (adding to position)
# ===========================================================================


class TestScaleIn:
    def test_two_buys_volume_weighted_avg(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 10.0, 100.0))
        t.on_fill(_trade("BUY", 10.0, 120.0))
        qty, avg = t.positions["SYM"]
        assert qty == pytest.approx(20.0)
        assert avg == pytest.approx(110.0)

    def test_unequal_sizes_weighted_avg(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 3.0, 100.0))
        t.on_fill(_trade("BUY", 7.0, 200.0))
        qty, avg = t.positions["SYM"]
        assert qty == pytest.approx(10.0)
        assert avg == pytest.approx((3 * 100 + 7 * 200) / 10)


# ===========================================================================
# Partial close
# ===========================================================================


class TestPartialClose:
    def test_partial_close_reduces_qty(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 10.0, 100.0))
        t.on_fill(_trade("SELL", 3.0, 110.0))
        qty, avg = t.positions["SYM"]
        assert qty == pytest.approx(7.0)

    def test_partial_close_avg_price_unchanged(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 10.0, 100.0))
        t.on_fill(_trade("SELL", 3.0, 110.0))
        _, avg = t.positions["SYM"]
        assert avg == pytest.approx(100.0)

    def test_partial_close_realised_pnl_correct(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 10.0, 100.0))
        t.on_fill(_trade("SELL", 3.0, 110.0))
        # closed 3 units at +10 profit each
        assert t.realised_pnl == pytest.approx(30.0)


# ===========================================================================
# Reversal
# ===========================================================================


class TestReversal:
    def test_reversal_closes_old_and_opens_new(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 5.0, 100.0))
        t.on_fill(_trade("SELL", 8.0, 120.0))
        qty, avg = t.positions["SYM"]
        assert qty == pytest.approx(-3.0)
        assert avg == pytest.approx(120.0)

    def test_reversal_realises_pnl_on_old_position(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 5.0, 100.0))
        t.on_fill(_trade("SELL", 8.0, 120.0))
        # Closed 5 longs at +20 each
        assert t.realised_pnl == pytest.approx(100.0)

    def test_short_reversal_to_long(self):
        t = _tracker()
        t.on_fill(_trade("SELL", 4.0, 200.0))
        t.on_fill(_trade("BUY", 6.0, 180.0))
        qty, avg = t.positions["SYM"]
        assert qty == pytest.approx(2.0)
        assert avg == pytest.approx(180.0)
        # Closed 4 shorts at +20 each (sold at 200, bought back at 180)
        assert t.realised_pnl == pytest.approx(80.0)


# ===========================================================================
# Mark-to-market
# ===========================================================================


class TestMarkToMarket:
    def test_no_positions_equity_equals_cash(self):
        t = _tracker(capital=50_000.0)
        t.mark_to_market(1_000, {})
        assert t.equity_curve[-1] == (1_000, 50_000.0)

    def test_open_long_position_adds_unrealised(self):
        t = _tracker(capital=100_000.0)
        t.on_fill(_trade("BUY", 10.0, 100.0))
        # cash = 99_000, position = 10 @ 100
        t.mark_to_market(2_000, {"SYM": 110.0})
        _, equity = t.equity_curve[-1]
        assert equity == pytest.approx(100_100.0)  # 99_000 + 10*110

    def test_open_short_position_reflected_correctly(self):
        t = _tracker(capital=100_000.0)
        t.on_fill(_trade("SELL", 10.0, 100.0))
        # cash = 101_000, position = -10 @ 100
        t.mark_to_market(2_000, {"SYM": 90.0})
        _, equity = t.equity_curve[-1]
        assert equity == pytest.approx(100_100.0)  # 101_000 + (-10)*90

    def test_mtm_does_not_modify_positions(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 5.0, 100.0))
        before = dict(t.positions)
        t.mark_to_market(1_000, {"SYM": 200.0})
        assert t.positions == before

    def test_mtm_does_not_modify_cash(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 5.0, 100.0))
        before_cash = t.cash
        t.mark_to_market(1_000, {"SYM": 200.0})
        assert t.cash == before_cash

    def test_mtm_appends_to_equity_curve(self):
        t = _tracker()
        t.mark_to_market(1_000, {})
        t.mark_to_market(2_000, {})
        assert len(t.equity_curve) == 2

    def test_symbol_not_in_prices_excluded(self):
        t = _tracker(capital=100_000.0)
        t.on_fill(_trade("BUY", 10.0, 100.0))
        # Pass empty prices — unknown symbol, so no unrealised
        t.mark_to_market(1_000, {})
        _, equity = t.equity_curve[-1]
        assert equity == pytest.approx(99_000.0)  # just cash


# ===========================================================================
# on_fill isolation
# ===========================================================================


class TestOnFillIsolation:
    def test_on_fill_does_not_append_to_equity_curve(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 1.0, 100.0))
        assert t.equity_curve == []


# ===========================================================================
# Cost model integration
# ===========================================================================


class TestCostModel:
    def test_cost_deducted_on_buy(self):
        t = _tracker(cost_model=FLAT_1)
        t.on_fill(_trade("BUY", 1.0, 100.0))
        assert t.cash == pytest.approx(100_000.0 - 100.0 - 1.0)

    def test_cost_deducted_on_sell(self):
        t = _tracker(cost_model=FLAT_1)
        t.on_fill(_trade("SELL", 1.0, 100.0))
        assert t.cash == pytest.approx(100_000.0 + 100.0 - 1.0)

    def test_zero_cost_cash_purely_from_trade_flow(self):
        t = _tracker(cost_model=ZERO_COST)
        t.on_fill(_trade("BUY", 5.0, 200.0))
        assert t.cash == pytest.approx(100_000.0 - 1000.0)


# ===========================================================================
# Multi-symbol
# ===========================================================================


class TestMultiSymbol:
    def test_independent_position_tracking(self):
        t = _tracker()
        t.on_fill(_trade("BUY", 1.0, 100.0, symbol="A"))
        t.on_fill(_trade("BUY", 2.0, 200.0, symbol="B"))
        assert t.positions["A"] == pytest.approx((1.0, 100.0))
        assert t.positions["B"] == pytest.approx((2.0, 200.0))

    def test_mtm_aggregates_multiple_positions(self):
        t = _tracker(capital=100_000.0)
        t.on_fill(_trade("BUY", 1.0, 100.0, symbol="A"))
        t.on_fill(_trade("BUY", 1.0, 200.0, symbol="B"))
        # cash = 100_000 - 100 - 200 = 99_700
        t.mark_to_market(1_000, {"A": 110.0, "B": 210.0})
        _, equity = t.equity_curve[-1]
        assert equity == pytest.approx(99_700.0 + 110.0 + 210.0)
