"""Tests for BaseCostModel and FlatPerTrade."""

import pytest

from src.analytics.cost_model import BaseCostModel, FlatPerTrade


class TestFlatPerTrade:
    def test_zero_cost_default(self):
        model = FlatPerTrade()
        assert model.calculate("BTC", "BUY", 1.0, 50000.0) == 0.0

    def test_explicit_zero(self):
        assert FlatPerTrade(0.0).calculate("X", "SELL", 100.0, 1.0) == 0.0

    def test_positive_cost(self):
        model = FlatPerTrade(1.50)
        assert model.calculate("GOOG", "BUY", 5.0, 200.0) == 1.50

    def test_cost_independent_of_symbol(self):
        model = FlatPerTrade(2.0)
        assert model.calculate("A", "BUY", 1.0, 1.0) == model.calculate("B", "BUY", 1.0, 1.0)

    def test_cost_independent_of_side(self):
        model = FlatPerTrade(2.0)
        assert model.calculate("A", "BUY", 1.0, 1.0) == model.calculate("A", "SELL", 1.0, 1.0)

    def test_cost_independent_of_quantity(self):
        model = FlatPerTrade(2.0)
        assert model.calculate("A", "BUY", 1.0, 1.0) == model.calculate("A", "BUY", 1000.0, 1.0)

    def test_cost_independent_of_fill_price(self):
        model = FlatPerTrade(2.0)
        assert model.calculate("A", "BUY", 1.0, 1.0) == model.calculate("A", "BUY", 1.0, 99999.0)

    def test_non_negative_return(self):
        assert FlatPerTrade(0.0).calculate("X", "BUY", 1.0, 1.0) >= 0.0
        assert FlatPerTrade(5.0).calculate("X", "BUY", 1.0, 1.0) >= 0.0


class TestBaseCostModelIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseCostModel()  # type: ignore[abstract]

    def test_subclass_must_implement_calculate(self):
        class Incomplete(BaseCostModel):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
