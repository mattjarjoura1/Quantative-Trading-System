"""Transaction cost models for backtesting analytics."""

from abc import ABC, abstractmethod


class BaseCostModel(ABC):
    """Abstract base for transaction cost calculation.

    Cost models are injected into PortfolioTracker at construction.
    They receive trade details and return a dollar cost to deduct from cash.
    The fill price in the trade log is never modified — costs are separate.
    """

    @abstractmethod
    def calculate(self, symbol: str, quantity: float, fill_price: float) -> float:
        """Return the dollar cost for this fill.

        Args:
            symbol: Asset identifier.
            quantity: Absolute (unsigned) trade quantity.
            fill_price: Execution price (unmodified market price).

        Returns:
            Dollar cost to deduct from cash. Non-negative.
        """


class FlatPerTrade(BaseCostModel):
    """Fixed dollar cost per trade regardless of size or symbol.

    Args:
        cost: Dollar amount charged per trade.
    """

    def __init__(self, cost: float = 0.0) -> None:
        self._cost = cost

    def calculate(self, symbol: str, quantity: float, fill_price: float) -> float:
        """Return the fixed cost per trade.

        Args:
            symbol: Unused.
            quantity: Unused.
            fill_price: Unused.

        Returns:
            The fixed dollar cost set at construction.
        """
        return self._cost
