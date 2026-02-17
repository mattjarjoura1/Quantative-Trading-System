"""Random strategy for testing the pipeline end-to-end."""

import numpy as np

from src.strategy.base import BaseStrategy
from src.types import Tick, Signal

# Probability of generating a signal on any given tick
RANDOM_THRESHOLD = 0.3
# Maximum position size in either direction (units of stock)
MAX_TRADE = 2


class RandomStrategy(BaseStrategy):
    """Emits random target positions for pipeline testing.

    Not a real strategy — used to verify that Data → Strategy → Execution
    wiring works correctly. To be removed once real strategies are in place.

    Args:
        tickers: List of assets to generate random positions for.
    """

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    def on_tick(self, tick: Tick) -> Signal | None:
        """Randomly emit a signal with probability RANDOM_THRESHOLD.

        Args:
            tick: Incoming price observation (used for timestamp only).

        Returns:
            A Signal with random positions per asset, or None.
        """
        if np.random.uniform() > RANDOM_THRESHOLD:
            return None

        positions = {
            tick.asset: np.random.uniform(-MAX_TRADE, MAX_TRADE)
        }

        return Signal(timestamp=tick.timestamp, positions=positions)