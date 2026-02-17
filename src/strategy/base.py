"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod

from src.types import Tick, Signal


class BaseStrategy(ABC):
    """Interface that all strategies must implement.

    A strategy receives individual price ticks and optionally emits
    a Signal representing a target portfolio position. All internal
    logic (hedging, modelling, warmup, refit cadence) lives inside
    the concrete implementation.
    """

    @abstractmethod
    def on_tick(self, tick: Tick) -> Signal | None:
        """Process a single price tick and optionally emit a signal.

        Args:
            tick: A single price observation for one asset.

        Returns:
            A Signal with target positions, or None if no action required.
        """
