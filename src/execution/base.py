"""Abstract base class for all execution handlers."""

from abc import ABC, abstractmethod

import pandas as pd

from src.types import Signal


class BaseExecution(ABC):
    """Interface that all execution handlers must implement.

    At minimum, an execution handler records signals and exposes
    the trade log. Concrete implementations may range from a simple
    ledger (backtesting) to a live exchange connector.
    """

    @abstractmethod
    def execute(self, signal: Signal) -> None:
        """Record or act on a target position signal.

        Args:
            signal: The desired portfolio state to move towards.
        """

    @abstractmethod
    def get_trade_log(self) -> pd.DataFrame:
        """Return the full history of executed signals.

        Returns:
            DataFrame with columns: timestamp, asset, target_position.
        """