"""Simulated execution handler for backtesting."""

import pandas as pd

from src.execution.base import BaseExecution
from src.types import Signal


class SimulationExecution(BaseExecution):
    """Records signals as a trade log without executing real orders.

    Stores each signal's individual asset positions in a flat list
    and converts to a DataFrame on demand for analysis.
    """

    def __init__(self) -> None:
        self.raw_trade_log: list[dict] = []

    def execute(self, signal: Signal) -> None:
        """Append each asset position from the signal to the trade log.

        Args:
            signal: Target portfolio state emitted by a strategy.
        """
        for asset, target_position in signal.positions.items():
            self.raw_trade_log.append({
                "timestamp": signal.timestamp,
                "asset": asset,
                "target_position": target_position,
            })

    def get_trade_log(self) -> pd.DataFrame:
        """Convert the raw trade log to a DataFrame.

        Returns:
            DataFrame with columns: timestamp, asset, target_position.
        """
        return pd.DataFrame(self.raw_trade_log)