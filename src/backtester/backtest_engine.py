"""Backtesting engine for generating P&L from trade logs and price data."""

import pandas as pd


class BacktestEngine:
    """Computes portfolio P&L by aligning trade signals with price history.

    Args:
        price_df: DataFrame of closing prices, indexed by date, one column per asset.
        trades_df: Trade log from execution layer with columns:
                   timestamp, asset, target_position.
    """

    def __init__(self, price_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
        self.price_df = price_df
        self.trades_df = trades_df

    def generate_pnl(self) -> pd.DataFrame:
        """Calculate portfolio-level daily and cumulative P&L.

        Returns:
            DataFrame indexed by date with columns: daily_pnl, cumulative_pnl.
        """
        # Pivot trades so each asset gets its own column of target positions
        positions = self.trades_df.pivot(
            index="timestamp", columns="asset", values="target_position"
        )

        # Reindex to full price timeline and forward-fill (hold until next signal)
        positions = positions.reindex(self.price_df.index).ffill().fillna(0)

        # Daily price changes per asset
        price_changes = self.price_df.diff()

        # Per-asset daily P&L: yesterday's position × today's price change
        daily_asset_pnl = positions.shift(1) * price_changes

        # Sum across assets for portfolio-level P&L
        daily_pnl = daily_asset_pnl.sum(axis=1).fillna(0)

        return pd.DataFrame({
            "daily_pnl": daily_pnl,
            "cumulative_pnl": daily_pnl.cumsum(),
        })