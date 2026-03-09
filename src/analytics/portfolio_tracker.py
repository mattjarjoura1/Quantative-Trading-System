"""Portfolio state machine for backtesting analytics."""

from src.analytics.cost_model import BaseCostModel
from src.types import TradeRecord


class PortfolioTracker:
    """Tracks portfolio state: cash, positions, and mark-to-market equity.

    Position ledger uses average cost basis. Quantity is signed —
    positive for long, negative for short. Realised PnL is tracked
    on position reductions and reversals.

    Args:
        initial_capital: Starting cash balance.
        cost_model: Transaction cost calculator (injected).
    """

    def __init__(self, initial_capital: float, cost_model: BaseCostModel) -> None:
        self.cash: float = initial_capital
        self.initial_capital: float = initial_capital
        self.positions: dict[str, tuple[float, float]] = {}   # {symbol: (qty, avg_price)}
        self.equity_curve: list[tuple[int, float]] = []        # [(timestamp_ms, equity)]
        self.realised_pnl: float = 0.0
        self._cost_model = cost_model

    def on_fill(self, trade: TradeRecord) -> None:
        """Process a fill: update position ledger and cash.

        Position update logic has five cases based on current state:
        1. Opening fresh position — avg_price = fill_price.
        2. Adding to existing (same direction) — volume-weighted average.
        3a. Fully closing — realise PnL, remove from ledger.
        3b. Partial close — avg unchanged, realise PnL on closed portion.
        3c. Reversal (crosses zero) — close old fully, open new at fill_price.

        Cash is adjusted by: -(signed_qty × fill_price) - cost.

        Args:
            trade: The executed TradeRecord from the pipeline.
        """
        symbol = trade.signal.symbol
        side = trade.signal.side
        quantity = trade.signal.quantity
        fill_price = trade.fill_price

        signed_qty = quantity if side == "BUY" else -quantity
        old_qty, old_avg = self.positions.get(symbol, (0.0, 0.0))
        new_qty = old_qty + signed_qty

        if old_qty == 0.0:
            # Case 1: Fresh open
            new_avg = fill_price
        elif (old_qty > 0) == (signed_qty > 0):
            # Case 2: Adding to existing position (same direction)
            new_avg = (old_qty * old_avg + signed_qty * fill_price) / new_qty
        elif abs(new_qty) < 1e-12:
            # Case 3a: Full close
            self.realised_pnl += abs(old_qty) * (fill_price - old_avg) * (1 if old_qty > 0 else -1)
            self.positions.pop(symbol, None)
            self.cash -= signed_qty * fill_price
            self.cash -= self._cost_model.calculate(symbol, side, quantity, fill_price)
            return
        elif (new_qty > 0) == (old_qty > 0):
            # Case 3b: Partial close (same sign, smaller magnitude)
            closed = abs(signed_qty)
            self.realised_pnl += closed * (fill_price - old_avg) * (1 if old_qty > 0 else -1)
            new_avg = old_avg
        else:
            # Case 3c: Reversal — cross zero
            self.realised_pnl += abs(old_qty) * (fill_price - old_avg) * (1 if old_qty > 0 else -1)
            new_avg = fill_price

        if abs(new_qty) < 1e-12:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = (new_qty, new_avg)

        self.cash -= signed_qty * fill_price
        self.cash -= self._cost_model.calculate(symbol, side, quantity, fill_price)

    def mark_to_market(self, timestamp_ms: int, prices: dict[str, float]) -> None:
        """Compute and record portfolio equity at a point in time.

        equity = cash + sum(qty × price for each open position)

        Args:
            timestamp_ms: Current timestamp in milliseconds.
            prices: {symbol: current_market_price} for all known symbols.
        """
        unrealised = sum(
            qty * prices[sym]
            for sym, (qty, _) in self.positions.items()
            if sym in prices
        )
        self.equity_curve.append((timestamp_ms, self.cash + unrealised))
