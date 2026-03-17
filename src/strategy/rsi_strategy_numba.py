"""RSI strategy using Numba-compiled math functions with zero per-tick allocation."""

import numpy as np

from src.bus import MessageBus, BufferView
from src.maths.indicators import rsi_update
from src.maths.pricing import mid_price as _mid_price, vwmp as _vwmp
from src.strategy import BaseStrategy
from src.types import OrderBookEntry, Signal


class RSIStrategyNumba(BaseStrategy[OrderBookEntry]):
    """RSI strategy backed by Numba-compiled incremental RSI and VWMP functions.

    Arrays for order book levels and the RSI price buffer are pre-allocated
    per symbol in __init__ and filled in-place each tick. No numpy allocations
    occur on the hot path.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for market data.
        publish_channel: Channel name to publish signals to.
        symbols: Symbols to track.
        rsi_period: Circular price buffer size. The RSI window covers
            rsi_period - 1 price changes.
        overbought: RSI threshold above which a short signal is emitted.
        oversold: RSI threshold below which a long signal is emitted.
        levels: Number of order book levels. Controls pre-allocated array size.
        vwmp: If True, use volume-weighted mid price. If False, use simple mid.
    """

    CONSUMES = OrderBookEntry

    def __init__(
        self,
        bus: MessageBus,
        listener_id: str,
        listen_channel: str,
        publish_channel: str,
        symbols: list[str],
        rsi_period: int,
        overbought: float,
        oversold: float,
        levels: int,
        vwmp: bool = True,
    ) -> None:
        """Validate parameters and pre-allocate per-symbol numpy arrays.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for market data.
            publish_channel: Channel name to publish signals to.
            symbols: Symbols to track.
            rsi_period: Circular price buffer size (must be >= 2).
            overbought: RSI threshold for short signals (must be > oversold).
            oversold: RSI threshold for long signals.
            levels: Number of order book levels for array pre-allocation (>= 1).
            vwmp: Use volume-weighted mid price if True, simple mid if False.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if rsi_period < 2:
            raise ValueError(f"rsi_period must be at least 2, got {rsi_period}")
        if overbought < 0 or overbought > 100:
            raise ValueError(f"overbought must be in [0, 100], got {overbought}")
        if oversold < 0 or oversold > 100:
            raise ValueError(f"oversold must be in [0, 100], got {oversold}")
        if overbought <= oversold:
            raise ValueError(
                f"overbought ({overbought}) must be greater than oversold ({oversold})"
            )
        if levels < 1:
            raise ValueError(f"levels must be at least 1, got {levels}")

        super().__init__(bus, listener_id, listen_channel, publish_channel)

        self._rsi_period = rsi_period
        self._overbought = overbought
        self._oversold = oversold
        self._use_vwmp = vwmp

        self._bid_p = {sym: np.empty(levels, dtype=np.float64) for sym in symbols}
        self._bid_q = {sym: np.empty(levels, dtype=np.float64) for sym in symbols}
        self._ask_p = {sym: np.empty(levels, dtype=np.float64) for sym in symbols}
        self._ask_q = {sym: np.empty(levels, dtype=np.float64) for sym in symbols}
        self._price_buf = {sym: np.empty(rsi_period, dtype=np.float64) for sym in symbols}
        self._idx = {sym: 0 for sym in symbols}
        self._gain_sum = {sym: 0.0 for sym in symbols}
        self._loss_sum = {sym: 0.0 for sym in symbols}
        self._views = {
            sym: BufferView(self._listen_ch.get_buffer(sym)) for sym in symbols
        }

    def on_data(self, dirty: set[str]) -> list[Signal]:
        """Process new order book snapshots and emit zero or one signal per symbol.

        Args:
            dirty: Symbols with new data since the last wake.

        Returns:
            List of Signals. At most one signal per dirty symbol, and only after
            the RSI warmup period (rsi_period ticks) has elapsed.
        """
        
        signals = []

        for symbol in dirty:
            
            
            
            if symbol not in self._views:
                continue

            entries: list[OrderBookEntry] = self._views[symbol].drain()
            if not entries:
                continue

            rsi_val = np.nan
            for entry in entries:
                for i, (p, q) in enumerate(entry.bids):
                    self._bid_p[symbol][i] = p
                    self._bid_q[symbol][i] = q
                for i, (p, q) in enumerate(entry.asks):
                    self._ask_p[symbol][i] = p
                    self._ask_q[symbol][i] = q

                if self._use_vwmp:
                    price = _vwmp(
                        self._bid_p[symbol], self._bid_q[symbol],
                        self._ask_p[symbol], self._ask_q[symbol],
                    )
                else:
                    price = _mid_price(self._bid_p[symbol][0], self._ask_p[symbol][0])
                
                rsi_val, self._gain_sum[symbol], self._loss_sum[symbol] = rsi_update(
                    self._price_buf[symbol],
                    self._idx[symbol],
                    price,
                    self._gain_sum[symbol],
                    self._loss_sum[symbol],
                    self._rsi_period,
                )
            
                self._idx[symbol] += 1

            if np.isnan(rsi_val):
                continue

            if rsi_val > self._overbought:
                target_position = -1.0
            elif rsi_val < self._oversold:
                target_position = 1.0
            else:
                target_position = 0.0

            last = entries[-1]
            
            signals.append(Signal(
                timestamp_ms=last.timestamp_ms,
                symbol=symbol,
                target_position=target_position,
                price=last.mtm_price(),
            ))

        
        
        return signals
