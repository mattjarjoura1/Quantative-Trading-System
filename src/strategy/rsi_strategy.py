import numpy as np

from collections import deque, defaultdict

from src.strategy import BaseStrategy
from src.types import OrderBookEntry, Signal
from src.bus import MessageBus, BufferView


class RSIStrategy(BaseStrategy):
    CONSUMES = OrderBookEntry
    
    def __init__(
        self, 
        bus: MessageBus,
        listener_id: str,
        listen_channel: str,
        publish_channel: str,
        symbols: list[str],
        rsi_period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        vwmp: bool = True
    ) -> None:
        
        super().__init__(bus, listener_id, listen_channel, publish_channel)
        
        self._views: dict[str, BufferView] = {
            symbol: BufferView(self._listen_ch.get_buffer(symbol))
            for symbol in symbols
        }
        
        if rsi_period < 0:
            raise ValueError(f"Invalid RSI period {rsi_period}")
        
        if overbought <= oversold:
            raise ValueError(f"The overbought period must be greater than the oversold")
        
        if overbought < 0 or overbought > 100:
            raise ValueError("The overbought period must be between 0 and 100")
        
        if oversold < 0 or oversold > 100:
            raise ValueError("The oversold period must be between 0 and 100")
        
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        
        self._prev_price = defaultdict(lambda: None)  # To store the last price for each symbol, if needed for VWMP calculation
        self._prev_rsi   = defaultdict(lambda: 0.0)
        self._gain_deque = defaultdict(lambda: deque(maxlen=self.rsi_period))  # To store gains for each symbol
        self._loss_deque = defaultdict(lambda: deque(maxlen=self.rsi_period))
        self._total_gain = defaultdict(lambda: 0.0)  # To store total gain for each symbol
        self._total_loss = defaultdict(lambda: 0.0)  # To store total
        
        self.mp_calculation = self._calculate_VWMP if vwmp else self._calculate_MP
        
        
    def on_data(self, dirty: set[str]) -> list[Signal]:
        
        signals = []
        
        for symbol in dirty:
            if symbol not in self._views:
                continue
            
            
            data: list[OrderBookEntry] = self._views[symbol].drain()
            
            if not data:
                continue
            
            # There is too much data to consume therefore we will only consume the 
            # part which will actually be relevant to the signal
            if len(data > self.rsi_period):
                data = data[-self.rsi_period : ]
                
            for entry in data:
                
                mid_price = self.mp_calculation(entry)
                
                self._prev_rsi[symbol] = self._calculate_RSI(symbol= symbol, current_price= mid_price)
                
            
            position = "HOLD"
            
            if self._prev_rsi[symbol] > self.overbought:
                position = "SELL"
            elif self._prev_rsi[symbol] < self.oversold:
                position = "BUY"
                
            return self._generate_signal(data[-1], position)
            
            
    
    def _generate_signal(self, snapshot: OrderBookEntry, position: str) -> Signal:
        
        return Signal(
            timestamp= snapshot.timestamp_ms,
            symbol=snapshot.symbol,
            best_ask=snapshot.asks[0,0],  # Assuming we want to hold at the best bid and ask price
            best_bid=snapshot.bids[0,0],
            position=position
        )
                  
    
    def _calculate_MP(self, snapshot: OrderBookEntry) -> float:
        """Calculate the simple mid price from best bid and ask.

        Args:
            snapshot: Order book snapshot.

        Returns:
            Mid price as (best_bid + best_ask) / 2.
        """
        
        return snapshot.mtm_price()
    
    def _calculate_VWMP(self, snapshot: OrderBookEntry) -> float:
        """Calculate volume-weighted mid price across all order book levels.

        Computes the VWAP independently for bids and asks then averages them.
        Fully vectorised using numpy dot products.

        Args:
            snapshot: Order book snapshot with bids and asks as (n, 2) float64 arrays.

        Returns:
            Volume-weighted mid price.
        """
        vwbid = np.dot(snapshot.bids[:,0], snapshot.bids[:,1]) / snapshot.bids[:,1].sum()
        vwask = np.dot(snapshot.asks[:,0], snapshot.asks[:,1]) / snapshot.asks[:,1].sum()
        
        vwap = (vwbid + vwask) / 2
        
        return vwap
    
    
    def _calculate_RSI(self, symbol: str, current_price: float) -> float | None:
        """Incrementally update RSI running totals and return the current RSI value.

        Uses an O(1) incremental approach: maintains running sums of gains and
        losses using fixed-length deques. Avoids full window recomputation on
        every tick.

        Args:
            symbol: Asset symbol used to index per-symbol state dictionaries.
            current_price: The latest VWMP value for this symbol.

        Returns:
            RSI in [0, 100], or None during the warmup period before
            rsi_period deltas have been accumulated.
        """
        prev_price = self._prev_price[symbol]
        if prev_price is None:
            self._prev_price[symbol] = current_price
            return None
        
        delta = current_price - prev_price
        self._prev_price[symbol] = current_price
        
        if delta > 0:
            new_gain = delta
            new_loss = 0.0
        else:
            new_gain = 0.0
            new_loss = -delta
            
        if len(self._gain_deque[symbol]) == self.rsi_period:
            
            self._total_gain[symbol] += new_gain - self._gain_deque[symbol][0]
            self._total_loss[symbol] += new_loss - self._loss_deque[symbol][0]
            
            self._gain_deque[symbol].append(new_gain)
            self._loss_deque[symbol].append(new_loss)
        
        else:
            self._total_gain[symbol] += new_gain
            self._total_loss[symbol] += new_loss
            
            self._gain_deque[symbol].append(new_gain)
            self._loss_deque[symbol].append(new_loss)
            
            if len(self._gain_deque[symbol]) < self.rsi_period:
                return None
        
        if self._total_loss[symbol] == 0:
            return 100.0
        else:
            rs = self._total_gain[symbol] / self._total_loss[symbol]
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi
                
            