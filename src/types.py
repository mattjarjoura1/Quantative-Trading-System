"""Shared dataclasses used as inter-component contracts throughout the pipeline."""

from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class OrderBookEntry:
    """A single order book snapshot for one symbol at one point in time.

    This is what data sources emit and what strategies consume. It lives on
    the hot path between the data source and the message bus, so construction
    must be cheap. Strategies convert to numpy at the consumer level.

    Attributes:
        symbol: Asset identifier (e.g. "btcusdt", "GOOG").
        timestamp_ms: Millisecond-precision unix timestamp.
        bids: Nested tuples of (price, quantity), best bid first (descending).
        asks: Nested tuples of (price, quantity), best ask first (ascending).
        _validate: When False, skips all validation. Data sources can disable
            on the hot path if they trust their own parsing. Always True in
            backtesting and tests.
    """

    symbol: str
    timestamp_ms: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    _validate: bool = field(default=True, compare=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        """Validate fields at construction time.

        Raises:
            ValueError: If any field fails its constraint.
        """
        if not self._validate:
            return

        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.timestamp_ms <= 0:
            raise ValueError("timestamp_ms must be a positive integer")
        if not self.bids:
            raise ValueError("bids must be non-empty")
        if not self.asks:
            raise ValueError("asks must be non-empty")

        for entry in self.bids:
            if len(entry) != 2:
                raise ValueError("each bid entry must have exactly 2 elements (price, quantity)")
            price, qty = entry
            if price <= 0:
                raise ValueError("bid price must be positive")
            if qty < 0:
                raise ValueError("bid quantity must be non-negative")

        for entry in self.asks:
            if len(entry) != 2:
                raise ValueError("each ask entry must have exactly 2 elements (price, quantity)")
            price, qty = entry
            if price <= 0:
                raise ValueError("ask price must be positive")
            if qty < 0:
                raise ValueError("ask quantity must be non-negative")

        bid_prices = [b[0] for b in self.bids]
        if bid_prices != sorted(bid_prices, reverse=True):
            raise ValueError("bids must be sorted descending by price")

        ask_prices = [a[0] for a in self.asks]
        if ask_prices != sorted(ask_prices):
            raise ValueError("asks must be sorted ascending by price")

    def fill_price(self, side: str) -> float:
        """Return the expected fill price for a given order side.

        BUY orders fill at the best ask (lowest available seller price).
        SELL orders fill at the best bid (highest available buyer price).

        Args:
            side: "BUY" or "SELL".

        Returns:
            Best ask price for BUY, best bid price for SELL.
        """
        return self.asks[0][0] if side == "BUY" else self.bids[0][0]

    def mtm_price(self) -> float:
        """Return the mid-price for mark-to-market portfolio valuation.

        Mid-price is the arithmetic mean of best bid and best ask.
        Standard institutional convention for continuous MtM valuation —
        avoids bid-ask bounce noise that depresses Sharpe ratios.

        Returns:
            (best_bid + best_ask) / 2
        """
        return (self.bids[0][0] + self.asks[0][0]) / 2

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict.

        Tuples are converted to lists so the output is valid JSON.
        _validate is excluded — it is a construction flag, not data.

        Returns:
            Dict with keys: symbol, timestamp_ms, bids, asks.
        """
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "bids": [list(level) for level in self.bids],
            "asks": [list(level) for level in self.asks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrderBookEntry":
        """Deserialise from a dict produced by to_dict().

        Restores nested list structure back to tuples. Validates by default.

        Args:
            d: Dict with keys: symbol, timestamp_ms, bids, asks.

        Returns:
            OrderBookEntry instance.
        """
        return cls(
            symbol=d["symbol"],
            timestamp_ms=d["timestamp_ms"],
            bids=tuple(tuple(level) for level in d["bids"]),
            asks=tuple(tuple(level) for level in d["asks"]),
        )


@dataclass(frozen=True)
class PriceTick:
    """A single price observation for one symbol at one point in time.

    Used by price-only data sources (Yahoo Finance, CSV historical data,
    L1 feeds). Simpler than OrderBookEntry — no bid/ask structure, just a
    price. Strategies consuming this type operate on close prices or similar
    scalar values.

    Attributes:
        symbol: Asset identifier (e.g. "BTC-USD", "GOOG").
        timestamp_ms: Millisecond-precision unix timestamp.
        price: The observed price (positive).
        _validate: When False, skips all validation. For hot-path replay.
    """

    symbol: str
    timestamp_ms: int
    price: float
    _validate: bool = field(default=True, compare=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        """Validate fields at construction time.

        Raises:
            ValueError: If any field fails its constraint.
        """
        if not self._validate:
            return
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string.")
        if self.timestamp_ms <= 0:
            raise ValueError("timestamp_ms must be a positive integer.")
        if self.price <= 0:
            raise ValueError("price must be positive.")

    def fill_price(self, side: str) -> float:
        """Return the observed price as the fill price.

        PriceTick has only one price — side is ignored and included only
        for interface consistency with OrderBookEntry.

        Args:
            side: Not used. Included for polymorphic compatibility.

        Returns:
            The observed price.
        """
        return self.price

    def mtm_price(self) -> float:
        """Return the mark-to-market price for portfolio valuation.

        PriceTick has a single price — returns it directly.

        Returns:
            The observed price.
        """
        return self.price

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict.

        _validate is excluded — it is a construction flag, not data.

        Returns:
            Dict with keys: symbol, timestamp_ms, price.
        """
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "price": self.price,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PriceTick":
        """Deserialise from a dict produced by to_dict().

        Validates by default.

        Args:
            d: Dict with keys: symbol, timestamp_ms, price.

        Returns:
            PriceTick instance.
        """
        return cls(
            symbol=d["symbol"],
            timestamp_ms=d["timestamp_ms"],
            price=d["price"],
        )


@dataclass(frozen=True)
class Signal:
    """A trading signal representing a strategy's desired position for a symbol.

    The strategy declares WHERE it wants to be, not HOW to get there.
    Execution computes the required delta from the current position.

    Flows from strategy → risk engine → execution. The metadata field carries
    strategy-specific diagnostics for logging and analysis; downstream
    components ignore it.

    Attributes:
        timestamp_ms: When the signal was generated (millisecond unix timestamp).
        symbol: Which asset this signal applies to.
        target_position: Desired absolute position. Signed — positive is long,
            negative is short, zero is flat.
        price: Reference price at time of signal (e.g. current market price).
        metadata: Optional strategy-specific data (e.g. {"rsi": 72.3}).
        _validate: When False, skips all validation.
    """

    timestamp_ms: int
    symbol: str
    target_position: float
    price: float
    metadata: dict = field(default_factory=dict, compare=False, hash=False)
    _validate: bool = field(default=True, compare=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        """Validate fields at construction time.

        Raises:
            ValueError: If any field fails its constraint.
        """
        if not self._validate:
            return

        if self.timestamp_ms <= 0:
            raise ValueError("timestamp_ms must be a positive integer")
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.price <= 0:
            raise ValueError("price must be positive")


@dataclass(frozen=True)
class TradeRecord:
    """An executed trade produced by the execution layer.

    Represents the actual position change made to satisfy a Signal.
    The signal is the strategy's intent; delta_quantity is what execution
    actually did.

    Attributes:
        signal: The approved Signal that triggered this trade.
        delta_quantity: Signed quantity traded. Positive = bought, negative = sold.
            This is the difference between the signal's target_position and
            the position held before execution.
        fill_price: Actual execution price from the market data.
        filled_at_ms: Millisecond timestamp when execution was processed.
    """

    signal: Signal
    delta_quantity: float
    fill_price: float
    filled_at_ms: int

    @property
    def side(self) -> str:
        """Derive the trade direction from delta_quantity.

        Returns:
            "BUY" if delta_quantity > 0, "SELL" if < 0, "HOLD" if zero.
        """
        if self.delta_quantity > 0:
            return "BUY"
        elif self.delta_quantity < 0:
            return "SELL"
        return "HOLD"
