"""Shared dataclasses used as inter-component contracts throughout the pipeline."""

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class Signal:
    """A trading signal emitted by a strategy for a single symbol.

    Flows from strategy → risk engine → execution. The metadata field carries
    strategy-specific diagnostics for logging and analysis; downstream
    components ignore it.

    Attributes:
        timestamp_ms: When the signal was generated (millisecond unix timestamp).
        symbol: Which asset this signal applies to.
        side: One of "BUY", "SELL", "HOLD".
        quantity: Target position size (absolute, not delta). Non-negative.
        price: Reference price at time of signal (e.g. best ask for buy).
        metadata: Optional strategy-specific data (e.g. {"rsi": 72.3}).
        _validate: When False, skips all validation.
    """

    timestamp_ms: int
    symbol: str
    side: str
    quantity: float
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
        if self.side not in ("BUY", "SELL", "HOLD"):
            raise ValueError("side must be one of 'BUY', 'SELL', 'HOLD'")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if self.price <= 0:
            raise ValueError("price must be positive")


@dataclass(frozen=True)
class TradeRecord:
    """An executed trade record produced by SimulationExecution.

    Wraps a Signal with execution metadata. Designed as an extension point:
    future implementations can add slippage, commission, or exchange-specific
    fields without modifying Signal.

    Attributes:
        signal: The approved Signal that triggered the trade.
        fill_price: Actual execution price. In simulation this equals
            signal.price (perfect fill); live execution can set it from
            the order book at time of fill.
        filled_at_ms: Millisecond timestamp when execution was processed.
    """

    signal: Signal
    fill_price: float
    filled_at_ms: int
