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
