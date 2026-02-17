"""Shared dataclasses used as interfaces between pipeline components."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Tick:
    """A single price observation for one asset.

    Attributes:
        timestamp: When the price was observed.
        asset: Identifier for the asset (e.g. "GOOG", "BTC").
        price: The observed price.
    """

    timestamp: datetime
    asset: str
    price: float
    
@dataclass(frozen=True)
class Signal:
    """Target portfolio position emitted by a strategy.

    Represents an atomic set of positions that must be executed together.
    Positive values are long, negative are short, zero is flat.

    Attributes:
        timestamp: When the signal was generated.
        positions: Target position per asset (e.g. {"BTC": 0.01, "ETH": -0.02}).
    """

    timestamp: datetime
    positions: dict[str, float]
