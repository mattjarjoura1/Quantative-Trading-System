from numba import njit
import numpy as np

@njit
def mid_price(best_bid: float, best_ask: float) -> float:
    """Compute the arithmetic mid price between best bid and best ask.

    Args:
        best_bid: Best bid price.
        best_ask: Best ask price.

    Returns:
        Arithmetic average of best_bid and best_ask.
    """
    return (best_bid + best_ask) / 2.0


@njit
def vwmp(bid_prices: np.ndarray, bid_quantities: np.ndarray,
         ask_prices: np.ndarray, ask_quantities: np.ndarray) -> float:
    """Compute the volume-weighted mid price across multiple book levels.

    Computes a volume-weighted average price for each side independently,
    then averages the two. Accounts for depth on both sides of the book
    without collapsing all quantity into a single price level.

    Args:
        bid_prices: Bid prices, best bid first.
        bid_quantities: Bid quantities corresponding to bid_prices.
        ask_prices: Ask prices, best ask first.
        ask_quantities: Ask quantities corresponding to ask_prices.

    Returns:
        Volume-weighted mid price.
    """
    vwbid = np.dot(bid_prices, bid_quantities) / bid_quantities.sum()
    vwask = np.dot(ask_prices, ask_quantities) / ask_quantities.sum()
    return (vwbid + vwask) / 2.0