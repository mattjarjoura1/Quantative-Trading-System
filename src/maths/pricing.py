from numba import njit
import numpy as np

@njit
def mid_price(best_bid: float, best_ask: float) -> float:
    return (best_bid + best_ask) / 2.0

@njit
def vwmp(bid_prices: np.ndarray, bid_quantities: np.ndarray,
         ask_prices: np.ndarray, ask_quantities: np.ndarray) -> float:
    vwbid = np.dot(bid_prices, bid_quantities) / bid_quantities.sum()
    vwask = np.dot(ask_prices, ask_quantities) / ask_quantities.sum()
    return (vwbid + vwask) / 2.0