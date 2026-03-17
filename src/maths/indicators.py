from numba import njit
import numpy as np

@njit
def rsi_update(prices: np.ndarray, idx: int, new_price: float,
               gain_sum: float, loss_sum: float, period: int):
    """Incremental RSI update using a pre-allocated circular price buffer.

    Processes a single price tick, maintaining a rolling window of `period`
    prices (which yields `period - 1` price changes). Returns NaN during the
    warmup phase (first `period` calls, i.e. while idx < period).

    The caller must increment `idx` after each call and pass the returned
    `gain_sum` and `loss_sum` back on the next call.

    Args:
        prices: Pre-allocated circular buffer of size `period`. Modified in-place.
        idx: Current tick index, starting at 0.
        new_price: Incoming price to process.
        gain_sum: Accumulated sum of positive price changes in the current window.
        loss_sum: Accumulated sum of absolute negative price changes in the window.
        period: Number of prices in the circular buffer.

    Returns:
        Tuple of (rsi, gain_sum, loss_sum). rsi is NaN while idx < period.
    """
    
    pos = idx % period

    if idx < 1:
        prices[pos] = new_price
        return np.nan, gain_sum, loss_sum

    prev = prices[(idx - 1) % period]
    new_delta = new_price - prev

    gain_sum += max(new_delta, 0.0)
    loss_sum += max(-new_delta, 0.0)

    if idx >= period:
        old_delta = prices[(idx + 1) % period] - prices[pos]
        gain_sum -= max(old_delta, 0.0)
        loss_sum -= max(-old_delta, 0.0)

    prices[pos] = new_price

    if idx < period:
        return np.nan, gain_sum, loss_sum

    if loss_sum < 1e-12:
        rsi = 100.0
    else:
        rs = gain_sum / loss_sum
        rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi, gain_sum, loss_sum


@njit
def rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Batch RSI over a full price array.

    Drives rsi_update sequentially over every element of `prices`. Equivalent
    to calling rsi_update in a loop with the state threaded through each call.

    Args:
        prices: 1-D array of prices in chronological order.
        period: Circular buffer size. The RSI window covers `period - 1`
            price changes.

    Returns:
        Array of RSI values the same length as `prices`. The first `period`
        values are NaN (warmup phase).
    """
    n = len(prices)
    result = np.empty(n)
    buf = np.empty(period)
    gain_sum = 0.0
    loss_sum = 0.0
    for idx in range(n):
        rsi_val, gain_sum, loss_sum = rsi_update(
            buf, idx, prices[idx], gain_sum, loss_sum, period
        )
        result[idx] = rsi_val
    return result