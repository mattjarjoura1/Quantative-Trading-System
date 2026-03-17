"""Tests for src/maths/pricing.py and src/maths/indicators.py."""

import numpy as np
import pytest

from src.maths.pricing import mid_price, vwmp
from src.maths.indicators import rsi_update, rsi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drive_rsi_update(prices: list[float], period: int) -> list[float]:
    """Drive rsi_update over a price list, returning one RSI value per tick.

    Args:
        prices: Price sequence to process.
        period: Circular buffer size passed to rsi_update.

    Returns:
        List of RSI values in tick order. NaN during warmup phase.
    """
    buf = np.empty(period, dtype=np.float64)
    gain_sum = 0.0
    loss_sum = 0.0
    results = []
    for idx, price in enumerate(prices):
        rsi_val, gain_sum, loss_sum = rsi_update(
            buf, idx, float(price), gain_sum, loss_sum, period
        )
        results.append(rsi_val)
    return results


# ===========================================================================
# mid_price
# ===========================================================================


class TestMidPrice:
    def test_compiles_on_first_call(self):
        assert mid_price(1.0, 2.0) == pytest.approx(1.5)

    def test_equal_bid_ask(self):
        assert mid_price(100.0, 100.0) == pytest.approx(100.0)

    def test_standard_spread(self):
        assert mid_price(99.0, 101.0) == pytest.approx(100.0)

    def test_asymmetric_spread(self):
        result = mid_price(99.0, 103.0)
        assert result == pytest.approx(101.0)
        assert 99.0 < result < 103.0


# ===========================================================================
# vwmp
# ===========================================================================


class TestVwmp:
    def test_compiles_on_first_call(self):
        result = vwmp(
            np.array([99.0]), np.array([1.0]),
            np.array([101.0]), np.array([1.0]),
        )
        assert np.isfinite(result)

    def test_single_level_equals_mid_price(self):
        # One level each side with equal qty: vwmp == simple mid.
        bp = np.array([99.0])
        bq = np.array([1.0])
        ap = np.array([101.0])
        aq = np.array([1.0])
        assert vwmp(bp, bq, ap, aq) == pytest.approx(mid_price(99.0, 101.0))

    def test_equal_quantities_multiple_levels(self):
        # Equal qty: vwbid = arithmetic mean of bids, vwask = mean of asks.
        # bid: [(100, 1), (99, 1)] → vwbid=99.5
        # ask: [(101, 1), (102, 1)] → vwask=101.5
        # result = (99.5 + 101.5) / 2 = 100.5
        bp = np.array([100.0, 99.0])
        bq = np.array([1.0, 1.0])
        ap = np.array([101.0, 102.0])
        aq = np.array([1.0, 1.0])
        assert vwmp(bp, bq, ap, aq) == pytest.approx(100.5)

    def test_skewed_bid_quantity_changes_result(self):
        # Heavy qty at best bid vs equal qty: heavier best bid pulls vwbid up.
        ap = np.array([101.0])
        aq = np.array([1.0])
        bp = np.array([100.0, 90.0])

        result_equal = vwmp(bp, np.array([1.0, 1.0]), ap, aq)
        result_heavy = vwmp(bp, np.array([3.0, 1.0]), ap, aq)

        assert result_heavy > result_equal

    def test_skewed_ask_quantity_changes_result(self):
        # Heavy qty at far ask vs equal qty: heavier far ask pulls vwask up.
        bp = np.array([100.0])
        bq = np.array([1.0])
        ap = np.array([101.0, 110.0])

        result_equal = vwmp(bp, bq, ap, np.array([1.0, 1.0]))
        result_heavy = vwmp(bp, bq, ap, np.array([1.0, 9.0]))

        assert result_heavy > result_equal

    def test_known_value_multi_level(self):
        # bid: [(100, 3), (90, 1)] → vwbid = (300 + 90) / 4 = 97.5
        # ask: [(101, 1)]          → vwask = 101.0
        # result = (97.5 + 101.0) / 2 = 99.25
        bp = np.array([100.0, 90.0])
        bq = np.array([3.0, 1.0])
        ap = np.array([101.0])
        aq = np.array([1.0])
        assert vwmp(bp, bq, ap, aq) == pytest.approx(99.25)


# ===========================================================================
# rsi_update — warmup phase
# ===========================================================================


class TestRsiUpdateWarmup:
    def test_compiles_on_first_call(self):
        buf = np.empty(3, dtype=np.float64)
        rsi_val, g, l = rsi_update(buf, 0, 100.0, 0.0, 0.0, 3)
        assert np.isnan(rsi_val)

    def test_idx_zero_returns_nan(self):
        buf = np.empty(5, dtype=np.float64)
        rsi_val, _, _ = rsi_update(buf, 0, 100.0, 0.0, 0.0, 5)
        assert np.isnan(rsi_val)

    def test_warmup_all_nan(self):
        # Exactly `period` calls should all return NaN.
        period = 4
        results = _drive_rsi_update([10.0, 11.0, 12.0, 13.0], period)
        assert len(results) == period
        assert all(np.isnan(v) for v in results)

    def test_first_steady_state_not_nan(self):
        # idx=0,1,2 → NaN; idx=3 (==period) → first real RSI.
        period = 3
        results = _drive_rsi_update([10.0, 11.0, 12.0, 13.0], period)
        assert all(np.isnan(v) for v in results[:period])
        assert not np.isnan(results[period])

    def test_gain_loss_sums_zero_on_flat_warmup(self):
        # Flat prices produce no deltas; sums stay zero throughout warmup.
        buf = np.empty(3, dtype=np.float64)
        gain_sum, loss_sum = 0.0, 0.0
        for idx in range(3):
            _, gain_sum, loss_sum = rsi_update(buf, idx, 100.0, gain_sum, loss_sum, 3)
        assert gain_sum == pytest.approx(0.0)
        assert loss_sum == pytest.approx(0.0)


# ===========================================================================
# rsi_update — steady state
# ===========================================================================


class TestRsiUpdateSteadyState:
    def test_all_gains_returns_100(self):
        # Monotone increasing → loss_sum < 1e-12 → RSI = 100.
        prices = [10.0 + i for i in range(8)]
        results = _drive_rsi_update(prices, 5)
        for v in results[5:]:
            assert v == pytest.approx(100.0)

    def test_all_losses_returns_zero(self):
        # Monotone decreasing → gain_sum = 0 → RSI = 0.
        prices = [100.0 - i for i in range(8)]
        results = _drive_rsi_update(prices, 5)
        for v in results[5:]:
            assert v == pytest.approx(0.0)

    def test_flat_prices_returns_100(self):
        # Zero deltas → both sums = 0 → loss_sum < 1e-12 guard → RSI = 100.
        prices = [50.0] * 10
        results = _drive_rsi_update(prices, 5)
        for v in results[5:]:
            assert v == pytest.approx(100.0)

    def test_known_mixed_sequence(self):
        # period=3, prices=[10, 12, 11, 13]
        # At idx=3 (first RSI), window covers deltas at idx=2 (-1) and idx=3 (+2).
        # gain=2, loss=1 → RS=2 → RSI = 100 - 100/(1+2) ≈ 66.667
        prices = [10.0, 12.0, 11.0, 13.0]
        results = _drive_rsi_update(prices, 3)
        expected = 100.0 - 100.0 / (1.0 + 2.0 / 1.0)
        assert results[3] == pytest.approx(expected, abs=1e-10)

    def test_gain_loss_sums_thread_correctly(self):
        # Verify the returned sums from one call feed the next correctly.
        # Drive to first steady-state RSI, then confirm the next call is consistent.
        period = 3
        buf = np.empty(period, dtype=np.float64)
        g, l = 0.0, 0.0
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        for idx, price in enumerate(prices):
            _, g, l = rsi_update(buf, idx, price, g, l, period)
        # All gains throughout: final RSI must be 100.
        assert _ == pytest.approx(100.0)


# ===========================================================================
# rsi_update — read-before-write ordering
# ===========================================================================


class TestRsiUpdateOldDeltaOrdering:
    def test_write_order_correct(self):
        # period=2, prices=[10, 14, 8].
        # At idx=2 the window (period-1=1 delta) contains only delta(idx=2)=-6.
        # gain=0, loss=6 → RSI=0.
        #
        # If the write to prices[pos] happened BEFORE reading old_delta, the
        # evicted delta would be computed from the new price (8) rather than the
        # old one (10), corrupting the sums and producing a wrong RSI.
        prices = [10.0, 14.0, 8.0]
        results = _drive_rsi_update(prices, 2)
        assert results[2] == pytest.approx(0.0)


# ===========================================================================
# rsi() — batch
# ===========================================================================


class TestRsiBatch:
    def test_compiles_on_first_call(self):
        prices = np.arange(10, dtype=np.float64)
        result = rsi(prices, 3)
        assert len(result) == 10

    def test_output_length_matches_input(self):
        for n in [5, 10, 20]:
            prices = np.arange(n, dtype=np.float64)
            assert len(rsi(prices, 3)) == n

    def test_first_period_values_are_nan(self):
        period = 5
        prices = np.arange(20, dtype=np.float64)
        result = rsi(prices, period)
        assert all(np.isnan(result[:period]))

    def test_remaining_values_not_nan(self):
        period = 5
        prices = np.arange(20, dtype=np.float64)
        result = rsi(prices, period)
        assert all(np.isfinite(result[period:]))

    def test_matches_rsi_update_sequential(self):
        prices_list = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 13.0, 15.0]
        period = 3
        incremental = _drive_rsi_update(prices_list, period)
        batch = rsi(np.array(prices_list, dtype=np.float64), period)
        assert np.allclose(batch, np.array(incremental), equal_nan=True)

    def test_all_gains_batch_returns_100(self):
        prices = np.arange(10, dtype=np.float64)
        period = 3
        result = rsi(prices, period)
        assert all(v == pytest.approx(100.0) for v in result[period:])

    def test_all_losses_batch_returns_zero(self):
        prices = np.arange(10, dtype=np.float64)[::-1].copy()
        period = 3
        result = rsi(prices, period)
        assert all(v == pytest.approx(0.0) for v in result[period:])
