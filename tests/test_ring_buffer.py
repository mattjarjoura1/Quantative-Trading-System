"""Tests for Layer 1: RingBuffer."""

import pytest

from src.bus.ring_buffer import RingBuffer
from src.types import OrderBookEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_buffer(capacity: int = 4) -> RingBuffer[int]:
    """Return an empty int RingBuffer with the given capacity."""
    return RingBuffer(capacity)


def filled_buffer(capacity: int, n: int | None = None) -> RingBuffer[int]:
    """Return a RingBuffer with n items appended (defaults to capacity)."""
    buf: RingBuffer[int] = RingBuffer(capacity)
    for i in range(n if n is not None else capacity):
        buf.append(i)
    return buf


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError):
            RingBuffer(0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError):
            RingBuffer(-1)

    def test_valid_capacity(self):
        buf = RingBuffer(10)
        assert buf.capacity == 10
        assert buf.count == 0
        assert buf.write_idx == 0


# ===========================================================================
# Append and retrieval
# ===========================================================================


class TestAppendAndRetrieve:
    def test_append_and_retrieve_single(self):
        buf: RingBuffer[int] = RingBuffer(4)
        buf.append(42)
        assert buf[0] == 42

    def test_append_multiple_retrievable(self):
        buf = filled_buffer(4, n=3)
        assert buf[0] == 0
        assert buf[1] == 1
        assert buf[2] == 2

    def test_append_fills_to_capacity(self):
        buf = filled_buffer(4)
        for i in range(4):
            assert buf[i] == i

    def test_write_idx_increments_each_append(self):
        buf = make_buffer()
        assert buf.write_idx == 0
        buf.append(1)
        assert buf.write_idx == 1
        buf.append(2)
        assert buf.write_idx == 2

    def test_write_idx_never_resets(self):
        n = 20
        buf: RingBuffer[int] = RingBuffer(4)
        for i in range(n):
            buf.append(i)
        assert buf.write_idx == n


# ===========================================================================
# Count
# ===========================================================================


class TestCount:
    def test_count_zero_on_empty(self):
        assert make_buffer().count == 0

    def test_count_before_capacity(self):
        buf = filled_buffer(4, n=2)
        assert buf.count == 2

    def test_count_at_capacity(self):
        buf = filled_buffer(4)
        assert buf.count == 4

    def test_count_stable_after_wraparound(self):
        buf = filled_buffer(4, n=10)
        assert buf.count == 4

    def test_count_capacity_one(self):
        buf: RingBuffer[int] = RingBuffer(1)
        buf.append(99)
        assert buf.count == 1
        buf.append(100)
        assert buf.count == 1


# ===========================================================================
# Wraparound behaviour
# ===========================================================================


class TestWraparound:
    def test_oldest_overwritten_after_wraparound(self):
        buf = filled_buffer(4, n=5)  # appended 0,1,2,3,4 — 0 is gone
        with pytest.raises(IndexError):
            _ = buf[0]

    def test_second_item_accessible_after_one_wrap(self):
        buf = filled_buffer(4, n=5)  # valid: [1, 2, 3, 4]
        assert buf[1] == 1

    def test_all_latest_items_accessible_after_wraparound(self):
        buf = filled_buffer(4, n=7)  # appended 0..6, valid window: [3,4,5,6]
        assert buf[3] == 3
        assert buf[4] == 4
        assert buf[5] == 5
        assert buf[6] == 6

    def test_overwritten_slots_raise(self):
        buf = filled_buffer(4, n=7)  # valid: indices 3..6
        for i in range(3):
            with pytest.raises(IndexError):
                _ = buf[i]

    def test_capacity_one_always_readable(self):
        buf: RingBuffer[int] = RingBuffer(1)
        for i in range(5):
            buf.append(i)
            assert buf[buf.write_idx - 1] == i

    def test_capacity_one_previous_gone(self):
        buf: RingBuffer[str] = RingBuffer(1)
        buf.append("a")
        buf.append("b")
        with pytest.raises(IndexError):
            _ = buf[0]
        assert buf[1] == "b"


# ===========================================================================
# __getitem__ edge cases
# ===========================================================================


class TestGetItem:
    def test_empty_buffer_raises(self):
        buf = make_buffer()
        with pytest.raises(IndexError):
            _ = buf[0]

    def test_future_index_raises(self):
        buf = filled_buffer(4, n=2)
        with pytest.raises(IndexError):
            _ = buf[buf.write_idx]  # one past the end

    def test_far_future_index_raises(self):
        buf = filled_buffer(4, n=2)
        with pytest.raises(IndexError):
            _ = buf[100]

    def test_last_valid_index(self):
        buf = filled_buffer(4, n=3)
        assert buf[buf.write_idx - 1] == 2

    def test_first_valid_index_after_warmup(self):
        buf = filled_buffer(4)
        assert buf[0] == 0

    def test_first_valid_index_after_wraparound(self):
        buf = filled_buffer(4, n=6)  # valid: [2,3,4,5]
        assert buf[buf.write_idx - buf.count] == 2


# ===========================================================================
# Generic type support
# ===========================================================================


class TestGenericTypes:
    def test_stores_strings(self):
        buf: RingBuffer[str] = RingBuffer(2)
        buf.append("hello")
        buf.append("world")
        assert buf[0] == "hello"
        assert buf[1] == "world"

    def test_stores_floats(self):
        buf: RingBuffer[float] = RingBuffer(3)
        buf.append(3.14)
        assert buf[0] == pytest.approx(3.14)

    def test_stores_order_book_entries(self):
        buf: RingBuffer[OrderBookEntry] = RingBuffer(2)
        entry = OrderBookEntry(
            symbol="btcusdt",
            timestamp_ms=1_000,
            bids=((100.0, 1.0),),
            asks=((101.0, 1.0),),
        )
        buf.append(entry)
        assert buf[0] is entry

    def test_stores_mixed_via_any(self):
        buf: RingBuffer = RingBuffer(3)
        buf.append(1)
        buf.append("two")
        buf.append(3.0)
        assert buf[0] == 1
        assert buf[1] == "two"
        assert buf[2] == pytest.approx(3.0)
