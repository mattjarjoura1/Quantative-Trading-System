"""Tests for BufferView — Layer 3."""

import pytest

from src.bus.ring_buffer import RingBuffer
from src.bus.buffer_view import BufferView


def _filled(capacity: int, n: int) -> RingBuffer[int]:
    """Return a RingBuffer[int] with n integers appended (values 0..n-1)."""
    buf: RingBuffer[int] = RingBuffer(capacity)
    for i in range(n):
        buf.append(i)
    return buf


class TestLatest:
    def test_empty_buffer_returns_none(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        assert view.latest() is None

    def test_returns_most_recent_item(self) -> None:
        buf = _filled(4, 3)
        view: BufferView[int] = BufferView(buf)
        assert view.latest() == 2

    def test_updates_as_buffer_grows(self) -> None:
        buf = _filled(4, 3)
        view: BufferView[int] = BufferView(buf)
        buf.append(99)
        assert view.latest() == 99

    def test_does_not_advance_read_cursor(self) -> None:
        buf = _filled(4, 3)
        view: BufferView[int] = BufferView(buf)
        before = view.read_idx
        view.latest()
        assert view.read_idx == before

    def test_latest_after_wraparound(self) -> None:
        buf = _filled(3, 5)  # capacity 3, wrote 0..4 — buffer holds 2,3,4
        view: BufferView[int] = BufferView(buf)
        assert view.latest() == 4


class TestLastN:
    def test_returns_correct_window(self) -> None:
        buf = _filled(8, 5)
        view: BufferView[int] = BufferView(buf)
        assert view.last_n(3) == [2, 3, 4]

    def test_ordered_oldest_to_newest(self) -> None:
        buf = _filled(8, 4)
        view: BufferView[int] = BufferView(buf)
        result = view.last_n(4)
        assert result == sorted(result)

    def test_n_greater_than_count_returns_all(self) -> None:
        buf = _filled(8, 3)
        view: BufferView[int] = BufferView(buf)
        assert view.last_n(10) == [0, 1, 2]

    def test_n_equals_count(self) -> None:
        buf = _filled(4, 4)
        view: BufferView[int] = BufferView(buf)
        assert view.last_n(4) == [0, 1, 2, 3]

    def test_n_equals_one(self) -> None:
        buf = _filled(4, 4)
        view: BufferView[int] = BufferView(buf)
        assert view.last_n(1) == [3]

    def test_does_not_advance_read_cursor(self) -> None:
        buf = _filled(4, 4)
        view: BufferView[int] = BufferView(buf)
        before = view.read_idx
        view.last_n(3)
        assert view.read_idx == before

    def test_last_n_after_wraparound(self) -> None:
        buf = _filled(3, 5)  # capacity 3, holds 2,3,4
        view: BufferView[int] = BufferView(buf)
        assert view.last_n(3) == [2, 3, 4]

    def test_empty_buffer_returns_empty(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        assert view.last_n(3) == []


class TestDrain:
    def test_returns_unseen_items(self) -> None:
        buf: RingBuffer[int] = RingBuffer(8)
        view: BufferView[int] = BufferView(buf)
        buf.append(10)
        buf.append(20)
        buf.append(30)
        assert view.drain() == [10, 20, 30]

    def test_advances_read_cursor(self) -> None:
        buf: RingBuffer[int] = RingBuffer(8)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        buf.append(2)
        view.drain()
        assert view.read_idx == buf.write_idx

    def test_second_drain_returns_empty(self) -> None:
        buf: RingBuffer[int] = RingBuffer(8)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        view.drain()
        assert view.drain() == []

    def test_drain_only_new_items_after_partial_read(self) -> None:
        buf: RingBuffer[int] = RingBuffer(8)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        buf.append(2)
        view.drain()
        buf.append(3)
        buf.append(4)
        assert view.drain() == [3, 4]

    def test_view_created_after_data_drain_returns_empty(self) -> None:
        buf = _filled(8, 5)
        view: BufferView[int] = BufferView(buf)  # initialised at write_idx=5
        assert view.drain() == []

    def test_view_created_after_data_sees_only_new_items(self) -> None:
        buf = _filled(8, 5)
        view: BufferView[int] = BufferView(buf)
        buf.append(99)
        assert view.drain() == [99]

    def test_drain_empty_buffer_returns_empty(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        assert view.drain() == []

    def test_drain_ordered_oldest_to_newest(self) -> None:
        buf: RingBuffer[int] = RingBuffer(8)
        view: BufferView[int] = BufferView(buf)
        for i in range(5):
            buf.append(i)
        result = view.drain()
        assert result == sorted(result)


class TestDrainOverflow:
    def test_gap_detected_flag_set_on_overflow(self) -> None:
        buf: RingBuffer[int] = RingBuffer(3)
        view: BufferView[int] = BufferView(buf)
        buf.append(0)  # view can drain this
        # now overwrite without draining — write 4 more items into capacity-3 buffer
        for i in range(1, 5):
            buf.append(i)
        view.drain()
        assert view.gap_detected is True

    def test_gap_not_detected_when_reader_keeps_up(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        for i in range(4):
            buf.append(i)
            view.drain()
        assert view.gap_detected is False

    def test_drain_returns_surviving_items_after_overflow(self) -> None:
        buf: RingBuffer[int] = RingBuffer(3)
        view: BufferView[int] = BufferView(buf)
        for i in range(6):  # write 0..5; buffer holds 3,4,5
            buf.append(i)
        result = view.drain()
        assert result == [3, 4, 5]

    def test_drain_does_not_crash_on_overflow(self) -> None:
        buf: RingBuffer[int] = RingBuffer(2)
        view: BufferView[int] = BufferView(buf)
        for i in range(10):
            buf.append(i)
        result = view.drain()  # should not raise
        assert len(result) == 2  # only capacity items survive


class TestHasNew:
    def test_false_on_empty_buffer(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        assert view.has_new() is False

    def test_true_after_append(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        assert view.has_new() is True

    def test_false_after_drain(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        view.drain()
        assert view.has_new() is False

    def test_true_again_after_new_append_post_drain(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        view.drain()
        buf.append(2)
        assert view.has_new() is True

    def test_false_for_view_created_after_data(self) -> None:
        buf = _filled(4, 4)
        view: BufferView[int] = BufferView(buf)
        assert view.has_new() is False

    def test_latest_does_not_affect_has_new(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        view.latest()
        assert view.has_new() is True

    def test_last_n_does_not_affect_has_new(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        view.last_n(1)
        assert view.has_new() is True


class TestReadIdx:
    def test_initialised_at_buffer_write_idx(self) -> None:
        buf = _filled(4, 3)
        view: BufferView[int] = BufferView(buf)
        assert view.read_idx == 3

    def test_advances_after_drain(self) -> None:
        buf: RingBuffer[int] = RingBuffer(4)
        view: BufferView[int] = BufferView(buf)
        buf.append(1)
        buf.append(2)
        view.drain()
        assert view.read_idx == 2

    def test_unchanged_after_latest(self) -> None:
        buf = _filled(4, 3)
        view: BufferView[int] = BufferView(buf)
        view.latest()
        assert view.read_idx == 3

    def test_unchanged_after_last_n(self) -> None:
        buf = _filled(4, 3)
        view: BufferView[int] = BufferView(buf)
        view.last_n(2)
        assert view.read_idx == 3


class TestFromStart:
    def test_default_starts_at_write_idx(self) -> None:
        buf = _filled(8, 5)
        view: BufferView[int] = BufferView(buf)
        assert view.read_idx == 5

    def test_from_start_empty_buffer_read_idx_is_zero(self) -> None:
        buf: RingBuffer[int] = RingBuffer(8)
        view: BufferView[int] = BufferView(buf, from_start=True)
        assert view.read_idx == 0

    def test_from_start_drain_returns_all_existing_items(self) -> None:
        buf = _filled(8, 5)
        view: BufferView[int] = BufferView(buf, from_start=True)
        assert view.drain() == [0, 1, 2, 3, 4]

    def test_from_start_wrapped_buffer_starts_at_oldest_valid(self) -> None:
        buf = _filled(3, 5)  # capacity 3, holds 2,3,4; write_idx=5, count=3
        view: BufferView[int] = BufferView(buf, from_start=True)
        assert view.read_idx == 2  # oldest valid = write_idx - count = 5 - 3

    def test_from_start_wrapped_buffer_drain_returns_surviving_items(self) -> None:
        buf = _filled(3, 5)  # capacity 3, holds 2,3,4
        view: BufferView[int] = BufferView(buf, from_start=True)
        assert view.drain() == [2, 3, 4]

    def test_from_start_false_drain_returns_empty_for_existing_data(self) -> None:
        buf = _filled(8, 5)
        view: BufferView[int] = BufferView(buf, from_start=False)
        assert view.drain() == []

    def test_from_start_subsequent_drain_sees_only_new_items(self) -> None:
        buf = _filled(8, 3)
        view: BufferView[int] = BufferView(buf, from_start=True)
        view.drain()  # consume existing [0, 1, 2]
        buf.append(99)
        assert view.drain() == [99]
