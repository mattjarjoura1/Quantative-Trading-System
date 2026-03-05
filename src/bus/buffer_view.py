"""Configurable reader over a RingBuffer. Tracks a read cursor per consumer."""

from typing import Generic, TypeVar

from src.bus.ring_buffer import RingBuffer

T = TypeVar("T")


class BufferView(Generic[T]):
    """A stateful lens over a RingBuffer that implements consumer read patterns.

    A BufferView wraps a RingBuffer reference and tracks a private read cursor.
    The cursor is initialised at the buffer's current write position so that a
    view created mid-stream does not retroactively see historical data on drain.

    Three read patterns are supported:
    - ``latest``: peek at the most recent item, no cursor movement.
    - ``last_n``: sliding window of the N most recent items, no cursor movement.
    - ``drain``: consume all items since the last drain, cursor advances.

    ``latest`` and ``last_n`` are stateless — they can be called any number of
    times without side effects. Only ``drain`` mutates ``_read_idx``.

    Attributes:
        _buffer: The underlying RingBuffer this view reads from.
        _read_idx: Absolute sequence position of the next item to drain.
        _gap_detected: Set to True if drain detects the buffer overwrote items
            the view had not yet consumed (reader fell behind writer).

    Args:
        buffer: The RingBuffer to attach this view to.
    """

    def __init__(self, buffer: RingBuffer[T], from_start: bool = False) -> None:
        """Attach a view to a buffer.

        Args:
            buffer: The RingBuffer to read from.
            from_start: If True, start the read cursor at the oldest item
                currently in the buffer so drain() sees pre-existing data.
                If False (default), start at the current write position so
                only items written after this view is created are visible.
        """
        self._buffer = buffer
        if from_start:
            self._read_idx: int = buffer.write_idx - buffer.count
        else:
            self._read_idx: int = buffer.write_idx
        self._gap_detected: bool = False

    def latest(self) -> T | None:
        """Return the most recently written item without advancing the cursor.

        Returns:
            The item at ``write_idx - 1``, or ``None`` if the buffer is empty.
        """
        if self._buffer.count == 0:
            return None
        return self._buffer[self._buffer.write_idx - 1]

    def last_n(self, n: int) -> list[T]:
        """Return the N most recent items, ordered oldest to newest.

        Does not advance the read cursor. If fewer than ``n`` items are in the
        buffer, returns all available items.

        Args:
            n: Number of items to return.

        Returns:
            A list of up to ``n`` items, oldest first.
        """
        available = self._buffer.count
        start = self._buffer.write_idx - min(n, available)
        return [self._buffer[i] for i in range(start, self._buffer.write_idx)]

    def drain(self) -> list[T]:
        """Return all items not yet consumed and advance the read cursor.

        If the buffer has wrapped past the read cursor (reader fell behind),
        ``_gap_detected`` is set to ``True``, the cursor is snapped forward to
        the oldest still-valid item, and only what remains is returned.

        Returns:
            A list of unread items, oldest first. Empty if nothing is new.
        """
        oldest_valid = self._buffer.write_idx - self._buffer.count

        if self._read_idx < oldest_valid:
            self._gap_detected = True
            self._read_idx = oldest_valid

        items = [self._buffer[i] for i in range(self._read_idx, self._buffer.write_idx)]
        self._read_idx = self._buffer.write_idx
        return items

    def has_new(self) -> bool:
        """Return True if there are items in the buffer that have not been drained.

        Returns:
            True if ``_read_idx < buffer.write_idx``.
        """
        return self._read_idx < self._buffer.write_idx

    @property
    def read_idx(self) -> int:
        """Current read cursor position (absolute sequence index)."""
        return self._read_idx

    @property
    def gap_detected(self) -> bool:
        """True if drain has ever snapped the cursor forward due to overflow."""
        return self._gap_detected
