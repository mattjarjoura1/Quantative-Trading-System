"""Fixed-capacity circular buffer. Single writer, multiple readers."""

from typing import Generic, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """A fixed-capacity, pre-allocated circular buffer.

    Designed for single-producer, multiple-consumer access. The buffer has no
    opinion about readers — read cursors belong to BufferView. When the buffer
    is full, the oldest item is silently overwritten on the next append.

    Indexing uses the absolute write-sequence position, not the physical slot.
    `buffer[0]` always refers to the first item ever appended (if still in the
    buffer). The physical slot is `idx % capacity`. This keeps BufferView
    arithmetic simple: a view stores an absolute read position and compares it
    directly against `write_idx`.

    Args:
        capacity: Maximum number of items the buffer holds. Must be positive.

    Raises:
        ValueError: If capacity is not positive.
    """

    def __init__(self, capacity: int) -> None:
        """Initialise the buffer with a fixed capacity.

        Args:
            capacity: Number of slots to pre-allocate. Must be > 0.

        Raises:
            ValueError: If capacity <= 0.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._buffer: list = [None] * capacity
        self._write_idx: int = 0
        self._count: int = 0

    def append(self, item: T) -> None:
        """Write an item at the current write position and advance the cursor.

        If the buffer is full, the oldest item is overwritten. This is the
        only mutation method — no locks, single writer by design.

        Args:
            item: The item to store.
        """
        self._buffer[self._write_idx % self._capacity] = item
        self._write_idx += 1
        self._count = min(self._count + 1, self._capacity)

    @property
    def write_idx(self) -> int:
        """Current write position (total number of appends ever made).

        Consumers use this to determine what is new since their last read.
        """
        return self._write_idx

    @property
    def capacity(self) -> int:
        """Maximum number of items the buffer can hold."""
        return self._capacity

    @property
    def count(self) -> int:
        """Number of valid (readable) items currently in the buffer.

        Before the buffer fills, this equals the number of appends. Once full,
        it stays at capacity. Important for warmup: a buffer of capacity 1024
        with 10 items written reports count = 10, not 1024.
        """
        return self._count

    def __getitem__(self, idx: int) -> T:
        """Retrieve the item at absolute write-sequence position idx.

        Args:
            idx: Absolute index (not a physical slot). Valid range is
                [write_idx - count, write_idx - 1].

        Returns:
            The item stored at position idx.

        Raises:
            IndexError: If idx is out of the valid range (not yet written, or
                already overwritten).
        """
        if idx >= self._write_idx or idx < self._write_idx - self._count:
            raise IndexError(
                f"index {idx} out of range "
                f"[{self._write_idx - self._count}, {self._write_idx - 1}]"
            )
        return self._buffer[idx % self._capacity]
