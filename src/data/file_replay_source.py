"""Replays a JSONL file through the pipeline as typed dataclass instances."""

import json
from typing import Generic, TypeVar

from src.bus.message_bus import MessageBus
from src.data.backtest_data_source import BacktestDataSource

T = TypeVar("T")


class FileReplaySource(BacktestDataSource[T], Generic[T]):
    """Replays a JSONL file through the pipeline as typed dataclass instances.

    Each line is parsed via data_cls.from_dict(json.loads(line)). Generic
    over T and sets PRODUCES dynamically at construction time — the only
    source that needs this because it can replay any data type.

    Loads all lines upfront for deterministic replay. Pacing is handled
    by BacktestDataSource.run().

    Args:
        bus: The shared message bus.
        publish_channel: Channel name to publish to.
        filepath: Path to the JSONL file to replay.
        data_cls: The dataclass type to deserialise into (e.g. PriceTick,
            OrderBookEntry). Must implement from_dict(d: dict).
    """

    def __init__(
        self,
        bus: MessageBus,
        publish_channel: str,
        filepath: str,
        data_cls: type[T],
    ) -> None:
        """Load all lines upfront and set PRODUCES to data_cls.

        Args:
            bus: The shared message bus.
            publish_channel: Channel name to publish to.
            filepath: Path to the JSONL file to replay.
            data_cls: The dataclass type to deserialise each line into.
        """
        super().__init__(bus, publish_channel)
        self.PRODUCES = data_cls
        self._data_cls = data_cls
        self._lines = open(filepath).readlines()
        self._index = 0

    async def fetch(self) -> T | None:
        """Return the next deserialised item, or stop if the file is exhausted.

        Returns:
            Next data item, or None after calling stop() at end of file.
        """
        if self._index >= len(self._lines):
            self.stop()
            return None
        line = self._lines[self._index]
        self._index += 1
        return self._data_cls.from_dict(json.loads(line), backtest=True)
