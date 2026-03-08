"""Orchestrator for recording live market data to JSONL files."""

import asyncio
import threading

from src.bus.message_bus import MessageBus
from src.data.market_data_dumper import MarketDataDumper
from src.registry import SOURCES


class RecordingOrchestrator:
    """Wires a live data source to a MarketDataDumper for data collection.

    Simplified orchestrator — no strategy, risk, or execution components.
    Creates a bus with a single market_data channel, runs the source async,
    and runs the dumper on a thread.

    Args:
        config: Configuration dict with keys: bus, source, recording.
            bus:       market_data_capacity
            source:    type (registry key), params (constructor kwargs)
            recording: listener_id, filepath, symbols

    Raises:
        KeyError: If the source type is not found in the SOURCES registry.
    """

    def __init__(self, config: dict) -> None:
        """Build bus, resolve source class, instantiate source and dumper.

        Args:
            config: Configuration dict as described in class docstring.

        Raises:
            KeyError: If the source type is not found in the SOURCES registry.
        """
        self._bus = MessageBus()
        self._bus.create_channel(
            "market_data", capacity=config["bus"]["market_data_capacity"]
        )

        source_cls = SOURCES[config["source"]["type"]]
        self._source = source_cls(
            self._bus,
            publish_channel="market_data",
            **config["source"]["params"],
        )

        rec = config["recording"]
        self._dumper = MarketDataDumper(
            bus=self._bus,
            listener_id=rec["listener_id"],
            listen_channel="market_data",
            filepath=rec["filepath"],
            symbols=rec["symbols"],
        )

    def run(self) -> None:
        """Run the source and dumper until the source stops.

        Starts the dumper on a thread, blocks on the async data source,
        then shuts down in order: source → dumper → join thread.
        """
        t = threading.Thread(target=self._dumper.run)
        t.start()
        try:
            asyncio.run(self._source.run())
        finally:
            self._source.stop()
            self._dumper.stop()
            t.join()
