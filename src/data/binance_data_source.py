"""Binance WebSocket order book data source."""

import time

import orjson
import websockets

from src.bus import MessageBus
from src.data import BaseDataSource
from src.types import OrderBookEntry

VALID_LEVELS = [5, 10, 20]
VALID_UPDATE_SPEEDS = [100, 1000]


class BinanceDataSource(BaseDataSource[OrderBookEntry]):
    PRODUCES = OrderBookEntry
    """Streams order book snapshots from the Binance combined WebSocket feed.

    Connects to the Binance partial depth stream, subscribes to the requested
    symbols, and publishes an OrderBookEntry to the bus on each message.

    Args:
        bus: The shared message bus.
        publish_channel: Name of the channel to publish market data to.
        symbols: List of Binance symbol strings (e.g. ["btcusdt", "ethusdt"]).
        levels: Depth levels per side. Must be one of 5, 10, or 20.
        update_speed: Stream update interval in ms. Must be 100 or 1000.
    """

    def __init__(
        self,
        bus: MessageBus,
        publish_channel: str = "market_data",
        symbols: list[str] = ["btcusdt"],
        levels: int = 5,
        update_speed: int = 1000,
    ) -> None:
        """Validate parameters and build the subscription stream list.

        Args:
            bus: The shared message bus.
            publish_channel: Name of the channel to publish market data to.
            symbols: Binance symbol strings to subscribe to.
            levels: Depth levels per side. Must be one of 5, 10, or 20.
            update_speed: Stream update interval in ms. Must be 100 or 1000.

        Raises:
            ValueError: If levels or update_speed are not valid Binance values.
        """
        super().__init__(bus=bus, publish_channel=publish_channel)

        if levels not in VALID_LEVELS:
            raise ValueError(f"Invalid levels: {levels}. Must be one of {VALID_LEVELS}.")

        if update_speed not in VALID_UPDATE_SPEEDS:
            raise ValueError(
                f"Invalid update_speed: {update_speed}. Must be one of {VALID_UPDATE_SPEEDS}."
            )

        self._levels = levels
        self._update_speed = update_speed
        self._symbols = symbols
        self._base_url = "wss://stream.binance.com:9443/stream"
        self._params = [
            f"{symbol}@depth{self._levels}@{self._update_speed}ms"
            for symbol in self._symbols
        ]

    async def run(self) -> None:
        """Open the WebSocket connection, subscribe, run the fetch loop, then unsubscribe."""
        async with websockets.connect(self._base_url) as websocket:
            self._ws = websocket
            await self._subscribe()
            await super().run()
            await self._unsubscribe()

    async def _subscribe(self) -> None:
        """Send a SUBSCRIBE request to the Binance stream."""
        await self._ws.send(orjson.dumps({
            "method": "SUBSCRIBE",
            "params": self._params,
            "id": 1,
        }).decode())

    async def _unsubscribe(self) -> None:
        """Send an UNSUBSCRIBE request to the Binance stream."""
        await self._ws.send(orjson.dumps({
            "method": "UNSUBSCRIBE",
            "params": self._params,
            "id": 2,
        }).decode())

    async def fetch(self) -> OrderBookEntry | None:
        """Receive one message from the WebSocket and return a parsed entry.

        Returns:
            An OrderBookEntry if the message is a depth snapshot, or None for
            subscription acknowledgements and other non-data messages.
        """
        try:
            data = orjson.loads(await self._ws.recv())
            if "stream" not in data:
                # Subscription ack or unknown envelope — skip
                return None
            return self._parse(data)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")
            return None

    def _parse(self, envelope: dict) -> OrderBookEntry:
        """Parse a Binance combined stream envelope into an OrderBookEntry.

        Args:
            envelope: Parsed JSON dict in Binance combined stream format:
                {"stream": "<symbol>@depth<n>", "data": {"bids": [...], "asks": [...]}}.

        Returns:
            OrderBookEntry with bids and asks as nested tuples of (price, quantity).
        """
        data = envelope["data"]
        return OrderBookEntry(
            symbol=envelope["stream"].split("@")[0],
            timestamp_ms=int(time.time() * 1000),
            bids=tuple(tuple((float(p), float(q))) for p, q in data["bids"]),
            asks=tuple(tuple((float(p), float(q))) for p, q in data["asks"]),
        )
