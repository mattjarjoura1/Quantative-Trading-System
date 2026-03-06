"""Yahoo Finance historical data source for backtesting."""

from datetime import datetime

import pandas as pd
import yfinance as yf

from src.bus.message_bus import MessageBus
from src.data.backtest_data_source import BacktestDataSource
from src.types import PriceTick

VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}


class YahooDataSource(BacktestDataSource[PriceTick]):
    """Streams historical close prices from Yahoo Finance as PriceTick objects.

    Downloads all data upfront at construction time — no network access
    during replay. Ticks are emitted chronologically via fetch(), one per
    call. Pacing is handled by BacktestDataSource.run().

    Args:
        bus: The shared message bus.
        publish_channel: Channel name to publish market data to.
        symbols: Yahoo Finance symbol strings (e.g. ["BTC-USD", "GOOG"]).
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        interval: Bar size. Must be one of VALID_INTERVALS. Defaults to "1d".

    Raises:
        ValueError: If symbols is empty, dates are invalid, or interval is
            unsupported.
    """

    PRODUCES = PriceTick

    def __init__(
        self,
        bus: MessageBus,
        publish_channel: str = "market_data",
        symbols: list[str] = ["BTC-USD"],
        start: str = "2020-01-01",
        end: str = "2021-01-01",
        interval: str = "1d",
    ) -> None:
        """Validate parameters and download data upfront.

        Args:
            bus: The shared message bus.
            publish_channel: Channel name to publish market data to.
            symbols: Yahoo Finance symbol strings to download.
            start: Start date in "YYYY-MM-DD" format.
            end: End date in "YYYY-MM-DD" format.
            interval: Bar size. Must be one of VALID_INTERVALS.

        Raises:
            ValueError: If symbols is empty, dates are invalid, or interval
                is unsupported.
        """
        super().__init__(bus, publish_channel)

        if not symbols:
            raise ValueError("symbols must be a non-empty list.")

        start_dt = _parse_date(start)
        end_dt = _parse_date(end)
        if end_dt <= start_dt:
            raise ValueError(f"end ({end}) must be after start ({start}).")

        if interval not in VALID_INTERVALS:
            raise ValueError(
                f"interval '{interval}' is not supported. "
                f"Must be one of: {sorted(VALID_INTERVALS)}"
            )

        self._ticks = self._download(symbols, start, end, interval)
        self._index = 0

    async def fetch(self) -> PriceTick | None:
        """Return the next tick in chronological order.

        Calls stop() when all ticks have been emitted. Pacing is handled
        by BacktestDataSource.run() — this method has no asyncio concerns.

        Returns:
            The next PriceTick, or None once the dataset is exhausted.
        """
        if self._index >= len(self._ticks):
            self.stop()
            return None
        tick = self._ticks[self._index]
        self._index += 1
        return tick

    def _download(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str,
    ) -> list[PriceTick]:
        """Download close prices and flatten to a chronologically sorted list.

        Args:
            symbols: Yahoo Finance symbol strings.
            start: Start date string.
            end: End date string.
            interval: Bar size string.

        Returns:
            List of PriceTick objects sorted by timestamp_ms ascending.
            Rows with NaN close prices are silently skipped.
        """
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )["Close"]

        # yfinance returns a Series for a single ticker — normalise to DataFrame
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=symbols[0])

        ticks = []
        for ts, row in raw.iterrows():
            ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
            for symbol in symbols:
                price = float(row[symbol])
                if pd.isna(price):
                    continue
                ticks.append(PriceTick(symbol=symbol, timestamp_ms=ts_ms, price=price))

        ticks.sort(key=lambda t: t.timestamp_ms)
        return ticks


def _parse_date(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD string into a datetime.

    Args:
        date_str: Date string to parse.

    Returns:
        Parsed datetime object.

    Raises:
        ValueError: If the string is not valid YYYY-MM-DD format.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
