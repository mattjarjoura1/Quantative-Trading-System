"""Orchestrator for live trading runs."""

import asyncio

from src.orchestrator.base_orchestrator import BaseOrchestrator


class LiveOrchestrator(BaseOrchestrator):
    """Runs the pipeline indefinitely until the data source is externally stopped.

    Minimal subclass — adds only run(). No market data recording, no
    post-processing. Live-specific concerns (dashboard streaming, database
    logging, graceful restart) are future additions.

    Args:
        config: Configuration dict (same format as BaseOrchestrator).
    """

    def __init__(self, config: dict) -> None:
        """Build pipeline via BaseOrchestrator.

        Args:
            config: Configuration dict as described in BaseOrchestrator.
        """
        super().__init__(config)

    def run(self) -> None:
        """Run the pipeline until the data source stops.

        Starts consumers on threads, blocks on the async data source, then
        shuts down all components and joins threads.
        """
        threads = self._start_consumers()
        asyncio.run(self._source.run())
        self._shutdown(threads)
