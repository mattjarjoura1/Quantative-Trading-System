"""Abstract base class for all orchestrators."""

import threading

from src.bus.message_bus import MessageBus
from src.registry import SOURCES, STRATEGIES, RISK_ENGINES, EXECUTORS


class BaseOrchestrator:
    """Wires pipeline components and manages their lifecycle.

    Owns bus creation, channel setup, class resolution, type validation,
    thread management, and shutdown. Does not define run() — subclasses
    differ in how the pipeline is started and terminated.

    Channel topology is fixed here:
      source → market_data → strategy → strategy_signals → risk → approved_signals → execution

    Args:
        config: Configuration dict with keys: bus, source, strategy, risk, execution.
            bus:       market_data_capacity, signal_capacity, approved_capacity
            source:    type (registry key), params (constructor kwargs)
            strategy:  type, params
            risk:      type, params
            execution: type, params

    Raises:
        KeyError: If any type string is not found in its registry.
        TypeError: If source.PRODUCES != strategy.CONSUMES.
    """

    def __init__(self, config: dict) -> None:
        """Build bus, resolve classes, validate types, instantiate components.

        Args:
            config: Configuration dict as described in class docstring.

        Raises:
            KeyError: If any type string is not found in its registry.
            TypeError: If source.PRODUCES != strategy.CONSUMES.
        """
        self._bus = MessageBus()
        self._bus.create_channel("market_data", capacity=config["bus"]["market_data_capacity"])
        self._bus.create_channel("strategy_signals", capacity=config["bus"]["signal_capacity"])
        self._bus.create_channel("approved_signals", capacity=config["bus"]["approved_capacity"])

        source_cls = SOURCES[config["source"]["type"]]
        strategy_cls = STRATEGIES[config["strategy"]["type"]]
        risk_cls = RISK_ENGINES[config["risk"]["type"]]
        execution_cls = EXECUTORS[config["execution"]["type"]]

        if source_cls.PRODUCES != strategy_cls.CONSUMES:
            raise TypeError(
                f"{source_cls.__name__} produces {source_cls.PRODUCES.__name__} "
                f"but {strategy_cls.__name__} consumes {strategy_cls.CONSUMES.__name__}"
            )

        self._source = source_cls(
            self._bus,
            publish_channel="market_data",
            **config["source"]["params"],
        )
        self._strategy = strategy_cls(
            self._bus,
            listen_channel="market_data",
            publish_channel="strategy_signals",
            **config["strategy"]["params"],
        )
        self._risk = risk_cls(
            self._bus,
            listen_channel="strategy_signals",
            publish_channel="approved_signals",
            **config["risk"]["params"],
        )
        self._execution = execution_cls(
            self._bus,
            listen_channel="approved_signals",
            market_channel="market_data",
            **config["execution"]["params"],
        )

    def _start_consumers(self) -> list[threading.Thread]:
        """Start strategy, risk, and execution each on their own thread.

        Returns:
            List of started daemon-less threads. Pass to _shutdown.
        """
        threads = [
            threading.Thread(target=self._strategy.run),
            threading.Thread(target=self._risk.run),
            threading.Thread(target=self._execution.run),
        ]
        for t in threads:
            t.start()
        return threads

    def _shutdown(self, threads: list[threading.Thread]) -> None:
        """Stop all components front-to-back and join threads.

        Stops in pipeline order: source first, execution last. Each stop()
        sets _running = False and unblocks the component's sleeping event.wait().

        Args:
            threads: The threads returned by _start_consumers.
        """
        self._source.stop()
        self._strategy.stop()
        self._risk.stop()
        self._execution.stop()
        for t in threads:
            t.join()
