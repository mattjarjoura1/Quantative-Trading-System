"""Class registries for orchestrator component resolution.

Adding a new component: import it and add one entry to the appropriate dict.
No orchestrator changes required.
"""

from src.data.binance_data_source import BinanceDataSource
from src.data.file_replay_source import FileReplaySource
from src.data.yahoo_data_source import YahooDataSource
from src.strategy.random_strategy_obe import RandomStrategyOBE
from src.strategy.random_strategy_pt import RandomStrategyPT
from src.risk.passthrough_risk import PassthroughRisk
from src.execution.simulation_execution import SimulationExecution
from src.types import OrderBookEntry, PriceTick

SOURCES: dict[str, type] = {
    "binance": BinanceDataSource,
    "file_replay": FileReplaySource,
    "yahoo": YahooDataSource,
}

DATA_TYPES: dict[str, type] = {
    "OrderBookEntry": OrderBookEntry,
    "PriceTick": PriceTick,
}

STRATEGIES: dict[str, type] = {
    "random_obe": RandomStrategyOBE,
    "random_pt": RandomStrategyPT,
}

RISK_ENGINES: dict[str, type] = {
    "passthrough": PassthroughRisk,
}

EXECUTORS: dict[str, type] = {
    "simulation": SimulationExecution,
}
