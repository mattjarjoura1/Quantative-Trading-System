# Quantitative Trading System

A modular pipeline for quantitative trading research and execution, built in Python. The focus of this project is the infrastructure: a type-safe, component-based system where new data sources, strategies, and execution modes integrate without modifying the core. Components communicate through typed dataclasses over a shared message bus, and the pipeline is generic over the market data type — compatibility between sources and strategies is enforced at startup.

This is my first ground-up quantitative trading project. The priority has been building a system that is well-structured, encapsulated, and extensible before layering on strategy complexity.

## Architecture

```mermaid
stateDiagram-v2
    direction LR

    [*] --> DataSource

    DataSource: BaseDataSource[T]
    DataSource: Awaits external data
    DataSource: Parses into typed dataclass
    DataSource: Publishes to market_data channel

    Strategy: BaseStrategy[T]
    Strategy: Wakes on new data via Event
    Strategy: Reads through BufferView
    Strategy: Computes signal
    Strategy: Publishes to strategy_signals channel

    RiskEngine: BaseRiskEngine
    RiskEngine: Wakes on new signal
    RiskEngine: Evaluates risk constraints
    RiskEngine: Publishes to approved_signals channel

    Execution: BaseExecution
    Execution: Wakes on approved signal
    Execution: Reads fill price from market_data
    Execution: Records trade

    DataSource --> Strategy
    Strategy --> RiskEngine
    RiskEngine --> Execution
    Execution --> [*]
```

Each stage runs on its own thread. The message bus sits between every stage — components never reference each other directly. An orchestrator reads a YAML config, resolves component classes from a registry, validates type compatibility, wires everything to the bus, and manages the lifecycle.

For detailed documentation of each subsystem (bus internals, data source hierarchy, component run loops, orchestrator wiring, data recording/replay), see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project Structure

```
├── config/                         # YAML run configurations
├── docs/
│   ├── math/                       # LaTeX derivations (Kalman, OU, Bertram)
│   └── design/                     # Architecture and design decision docs
├── data/
│   ├── raw/                        # Immutable original data and JSONL recordings
│   └── processed/                  # Derived datasets
├── notebooks/                      # Exploratory analysis and demos
├── src/
│   ├── types.py                    # Shared dataclasses (OrderBookEntry, PriceTick, Signal, TradeRecord)
│   ├── registry.py                 # Class registries for orchestrator resolution
│   ├── bus/                        # RingBuffer, Channel, MessageBus, BufferView
│   ├── data/                       # Data source ABCs and implementations
│   ├── strategy/                   # Strategy ABC and implementations
│   ├── risk/                       # Risk engine ABC and implementations
│   ├── execution/                  # Execution ABC and implementations
│   └── orchestrator/               # Pipeline wiring and lifecycle management
├── tests/                          # pytest test suite
├── scripts/                        # Entry point scripts
└── Personal/                       # Sandbox notebooks (gitignored)
```

## Usage

```python
import yaml
from src.orchestrator import BacktestOrchestrator

config = yaml.safe_load(open("config/backtest_random.yaml"))
orchestrator = BacktestOrchestrator(config)
trade_log = orchestrator.run()
```

## Setup

```bash
make install   # Create venv and install dependencies
make run       # Activate the venv
make clean     # Remove the venv
```

## Future Development

The infrastructure is in place. What comes next is the strategy layer:

- **Kalman Filter** — Dynamic hedge ratio estimation via recursive state-space filtering
- **Ornstein-Uhlenbeck Process** — Mean-reversion parameter estimation (θ, μ, σ)
- **Bertram Optimal Thresholds** — Analytically derived entry/exit levels maximising expected return per unit time
- **RSI** — Relative Strength Index on VWMP with incremental computation

## Standards

See [STANDARDS.md](STANDARDS.md) for coding conventions, naming rules, and documentation requirements.