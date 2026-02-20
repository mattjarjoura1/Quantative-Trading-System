# Statistical Arbitrage Project

A modular trading strategy laboratory for pairs trading and statistical arbitrage research.

## What This Is

A personal research repository for building, testing, and iterating on quantitative trading strategies. This is my first ground-up quantitative finance project, so I have attempted to compose it in a manner that is modular and interchangeable — beginning with the simplest approaches, which can be continually improved through experience and research.

## Project Structure

```
├── config/              # Strategy parameters (YAML)
├── docs/
│   ├── math/            # LaTeX documentation of mathematical models
│   └── design/          # Architecture and design decision docs
├── data/
│   ├── raw/             # Immutable original data
│   └── processed/       # Derived datasets
├── notebooks/           # Exploratory analysis and demos
├── src/
│   ├── types.py         # Shared dataclasses (Tick, Signal)
│   ├── data/            # Data sources (Yahoo, CSV replay, Binance)
│   ├── strategy/        # Trading strategies (prices in, signals out)
│   ├── execution/       # Order management and fill simulation
│   └── backtester/      # Backtest engine and performance metrics
├── tests/               # pytest test suite
├── scripts/             # Entry point scripts
├── Personal/            # Sandbox notebooks (gitignored)
└── readings/            # Reference papers (gitignored)
```


## Key Mathematical Models

- **Kalman Filter** — Dynamic hedge ratio estimation via recursive state-space filtering
- **Ornstein-Uhlenbeck Process** — Mean-reversion parameter estimation (θ, μ, σ)
- **Bertram Optimal Thresholds** — Analytically derived entry/exit levels maximising expected return per unit time

See `docs/math/` for full derivations. These will be continually improved — check back for updates.

## Setup

The requirements for running this project are listed in `requirements.txt`. A Makefile is provided for convenience:

```bash
make install   # Create venv and install dependencies
make run       # Activate the venv
make clean     # Remove the venv
```

## Standards

See [STANDARDS.md](STANDARDS.md) for coding conventions, naming rules, and documentation requirements.

## Future Development

### Live Execution: Threading the Pipeline

The current run loop is sequential — data, strategy, and execution operate on a single thread:

```python
for tick in data:
    signal = strategy.on_tick(tick)  # blocks until maths is done
    if signal:
        execution.execute(signal)
```

This is fine for backtesting on daily bars, but **will not work for live trading at high tick rates**. If the strategy computation (Kalman + OU refit + Bertram) takes longer than the interval between incoming ticks, the pipeline falls behind. The data the strategy operates on becomes stale, and the lag compounds over time.

The fix is to decouple ingestion from computation using threads and queues:

- **Data thread**: receives ticks from the websocket and writes to a shared buffer. Never blocked by strategy computation.
- **Strategy thread**: reads the latest available data from the buffer at its own cadence, runs the maths, and emits signals downstream.
- **Execution thread**: receives signals and acts independently.

Components can reference downstream (strategy knows about execution) but never upstream (execution never calls back into strategy, strategy never blocks data). This is a plumbing change in the run script — the ABCs, dataclasses, and concrete implementations remain unchanged.
