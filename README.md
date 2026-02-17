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
│   ├── types.py         # Shared dataclasses (MarketTick, SpreadState, etc.)
│   ├── data/            # Data sources (Yahoo, CSV replay, Binance)
│   ├── hedgers/         # Beta estimation (static OLS, Kalman)
│   ├── models/          # Spread modelling (OU process, half-life)
│   ├── decisions/       # Signal generation (z-score rules, Bertram)
│   ├── backtester/      # Backtest engine and performance metrics
│   └── execution/       # Fee and slippage models
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
