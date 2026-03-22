"""Kalman-OU-Bertram pairs trading strategy."""


import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import erf
from collections import deque

from src.bus import BufferView, MessageBus
from src.strategy.base_strategy import BaseStrategy
from src.types import PriceTick, Signal


class KalmanOUBertramStrategy(BaseStrategy[PriceTick]):
    """Pairs trading strategy using a Kalman filter to track the hedge ratio.

    Receives price ticks for two cointegrated assets, estimates the hedge ratio
    dynamically via a Kalman filter, fits an Ornstein-Uhlenbeck process to the
    resulting spread, and derives analytically optimal entry/exit thresholds via
    Bertram's framework.

    Uses ``latest()`` on BufferViews — only the most recent price per symbol
    matters. Processing is gated on both symbols having fresh data.

    Args:
        bus: The shared message bus.
        listener_id: Unique identifier for this listener's dirty-set slot.
        listen_channel: Channel name to listen on for market data.
        publish_channel: Channel name to publish signals to.
        symbol_y: Dependent asset (Y in Y = βX).
        symbol_x: Independent asset (X).
        delta: Kalman process noise (controls how quickly β can change).
        r: Kalman observation noise.
        warmup_ticks: Number of price pairs for OLS initialisation before
            the Kalman filter begins.
        spread_buffer_size: Maximum spread history length (ring).
        refit_interval: Ticks between OU/Bertram refits.
        min_half_life: Reject OU fit if estimated half-life is below this.
        max_half_life: Reject OU fit if estimated half-life is above this.
        cost: Round-trip transaction cost as a fraction of notional.
        threshold_buffer: Widen Bertram thresholds by this factor.
    """

    CONSUMES = PriceTick

    def __init__(
        self,
        bus: MessageBus,
        listener_id: str,
        listen_channel: str,
        publish_channel: str,
        symbol_y: str,
        symbol_x: str,
        delta: float = 1e-4,
        r: float = 1e-3,
        warmup_ticks: int = 100,
        spread_buffer_size: int = 3600,
        refit_interval: int = 500,
        min_half_life: float = 30.0,
        max_half_life: float = 1800.0,
        cost: float = 0.003,
        threshold_buffer: float = 1.15,
    ) -> None:
        """Initialise state, BufferViews, and all parameter fields.

        Args:
            bus: The shared message bus.
            listener_id: Unique identifier for this listener's dirty-set slot.
            listen_channel: Channel name to listen on for market data.
            publish_channel: Channel name to publish signals to.
            symbol_y: Dependent asset.
            symbol_x: Independent asset.
            delta: Kalman process noise.
            r: Kalman observation noise.
            warmup_ticks: OLS warmup length in paired price observations.
            spread_buffer_size: Spread ring buffer capacity.
            refit_interval: Ticks between OU/Bertram refits.
            min_half_life: Minimum acceptable OU half-life.
            max_half_life: Maximum acceptable OU half-life.
            cost: Round-trip transaction cost fraction.
            threshold_buffer: Bertram threshold widening factor.
        """
        super().__init__(bus, listener_id, listen_channel, publish_channel)

        self._symbol_y = symbol_y
        self._symbol_x = symbol_x
        self._delta = delta
        self._r = r
        self._warmup_ticks = warmup_ticks
        self._refit_interval = refit_interval
        self._min_half_life = min_half_life
        self._max_half_life = max_half_life
        self._cost = cost
        self._threshold_buffer = threshold_buffer

        # BufferViews — latest() is stateless so from_start is irrelevant here
        self._views: dict[str, BufferView[PriceTick]] = {
            symbol_y: BufferView(self._listen_ch.get_buffer(symbol_y)),
            symbol_x: BufferView(self._listen_ch.get_buffer(symbol_x)),
        }

        # Price tracking
        self._latest_price: dict[str, float | None] = {symbol_y: None, symbol_x: None}
        self._is_fresh: dict[str, bool] = {symbol_y: False, symbol_x: False}

        # Kalman state (overwritten by OLS at end of warmup)
        self._beta: float = 1.0
        self._P: float = 1.0
        self._warmup_prices_x: list[float] = []
        self._warmup_prices_y: list[float] = []
        self._is_warmed_up: bool = False

        # Spread buffer
        self._spread_buffer = deque(maxlen=spread_buffer_size)
        self._tick_count: int = 0

        # OU parameters (None until first successful fit)
        self._speed: float | None = None
        self._mean: float | None = None
        self._sigma: float | None = None
        self._half_life: float | None = None

        # Bertram thresholds (None until first successful computation)
        self._long_threshold: float | None = None
        self._short_threshold: float | None = None

        # State machine
        self._state: str = "FLAT"

    def on_data(self, dirty: set[str]) -> list[Signal]:
        """Process new price ticks and emit zero or more signals.

        Gates on both symbols having fresh data before running the
        Kalman-OU-Bertram pipeline. Returns an empty list during warmup
        and before the first successful OU/Bertram fit.

        Args:
            dirty: Symbols with new data since the last wake.

        Returns:
            List of Signals. Empty during warmup and before thresholds are set.
        """
        # 1. Update latest prices for dirty symbols
        for symbol in dirty:
            if symbol not in self._views:
                continue
            tick = self._views[symbol].latest()
            if tick is not None:
                self._latest_price[symbol] = tick.price
                self._is_fresh[symbol] = True

        # 2. Gate on both symbols being fresh
        if not (self._is_fresh[self._symbol_x] and self._is_fresh[self._symbol_y]):
            return []

        self._is_fresh[self._symbol_x] = False
        self._is_fresh[self._symbol_y] = False

        price_x = self._latest_price[self._symbol_x]
        price_y = self._latest_price[self._symbol_y]

        # 3. Warmup phase
        if not self._is_warmed_up:
            return self._handle_warmup(price_x, price_y)

        # 4. Kalman update
        spread = self._kalman_update(price_x, price_y)
        timestamp = max(
            self._views[self._symbol_y].latest().timestamp_ms,
            self._views[self._symbol_x].latest().timestamp_ms,
        )
        self._spread_buffer.append((timestamp, spread))
        self._tick_count += 1

        # 5. Periodic OU/Bertram refit
        if self._tick_count % self._refit_interval == 0:
            self._fit_ou_and_bertram()

        # 6. Signal generation
        return self._evaluate_signal(price_x, price_y)

    def _handle_warmup(self, price_x: float, price_y: float) -> list[Signal]:
        """Collect warmup prices and initialise the Kalman filter via OLS.

        Accumulates paired observations until ``warmup_ticks`` pairs are
        available, then runs OLS to estimate the initial β and P.

        Args:
            price_x: Current price of the independent asset.
            price_y: Current price of the dependent asset.

        Returns:
            Always an empty list — no signals during warmup.
        """
        self._warmup_prices_x.append(price_x)
        self._warmup_prices_y.append(price_y)

        if len(self._warmup_prices_x) < self._warmup_ticks:
            return []

        X = np.array(self._warmup_prices_x)
        Y = np.array(self._warmup_prices_y)
        beta, alpha = np.polyfit(X, Y, 1)

        residuals = Y - beta * X - alpha
        self._P = float(np.var(residuals) / np.var(X))
        self._beta = float(beta)
        self._is_warmed_up = True

        return []

    def _kalman_update(self, price_x: float, price_y: float) -> float:
        """Update the Kalman filter estimate of β with a new price pair.

        Scalar (1D) Kalman update tracking only the hedge ratio β, with no
        intercept term. The error (spread before the update) is returned and
        appended to the spread buffer by the caller.

        Args:
            price_x: Current price of the independent asset.
            price_y: Current price of the dependent asset.

        Returns:
            The prediction error (spread) computed with the pre-update β.
        """
        P_pred = self._P + self._delta
        error = price_y - self._beta * price_x
        S = P_pred * price_x ** 2 + self._r
        K = (P_pred * price_x) / S
        self._beta = self._beta + K * error
        self._P = P_pred - K * price_x * P_pred
        return error

    def _fit_ou_and_bertram(self) -> None:
        """Refit OU parameters then Bertram thresholds if the OU fit succeeds.

        Called every ``refit_interval`` ticks from ``on_data``. A failed OU
        fit leaves all OU and Bertram state unchanged.
        """
        if self._fit_ou():
            self._compute_bertram()

    def _fit_ou(self) -> bool:
        """Fit OU parameters to the spread buffer via AR(1) regression.

        Uses a dynamic window of ``5 × previous half-life`` entries if a prior
        half-life estimate exists, otherwise uses the full buffer. Applies four
        validation gates: mean-reversion coefficient, half-life bounds, and
        positive volatility.

        Returns:
            True if the fit is valid and OU parameters have been updated.
            False if any validation gate fails; existing parameters are unchanged.
        """
        if self._half_life is not None:
            window = min(int(5 * self._half_life), len(self._spread_buffer))
        else:
            window = len(self._spread_buffer)

        if window < 30:
            return False

        timestamps = np.array([t for t, _ in list(self._spread_buffer)[-window:]])
        spreads = np.array([s for _, s in list(self._spread_buffer)[-window:]])
            
        x = spreads[:-1]
        y = spreads[1:]

        a, b = np.polyfit(x, y, 1)

        if a >= 1.0 or a <= 0.0:
            return False

        dt = np.mean(np.diff(timestamps)) / 1000.0 # convert ms to seconds
        print(dt)
        
        speed = -np.log(a) / dt
        mean = b / (1 - a)
        residuals = y - (a * x + b)
        residual_std = float(np.std(residuals))
        sigma = residual_std * np.sqrt(2 * speed / (1 - a ** 2))
        half_life = np.log(2) / speed
        
        print(f"OU fit results: speed={speed:.4f}, mean={mean:.4f}, sigma={sigma:.4f}, half-life={half_life:.2f} seconds")

        if half_life < self._min_half_life * dt or half_life > self._max_half_life * dt:
            print(f"OU fit rejected: half-life {half_life:.2f} out of bounds")
            print(f"Lower bound: {self._min_half_life * dt:.2f} seconds, Upper bound: {self._max_half_life * dt:.2f} seconds")
            return False
        if sigma <= 0:
            print(f"OU fit rejected: non-positive sigma {sigma:.4f}")
            return False

        self._speed = float(speed)
        self._mean = float(mean)
        self._sigma = float(sigma)
        self._half_life = float(half_life)
        print("Returning from OU fit with parameters:")
        return True

    def _compute_bertram(self) -> bool:
        """Compute optimal Bertram entry/exit thresholds from the current OU fit.

        Transforms the problem to dimensionless units, maximises the per-unit-time
        return function G(d) via bounded scalar optimisation, then maps the optimal
        threshold back to real spread units. Sets state to INACTIVE if the strategy
        is not profitable at any threshold.

        Returns:
            True if a profitable threshold exists and thresholds have been set.
            False if G(d_optimal) <= 0; state is set to INACTIVE.
        """
        sd = self._sigma / np.sqrt(2 * self._speed)
        c_dimless = self._cost / sd

        def expected_time(d: float) -> float:
            result, _ = quad(
                lambda y: (np.sqrt(np.pi) / 2) * np.exp(y ** 2) * (1 + erf(y)),
                -d, d,
            )
            return result

        def G(d: float) -> float:
            et = expected_time(d)
            if et < 1e-12:
                return 0.0
            return (2 * d - c_dimless) / (2 * et)

        # Feasibility gate: lower bound must be below the fixed upper bound of 4
        if c_dimless / 2 >= 4:
            self._state = "INACTIVE"
            print("Breakpoint 1")
            return False

        res = minimize_scalar(lambda d: -G(d), bounds=(c_dimless / 2, 4), method="bounded")
        d_optimal = res.x

        if G(d_optimal) <= 0:
            self._state = "INACTIVE"
            print("Breakpoint 2")
            return False

        d_practical = d_optimal * self._threshold_buffer
        self._long_threshold = self._mean - d_practical * sd
        self._short_threshold = self._mean + d_practical * sd
        
        print(f"Current Price of {self._symbol_x}: {self._latest_price[self._symbol_x]:.2f}, ")
        print(f"Current Price of {self._symbol_y}: {self._latest_price[self._symbol_y]:.2f}")
        print(f"Thresholds set: LONG below {self._long_threshold:.4f}, SHORT above {self._short_threshold:.4f}")
        
        return True

    def _evaluate_signal(self, price_x: float, price_y: float) -> list[Signal]:
        """Evaluate the state machine against current spread and emit signals.

        No signals are emitted while thresholds are unset or state is INACTIVE.
        Transitions happen at most once per call; if state does not change, an
        empty list is returned.

        Args:
            price_x: Current price of the independent asset.
            price_y: Current price of the dependent asset.

        Returns:
            Two-element list of Signals on a state transition, empty list otherwise.
        """
        if self._long_threshold is None or self._short_threshold is None:
            return []
        if self._state == "INACTIVE":
            return []

        spread = price_y - self._beta * price_x
        new_state = self._state

        if self._state == "FLAT":
            if spread < self._long_threshold:
                new_state = "LONG"
            elif spread > self._short_threshold:
                new_state = "SHORT"
        elif self._state == "LONG":
            if spread >= self._mean:
                new_state = "FLAT"
        elif self._state == "SHORT":
            if spread <= self._mean:
                new_state = "FLAT"

        if new_state == self._state:
            return []

        self._state = new_state
        return self._build_signals(price_x, price_y)

    def _build_signals(self, price_x: float, price_y: float) -> list[Signal]:
        """Construct the two-legged position signals for the current state.

        β-weighted notional allocation: Y leg gets capital/(1+|β|), X leg gets
        |β|·capital/(1+|β|). When LONG the spread, buy Y and sell X. When SHORT
        the spread, sell Y and buy X. When FLAT, both legs go to zero.

        Args:
            price_x: Current price of the independent asset.
            price_y: Current price of the dependent asset.

        Returns:
            List of two Signals, one per leg.
        """
        timestamp = max(
            self._views[self._symbol_y].latest().timestamp_ms,
            self._views[self._symbol_x].latest().timestamp_ms,
        )
        
        if self._state == "LONG":
            target_y = 1.0
            target_x = -abs(self._beta)
        elif self._state == "SHORT":
            target_y = -1.0
            target_x = abs(self._beta)
        else:
            target_y = 0.0
            target_x = 0.0
            
        return [
            Signal(timestamp_ms=timestamp, symbol=self._symbol_y,
                   target_position=target_y, price=price_y),
            Signal(timestamp_ms=timestamp, symbol=self._symbol_x,
                   target_position=target_x, price=price_x),
        ]
        
