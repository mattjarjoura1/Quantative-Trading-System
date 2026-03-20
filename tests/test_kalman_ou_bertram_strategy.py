"""Tests for KalmanOUBertramStrategy — Steps 1 & 2: warmup, Kalman update, OU fit."""

import numpy as np
import pytest

from src.bus.message_bus import MessageBus
from src.strategy.kalman_ou_bertram_strategy import KalmanOUBertramStrategy
from src.types import PriceTick, Signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOL_Y = "ETH-USD"
SYMBOL_X = "BTC-USD"
LISTEN_CH = "market_data"
PUBLISH_CH = "strategy_signals"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bus() -> MessageBus:
    """Return a MessageBus with the two channels used by the strategy."""
    bus = MessageBus()
    bus.create_channel(LISTEN_CH, capacity=256)
    bus.create_channel(PUBLISH_CH, capacity=256)
    return bus


def _make_tick(symbol: str, price: float, ts: int = 1_000_000) -> PriceTick:
    """Return a PriceTick for the given symbol."""
    return PriceTick(symbol=symbol, timestamp_ms=ts, price=price)


def _make_strategy(bus: MessageBus, **overrides) -> KalmanOUBertramStrategy:
    """Construct a KalmanOUBertramStrategy with sensible test defaults."""
    defaults = dict(
        listener_id="kalman",
        symbol_y=SYMBOL_Y,
        symbol_x=SYMBOL_X,
        delta=1e-4,
        r=1e-3,
        warmup_ticks=10,
        spread_buffer_size=100,
        refit_interval=50,
        min_half_life=1.0,
        max_half_life=100.0,
        cost=0.003,
        threshold_buffer=1.15,
        capital=10000.0,
    )
    defaults.update(overrides)
    return KalmanOUBertramStrategy(
        bus=bus,
        listen_channel=LISTEN_CH,
        publish_channel=PUBLISH_CH,
        **defaults,
    )


def _feed_pair(
    strategy: KalmanOUBertramStrategy,
    bus: MessageBus,
    price_x: float,
    price_y: float,
    ts: int = 1_000_000,
) -> list[Signal]:
    """Publish one tick per symbol and call on_data with both symbols dirty."""
    ch = bus.channel(LISTEN_CH)
    ch.publish(SYMBOL_X, _make_tick(SYMBOL_X, price_x, ts=ts))
    ch.publish(SYMBOL_Y, _make_tick(SYMBOL_Y, price_y, ts=ts))
    return strategy.on_data({SYMBOL_X, SYMBOL_Y})


def _do_warmup(
    strategy: KalmanOUBertramStrategy,
    bus: MessageBus,
    n: int,
    base_x: float = 10.0,
    true_beta: float = 2.0,
) -> None:
    """Feed n pairs along Y = true_beta * X with slight X variation for OLS."""
    for i in range(n):
        x = base_x + i * 0.1
        y = true_beta * x
        _feed_pair(strategy, bus, price_x=x, price_y=y, ts=i + 1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_consumes_is_price_tick(self):
        assert KalmanOUBertramStrategy.CONSUMES is PriceTick

    def test_initial_not_warmed_up(self):
        strat = _make_strategy(_make_bus())
        assert not strat._is_warmed_up

    def test_initial_state_flat(self):
        strat = _make_strategy(_make_bus())
        assert strat._state == "FLAT"

    def test_thresholds_none_on_construction(self):
        strat = _make_strategy(_make_bus())
        assert strat._long_threshold is None
        assert strat._short_threshold is None

    def test_ou_params_none_on_construction(self):
        strat = _make_strategy(_make_bus())
        assert strat._speed is None
        assert strat._mean is None
        assert strat._sigma is None
        assert strat._half_life is None

    def test_spread_buffer_empty_on_construction(self):
        strat = _make_strategy(_make_bus())
        assert len(strat._spread_buffer) == 0

    def test_tick_count_zero_on_construction(self):
        strat = _make_strategy(_make_bus())
        assert strat._tick_count == 0


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_no_signals_during_warmup(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5)
        for i in range(5):
            signals = _feed_pair(strat, bus, price_x=10.0 + i * 0.1, price_y=20.0 + i * 0.2, ts=i + 1)
            assert signals == [], f"expected no signal at warmup tick {i}"

    def test_not_warmed_up_before_last_warmup_tick(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5)
        for i in range(4):
            _feed_pair(strat, bus, price_x=10.0 + i * 0.1, price_y=20.0 + i * 0.2, ts=i + 1)
        assert not strat._is_warmed_up

    def test_warmed_up_after_warmup_ticks(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5)
        for i in range(5):
            _feed_pair(strat, bus, price_x=10.0 + i * 0.1, price_y=20.0 + i * 0.2, ts=i + 1)
        assert strat._is_warmed_up

    def test_beta_finite_after_warmup(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=10)
        _do_warmup(strat, bus, 10)
        assert np.isfinite(strat._beta)

    def test_P_non_negative_after_warmup(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=10)
        _do_warmup(strat, bus, 10)
        assert strat._P >= 0

    def test_beta_close_to_true_value_after_ols(self):
        """OLS warmup on Y = 2X should yield beta near 2.0."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=20)
        rng = np.random.default_rng(42)
        for i in range(20):
            x = 100.0 + i
            y = 2.0 * x + rng.normal(0, 0.01)
            _feed_pair(strat, bus, price_x=x, price_y=y, ts=i + 1)
        assert abs(strat._beta - 2.0) < 0.05

    def test_freshness_gate_only_x_dirty(self):
        """Dirty set with only X → no processing, empty return."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=10)
        ch = bus.channel(LISTEN_CH)
        ch.publish(SYMBOL_X, _make_tick(SYMBOL_X, 100.0, ts=1))
        signals = strat.on_data({SYMBOL_X})
        assert signals == []
        assert not strat._is_fresh[SYMBOL_Y]

    def test_freshness_gate_only_y_dirty(self):
        """Dirty set with only Y → no processing, empty return."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=10)
        ch = bus.channel(LISTEN_CH)
        ch.publish(SYMBOL_Y, _make_tick(SYMBOL_Y, 200.0, ts=1))
        signals = strat.on_data({SYMBOL_Y})
        assert signals == []

    def test_unknown_symbol_in_dirty_is_ignored(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5)
        assert strat.on_data({"UNKNOWN"}) == []

    def test_empty_dirty_set_returns_empty(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5)
        assert strat.on_data(set()) == []

    def test_partial_warmup_does_not_fill_spread_buffer(self):
        """Warmup ticks must not contribute to the spread buffer."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5)
        for i in range(4):
            _feed_pair(strat, bus, price_x=10.0 + i * 0.1, price_y=20.0 + i * 0.2, ts=i + 1)
        assert len(strat._spread_buffer) == 0

    def test_warmup_does_not_increment_tick_count(self):
        """tick_count must remain 0 during warmup."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5)
        _do_warmup(strat, bus, 5)
        assert strat._tick_count == 0


# ---------------------------------------------------------------------------
# Kalman update (post-warmup)
# ---------------------------------------------------------------------------


class TestKalmanUpdate:
    def test_no_signals_after_warmup_before_thresholds(self):
        """Post-warmup Kalman ticks return [] until thresholds are set."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5, refit_interval=100)
        _do_warmup(strat, bus, 5)
        signals = _feed_pair(strat, bus, price_x=10.5, price_y=21.0, ts=100)
        assert signals == []

    def test_spread_appended_after_warmup(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5, refit_interval=100)
        _do_warmup(strat, bus, 5)
        assert len(strat._spread_buffer) == 0
        _feed_pair(strat, bus, price_x=10.5, price_y=21.0, ts=100)
        assert len(strat._spread_buffer) == 1

    def test_spread_buffer_grows_each_post_warmup_tick(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5, refit_interval=100)
        _do_warmup(strat, bus, 5)
        for i in range(5):
            _feed_pair(strat, bus, price_x=10.0 + i * 0.1, price_y=20.0 + i * 0.2, ts=100 + i)
        assert len(strat._spread_buffer) == 5

    def test_tick_count_increments_each_post_warmup_tick(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5, refit_interval=100)
        _do_warmup(strat, bus, 5)
        assert strat._tick_count == 0
        for i in range(3):
            _feed_pair(strat, bus, price_x=10.0 + i * 0.1, price_y=20.0 + i * 0.2, ts=100 + i)
        assert strat._tick_count == 3

    def test_kalman_update_known_values(self):
        """Known initial state produces the expected beta and P after one update."""
        bus = _make_bus()
        # Warmup on Y = 2X exactly → beta = 2.0, P = 0.0
        strat = _make_strategy(bus, warmup_ticks=5, delta=1e-4, r=1e-3, refit_interval=100)
        _do_warmup(strat, bus, 5, true_beta=2.0)

        beta_before = strat._beta
        P_before = strat._P
        price_x = 10.0
        price_y = 21.0

        _feed_pair(strat, bus, price_x=price_x, price_y=price_y, ts=100)

        # Expected values from the Kalman equations
        P_pred = P_before + 1e-4
        error = price_y - beta_before * price_x
        S = P_pred * price_x ** 2 + 1e-3
        K = (P_pred * price_x) / S
        expected_beta = beta_before + K * error
        expected_P = P_pred - K * price_x * P_pred

        assert strat._beta == pytest.approx(expected_beta, rel=1e-6)
        assert strat._P == pytest.approx(expected_P, rel=1e-6)

    def test_P_stays_non_negative(self):
        """Kalman covariance must remain non-negative after many updates."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5, delta=1e-4, r=1e-3, refit_interval=200)
        _do_warmup(strat, bus, 5)
        for i in range(50):
            _feed_pair(strat, bus, price_x=10.0 + i * 0.05, price_y=20.0 + i * 0.1, ts=100 + i)
            assert strat._P >= 0, f"P went negative at post-warmup tick {i}"

    def test_P_smaller_after_many_updates(self):
        """After many updates, P should settle well below its initial OLS value."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5, delta=1e-4, r=1e-3, refit_interval=200)
        _do_warmup(strat, bus, 5)
        # Force a known large initial P by setting it directly
        strat._P = 1.0
        for i in range(30):
            _feed_pair(strat, bus, price_x=10.0 + i * 0.05, price_y=20.0 + i * 0.1, ts=100 + i)
        assert strat._P < 0.01

    def test_beta_moves_toward_truth(self):
        """When true beta > estimated beta, each update must increase beta."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5, delta=1e-4, r=1e-3, refit_interval=200)
        _do_warmup(strat, bus, 5, true_beta=2.0)  # beta ≈ 2.0 after warmup
        beta_before = strat._beta
        # Feed a tick where true beta = 3.0 → error = (3.0 - 2.0) * price_x > 0
        _feed_pair(strat, bus, price_x=10.0, price_y=30.0, ts=100)
        assert strat._beta > beta_before

    def test_beta_tracks_shifting_relationship(self):
        """Beta should converge toward a new true value after a gradual shift."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=50, delta=1e-3, r=1e-3, refit_interval=500)
        # Warmup with true beta = 2.0
        _do_warmup(strat, bus, 50, base_x=100.0, true_beta=2.0)
        # Post-warmup: true beta drifts gradually from 2.0 to 3.0 over 200 ticks
        for i in range(200):
            true_beta = 2.0 + (i / 200) * 1.0
            _feed_pair(strat, bus, price_x=100.0, price_y=true_beta * 100.0, ts=1000 + i)
        # After tracking 200 ticks of drift, beta should be well above 2.5
        assert strat._beta > 2.5, f"beta {strat._beta:.4f} did not track the shift toward 3.0"

    def test_spread_value_is_pre_update_error(self):
        """The spread appended to the buffer must equal price_y - beta_old * price_x."""
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5, delta=1e-4, r=1e-3, refit_interval=100)
        _do_warmup(strat, bus, 5, true_beta=2.0)

        beta_before = strat._beta
        price_x = 10.0
        price_y = 21.0
        expected_spread = price_y - beta_before * price_x

        _feed_pair(strat, bus, price_x=price_x, price_y=price_y, ts=100)
        assert strat._spread_buffer[-1] == pytest.approx(expected_spread, rel=1e-10)


# ---------------------------------------------------------------------------
# OU fitting
# ---------------------------------------------------------------------------


def _ar1_spread(
    n: int,
    a: float = 0.9,
    b: float = 0.5,
    noise: float = 0.05,
    seed: int = 0,
) -> list[float]:
    """Generate an AR(1) process: x_{t+1} = a·x_t + b + N(0, noise)."""
    rng = np.random.default_rng(seed)
    spreads = [b / (1 - a)]  # start near the long-run mean
    for _ in range(n - 1):
        spreads.append(a * spreads[-1] + b + rng.normal(0, noise))
    return spreads


def _strat_with_spread(spread: list[float], **overrides) -> KalmanOUBertramStrategy:
    """Return a warmed-up strategy with the spread buffer pre-loaded."""
    bus = _make_bus()
    kw = dict(warmup_ticks=5, refit_interval=500, min_half_life=1.0, max_half_life=100.0)
    kw.update(overrides)
    strat = _make_strategy(bus, **kw)
    _do_warmup(strat, bus, 5)
    strat._spread_buffer.extend(spread)
    return strat


class TestOUFit:
    def test_mean_reverting_returns_true(self):
        """AR(1) with a=0.9 should pass all validation gates."""
        strat = _strat_with_spread(_ar1_spread(100, a=0.9, b=0.5, noise=0.05))
        assert strat._fit_ou() is True

    def test_mean_reverting_stores_params(self):
        """Successful fit must populate speed, mean, sigma, and half_life."""
        strat = _strat_with_spread(_ar1_spread(100, a=0.9, b=0.5, noise=0.05))
        strat._fit_ou()
        assert strat._speed is not None and strat._speed > 0
        assert strat._mean is not None
        assert strat._sigma is not None and strat._sigma > 0
        assert strat._half_life is not None and strat._half_life > 0

    def test_speed_positive(self):
        strat = _strat_with_spread(_ar1_spread(100, a=0.9, noise=0.05))
        strat._fit_ou()
        assert strat._speed > 0

    def test_half_life_consistent_with_speed(self):
        """half_life must equal log(2) / speed."""
        strat = _strat_with_spread(_ar1_spread(100, a=0.9, noise=0.05))
        strat._fit_ou()
        assert strat._half_life == pytest.approx(np.log(2) / strat._speed, rel=1e-6)

    def test_mean_close_to_true_mean(self):
        """AR(1) with a=0.9, b=1.0 has long-run mean = b/(1-a) = 10.0."""
        strat = _strat_with_spread(_ar1_spread(500, a=0.9, b=1.0, noise=0.05, seed=7))
        strat._fit_ou()
        assert strat._mean == pytest.approx(10.0, abs=0.5)

    def test_oscillating_process_fails_a_gate(self):
        """AR(1) with a=-0.5 produces a < 0, rejected by the first gate."""
        rng = np.random.default_rng(3)
        spread = [0.0]
        for _ in range(99):
            spread.append(-0.5 * spread[-1] + rng.normal(0, 0.01))
        strat = _strat_with_spread(spread)
        assert strat._fit_ou() is False

    def test_unit_root_fails_a_gate(self):
        """Linear trend gives OLS a = 1.0 exactly, rejected by the a >= 1.0 gate."""
        spread = [float(i) for i in range(100)]
        strat = _strat_with_spread(spread)
        assert strat._fit_ou() is False

    def test_slow_mean_reversion_exceeds_max_half_life(self):
        """a=0.99, n=2000 → OLS half-life ≈ 44 ticks, exceeds max_half_life=20."""
        strat = _strat_with_spread(
            _ar1_spread(2000, a=0.99, b=0.01, noise=0.1, seed=2),
            spread_buffer_size=3000,  # must hold all 2000 entries
            min_half_life=1.0,
            max_half_life=20.0,
        )
        assert strat._fit_ou() is False

    def test_fast_mean_reversion_below_min_half_life(self):
        """a=0.1 gives half-life ≈ 0.30 ticks, which is below min_half_life=1.0."""
        strat = _strat_with_spread(
            _ar1_spread(100, a=0.1, b=0.5, noise=0.1, seed=1),
            min_half_life=1.0,
            max_half_life=100.0,
        )
        assert strat._fit_ou() is False

    def test_too_few_points_returns_false(self):
        """Buffer with < 30 entries must return False before attempting regression."""
        strat = _strat_with_spread(_ar1_spread(20, a=0.9))
        assert strat._fit_ou() is False

    def test_exactly_thirty_points_attempts_fit(self):
        """Window == 30 must not be rejected by the minimum-points gate."""
        strat = _strat_with_spread(
            _ar1_spread(30, a=0.9, noise=0.01),
            min_half_life=1.0,
            max_half_life=100.0,
        )
        assert strat._fit_ou() is True

    def test_failed_fit_does_not_overwrite_params(self):
        """A failed fit must leave previously stored OU params unchanged."""
        spread = _ar1_spread(100, a=0.9, noise=0.05)
        strat = _strat_with_spread(spread)
        strat._fit_ou()  # first fit succeeds
        speed_before = strat._speed
        mean_before = strat._mean
        # Replace buffer with oscillating data → a < 0 → fit fails
        strat._spread_buffer.clear()
        rng = np.random.default_rng(99)
        osc = [0.0]
        for _ in range(99):
            osc.append(-0.5 * osc[-1] + rng.normal(0, 0.01))
        strat._spread_buffer.extend(osc)
        strat._fit_ou()
        assert strat._speed == speed_before
        assert strat._mean == mean_before

    def test_dynamic_window_uses_prior_half_life(self):
        """When _half_life=3.0, window = 5*3 = 15 < 30 → returns False."""
        strat = _strat_with_spread(_ar1_spread(200, a=0.9))
        strat._half_life = 3.0  # forces window = 15
        assert strat._fit_ou() is False

    def test_first_fit_uses_full_buffer_length(self):
        """With _half_life=None and 20 entries, window=20 < 30 → returns False."""
        strat = _strat_with_spread(_ar1_spread(20, a=0.9))
        assert strat._half_life is None
        assert strat._fit_ou() is False

    def test_dynamic_window_capped_at_buffer_size(self):
        """half_life=100 → 5*100=500, but buffer has 200 → window=200 → succeeds."""
        strat = _strat_with_spread(
            _ar1_spread(200, a=0.9, noise=0.05),
            min_half_life=1.0,
            max_half_life=1000.0,
        )
        strat._half_life = 100.0  # 5*100=500 > 200, capped at 200
        assert strat._fit_ou() is True

    def test_fit_ou_and_bertram_triggers_fit(self):
        """_fit_ou_and_bertram must call _fit_ou; OU params are set after call."""
        strat = _strat_with_spread(_ar1_spread(100, a=0.9, noise=0.05))
        assert strat._speed is None
        strat._fit_ou_and_bertram()
        assert strat._speed is not None

    def test_constant_spread_sigma_zero_rejected(self):
        """A perfectly constant spread gives sigma=0, which must be rejected."""
        strat = _strat_with_spread([5.0] * 50)
        assert strat._fit_ou() is False


# ---------------------------------------------------------------------------
# Bertram threshold computation
# ---------------------------------------------------------------------------


def _strat_with_ou_params(
    speed: float = 0.1,
    mean: float = 0.0,
    sigma: float = 0.5,
    cost: float = 0.003,
    threshold_buffer: float = 1.0,
    **overrides,
) -> KalmanOUBertramStrategy:
    """Return a strategy with OU parameters pre-loaded, ready for _compute_bertram."""
    bus = _make_bus()
    kw = dict(
        warmup_ticks=5,
        cost=cost,
        threshold_buffer=threshold_buffer,
    )
    kw.update(overrides)
    strat = _make_strategy(bus, **kw)
    _do_warmup(strat, bus, 5)
    strat._speed = speed
    strat._sigma = sigma
    strat._mean = mean
    strat._half_life = np.log(2) / speed
    return strat


class TestBertram:
    def test_valid_params_returns_true(self):
        """Reasonable OU params with low cost should produce profitable thresholds."""
        strat = _strat_with_ou_params(speed=0.1, sigma=0.5, cost=0.003)
        assert strat._compute_bertram() is True

    def test_thresholds_set_after_successful_compute(self):
        strat = _strat_with_ou_params()
        strat._compute_bertram()
        assert strat._long_threshold is not None
        assert strat._short_threshold is not None

    def test_long_threshold_below_mean(self):
        """Long (buy spread) threshold must be below the OU mean."""
        strat = _strat_with_ou_params(mean=0.0)
        strat._compute_bertram()
        assert strat._long_threshold < strat._mean

    def test_short_threshold_above_mean(self):
        """Short (sell spread) threshold must be above the OU mean."""
        strat = _strat_with_ou_params(mean=0.0)
        strat._compute_bertram()
        assert strat._short_threshold > strat._mean

    def test_thresholds_symmetric_around_mean(self):
        """long and short thresholds must be equidistant from the OU mean."""
        strat = _strat_with_ou_params(mean=2.5)
        strat._compute_bertram()
        dist_long = strat._mean - strat._long_threshold
        dist_short = strat._short_threshold - strat._mean
        assert dist_long == pytest.approx(dist_short, rel=1e-6)

    def test_threshold_buffer_widens_thresholds(self):
        """A larger threshold_buffer must produce wider thresholds."""
        strat_tight = _strat_with_ou_params(threshold_buffer=1.0)
        strat_wide = _strat_with_ou_params(threshold_buffer=1.5)
        strat_tight._compute_bertram()
        strat_wide._compute_bertram()
        tight_width = strat_tight._short_threshold - strat_tight._long_threshold
        wide_width = strat_wide._short_threshold - strat_wide._long_threshold
        assert wide_width > tight_width

    def test_threshold_buffer_scales_exactly(self):
        """Thresholds with buffer=1.5 should be exactly 1.5× wider than buffer=1.0."""
        strat_1 = _strat_with_ou_params(threshold_buffer=1.0)
        strat_15 = _strat_with_ou_params(threshold_buffer=1.5)
        strat_1._compute_bertram()
        strat_15._compute_bertram()
        width_1 = strat_1._short_threshold - strat_1._long_threshold
        width_15 = strat_15._short_threshold - strat_15._long_threshold
        assert width_15 == pytest.approx(1.5 * width_1, rel=1e-6)

    def test_high_cost_returns_false(self):
        """When cost >> spread volatility, G(d) <= 0 everywhere → returns False."""
        strat = _strat_with_ou_params(speed=0.01, sigma=0.01, cost=100.0)
        assert strat._compute_bertram() is False

    def test_high_cost_sets_state_inactive(self):
        """Unprofitable conditions must set state to INACTIVE."""
        strat = _strat_with_ou_params(speed=0.01, sigma=0.01, cost=100.0)
        strat._compute_bertram()
        assert strat._state == "INACTIVE"

    def test_high_cost_leaves_thresholds_none(self):
        """Thresholds must not be set when the strategy is unprofitable."""
        strat = _strat_with_ou_params(speed=0.01, sigma=0.01, cost=100.0)
        strat._compute_bertram()
        assert strat._long_threshold is None
        assert strat._short_threshold is None

    def test_state_unchanged_on_profitable_compute(self):
        """A profitable compute must not change _state (state is FLAT by default)."""
        strat = _strat_with_ou_params()
        strat._compute_bertram()
        assert strat._state == "FLAT"

    def test_sd_scales_thresholds(self):
        """Larger sigma (higher OU volatility) → wider thresholds in real units."""
        strat_low = _strat_with_ou_params(sigma=0.1, threshold_buffer=1.0)
        strat_high = _strat_with_ou_params(sigma=1.0, threshold_buffer=1.0)
        strat_low._compute_bertram()
        strat_high._compute_bertram()
        width_low = strat_low._short_threshold - strat_low._long_threshold
        width_high = strat_high._short_threshold - strat_high._long_threshold
        assert width_high > width_low

    def test_fit_ou_and_bertram_sets_thresholds(self):
        """Full pipeline: valid spread buffer → _fit_ou_and_bertram → thresholds set."""
        strat = _strat_with_spread(
            _ar1_spread(100, a=0.9, b=0.5, noise=0.05),
            min_half_life=1.0,
            max_half_life=100.0,
            cost=0.001,
        )
        assert strat._long_threshold is None
        strat._fit_ou_and_bertram()
        assert strat._long_threshold is not None
        assert strat._short_threshold is not None


# ---------------------------------------------------------------------------
# State machine and signal generation
# ---------------------------------------------------------------------------


def _strat_with_thresholds(
    long_threshold: float = -1.0,
    short_threshold: float = 1.0,
    mean: float = 0.0,
    beta: float = 2.0,
    capital: float = 10000.0,
    **overrides,
) -> KalmanOUBertramStrategy:
    """Return a warmed-up strategy with thresholds pre-set, ready for signal tests."""
    bus = _make_bus()
    kw = dict(warmup_ticks=5, capital=capital, refit_interval=500)
    kw.update(overrides)
    strat = _make_strategy(bus, **kw)
    _do_warmup(strat, bus, 5)
    strat._long_threshold = long_threshold
    strat._short_threshold = short_threshold
    strat._mean = mean
    strat._beta = beta
    return strat


def _eval(strat: KalmanOUBertramStrategy, price_x: float, price_y: float) -> list[Signal]:
    """Publish ticks and call on_data to drive the strategy."""
    ch = _make_bus().channel(LISTEN_CH)  # wrong bus — use direct method call
    return strat._evaluate_signal(price_x, price_y)


class TestStateMachine:
    # ------------------------------------------------------------------ FLAT
    def test_no_signal_when_spread_inside_thresholds(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=1.0)
        # spread = price_y - beta * price_x = 5.0 - 5.0 = 0.0, inside [-1, 1]
        assert strat._evaluate_signal(5.0, 5.0) == []

    def test_flat_to_long_when_spread_below_long_threshold(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=1.0)
        # spread = 1.0 - 2.5 = -1.5 < -1.0
        signals = strat._evaluate_signal(2.5, 1.0)
        assert len(signals) == 2
        assert strat._state == "LONG"

    def test_flat_to_short_when_spread_above_short_threshold(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=1.0)
        # spread = 3.0 - 1.5 = 1.5 > 1.0
        signals = strat._evaluate_signal(1.5, 3.0)
        assert len(signals) == 2
        assert strat._state == "SHORT"

    def test_flat_no_transition_at_exact_long_threshold(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, beta=1.0)
        # spread = -1.0 = long_threshold (not strictly below)
        assert strat._evaluate_signal(2.0, 1.0) == []

    def test_flat_no_transition_at_exact_short_threshold(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, beta=1.0)
        # spread = 1.0 = short_threshold (not strictly above)
        assert strat._evaluate_signal(0.0, 1.0) == []

    # ------------------------------------------------------------------ LONG
    def test_long_to_flat_when_spread_reaches_mean(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=1.0)
        strat._state = "LONG"
        # spread = 2.0 - 2.0 = 0.0 >= mean
        signals = strat._evaluate_signal(2.0, 2.0)
        assert len(signals) == 2
        assert strat._state == "FLAT"

    def test_long_stays_long_while_spread_below_mean(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=1.0)
        strat._state = "LONG"
        # spread = 1.0 - 2.0 = -1.0 < mean=0.0
        assert strat._evaluate_signal(2.0, 1.0) == []
        assert strat._state == "LONG"

    def test_long_exits_at_exactly_mean(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.5, beta=1.0)
        strat._state = "LONG"
        # spread = 2.5 - 2.0 = 0.5 = mean (>= mean → exit)
        signals = strat._evaluate_signal(2.0, 2.5)
        assert len(signals) == 2
        assert strat._state == "FLAT"

    # ---------------------------------------------------------------- SHORT
    def test_short_to_flat_when_spread_reaches_mean(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=1.0)
        strat._state = "SHORT"
        # spread = 0.5 - 0.5 = 0.0 <= mean
        signals = strat._evaluate_signal(0.5, 0.5)
        assert len(signals) == 2
        assert strat._state == "FLAT"

    def test_short_stays_short_while_spread_above_mean(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=1.0)
        strat._state = "SHORT"
        # spread = 3.0 - 1.5 = 1.5 > mean=0.0
        assert strat._evaluate_signal(1.5, 3.0) == []
        assert strat._state == "SHORT"

    # ------------------------------------------------------------- Guards
    def test_no_signal_when_thresholds_none(self):
        bus = _make_bus()
        strat = _make_strategy(bus, warmup_ticks=5)
        _do_warmup(strat, bus, 5)
        assert strat._evaluate_signal(10.0, 20.0) == []

    def test_no_signal_when_inactive(self):
        strat = _strat_with_thresholds()
        strat._state = "INACTIVE"
        assert strat._evaluate_signal(10.0, 20.0) == []

    # -------------------------------------------------------- Signal shape
    def test_long_signal_y_positive_x_negative(self):
        """LONG the spread = buy Y, sell X."""
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, beta=1.0)
        # spread < long_threshold → go LONG
        signals = strat._evaluate_signal(2.5, 1.0)
        sig_y = next(s for s in signals if s.symbol == SYMBOL_Y)
        sig_x = next(s for s in signals if s.symbol == SYMBOL_X)
        assert sig_y.target_position > 0
        assert sig_x.target_position < 0

    def test_short_signal_y_negative_x_positive(self):
        """SHORT the spread = sell Y, buy X."""
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, beta=1.0)
        signals = strat._evaluate_signal(1.5, 3.0)
        sig_y = next(s for s in signals if s.symbol == SYMBOL_Y)
        sig_x = next(s for s in signals if s.symbol == SYMBOL_X)
        assert sig_y.target_position < 0
        assert sig_x.target_position > 0

    def test_flat_signal_both_zero(self):
        """FLAT signals must set both legs to target_position=0."""
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=1.0)
        strat._state = "LONG"
        signals = strat._evaluate_signal(2.0, 2.0)  # spread = 0 >= mean → FLAT
        for sig in signals:
            assert sig.target_position == 0.0

    def test_two_signals_emitted_on_every_transition(self):
        """Every state change must produce exactly 2 signals (one per leg)."""
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, beta=1.0)
        assert len(strat._evaluate_signal(2.5, 1.0)) == 2   # FLAT→LONG
        assert len(strat._evaluate_signal(2.0, 2.0)) == 2   # LONG→FLAT
        assert len(strat._evaluate_signal(1.5, 3.0)) == 2   # FLAT→SHORT
        assert len(strat._evaluate_signal(0.5, 0.5)) == 2   # SHORT→FLAT

    def test_signal_symbols_correct(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, beta=1.0)
        signals = strat._evaluate_signal(2.5, 1.0)
        symbols = {s.symbol for s in signals}
        assert symbols == {SYMBOL_Y, SYMBOL_X}

    def test_signal_prices_correct(self):
        strat = _strat_with_thresholds(long_threshold=-1.0, short_threshold=1.0, beta=1.0)
        px, py = 2.5, 1.0
        signals = strat._evaluate_signal(px, py)
        sig_y = next(s for s in signals if s.symbol == SYMBOL_Y)
        sig_x = next(s for s in signals if s.symbol == SYMBOL_X)
        assert sig_y.price == py
        assert sig_x.price == px


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    # Use px=10, py=1 → spread = 1 - beta*10. With beta ≥ 1 this is negative and
    # well below long_threshold=-1, so we reliably enter LONG without negative prices.

    def _long_signals(self, beta: float, capital: float) -> tuple[Signal, Signal]:
        """Drive strategy to LONG and return (sig_y, sig_x)."""
        strat = _strat_with_thresholds(
            long_threshold=-1.0, short_threshold=1.0, mean=0.0,
            beta=beta, capital=capital,
        )
        signals = strat._evaluate_signal(10.0, 1.0)  # spread = 1 - beta*10 << -1
        assert len(signals) == 2, "failed to enter LONG"
        return (
            next(s for s in signals if s.symbol == SYMBOL_Y),
            next(s for s in signals if s.symbol == SYMBOL_X),
        )

    def test_notional_sums_to_capital(self):
        """qty_y * price_y + qty_x * price_x must equal capital."""
        beta, capital = 2.0, 10000.0
        sig_y, sig_x = self._long_signals(beta, capital)
        notional = sig_y.target_position * sig_y.price + abs(sig_x.target_position) * sig_x.price
        assert notional == pytest.approx(capital, rel=1e-6)

    def test_y_leg_fraction_is_correct(self):
        """Y-leg notional must equal capital / (1 + |β|)."""
        beta, capital = 3.0, 10000.0
        sig_y, _ = self._long_signals(beta, capital)
        expected = capital / (1 + abs(beta))
        assert sig_y.target_position * sig_y.price == pytest.approx(expected, rel=1e-6)

    def test_x_leg_fraction_is_correct(self):
        """X-leg notional must equal |β| * capital / (1 + |β|)."""
        beta, capital = 3.0, 10000.0
        _, sig_x = self._long_signals(beta, capital)
        expected = abs(beta) * capital / (1 + abs(beta))
        assert abs(sig_x.target_position) * sig_x.price == pytest.approx(expected, rel=1e-6)

    def test_higher_beta_shifts_weight_to_x(self):
        """Increasing β reduces the Y-leg quantity and increases the X-leg quantity."""
        sig_y_low, _ = self._long_signals(beta=1.0, capital=10000.0)
        sig_y_high, _ = self._long_signals(beta=4.0, capital=10000.0)
        assert sig_y_high.target_position < sig_y_low.target_position

    def test_short_notional_sums_to_capital(self):
        """SHORT allocation must also satisfy qty_y*py + qty_x*px == capital."""
        beta, capital = 2.0, 10000.0
        strat = _strat_with_thresholds(
            long_threshold=-1.0, short_threshold=1.0, mean=0.0,
            beta=beta, capital=capital,
        )
        # spread = 50 - 2*1 = 48 >> short_threshold=1
        signals = strat._evaluate_signal(1.0, 50.0)
        sig_y = next(s for s in signals if s.symbol == SYMBOL_Y)
        sig_x = next(s for s in signals if s.symbol == SYMBOL_X)
        notional = abs(sig_y.target_position) * sig_y.price + abs(sig_x.target_position) * sig_x.price
        assert notional == pytest.approx(capital, rel=1e-6)

    def test_flat_positions_are_zero(self):
        """FLAT signals must have target_position=0 regardless of prices."""
        strat = _strat_with_thresholds(
            long_threshold=-1.0, short_threshold=1.0, mean=0.0, beta=2.0, capital=10000.0
        )
        strat._state = "LONG"
        # spread = 12 - 2*5 = 2 >= mean=0 → exits to FLAT
        signals = strat._evaluate_signal(5.0, 12.0)
        for sig in signals:
            assert sig.target_position == 0.0
