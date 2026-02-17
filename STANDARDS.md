# Coding Standards

## Naming

| Thing       | Convention      | Example                              |
|-------------|-----------------|--------------------------------------|
| Files       | `snake_case.py` | `kalman_filter.py`, `static_ols.py`  |
| Classes     | `PascalCase`    | `KalmanHedger`, `YahooSource`        |
| ABCs        | `Base` prefix   | `BaseDataSource`                     |
| Functions   | `snake_case`    | `fit_ou_process()`, `_validate()`    |
| Variables   | `snake_case`    | `hedge_ratio`, `rolling_mean`        |
| Constants   | `UPPER_SNAKE`   | `DEFAULT_WINDOW = 30`               |
| Booleans    | `is_` / `has_`  | `is_stationary()`                    |

Single-letter math variables (`theta`, `mu`, `sigma`, `dt`) are fine. Vague names (`a`, `tmp`) are not.

## Function Signatures

Type hints and a Google-style docstring on every function.

```python
def calculate_half_life(spread: np.ndarray, window: int = 60) -> float:
    """Estimate mean-reversion half-life via AR(1) regression.

    Args:
        spread: Array of spread values.
        window: Number of recent observations to use.

    Returns:
        Half-life in the same time units as the input data.
    """
```
