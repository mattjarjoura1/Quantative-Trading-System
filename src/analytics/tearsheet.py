"""Backtest tearsheet renderer."""

import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from src.analytics.backtest_analyser import BacktestResult


def render_tearsheet(result: BacktestResult) -> plt.Figure:
    """Render a four-panel backtest tear sheet.

    Panels:
        1. Equity curve with initial capital reference line.
        2. Drawdown percentage (filled area below zero).
        3. Summary statistics table.
        4. Per-trade PnL distribution histogram.

    Args:
        result: BacktestResult from BacktestAnalyser.run().

    Returns:
        matplotlib Figure. Caller can display, save to PDF/PNG, or close.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], figure=fig)

    ax_equity = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_equity)
    ax_stats = fig.add_subplot(gs[2, 0])
    ax_hist = fig.add_subplot(gs[2, 1])

    _plot_equity(ax_equity, result)
    _plot_drawdown(ax_dd, result)
    _plot_stats(ax_stats, result)
    _plot_histogram(ax_hist, result)

    fig.tight_layout(pad=1.5)
    return fig


def _plot_equity(ax: plt.Axes, result: BacktestResult) -> None:
    """Plot equity curve with initial capital reference line.

    Args:
        ax: Target axes.
        result: BacktestResult containing equity_curve and tracker.
    """
    curve = result.equity_curve
    if curve:
        timestamps = np.array([t for t, _ in curve], dtype=np.float64)
        equity = np.array([e for _, e in curve], dtype=np.float64)
        dates = [datetime.datetime.fromtimestamp(t / 1000.0, tz=datetime.timezone.utc) for t in timestamps]
        ax.plot(dates, equity, color="black", linewidth=1)
        ax.axhline(result.tracker.initial_capital, color="grey", linestyle="--", linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.set_title("Portfolio Equity")
    ax.set_ylabel("Equity ($)")
    ax.tick_params(axis="x", labelbottom=False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _plot_drawdown(ax: plt.Axes, result: BacktestResult) -> None:
    """Plot drawdown percentage as a filled area.

    Args:
        ax: Target axes.
        result: BacktestResult containing equity_curve.
    """
    curve = result.equity_curve
    if curve:
        timestamps = np.array([t for t, _ in curve], dtype=np.float64)
        equity = np.array([e for _, e in curve], dtype=np.float64)
        dates = [datetime.datetime.fromtimestamp(t / 1000.0, tz=datetime.timezone.utc) for t in timestamps]
        peaks = np.maximum.accumulate(equity)
        drawdown_pct = (equity - peaks) / peaks * 100.0
        ax.fill_between(dates, drawdown_pct, 0, color="red", alpha=0.4)
        ax.plot(dates, drawdown_pct, color="red", linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _plot_stats(ax: plt.Axes, result: BacktestResult) -> None:
    """Render summary statistics as formatted text.

    Args:
        ax: Target axes.
        result: BacktestResult containing metrics.
    """
    ax.axis("off")
    m = result.metrics

    def _fmt_dollar(v: float) -> str:
        return "N/A" if np.isnan(v) else f"${v:,.2f}"

    def _fmt_pct(v: float) -> str:
        return "N/A" if np.isnan(v) else f"{v * 100:.2f}%"

    def _fmt_ratio(v: float) -> str:
        return "N/A" if np.isnan(v) else f"{v:.2f}"

    def _fmt_inf(v: float) -> str:
        if np.isnan(v):
            return "N/A"
        if np.isinf(v):
            return "Inf"
        return f"{v:.2f}"

    dd_days = m.max_drawdown_duration_ms / (24 * 3600 * 1000)

    rows = [
        ("Total PnL",        _fmt_dollar(m.total_pnl)),
        ("Total Return",     _fmt_pct(m.total_return_pct)),
        ("CAGR",             _fmt_pct(m.cagr)),
        ("Sharpe",           _fmt_ratio(m.annualised_sharpe)),
        ("Max Drawdown",     f"-{m.max_drawdown_pct * 100:.2f}%" if not np.isnan(m.max_drawdown_pct) else "N/A"),
        ("Max DD Duration",  f"{dd_days:.0f} days"),
        ("Trades",           str(m.num_trades)),
        ("Win Rate",         _fmt_pct(m.win_rate)),
        ("Profit Factor",    _fmt_inf(m.profit_factor)),
    ]

    line_height = 0.10
    y = 0.95
    for label, value in rows:
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=9,
                fontfamily="monospace", va="top")
        ax.text(0.98, y, value, transform=ax.transAxes, fontsize=9,
                fontfamily="monospace", va="top", ha="right")
        y -= line_height

    ax.set_title("Summary Statistics")


def _plot_histogram(ax: plt.Axes, result: BacktestResult) -> None:
    """Plot per-trade PnL distribution histogram.

    Args:
        ax: Target axes.
        result: BacktestResult containing tracker.trade_pnls.
    """
    pnls = result.tracker.trade_pnls
    if not pnls:
        ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
    else:
        ax.hist(pnls, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_title("Trade PnL Distribution")
    ax.set_xlabel("PnL ($)")
    ax.set_ylabel("Count")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
