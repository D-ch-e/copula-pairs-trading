"""Report figures. Static matplotlib, one y-scale per panel, fixed series colors.

Colors follow the entity, not its rank: each strategy/benchmark keeps the same
color in every figure.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter

from cpt.config import PROJECT_ROOT
from cpt.metrics import drawdown

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

SURFACE = "#fcfcfb"
TEXT, TEXT_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENCH = {"GLD": "#3d3c39", "SPY": "#8a8984", "50/50 GLD+SLV": "#b5b4ae"}

SPEC_COLORS = {
    "Copula t (main)": SERIES[0],
    "Copula Gaussian": SERIES[1],
    "Copula t, paper exit": SERIES[2],
    "z-score baseline": SERIES[3],
    **BENCH,
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": TEXT_2, "ytick.labelcolor": TEXT_2,
        "legend.frameon": False, "legend.labelcolor": TEXT_2, "font.size": 10,
        "lines.linewidth": 2, "figure.dpi": 110, "axes.axisbelow": True,
    })


def save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def prices_and_ratio(prices: pd.DataFrame) -> plt.Figure:
    """Fig 1: both legs indexed to 100, and their price ratio, as two panels."""
    leg1, leg2 = prices.columns
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 2])
    indexed = 100 * prices / prices.iloc[0]
    ax1.plot(indexed[leg1], color=SERIES[0], label=leg1)
    ax1.plot(indexed[leg2], color=SERIES[1], label=leg2)
    ax1.set_title(f"{leg1} and {leg2}, indexed to 100 at {prices.index[0]:%Y-%m}")
    ax1.legend(loc="upper left")
    ratio = prices[leg1] / prices[leg2]
    ax2.plot(ratio, color=TEXT_2, linewidth=1.5)
    ax2.set_title(f"{leg1} / {leg2} price ratio")
    fig.tight_layout()
    return fig


def _band_contours(ax, h, fit, alpha):
    grid = np.linspace(0.002, 0.998, 300)
    g1, g2 = np.meshgrid(grid, grid)
    h1, h2 = h(g1, g2, fit), h(g2, g1, fit)
    region_a = (h1 < alpha) & (h2 > 1 - alpha)
    region_b = (h1 > 1 - alpha) & (h2 < alpha)
    ax.contourf(g1, g2, region_a.astype(float), levels=[0.5, 1.5], colors=[SERIES[0]], alpha=0.18)
    ax.contourf(g1, g2, region_b.astype(float), levels=[0.5, 1.5], colors=[SERIES[1]], alpha=0.18)
    for field in (h1, h2):
        ax.contour(g1, g2, field, levels=[alpha, 1 - alpha], colors=[MUTED], linewidths=1)


def copula_bands(train_u: tuple, test_u: tuple, fit_g: dict, fit_t: dict, alpha: float,
                 legs: tuple[str, str], month: str) -> plt.Figure:
    """Fig 2: training pseudo-observations with Gaussian vs t signal regions A/B."""
    from cpt.backtest import H_FUNCTIONS

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, family, fit in [(axes[0], "gaussian", fit_g), (axes[1], "t", fit_t)]:
        _band_contours(ax, H_FUNCTIONS[family], fit, alpha)
        ax.scatter(*train_u, s=8, color=MUTED, alpha=0.5, linewidths=0, label="Estimation window")
        ax.scatter(*test_u, s=36, color=TEXT, edgecolors=SURFACE, linewidths=1.5, label=f"{month} (out of sample)")
        params = f"ρ={fit['rho']:.2f}" + (f", ν={fit['nu']:.1f}" if family == "t" else "")
        ax.set_title(f"{'Gaussian' if family == 'gaussian' else 'Student-t'} copula ({params})")
        ax.set_xlabel(f"u1 ({legs[0]})")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal")
        ax.text(0.04, 0.94, "A", color=SERIES[0], fontweight="bold")
        ax.text(0.93, 0.03, "B", color=SERIES[1], fontweight="bold")
    axes[0].set_ylabel(f"u2 ({legs[1]})")
    axes[1].legend(loc="lower right", bbox_to_anchor=(1, -0.3), ncol=2)
    fig.tight_layout()
    fig.suptitle(f"Signal regions, α = {alpha:.0%}: A = long {legs[0]} / short {legs[1]}, B = reverse",
                 x=0.01, y=1.04, ha="left", color=TEXT_2, fontsize=10)
    return fig


def rolling_parameters(fits: pd.DataFrame, pair: str) -> plt.Figure:
    """Fig 3: monthly rho (both families), t-copula nu, and implied tail dependence."""
    x = pd.PeriodIndex(fits["month"], freq="M").to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, fits["t_rho"], color=SERIES[0], label="t-copula ρ")
    axes[0].plot(x, fits["gaussian_rho"], color=SERIES[1], linewidth=1.5, label="Gaussian copula ρ")
    axes[0].set_title(f"{pair}: copula correlation (12-month window, refit monthly)")
    axes[0].legend(loc="lower left")
    axes[1].plot(x, fits["t_nu"], color=SERIES[0])
    axes[1].set_yscale("log")
    axes[1].set_title("t-copula ν (log scale; small ν = fatter joint tails, ν ≈ 100 = indistinguishable from Gaussian)")
    axes[2].plot(x, fits["t_tail_dep"], color=SERIES[0])
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set_title("Implied tail dependence λ of the t-copula (Gaussian λ = 0)")
    fig.tight_layout()
    return fig


def tail_calibration(table: pd.DataFrame, q: float) -> plt.Figure:
    """Fig 4: observed vs predicted out-of-sample joint-tail days, as % of days, per pair."""
    cols = [("observed_lower", "Observed, lower tail", SERIES[0]),
            ("observed_upper", "Observed, upper tail", SERIES[2]),
            ("pred_gaussian", "Predicted, Gaussian", SERIES[1]),
            ("pred_t", "Predicted, t", SERIES[3])]
    pairs = table.index
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, (col, label, color) in enumerate(cols):
        x = np.arange(len(pairs)) + (k - 1.5) * width
        ax.bar(x, table[col] / table["days"], width=width * 0.9, color=color, label=label)
    ax.axhline(q**2, color=MUTED, linewidth=1, linestyle="--", label=f"Independence ({q**2:.2%})")
    ax.set_xticks(np.arange(len(pairs)), pairs)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Out-of-sample share of days with both legs beyond their {q:.0%} quantile")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    return fig


def equity_and_drawdown(returns: dict[str, pd.Series], title: str) -> plt.Figure:
    """Fig 5: growth of $1 (log scale) and drawdown, two panels."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 2])
    for name, r in returns.items():
        color = SPEC_COLORS[name]
        lw = 2 if name in BENCH else 2.2
        ax1.plot((1 + r).cumprod(), color=color, linewidth=lw, label=name)
        if name not in BENCH or name == "GLD":
            ax2.plot(drawdown(r), color=color, linewidth=1.3)
    ax1.set_yscale("log")
    lo, hi = ax1.get_ylim()
    ax1.set_yticks([v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if lo <= v <= hi])
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.axhline(1, color=MUTED, linewidth=1)
    ax1.set_title(title)
    ax1.legend(loc="upper left", ncol=2)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_title("Drawdown from previous peak (strategies and GLD)")
    fig.tight_layout()
    return fig


def cost_sensitivity(table: pd.DataFrame, pair: str) -> plt.Figure:
    """Fig 6: annualized return vs round-trip cost; rows = spec name, columns = bps."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, row in table.iterrows():
        ax.plot(row.index, row.values, color=SPEC_COLORS[name], marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Round-trip cost per pair trade (bps)")
    ax.set_title(f"{pair}: annualized return vs transaction cost")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def annual_returns(returns: dict[str, pd.Series], title: str) -> plt.Figure:
    """Fig 7: calendar-year returns, grouped bars."""
    yearly = pd.DataFrame({k: (1 + r).groupby(r.index.year).prod() - 1 for k, r in returns.items()})
    width = 0.8 / len(yearly.columns)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, name in enumerate(yearly.columns):
        x = np.arange(len(yearly)) + (k - (len(yearly.columns) - 1) / 2) * width
        ax.bar(x, yearly[name], width=width * 0.9, color=SPEC_COLORS[name], label=name)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(np.arange(len(yearly)), yearly.index, rotation=45)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title)
    ax.legend(ncol=len(yearly.columns), loc="upper left", bbox_to_anchor=(0, -0.15))
    fig.tight_layout()
    return fig


def holding_periods(trades: dict[str, pd.DataFrame], pair: str) -> plt.Figure:
    """Fig 8: holding-period distribution per spec (exit reasons are shown as a table)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.arange(0, 22) - 0.5
    for name, t in trades.items():
        ax.hist(t["holding_days"], bins=bins, histtype="step", linewidth=2,
                color=SPEC_COLORS[name], label=name)
    ax.set_xlabel("Holding period (trading days)")
    ax.set_ylabel("Trades")
    ax.set_title(f"{pair}: holding period per trade")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
