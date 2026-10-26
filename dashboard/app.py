"""Backtest results dashboard. Run with: streamlit run dashboard/app.py

Every number is read from data/pairs.sqlite (built by scripts/run_experiments.py);
nothing here is hard-coded.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from cpt import analysis  # noqa: E402
from cpt import dashboard_data as dd  # noqa: E402
from cpt.config import db_path, load_config  # noqa: E402
from cpt.db import connect  # noqa: E402
from cpt.metrics import drawdown  # noqa: E402
from cpt.plots import GRID, MUTED, SERIES, SPEC_COLORS, SURFACE, TEXT_2  # noqa: E402

st.set_page_config(page_title="Copula Pairs Trading", page_icon="📉", layout="wide")

CFG = load_config()
DB = db_path(CFG)
MAIN_PAIR = "-".join(CFG["pairs"]["main"])
PAIRS = [MAIN_PAIR, *("-".join(p) for p in CFG["pairs"]["robustness"])]
MAIN_KEY = f"copula|{CFG['strategy']['family']}|{CFG['strategy']['exit_rule']}"
MAIN_NAME = dd.SPEC_NAMES[MAIN_KEY]
DEFAULT_COST = CFG["costs"]["round_trip_bps"]
BENCHMARKS = ["GLD", "SPY", "50/50 GLD+SLV"]


# ---------- data (cached) ----------

@st.cache_data
def runs() -> pd.DataFrame:
    with connect(DB) as c:
        return dd.load_runs(c)


@st.cache_data
def returns(ids: dict[str, str]) -> pd.DataFrame:
    with connect(DB) as c:
        return dd.load_returns(c, ids)


@st.cache_data
def trades(rid: str) -> pd.DataFrame:
    with connect(DB) as c:
        return dd.load_trades(c, rid)


@st.cache_data
def fits(pair: str) -> pd.DataFrame:
    with connect(DB) as c:
        return dd.load_fits(c, pair)


@st.cache_data
def copula_daily(pair: str) -> pd.DataFrame:
    with connect(DB) as c:
        return dd.load_copula_daily(c, pair)


@st.cache_data
def tail_tables(pair: str, q: float = 0.05) -> tuple[dict, dict, dict]:
    f, daily = fits(pair), copula_daily(pair)
    raw = analysis.oos_tail_calibration(f, daily, q).drop(columns="month").sum()
    raw = (raw.drop("days") / raw["days"]).to_dict()
    return analysis.marginal_exceedance(daily, q), raw, analysis.recalibrated_tail_calibration(f, daily, q)


# ---------- chart helpers ----------

def style(fig: go.Figure, height: int, title: str, legend_y: float = -0.12) -> go.Figure:
    fig.update_layout(
        height=height, title=dict(text=title, x=0, font=dict(size=15)), paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE, font=dict(color=TEXT_2), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=legend_y, x=0), hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def chart(fig: go.Figure, height: int, title: str, legend_y: float = -0.12) -> None:
    st.plotly_chart(style(fig, height, title, legend_y), use_container_width=True,
                    config={"displayModeBar": False})


def pct(x: float, digits: int = 1) -> str:
    return "–" if pd.isna(x) else f"{x:.{digits}%}"


# ---------- guard ----------

if not DB.exists():
    st.title("Copula pairs trading: backtest results")
    st.warning("No results database found. Run `python -m scripts.download_data` and "
               "`python -m scripts.run_experiments` first.")
    st.stop()

RUNS = runs()

# ---------- header ----------

st.title("Copula pairs trading on gold and silver")
st.caption("Backtest of the Krauss & Stübinger (2015) copula pairs-trading framework on GLD/SLV and three "
           "robustness pairs, daily data 2007–2025. Research project, not investment advice.")

all_fits = {p: fits(p) for p in PAIRS}
kpi = dd.headline_kpis(RUNS, all_fits, MAIN_PAIR, MAIN_KEY, DEFAULT_COST)
st.markdown(
    f"**Result:** the t-copula describes the joint tails better than a Gaussian copula, but the trading "
    f"strategy built on it does not pay. On {MAIN_PAIR}, net of a {DEFAULT_COST} bps round-trip cost, the main "
    f"specification returns {pct(kpi['main_ann_return'])} a year (Sharpe {kpi['main_sharpe']:.2f}) against "
    f"{pct(kpi['gld_ann_return'])} for buy-and-hold GLD (Sharpe {kpi['gld_sharpe']:.2f})."
)
KPI = dict(delta_color="off", delta_arrow="off", border=True)
main_help = f"{MAIN_NAME} on {MAIN_PAIR}, net of {DEFAULT_COST} bps, 200% gross exposure."
cols = st.columns(5)
cols[0].metric("Net Sharpe", f"{kpi['main_sharpe']:.2f}", f"gross {kpi['main_gross_sharpe']:.2f}",
               help=main_help, **KPI)
cols[1].metric("Annual return", pct(kpi["main_ann_return"]), f"GLD {pct(kpi['gld_ann_return'])}",
               help=main_help + " Compared with buy-and-hold GLD.", **KPI)
cols[2].metric("Max drawdown", pct(kpi["main_max_drawdown"]), f"GLD {pct(kpi['gld_max_drawdown'])}",
               help=main_help + " Compared with buy-and-hold GLD.", **KPI)
cols[3].metric("Break-even cost", f"{kpi['breakeven_bps']:.0f} bps", "per round trip",
               help="Round-trip cost at which the average trade nets zero; negative means the average "
                    "trade loses money before any cost.", **KPI)
cols[4].metric("t beats Gaussian", f"{pct(kpi['t_better_oos_min'], 0)}–{pct(kpi['t_better_oos_max'], 0)}",
               "of months", help="Share of months in which the t-copula has the higher out-of-sample "
                                 "log-likelihood, lowest to highest across the four pairs.", **KPI)

# ---------- sidebar ----------

with st.sidebar:
    st.header("Filters")
    pair = st.selectbox("Pair", PAIRS)
    cost = st.select_slider("Round-trip cost (bps)", CFG["costs"]["sensitivity_bps"], value=DEFAULT_COST)
    exposure = 2.0
    if pair == MAIN_PAIR:
        exposure = st.radio("Gross exposure", [2.0, 1.0], format_func=lambda g: f"{g:.0%}",
                            help="200% = $1 long + $1 short per $1 of NAV (the paper's convention). "
                                 "100% is available for the main specification only.")
    specs = st.multiselect("Strategies", list(dd.SPEC_NAMES.values()), default=list(dd.SPEC_NAMES.values()))
    show_bench = pair == MAIN_PAIR and st.checkbox("Show buy-and-hold benchmarks", value=True)
    st.caption(f"Main specification: {MAIN_NAME}, {DEFAULT_COST} bps, one-day execution lag.")

spec_keys = {v: k for k, v in dd.SPEC_NAMES.items()}
if exposure != 2.0:
    specs = [s for s in specs if s == MAIN_NAME]
    if not specs:
        specs = [MAIN_NAME]
series_ids = {s: dd.run_id(pair, spec_keys[s], cost, exposure) for s in specs}
if show_bench:
    series_ids |= {b: f"{pair}|buy_hold|{b}" for b in BENCHMARKS}

tab_perf, tab_dep, tab_sub, tab_trades, tab_method = st.tabs(
    ["Performance", "Dependence", "Subperiods", "Trades", "Method & limitations"])

# ---------- performance ----------

with tab_perf:
    if exposure != 2.0:
        st.info("100% gross exposure: only the main specification is shown.")
    if not series_ids:
        st.info("Select at least one strategy in the sidebar.")
    else:
        rets = returns(series_ids)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08,
                            subplot_titles=("Growth of $1 NAV (log scale)", "Drawdown (strategies and GLD)"))
        for name in rets.columns:
            r = rets[name].dropna()
            width = 1.5 if name in BENCHMARKS else 2.2
            color = SPEC_COLORS[name]
            fig.add_trace(go.Scatter(x=r.index, y=(1 + r).cumprod(), name=name, line=dict(color=color, width=width),
                                     hovertemplate="$%{y:.2f}"), row=1, col=1)
            if name not in BENCHMARKS or name == "GLD":
                fig.add_trace(go.Scatter(x=r.index, y=drawdown(r), name=name, showlegend=False,
                                         line=dict(color=color, width=1.2), hovertemplate="%{y:.1%}"), row=2, col=1)
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${v:g}" for v in ticks], row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        exposure_label = f"{exposure:.0%} gross"
        chart(fig, 620, f"{pair}: strategies net of {cost} bps, {exposure_label}")

        table = RUNS.set_index("run_id").loc[list(series_ids.values())].set_index("spec")
        columns = {  # metric: (label, format)
            "ann_return": ("Annual return", "{:.1%}"), "ann_vol": ("Volatility", "{:.1%}"),
            "sharpe": ("Sharpe", "{:.2f}"), "sortino": ("Sortino", "{:.2f}"),
            "max_drawdown": ("Max drawdown", "{:.1%}"), "var_95": ("Daily VaR 95%", "{:.2%}"),
            "cvar_95": ("Daily CVaR 95%", "{:.2%}"), "n_trades": ("Trades", "{:.0f}"),
            "avg_holding_days": ("Avg hold (days)", "{:.1f}"), "win_rate": ("Win rate", "{:.0%}"),
            "breakeven_bps": ("Break-even (bps)", "{:.0f}"),
        }
        table = pd.DataFrame({label: table[metric].astype(float).map(lambda v, f=fmt: "–" if pd.isna(v) else f.format(v))
                              for metric, (label, fmt) in columns.items()})
        table.index.name = None
        st.dataframe(table, use_container_width=True)
        st.caption("Strategy returns are on \\$1 of NAV; benchmarks are \\$1 long-only. Sharpe uses the daily "
                   "Fama/French risk-free rate. Break-even = round-trip cost at which the average trade nets zero.")

    sens = RUNS[(RUNS["pair"] == pair) & (RUNS["model"] != "buy_hold") & (RUNS["gross_exposure"] == 2.0)]
    sens = sens.pivot(index="cost_bps", columns="spec", values="ann_return")[list(dd.SPEC_NAMES.values())]
    fig = go.Figure()
    for name in sens.columns:
        fig.add_trace(go.Scatter(x=sens.index, y=sens[name], name=name, mode="lines+markers",
                                 line=dict(color=SPEC_COLORS[name], width=2), marker=dict(size=8),
                                 hovertemplate="%{y:.1%}"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title="Round-trip cost per pair trade (bps)")
    chart(fig, 400, f"{pair}: annualized return vs transaction cost (200% gross)", legend_y=-0.25)

# ---------- dependence ----------

with tab_dep:
    f = fits(pair)
    month = pd.PeriodIndex(f["month"], freq="M").to_timestamp()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Copula correlation ρ", "t-copula degrees of freedom ν (log scale; 200 = upper bound ≈ Gaussian)",
                                        "Implied tail dependence λ of the t-copula"))
    fig.add_trace(go.Scatter(x=month, y=f["t_rho"], name="t-copula", line=dict(color=SERIES[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=month, y=f["gaussian_rho"], name="Gaussian copula", line=dict(color=SERIES[1], width=1.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=month, y=f["t_nu"], name="ν", showlegend=False, line=dict(color=SERIES[0])), row=2, col=1)
    fig.add_trace(go.Scatter(x=month, y=f["t_tail_dep"], name="λ", showlegend=False, line=dict(color=SERIES[0]),
                             hovertemplate="%{y:.0%}"), row=3, col=1)
    nu_ticks = [2, 5, 10, 20, 50, 100, 200]
    fig.update_yaxes(type="log", tickvals=nu_ticks, ticktext=[str(v) for v in nu_ticks], row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    chart(fig, 640, f"{pair}: monthly copula fits (12-month window)")

    clean = f[f["t_converged"] & ~f["t_nu_at_bound"] & ~f["t_rho_at_bound"]]
    comp = pd.DataFrame({"All months": analysis.fit_comparison(f), "Excluding fits on a bound": analysis.fit_comparison(clean)})
    comp = comp.loc[["months", "t_better_aic_share", "t_better_oos_share", "median_nu", "median_tail_dep"]]
    comp.index = ["Months", "t better by AIC", "t better out of sample", "Median ν", "Median λ"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("t vs Gaussian fit")
        st.dataframe(comp.apply(lambda col: [f"{col.iloc[0]:.0f}", pct(col.iloc[1], 0), pct(col.iloc[2], 0),
                                             f"{col.iloc[3]:.1f}", pct(col.iloc[4], 0)]),
                     use_container_width=True)
    marginal, raw, recal = tail_tables(pair)
    with c2:
        st.subheader("Marginal drift out of sample")
        st.dataframe(pd.DataFrame({"Lower 5% tail": [marginal["u1_lower"], marginal["u2_lower"]],
                                   "Upper 5% tail": [marginal["u1_upper"], marginal["u2_upper"]]},
                                  index=pair.split("-")).style.format("{:.1%}"), use_container_width=True)
        st.caption("Share of days each leg alone lands in its tail; the copula predictions assume exactly 5%.")

    labels = ["Observed, lower tail", "Observed, upper tail", "Predicted, Gaussian", "Predicted, t"]
    keys = ["observed_lower", "observed_upper", "pred_gaussian", "pred_t"]
    fig = go.Figure()
    for k, (label, key) in enumerate(zip(labels, keys)):
        fig.add_trace(go.Bar(x=["Raw", "Marginal drift removed"], y=[raw[key], recal[key]], name=label,
                             marker_color=SERIES[[0, 2, 1, 3][k]], hovertemplate="%{y:.2%}"))
    fig.add_hline(y=0.05**2, line=dict(color=MUTED, dash="dash", width=1), annotation_text="independence")
    fig.update_yaxes(tickformat=".1%")
    chart(fig, 400, f"{pair}: share of out-of-sample days with both legs in their 5% tail", legend_y=-0.18)

# ---------- subperiods ----------

with tab_sub:
    periods = CFG["subperiods"]
    labels = {k: v["label"] for k, v in periods.items()} | {"other": "All other days"}
    if series_ids:
        rets = returns(series_ids)
        sub = pd.DataFrame({name: analysis.subperiod_performance(rets[name].dropna(), periods)["total_return"]
                            for name in rets.columns})
        stress = sub.drop(index="other").dropna(how="all")
        fig = go.Figure()
        for name in stress.columns:
            fig.add_trace(go.Bar(x=[labels[p] for p in stress.index], y=stress[name], name=name,
                                 marker_color=SPEC_COLORS[name], hovertemplate="%{y:.1%}"))
        fig.update_yaxes(tickformat=".0%")
        chart(fig, 440, f"{pair}: total return per stress period (net of {cost} bps)", legend_y=-0.18)
        st.dataframe(sub.rename(index=labels).style.format("{:.1%}", na_rep="–"), use_container_width=True)
        st.caption("Periods were registered in config.yaml before results were computed and are dated by "
                   "external events. Returns are total over each period, not annualized.")

# ---------- trades ----------

with tab_trades:
    spec = st.selectbox("Strategy", specs or [MAIN_NAME], key="trade_spec")
    t = trades(dd.run_id(pair, spec_keys[spec], cost, exposure))
    if t.empty:
        st.info("No trades for this selection.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure(go.Histogram(x=t["holding_days"], xbins=dict(start=-0.5, end=20.5, size=1),
                                         marker_color=SPEC_COLORS[spec]))
            fig.update_xaxes(title="Holding period (trading days)")
            fig.update_yaxes(title="Trades")
            chart(fig, 340, f"{spec}: holding period per trade")
        with c2:
            st.subheader("Exit reasons")
            reasons = t["exit_reason"].str.replace("_", " ").value_counts(normalize=True)
            st.dataframe(reasons.map("{:.1%}".format).rename("Share").rename_axis("Exit reason").to_frame(),
                         use_container_width=True)
            st.metric("Trades", len(t))
            st.metric("Gross win rate", pct((t["gross_return"] > 0).mean(), 0))
        shown = pd.DataFrame({
            "Signal": t["signal_date"].dt.date, "Entry": t["entry_date"].dt.date, "Exit": t["exit_date"].dt.date,
            "Direction": t["direction"], "Exit reason": t["exit_reason"].str.replace("_", " "),
            "Days held": t["holding_days"], "Gross return": t["gross_return"], "Net return": t["net_return"],
        })
        st.dataframe(shown, use_container_width=True, hide_index=True, column_config={
            "Gross return": st.column_config.NumberColumn(format="percent"),
            "Net return": st.column_config.NumberColumn(format="percent"),
        })
        st.caption("direction +1 = long leg 1 / short leg 2. Returns are on \\$1 of NAV.")

# ---------- method ----------

with tab_method:
    st.subheader("Paper vs this project")
    st.markdown("""
| Paper (cited) | This project | Extension |
|---|---|---|
| t-copula, semiparametric fit, conditional-probability signals (α = 5%) | Same signal rule on pre-selected pairs | Gaussian vs t comparison: AIC, out-of-sample log-likelihood, tail calibration |
| S&P 100, 1990–2014, copula-based pair selection | Free data (yfinance), 2006–2025, monthly walk-forward refit | z-score spread baseline |
| Quantile take-profit / stop-loss exit | Second, pre-registered specification | Probability-reversion exit as main specification |
| 50 bps round-trip cost, one-day waiting rule | Same, as main specification | Cost sensitivity, Newey-West tests, block-bootstrap Sharpe intervals, stress periods |
""")
    st.subheader("Limitations")
    st.markdown("""
- One price path per pair, four pairs chosen ex ante; p-values are not adjusted for multiple testing.
- Close-to-close execution with no slippage model; short-selling assumed available with no borrow fee.
- Out-of-sample margins drift away from the training ECDF; the drift-corrected tail check is a diagnostic only.
- The paper's quantile exit occasionally produces degenerate thresholds; it was kept as specified.
- Free data: yfinance prices can be revised; see the data manifest below.
""")
    st.subheader("Data behind these results")
    manifest_path = ROOT / "data" / "manifest.json"
    if manifest_path.exists():
        manifest = pd.DataFrame(json.loads(manifest_path.read_text())).T
        st.dataframe(manifest[["rows", "start", "end", "downloaded_at"]], use_container_width=True)
        st.caption("Checksums are in data/manifest.json; `python -m scripts.verify_data` compares a fresh download.")
    st.caption(f"Full analysis: notebooks/analysis.ipynb. Seed {CFG['seed']}. "
               f"Results database: {DB.relative_to(ROOT)}.")
