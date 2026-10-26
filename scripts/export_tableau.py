"""Export backtest results as CSV files for the Tableau Public dashboard.

Long-format tables keyed by run_id (see docs/tableau_data_dictionary.md). Tableau
Public makes all data public, so buy-and-hold benchmarks (derived from Yahoo prices)
are exported as monthly returns only; daily rows are strategy returns we computed.

Usage: python -m scripts.export_tableau   (after scripts.run_experiments)
"""
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from cpt import analysis, inference, results
from cpt.config import PROJECT_ROOT, db_path, load_config
from cpt.db import connect
from cpt.metrics import drawdown

OUT_DIR = PROJECT_ROOT / "exports" / "tableau"
DICTIONARY = PROJECT_ROOT / "docs" / "tableau_data_dictionary.md"
RUN_COLUMNS = ["run_id", "pair", "spec", "model", "family", "exit_rule", "cost_bps", "gross_exposure"]


def _main_key(cfg: dict) -> str:
    return f"copula|{cfg['strategy']['family']}|{cfg['strategy']['exit_rule']}"


def runs_table(conn: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    runs = results.load_runs(conn).drop(columns="created_at")
    main_id = results.run_id("-".join(cfg["pairs"]["main"]), _main_key(cfg), cfg["costs"]["round_trip_bps"])
    runs["is_main_spec"] = runs["run_id"] == main_id
    runs["is_benchmark"] = runs["model"] == "buy_hold"
    return runs


def _with_wealth(df: pd.DataFrame, key: str, date: str) -> pd.DataFrame:
    """Add wealth (growth of $1) and drawdown per `key`, ordered by `date`."""
    df = df.sort_values([key, date])
    df["wealth"] = df.groupby(key)["ret"].transform(lambda r: (1 + r).cumprod())
    df["drawdown"] = df.groupby(key)["ret"].transform(drawdown)
    return df


def daily_returns(conn: sqlite3.Connection) -> pd.DataFrame:
    """All daily returns, strategies and benchmarks; benchmark dailies are used internally only."""
    return pd.read_sql("SELECT run_id, date, ret FROM strategy_returns", conn, parse_dates=["date"])


def strategy_daily(daily: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    strategy_ids = runs.loc[~runs["is_benchmark"], "run_id"]
    return _with_wealth(daily[daily["run_id"].isin(strategy_ids)].copy(), "run_id", "date")


def monthly_returns(daily: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    df = daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp())
    monthly = df.groupby(["run_id", "month"])["ret"].apply(lambda r: (1 + r).prod() - 1).reset_index()
    monthly = _with_wealth(monthly, "run_id", "month")
    return monthly.merge(runs[["run_id", "pair", "spec", "cost_bps", "gross_exposure", "is_benchmark"]], on="run_id")


def subperiods_table(daily: pd.DataFrame, runs: pd.DataFrame, periods: dict) -> pd.DataFrame:
    labels = {k: v["label"] for k, v in periods.items()} | {"other": "All other days"}
    frames = []
    for rid, df in daily.groupby("run_id"):
        perf = analysis.subperiod_performance(df.set_index("date")["ret"], periods)
        frames.append(perf.rename_axis("period").reset_index().assign(run_id=rid))
    out = pd.concat(frames)
    out["period_label"] = out["period"].map(labels)
    order = {p: k for k, p in enumerate(labels)}
    out["period_order"] = out["period"].map(order)
    return out.merge(runs[RUN_COLUMNS[:4] + ["cost_bps", "gross_exposure", "is_benchmark"]], on="run_id")


def tail_table(conn: sqlite3.Connection, pairs: list[str], q: float = 0.05) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        fits, daily = results.load_fits(conn, pair), results.load_copula_daily(conn, pair)
        raw = analysis.oos_tail_calibration(fits, daily, q).drop(columns="month").sum()
        raw = raw.drop("days") / raw["days"]
        recal = pd.Series(analysis.recalibrated_tail_calibration(fits, daily, q)).drop("days")
        marginal = pd.Series(analysis.marginal_exceedance(daily, q))
        for version, values in [("Raw", raw), ("Marginal drift removed", recal), ("Marginal exceedance", marginal)]:
            rows += [{"pair": pair, "version": version, "measure": m, "value": float(v)} for m, v in values.items()]
    return pd.DataFrame(rows)


def significance_table(daily: pd.DataFrame, runs: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for rid, df in daily.groupby("run_id"):
        r = df.sort_values("date")["ret"]
        nw, ci = inference.newey_west_mean_test(r), inference.bootstrap_ci(r, reps=reps, seed=seed)
        rows.append({"run_id": rid, "ann_mean": nw["mean"] * 252, "t_nw": nw["t_nw"], "p_nw": nw["p_nw"],
                     "sharpe_raw": ci["estimate"], "sharpe_ci_low": ci["ci_low"], "sharpe_ci_high": ci["ci_high"]})
    return pd.DataFrame(rows).merge(runs[RUN_COLUMNS], on="run_id")


def export_all(conn: sqlite3.Connection, cfg: dict, out_dir: Path, bootstrap_reps: int = 2000) -> dict:
    pairs = ["-".join(cfg["pairs"]["main"]), *("-".join(p) for p in cfg["pairs"]["robustness"])]
    runs = runs_table(conn, cfg)
    daily = daily_returns(conn)
    tables = {
        "runs": runs,
        "strategy_daily": strategy_daily(daily, runs),
        "monthly_returns": monthly_returns(daily, runs),
        "trades": pd.read_sql("SELECT * FROM trades", conn),
        "copula_fits": pd.read_sql("SELECT * FROM copula_fits", conn).assign(
            month=lambda f: pd.PeriodIndex(f["month"], freq="M").to_timestamp()),
        "subperiods": subperiods_table(daily, runs, cfg["subperiods"]),
        "tail_calibration": tail_table(conn, pairs),
        "significance": significance_table(daily, runs, bootstrap_reps, cfg["seed"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    if DICTIONARY.exists():
        shutil.copy(DICTIONARY, out_dir / "data_dictionary.md")
    return tables


def checklist(tables: dict, cfg: dict) -> list[str]:
    """Numbers the finished dashboard must match."""
    runs = tables["runs"].set_index("run_id")
    main = runs[runs["is_main_spec"]].iloc[0]
    pair = main["pair"]
    gld = runs.loc[f"{pair}|buy_hold|GLD"]
    main_trades = tables["trades"][tables["trades"]["run_id"] == main.name]
    fits = tables["copula_fits"]
    oos = fits.groupby("pair").apply(lambda f: (f["t_oos_loglik"] > f["gaussian_oos_loglik"]).mean(),
                                     include_groups=False)
    return [
        f"Main spec ({main.name}): net Sharpe {main['sharpe']:.2f}, annual return {main['ann_return']:.1%}, "
        f"max drawdown {main['max_drawdown']:.1%}, {len(main_trades)} trades",
        f"Buy-and-hold GLD (daily, from run_metrics): annual return {gld['ann_return']:.1%}, Sharpe {gld['sharpe']:.2f}",
        f"Break-even cost: {main['breakeven_bps']:.0f} bps",
        f"t beats Gaussian out of sample: {oos.min():.0%}–{oos.max():.0%} of months",
    ]


def main() -> None:
    cfg = load_config()
    with connect(db_path(cfg)) as conn:
        tables = export_all(conn, cfg, OUT_DIR)
    for name, df in tables.items():
        print(f"{name}.csv: {len(df):,} rows")
    print("\nThe dashboard should show:")
    print("\n".join(f"  - {line}" for line in checklist(tables, cfg)))


if __name__ == "__main__":
    main()
