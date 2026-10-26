"""Read backtest results from SQLite for the dashboard. Pure functions of a connection."""
import sqlite3

import pandas as pd

SPEC_NAMES = {
    "copula|t|probability": "Copula t (main)",
    "copula|gaussian|probability": "Copula Gaussian",
    "copula|t|paper_quantile": "Copula t, paper exit",
    "zscore|None|None": "z-score baseline",
}
FIT_FLAGS = ["t_converged", "t_nu_at_bound", "t_rho_at_bound"]


def run_id(pair: str, spec_key: str, cost_bps: float, gross_exposure: float = 2.0) -> str:
    """run_id as written by scripts/run_experiments.py."""
    rid = f"{pair}|{spec_key}|{cost_bps:g}"
    return rid if gross_exposure == 2.0 else f"{rid}|gross{gross_exposure:g}"


def load_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per run: run attributes, every metric as a column, and a display name `spec`."""
    runs = pd.read_sql("SELECT * FROM experiment_runs", conn)
    metrics = pd.read_sql("SELECT * FROM run_metrics", conn).pivot(index="run_id", columns="metric", values="value")
    runs = runs.join(metrics, on="run_id")
    key = runs["model"] + "|" + runs["family"].fillna("None") + "|" + runs["exit_rule"].fillna("None")
    runs["spec"] = key.map(SPEC_NAMES).where(runs["model"] != "buy_hold", runs["family"])
    return runs


def load_returns(conn: sqlite3.Connection, run_ids: dict[str, str]) -> pd.DataFrame:
    """Daily returns, one column per label in `run_ids` ({label: run_id})."""
    placeholders = ", ".join("?" for _ in run_ids)
    df = pd.read_sql(f"SELECT run_id, date, ret FROM strategy_returns WHERE run_id IN ({placeholders})",
                     conn, params=list(run_ids.values()), parse_dates=["date"])
    wide = df.pivot(index="date", columns="run_id", values="ret")
    return wide[list(run_ids.values())].set_axis(list(run_ids), axis=1)


def load_trades(conn: sqlite3.Connection, rid: str) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM trades WHERE run_id = ? ORDER BY trade_no", conn, params=[rid],
                       parse_dates=["signal_date", "entry_date", "exit_date"])


def load_fits(conn: sqlite3.Connection, pair: str) -> pd.DataFrame:
    fits = pd.read_sql("SELECT * FROM copula_fits WHERE pair = ? ORDER BY month", conn, params=[pair])
    return fits.astype({flag: bool for flag in FIT_FLAGS})


def load_copula_daily(conn: sqlite3.Connection, pair: str) -> pd.DataFrame:
    return pd.read_sql("SELECT date, u1, u2, h1, h2, signal FROM copula_daily WHERE pair = ? ORDER BY date",
                       conn, params=[pair], parse_dates=["date"]).set_index("date")


def headline_kpis(runs: pd.DataFrame, fits: dict[str, pd.DataFrame], main_pair: str,
                  main_key: str, cost_bps: float) -> dict:
    """Numbers for the dashboard header: main spec vs buy-and-hold GLD, and copula fit quality."""
    by_id = runs.set_index("run_id")
    main = by_id.loc[run_id(main_pair, main_key, cost_bps)]
    gross = by_id.loc[run_id(main_pair, main_key, 0)]
    gld = by_id.loc[f"{main_pair}|buy_hold|GLD"]
    oos_share = {p: (f["t_oos_loglik"] > f["gaussian_oos_loglik"]).mean() for p, f in fits.items()}
    return {
        "main_sharpe": main["sharpe"], "main_ann_return": main["ann_return"],
        "main_max_drawdown": main["max_drawdown"], "main_gross_sharpe": gross["sharpe"],
        "breakeven_bps": main["breakeven_bps"],
        "gld_sharpe": gld["sharpe"], "gld_ann_return": gld["ann_return"], "gld_max_drawdown": gld["max_drawdown"],
        "t_better_oos_min": min(oos_share.values()), "t_better_oos_max": max(oos_share.values()),
    }
