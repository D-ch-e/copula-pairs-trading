"""Run every pre-registered spec on every pair and store results in SQLite.

Specs per pair: copula-t / copula-gaussian with the probability exit (main),
copula-t with the paper's quantile exit, and the z-score baseline; each at
every cost in costs.sensitivity_bps. Buy-and-hold benchmarks for the main pair.
"""
from datetime import datetime, timezone

import pandas as pd

from cpt.backtest import copula_frame, simulate, zscore_frame
from cpt.config import db_path, load_config
from cpt.db import connect, init_db, load_adj_close, upsert
from cpt.metrics import performance_summary

RESULT_TABLES = ["experiment_runs", "run_metrics", "strategy_returns", "trades", "copula_fits"]
FIT_COLUMNS = ["gaussian_rho", "gaussian_aic", "t_rho", "t_nu", "t_aic", "t_tail_dep",
               "gaussian_oos_loglik", "t_oos_loglik", "t_converged", "t_nu_at_bound", "t_rho_at_bound"]


def _replace_run(conn, run: dict, daily: pd.Series, position: pd.Series | None,
                 trades: pd.DataFrame, rf: pd.Series) -> dict:
    run_id = run["run_id"]
    for table in ["experiment_runs", "run_metrics", "strategy_returns", "trades"]:
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))

    metrics = performance_summary(daily, rf)
    if not trades.empty:
        metrics.update({
            "n_trades": len(trades),
            "avg_holding_days": trades["holding_days"].mean(),
            "win_rate": (trades["net_return"] > 0).mean(),
            # round-trip cost (quoted per pair at $1 per leg) at which the average trade nets zero
            "breakeven_bps": (trades["gross_return"] / trades["leg_notional"]).mean() * 1e4,
        })

    upsert(conn, "experiment_runs", pd.DataFrame([{
        **run, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}]))
    upsert(conn, "run_metrics", pd.DataFrame(
        {"run_id": run_id, "metric": list(metrics), "value": [float(v) for v in metrics.values()]}))
    upsert(conn, "strategy_returns", pd.DataFrame({
        "run_id": run_id, "date": daily.index.strftime("%Y-%m-%d"), "ret": daily.to_numpy(),
        "position": None if position is None else position.to_numpy()}))
    if not trades.empty:
        t = trades.copy()
        for col in ["signal_date", "entry_date", "exit_date"]:
            t[col] = pd.to_datetime(t[col]).dt.strftime("%Y-%m-%d")
        upsert(conn, "trades", t.assign(run_id=run_id, trade_no=range(1, len(t) + 1)))
    return metrics


def main() -> None:
    cfg = load_config()
    s = {**cfg["strategy"], "paper_quantile": cfg["paper_quantile"]}
    costs = cfg["costs"]["sensitivity_bps"]
    pairs = [cfg["pairs"]["main"], *cfg["pairs"]["robustness"]]

    with connect(db_path(cfg)) as conn:
        for table in RESULT_TABLES:  # derived data: rebuilt from the current schema on every run
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        init_db(conn)
        gross, gross_cmp = cfg["capital"]["gross_exposure"], cfg["capital"]["comparison_gross_exposure"]
        rf = pd.read_sql("SELECT date, rf_daily FROM risk_free", conn,
                         parse_dates=["date"]).set_index("date")["rf_daily"]

        for leg1, leg2 in pairs:
            pair = f"{leg1}-{leg2}"
            prices = load_adj_close(conn, [leg1, leg2])
            specs = {}
            for family, exit_rule in [("t", "probability"), ("gaussian", "probability"), ("t", "paper_quantile")]:
                frame, fits = copula_frame(prices, s, family, exit_rule)
                specs[("copula", family, exit_rule)] = frame
                if (family, exit_rule) == ("t", "probability"):
                    upsert(conn, "copula_fits", fits[["month", *FIT_COLUMNS]].assign(pair=pair))
            specs[("zscore", None, None)] = zscore_frame(prices, s, cfg["zscore"])

            for (model, family, exit_rule), frame in specs.items():
                for cost in costs:
                    daily, trades = simulate(prices, frame, s, cost, gross)
                    run = {"run_id": f"{pair}|{model}|{family}|{exit_rule}|{cost}", "pair": pair,
                           "model": model, "family": family, "exit_rule": exit_rule, "cost_bps": cost,
                           "gross_exposure": gross}
                    _replace_run(conn, run, daily["ret"], daily["position"], trades, rf)

            if [leg1, leg2] == cfg["pairs"]["main"]:
                frame = specs[("copula", s["family"], s["exit_rule"])]
                for cost in costs:
                    daily, trades = simulate(prices, frame, s, cost, gross_cmp)
                    run = {"run_id": f"{pair}|copula|{s['family']}|{s['exit_rule']}|{cost}|gross{gross_cmp:g}",
                           "pair": pair, "model": "copula", "family": s["family"], "exit_rule": s["exit_rule"],
                           "cost_bps": cost, "gross_exposure": gross_cmp}
                    _replace_run(conn, run, daily["ret"], daily["position"], trades, rf)
            print(f"{pair}: {len(specs) * len(costs)} runs, "
                  f"{frame.index.min().date()} -> {frame.index.max().date()}")

            if [leg1, leg2] == cfg["pairs"]["main"]:
                period = frame.index
                bench = load_adj_close(conn, sorted({*cfg["benchmarks"], leg1, leg2})).pct_change()
                bench = bench.loc[period]
                series = {t: bench[t] for t in cfg["benchmarks"]}
                series[f"50/50 {leg1}+{leg2}"] = 0.5 * bench[leg1] + 0.5 * bench[leg2]
                for name, r in series.items():
                    run = {"run_id": f"{pair}|buy_hold|{name}", "pair": pair, "model": "buy_hold",
                           "family": name, "exit_rule": None, "cost_bps": 0,
                           "gross_exposure": 1.0}
                    _replace_run(conn, run, r, None, pd.DataFrame(), rf)
        conn.commit()


if __name__ == "__main__":
    main()
