import copy

import numpy as np
import pandas as pd
import pytest

from cpt import copula
from cpt.config import load_config
from cpt.db import connect, init_db, upsert
from scripts.export_tableau import checklist, export_all

PAIR = "GLD-SLV"
MAIN = f"{PAIR}|copula|t|probability|50"
GLD = f"{PAIR}|buy_hold|GLD"


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["pairs"]["robustness"] = []
    return c


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.sqlite")
    init_db(c)
    upsert(c, "experiment_runs", pd.DataFrame([
        {"run_id": MAIN, "pair": PAIR, "model": "copula", "family": "t", "exit_rule": "probability",
         "cost_bps": 50, "gross_exposure": 2.0, "created_at": "2026-01-01"},
        {"run_id": GLD, "pair": PAIR, "model": "buy_hold", "family": "GLD", "exit_rule": None,
         "cost_bps": 0, "gross_exposure": 1.0, "created_at": "2026-01-01"}]))
    for rid in [MAIN, GLD]:
        upsert(c, "run_metrics", pd.DataFrame({"run_id": rid, "metric": ["sharpe", "ann_return", "max_drawdown", "breakeven_bps"],
                                               "value": [-1.0, -0.08, 0.5, -17.0]}))
    dates = pd.bdate_range("2020-01-01", "2020-02-28")
    rets = {MAIN: np.r_[-0.10, np.full(len(dates) - 1, 0.001)], GLD: np.random.default_rng(1).normal(0.001, 0.01, len(dates))}
    for rid, r in rets.items():
        upsert(c, "strategy_returns", pd.DataFrame({"run_id": rid, "date": dates.strftime("%Y-%m-%d"), "ret": r, "position": 0}))
    upsert(c, "trades", pd.DataFrame([{"run_id": MAIN, "trade_no": 1, "entry_date": "2020-01-02", "exit_date": "2020-01-03",
                                       "direction": 1, "exit_reason": "reversion", "holding_days": 1,
                                       "gross_return": 0.01, "net_return": 0.005}]))
    upsert(c, "copula_fits", pd.DataFrame({
        "pair": PAIR, "month": ["2020-01", "2020-02"], "gaussian_rho": 0.6, "gaussian_aic": -10.0, "t_rho": 0.6,
        "t_nu": 5.0, "t_aic": -12.0, "t_tail_dep": 0.2, "gaussian_oos_loglik": [1.0, 2.0], "t_oos_loglik": [2.0, 1.0],
        "t_converged": 1, "t_nu_at_bound": 0, "t_rho_at_bound": 0}))
    u1, u2 = copula.sample_t(len(dates), rho=0.6, nu=5, seed=0)
    upsert(c, "copula_daily", pd.DataFrame({"pair": PAIR, "date": dates.strftime("%Y-%m-%d"), "u1": u1, "u2": u2,
                                            "h1": 0.5, "h2": 0.5, "signal": 0}))
    return c


def test_export_writes_all_files(conn, cfg, tmp_path):
    out = tmp_path / "out"
    tables = export_all(conn, cfg, out, bootstrap_reps=50)
    for name in tables:
        assert (out / f"{name}.csv").exists()
    assert {"runs", "strategy_daily", "monthly_returns", "trades", "copula_fits",
            "subperiods", "tail_calibration", "significance"} == set(tables)


def test_benchmarks_only_at_monthly_grain(conn, cfg, tmp_path):
    tables = export_all(conn, cfg, tmp_path / "out", bootstrap_reps=50)
    assert GLD not in set(tables["strategy_daily"]["run_id"])  # the only daily-grain file
    assert GLD in set(tables["monthly_returns"]["run_id"])
    assert len(tables["monthly_returns"].query("run_id == @GLD")) == 2


def test_wealth_drawdown_and_monthly_compounding(conn, cfg, tmp_path):
    tables = export_all(conn, cfg, tmp_path / "out", bootstrap_reps=50)
    d = tables["strategy_daily"].query("run_id == @MAIN").sort_values("date")
    np.testing.assert_allclose(d["wealth"], (1 + d["ret"]).cumprod())
    assert d["drawdown"].iloc[0] == pytest.approx(-0.10)  # first-day loss counts against starting NAV
    m = tables["monthly_returns"].query("run_id == @MAIN").sort_values("month")
    jan = d[d["date"].dt.month == 1]["ret"]
    assert m["ret"].iloc[0] == pytest.approx((1 + jan).prod() - 1)


def test_export_is_reproducible(conn, cfg, tmp_path):
    export_all(conn, cfg, tmp_path / "a", bootstrap_reps=50)
    export_all(conn, cfg, tmp_path / "b", bootstrap_reps=50)
    for f in (tmp_path / "a").glob("*.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(f), pd.read_csv(tmp_path / "b" / f.name))


def test_checklist_and_flags(conn, cfg, tmp_path):
    tables = export_all(conn, cfg, tmp_path / "out", bootstrap_reps=50)
    runs = tables["runs"].set_index("run_id")
    assert runs.loc[MAIN, "is_main_spec"] and runs.loc[GLD, "is_benchmark"]
    lines = checklist(tables, cfg)
    assert "1 trades" in lines[0] and "50%" in lines[3]
    tail = tables["tail_calibration"]
    assert set(tail["version"]) == {"Raw", "Marginal drift removed", "Marginal exceedance"}
