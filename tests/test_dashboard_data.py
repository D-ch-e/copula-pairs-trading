from pathlib import Path

import pandas as pd
import pytest

from cpt import dashboard_data as dd
from cpt.config import PROJECT_ROOT, db_path, load_config
from cpt.db import connect, init_db, upsert

PAIR = "GLD-SLV"


def _metrics(rid, **values):
    return pd.DataFrame({"run_id": rid, "metric": list(values), "value": list(values.values())})


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "d.sqlite")
    init_db(c)
    runs = [
        (f"{PAIR}|copula|t|probability|50", "copula", "t", "probability", 50, 2.0),
        (f"{PAIR}|copula|t|probability|0", "copula", "t", "probability", 0, 2.0),
        (f"{PAIR}|copula|t|probability|50|gross1", "copula", "t", "probability", 50, 1.0),
        (f"{PAIR}|zscore|None|None|50", "zscore", None, None, 50, 2.0),
        (f"{PAIR}|buy_hold|GLD", "buy_hold", "GLD", None, 0, 1.0),
    ]
    upsert(c, "experiment_runs", pd.DataFrame(
        [{"run_id": r, "pair": PAIR, "model": m, "family": f, "exit_rule": e, "cost_bps": b,
          "gross_exposure": g, "created_at": "2026-01-01"} for r, m, f, e, b, g in runs]))
    for k, (rid, *_) in enumerate(runs):
        upsert(c, "run_metrics", _metrics(rid, sharpe=-1.0 + k, ann_return=-0.08 + k / 100,
                                          max_drawdown=0.5, breakeven_bps=-17.0))
    dates = ["2020-01-02", "2020-01-03"]
    for rid, ret in [(runs[0][0], [0.01, -0.02]), (runs[4][0], [0.03, 0.0])]:
        upsert(c, "strategy_returns", pd.DataFrame({"run_id": rid, "date": dates, "ret": ret, "position": 0}))
    upsert(c, "copula_fits", pd.DataFrame({
        "pair": PAIR, "month": ["2020-01", "2020-02"], "t_oos_loglik": [2.0, 0.0], "gaussian_oos_loglik": [1.0, 1.0],
        "t_converged": [1, 1], "t_nu_at_bound": [0, 1], "t_rho_at_bound": [0, 0]}))
    return c


def test_run_id_matches_runner_format():
    assert dd.run_id(PAIR, "copula|t|probability", 50) == f"{PAIR}|copula|t|probability|50"
    assert dd.run_id(PAIR, "copula|t|probability", 50, 1.0) == f"{PAIR}|copula|t|probability|50|gross1"


def test_load_runs_pivots_metrics_and_names_specs(conn):
    runs = dd.load_runs(conn).set_index("run_id")
    assert runs.loc[f"{PAIR}|copula|t|probability|50", "spec"] == "Copula t (main)"
    assert runs.loc[f"{PAIR}|zscore|None|None|50", "spec"] == "z-score baseline"
    assert runs.loc[f"{PAIR}|buy_hold|GLD", "spec"] == "GLD"
    assert {"sharpe", "ann_return", "max_drawdown"} <= set(runs.columns)


def test_load_returns_labels_columns(conn):
    df = dd.load_returns(conn, {"main": f"{PAIR}|copula|t|probability|50", "GLD": f"{PAIR}|buy_hold|GLD"})
    assert list(df.columns) == ["main", "GLD"]
    assert df["main"].tolist() == [0.01, -0.02]


def test_load_fits_casts_flags_to_bool(conn):
    fits = dd.load_fits(conn, PAIR)
    assert fits["t_nu_at_bound"].dtype == bool and fits["t_nu_at_bound"].tolist() == [False, True]


def test_headline_kpis(conn):
    runs = dd.load_runs(conn)
    kpi = dd.headline_kpis(runs, {PAIR: dd.load_fits(conn, PAIR)}, PAIR, "copula|t|probability", 50)
    assert kpi["main_sharpe"] == -1.0 and kpi["main_gross_sharpe"] == 0.0
    assert kpi["gld_sharpe"] == 3.0
    assert kpi["t_better_oos_min"] == kpi["t_better_oos_max"] == 0.5


REAL_DB = db_path(load_config())


@pytest.mark.skipif(not Path(REAL_DB).exists(), reason="needs data/pairs.sqlite from run_experiments")
def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert len(app.metric) >= 4

    app.sidebar.radio[0].set_value(1.0).run()  # 100% gross exposure, main pair
    assert not app.exception

    app.sidebar.selectbox[0].set_value("KO-PEP").run()  # robustness pair: no exposure switch, no benchmarks
    assert not app.exception and not app.sidebar.radio

    app.sidebar.multiselect[0].set_value([]).run()  # nothing selected
    assert not app.exception
