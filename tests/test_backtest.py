import copy

import numpy as np
import pandas as pd
import pytest

from cpt.backtest import run_backtest
from cpt.config import load_config


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["strategy"]["estimation_window_days"] = 120
    return c


def _synthetic_prices(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.standard_t(4, n) * 0.01
    r1 = common + rng.normal(0, 0.006, n)
    r2 = common + rng.normal(0, 0.006, n)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"A": 100 * np.exp(np.cumsum(r1)), "B": 50 * np.exp(np.cumsum(r2))}, index=dates)


def test_no_look_ahead(cfg):
    prices = _synthetic_prices()
    cutoff = prices.index[350]
    shocked = prices.copy()
    shocked.loc[shocked.index > cutoff] *= np.linspace(1.5, 0.5, (shocked.index > cutoff).sum())[:, None]

    base = run_backtest(prices, cfg)["daily"]
    alt = run_backtest(shocked, cfg)["daily"]
    before = base.index <= cutoff
    pd.testing.assert_frame_equal(base[before], alt[alt.index <= cutoff])


def test_costs_reduce_returns_by_cost_per_trade(cfg):
    prices = _synthetic_prices(seed=1)
    gross = run_backtest(prices, cfg, cost_bps=0)
    net = run_backtest(prices, cfg, cost_bps=50)
    n_trades = len(net["trades"])
    assert n_trades > 0
    diff = gross["daily"]["ret"].sum() - net["daily"]["ret"].sum()
    assert diff == pytest.approx(n_trades * 0.005)


def test_trade_returns_match_daily_returns(cfg):
    out = run_backtest(_synthetic_prices(seed=2), cfg)
    assert out["daily"]["ret"].sum() == pytest.approx(out["trades"]["net_return"].sum())


def test_execution_lag_delays_entry(cfg):
    out = run_backtest(_synthetic_prices(seed=3), cfg)
    trades = out["trades"]
    dates = list(out["daily"].index)
    gaps = [dates.index(e) - dates.index(s) for s, e in zip(trades["signal_date"], trades["entry_date"])]
    assert set(gaps) == {cfg["strategy"]["execution_lag_days"]}


def test_gaussian_family_runs(cfg):
    c = copy.deepcopy(cfg)
    out = run_backtest(_synthetic_prices(seed=4), c, family="gaussian")
    assert {"t_aic", "gaussian_aic", "t_oos_loglik"} <= set(out["fits"].columns)
