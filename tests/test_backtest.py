import copy

import numpy as np
import pandas as pd
import pytest

from cpt.backtest import paper_quantile_thresholds, run_backtest
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


@pytest.mark.parametrize("kwargs", [
    {},
    {"exit_rule": "paper_quantile"},
    {"model": "zscore"},
])
def test_no_look_ahead(cfg, kwargs):
    prices = _synthetic_prices()
    cutoff = prices.index[350]
    shocked = prices.copy()
    shocked.loc[shocked.index > cutoff] *= np.linspace(1.5, 0.5, (shocked.index > cutoff).sum())[:, None]

    base = run_backtest(prices, cfg, **kwargs)["daily"]
    alt = run_backtest(shocked, cfg, **kwargs)["daily"]
    pd.testing.assert_frame_equal(base[base.index <= cutoff], alt[alt.index <= cutoff])


def test_paper_quantile_thresholds_hand_example():
    p1 = np.array([100, 101, 102, 103, 104.0])
    p2 = np.array([100, 100, 100, 100, 100.0])
    signal = np.array([1, 0, 0, 0, 0])
    tp, sl, n = paper_quantile_thresholds(p1, p2, signal, horizon=2, tp_q=1.0, sl_q=0.0)
    # only j=0 has a full 2-day path: CR = [0.01, 0.02]
    assert n == 1
    assert tp == pytest.approx(0.02) and sl == pytest.approx(0.01)


def test_zscore_baseline_trades_and_reverts(cfg):
    out = run_backtest(_synthetic_prices(seed=5), cfg, model="zscore")
    assert len(out["trades"]) > 0
    assert out["fits"] is None
    assert out["daily"]["ret"].sum() == pytest.approx(out["trades"]["net_return"].sum())


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
