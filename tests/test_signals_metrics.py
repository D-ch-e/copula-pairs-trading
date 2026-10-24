import numpy as np
import pandas as pd
import pytest

from cpt import metrics
from cpt.signals import classify_regions


def test_classify_regions():
    h1 = pd.Series([0.01, 0.99, 0.01, 0.5])
    h2 = pd.Series([0.99, 0.01, 0.50, 0.5])
    assert classify_regions(h1, h2, alpha=0.05).tolist() == [1, -1, 0, 0]


def test_max_drawdown_hand_example():
    # wealth: 1.1, 0.99, 1.089 -> peak 1.1, trough 0.99 -> 10% drawdown
    r = pd.Series([0.10, -0.10, 0.10])
    assert metrics.max_drawdown(r) == pytest.approx(0.10)


def test_historical_var_cvar():
    r = pd.Series(np.arange(-50, 50) / 1000)  # -0.050 .. 0.049
    var, cvar = metrics.historical_var_cvar(r, 0.95)
    assert var == pytest.approx(r.quantile(0.05))
    assert cvar <= var < 0


def test_sharpe_sortino_and_rf():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    assert metrics.sharpe_ratio(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    rf = pd.Series(0.0002, index=r.index)
    ex = r - 0.0002
    assert metrics.sharpe_ratio(r, rf) == pytest.approx(ex.mean() / ex.std() * np.sqrt(252))
    assert metrics.sortino_ratio(pd.Series([0.02, -0.01, 0.03, -0.01])) > 0


def test_annualized_return_constant_growth():
    r = pd.Series([0.001] * 252)
    assert metrics.annualized_return(r) == pytest.approx(1.001**252 - 1)
