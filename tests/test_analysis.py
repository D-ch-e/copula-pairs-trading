import numpy as np
import pandas as pd
import pytest

from cpt import analysis, copula


def test_joint_tail_prob_independence_and_tail_dependence():
    q = 0.05
    assert analysis.joint_tail_prob("gaussian", 0.0, q) == pytest.approx(q**2, rel=1e-3)
    # t-copula has tail dependence even at rho = 0
    assert analysis.joint_tail_prob("t", 0.0, q, nu=3) > q**2
    assert analysis.joint_tail_prob("t", 0.6, q, nu=4) > analysis.joint_tail_prob("gaussian", 0.6, q)


def test_joint_tail_prob_matches_simulation():
    u1, u2 = copula.sample_t(200_000, rho=0.5, nu=4, seed=3)
    simulated = np.mean((u1 < 0.05) & (u2 < 0.05))
    assert analysis.joint_tail_prob("t", 0.5, 0.05, nu=4) == pytest.approx(simulated, rel=0.05)


def test_dependence_summary_on_t_data():
    u1, u2 = copula.sample_t(5000, rho=0.6, nu=3, seed=4)
    r = pd.DataFrame({"a": np.log(u1 / (1 - u1)), "b": np.log(u2 / (1 - u2))})
    out = analysis.dependence_summary(r)
    assert out["pearson_implied_by_kendall"] == pytest.approx(0.6, abs=0.03)
    assert out["joint_lower_tail"] > out["independence_tail"]


PERIODS = {"a": {"start": "2020-01-02", "end": "2020-01-03"}, "b": {"start": "2020-01-07", "end": "2020-01-08"}}


def test_period_masks_partition_days():
    idx = pd.bdate_range("2020-01-01", periods=8)
    masks = analysis.period_masks(idx, PERIODS)
    total = sum(m.astype(int) for m in masks.values())
    assert (total == 1).all()
    assert masks["a"].sum() == 2 and masks["b"].sum() == 2 and masks["other"].sum() == 4


def test_subperiod_performance_total_return():
    idx = pd.bdate_range("2020-01-01", periods=8)
    r = pd.Series([0.0, 0.10, -0.10, 0.0, 0.05, 0.05, 0.0, 0.0], index=idx)
    out = analysis.subperiod_performance(r, PERIODS)
    assert out.loc["a", "total_return"] == pytest.approx(1.1 * 0.9 - 1)
    assert out.loc["b", "total_return"] == pytest.approx(1.05**2 - 1)
    assert out.loc["a", "max_drawdown"] == pytest.approx(0.10)


def test_subperiod_copula_uses_month_start():
    fits = pd.DataFrame({"month": ["2020-01", "2020-02"], "t_rho": [0.5, 0.7], "t_nu": [4, 8],
                         "t_tail_dep": [0.2, 0.1], "t_oos_loglik": [1, 0], "gaussian_oos_loglik": [0, 1]})
    periods = {"jan": {"start": "2020-01-01", "end": "2020-01-31"}}
    out = analysis.subperiod_copula(fits, periods)
    assert out.loc["jan", "months"] == 1 and out.loc["jan", "median_t_nu"] == 4
    assert out.loc["other", "t_better_oos_share"] == 0
