import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cpt import inference


def test_newey_west_matches_plain_t_for_zero_lags():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.001, 0.01, 2000))
    out = inference.newey_west_mean_test(x, lags=0)
    plain = stats.ttest_1samp(x, 0.0)
    # HAC with 0 lags uses a 1/n variance instead of 1/(n-1)
    assert out["t_nw"] == pytest.approx(plain.statistic, rel=1e-3)


def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 0.01, 3000)
    x = pd.Series(np.convolve(e, np.ones(10) / 10, mode="valid"))  # MA(9): strong autocorrelation
    assert inference.newey_west_mean_test(x, lags=15)["se_nw"] > inference.newey_west_mean_test(x, lags=0)["se_nw"]


def test_bootstrap_indices_are_valid_and_blocky():
    rng = np.random.default_rng(2)
    idx = inference.stationary_bootstrap_indices(1000, block=20, rng=rng)
    assert idx.min() >= 0 and idx.max() < 1000
    consecutive = np.mean(np.diff(idx) == 1)
    assert consecutive > 0.9  # mean block length 20 -> ~95% of steps continue the block


def test_bootstrap_sharpe_ci_covers_truth_for_iid_data():
    rng = np.random.default_rng(3)
    true_sharpe = 1.0
    x = pd.Series(rng.normal(true_sharpe * 0.01 / np.sqrt(252), 0.01, 252 * 20))
    ci = inference.bootstrap_ci(x, reps=500, seed=4)
    assert ci["ci_low"] < true_sharpe < ci["ci_high"]


def test_paired_difference_of_identical_series_is_zero():
    x = pd.Series(np.random.default_rng(5).normal(0, 0.01, 500))
    out = inference.paired_bootstrap_difference(x, x, reps=50)
    assert out["estimate"] == 0 and out["ci_low"] == 0 and out["ci_high"] == 0
