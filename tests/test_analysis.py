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
