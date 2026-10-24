"""Dependence analysis for RQ1 (nonlinear / tail dependence) and RQ2 (t vs Gaussian)."""
import numpy as np
import pandas as pd
from scipy import stats

from cpt import copula


def dependence_summary(log_returns: pd.DataFrame, q: float = 0.05) -> dict:
    """Linear vs rank correlation and empirical joint-tail frequencies of one pair."""
    r = log_returns.dropna()
    x, y = r.iloc[:, 0], r.iloc[:, 1]
    u1, u2 = copula.pseudo_observations(x), copula.pseudo_observations(y)
    kendall = stats.kendalltau(x, y)[0]
    return {
        "n": len(r),
        "pearson": stats.pearsonr(x, y)[0],
        "spearman": stats.spearmanr(x, y)[0],
        "kendall": kendall,
        "pearson_implied_by_kendall": np.sin(np.pi * kendall / 2),
        "joint_lower_tail": np.mean((u1 < q) & (u2 < q)),
        "joint_upper_tail": np.mean((u1 > 1 - q) & (u2 > 1 - q)),
        "independence_tail": q**2,
    }


def joint_tail_prob(family: str, rho: float, q: float, nu: float | None = None) -> float:
    """C(q, q) = P(U1 < q, U2 < q). Elliptical copulas are symmetric, so it also
    equals the upper-tail probability P(U1 > 1-q, U2 > 1-q)."""
    shape = [[1.0, rho], [rho, 1.0]]
    if family == "gaussian":
        z = stats.norm.ppf(q)
        return float(stats.multivariate_normal(mean=[0, 0], cov=shape).cdf([z, z]))
    z = stats.t.ppf(q, nu)
    return float(stats.multivariate_t(shape=shape, df=nu).cdf([z, z], random_state=0))


def oos_tail_calibration(fits: pd.DataFrame, probs: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """Observed vs predicted joint-tail days, out of sample.

    `fits` (one row per month, fitted on the prior window) and `probs` (that
    month's u1, u2 from the training ECDF) come from `backtest.copula_frame`.
    """
    month = probs.index.to_period("M").astype(str)
    rows = []
    for fit in fits.itertuples():
        u = probs[month == fit.month]
        rows.append({
            "month": fit.month, "days": len(u),
            "observed_lower": int(((u.u1 < q) & (u.u2 < q)).sum()),
            "observed_upper": int(((u.u1 > 1 - q) & (u.u2 > 1 - q)).sum()),
            "pred_gaussian": len(u) * joint_tail_prob("gaussian", fit.gaussian_rho, q),
            "pred_t": len(u) * joint_tail_prob("t", fit.t_rho, q, fit.t_nu),
        })
    return pd.DataFrame(rows)


def fit_comparison(fits: pd.DataFrame) -> dict:
    """Share of months where t beats Gaussian, in sample (AIC) and out of sample (log-lik)."""
    oos_diff = fits["t_oos_loglik"] - fits["gaussian_oos_loglik"]
    return {
        "months": len(fits),
        "t_better_aic_share": (fits["t_aic"] < fits["gaussian_aic"]).mean(),
        "t_better_oos_share": (oos_diff > 0).mean(),
        "oos_loglik_diff_total": oos_diff.sum(),
        "median_nu": fits["t_nu"].median(),
        "median_tail_dep": fits["t_tail_dep"].median(),
    }
