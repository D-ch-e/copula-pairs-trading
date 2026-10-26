"""Dependence analysis for RQ1 (nonlinear / tail dependence) and RQ2 (t vs Gaussian),
and subperiod breakdowns for RQ5."""
import numpy as np
import pandas as pd
from scipy import stats

from cpt import copula
from cpt.metrics import max_drawdown, sharpe_ratio


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


def marginal_exceedance(probs: pd.DataFrame, q: float = 0.05) -> dict:
    """Share of out-of-sample days on which each leg alone is in its lower / upper q tail.

    The training ECDF is only an estimate of next month's margins: if returns get more
    volatile, more than q of days land in each tail. The copula's joint-tail prediction
    C(q, q) assumes exactly q, so marginal drift alone moves the joint count.
    """
    return {
        "u1_lower": np.mean(probs.u1 < q), "u1_upper": np.mean(probs.u1 > 1 - q),
        "u2_lower": np.mean(probs.u2 < q), "u2_upper": np.mean(probs.u2 > 1 - q),
    }


def recalibrated_tail_calibration(fits: pd.DataFrame, probs: pd.DataFrame, q: float = 0.05) -> dict:
    """Joint-tail frequency after removing marginal drift, vs the copulas' average prediction.

    All out-of-sample u's are re-ranked per leg over the whole test sample, so each leg is
    exactly uniform and the remaining joint-tail frequency reflects dependence only. This
    uses the full test sample and is a diagnostic, never an input to trading.
    """
    v1, v2 = copula.pseudo_observations(probs.u1.to_numpy()), copula.pseudo_observations(probs.u2.to_numpy())
    raw = oos_tail_calibration(fits, probs, q)
    days = raw["days"].sum()
    return {
        "days": days,
        "observed_lower": np.mean((v1 < q) & (v2 < q)),
        "observed_upper": np.mean((v1 > 1 - q) & (v2 > 1 - q)),
        "pred_gaussian": raw["pred_gaussian"].sum() / days,
        "pred_t": raw["pred_t"].sum() / days,
    }


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


def period_masks(index: pd.DatetimeIndex, periods: dict) -> dict[str, np.ndarray]:
    """Boolean mask per configured subperiod, plus "other" for days outside all of them."""
    masks = {name: np.asarray((index >= pd.Timestamp(p["start"])) & (index <= pd.Timestamp(p["end"])))
             for name, p in periods.items()}
    masks["other"] = ~np.logical_or.reduce(list(masks.values()))
    return masks


def subperiod_performance(returns: pd.Series, periods: dict) -> pd.DataFrame:
    """Total return (not annualized: periods are short), annualized Sharpe and max drawdown per period."""
    rows = {}
    for name, mask in period_masks(returns.index, periods).items():
        r = returns[mask]
        rows[name] = {
            "days": len(r),
            "total_return": (1 + r).prod() - 1 if len(r) else np.nan,
            "sharpe": sharpe_ratio(r) if len(r) > 1 else np.nan,
            "max_drawdown": max_drawdown(r) if len(r) else np.nan,
        }
    return pd.DataFrame(rows).T


def subperiod_copula(fits: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """Median t-copula parameters over the months whose start falls in each period."""
    months = pd.PeriodIndex(fits["month"], freq="M").to_timestamp()
    rows = {}
    for name, mask in period_masks(months, periods).items():
        f = fits[mask]
        rows[name] = {
            "months": len(f),
            "median_t_rho": f["t_rho"].median(),
            "median_t_nu": f["t_nu"].median(),
            "median_tail_dep": f["t_tail_dep"].median(),
            "t_better_oos_share": (f["t_oos_loglik"] > f["gaussian_oos_loglik"]).mean() if len(f) else np.nan,
        }
    return pd.DataFrame(rows).T
