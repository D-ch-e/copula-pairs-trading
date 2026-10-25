"""Significance tests that respect autocorrelation in daily strategy returns.

- Newey-West t-statistic of the mean (paper's Table 4 uses NW standard errors)
- stationary block bootstrap (Politis & Romano 1994) confidence intervals
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

DAYS_PER_YEAR = 252


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb lag length floor(4 * (n / 100) ** (2 / 9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_mean_test(x: pd.Series, lags: int | None = None) -> dict:
    """Mean of x with HAC standard error; H0: mean = 0."""
    x = pd.Series(x).dropna().to_numpy()
    lags = newey_west_lags(len(x)) if lags is None else lags
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean": res.params[0], "se_nw": res.bse[0], "t_nw": res.tvalues[0],
            "p_nw": res.pvalues[0], "lags": lags, "n": len(x)}


def stationary_bootstrap_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of positions 0..n-1 with geometric block lengths of mean `block`."""
    restart = rng.random(n) < 1.0 / block
    restart[0] = True
    pos = np.arange(n)
    block_start_pos = np.maximum.accumulate(np.where(restart, pos, 0))
    starts = rng.integers(n, size=n)
    return (starts[block_start_pos] + pos - block_start_pos) % n


def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return np.nan if sd == 0 else x.mean() / sd * np.sqrt(DAYS_PER_YEAR)


def bootstrap_ci(x: pd.Series, stat=_sharpe, reps: int = 2000, block: float = 20,
                 level: float = 0.95, seed: int = 0) -> dict:
    """Percentile CI of `stat` (default: annualized Sharpe) under the stationary bootstrap."""
    x = pd.Series(x).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.array([stat(x[stationary_bootstrap_indices(len(x), block, rng)]) for _ in range(reps)])
    lo, hi = np.nanquantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": stat(x), "ci_low": lo, "ci_high": hi, "reps": reps, "block": block}


def paired_bootstrap_difference(a: pd.Series, b: pd.Series, reps: int = 2000, block: float = 20,
                                level: float = 0.95, seed: int = 0) -> dict:
    """CI for the annualized mean return difference a - b on common dates, resampling days jointly."""
    df = pd.concat([a, b], axis=1, join="inner").dropna().to_numpy()
    diff = df[:, 0] - df[:, 1]
    ann_mean = lambda d: d.mean() * DAYS_PER_YEAR
    return bootstrap_ci(pd.Series(diff), stat=ann_mean, reps=reps, block=block, level=level, seed=seed)
