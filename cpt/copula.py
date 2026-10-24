"""Bivariate Gaussian and Student-t copulas (semiparametric fit).

Follows Krauss & Stübinger (2015), eq. 3-4: margins are empirical CDFs, the
copula parameters are fitted by maximum likelihood on the pseudo-observations,
and trading signals come from the h-functions h(u1|u2), h(u2|u1).
"""
import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

RHO_BOUND = 0.999
NU_MIN, NU_MAX = 2.01, 200.0
NU_GRID = [2.5, 3, 4, 6, 8, 12, 20, 30, 50, 100]


def pseudo_observations(x: np.ndarray) -> np.ndarray:
    """Scaled ranks rank / (n + 1), strictly inside (0, 1)."""
    x = np.asarray(x, dtype=float)
    return stats.rankdata(x) / (len(x) + 1)


def ecdf_transform(train: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Map new observations through the empirical CDF of `train` (no leakage).

    For points inside `train` this equals `pseudo_observations(train)` when there
    are no ties. Values outside the training range are clipped into (0, 1).
    """
    train = np.sort(np.asarray(train, dtype=float))
    n = len(train)
    counts = np.searchsorted(train, np.asarray(new, dtype=float), side="right")
    return np.clip(counts, 0.5, n + 0.5) / (n + 1)


def logpdf_gaussian(u1: np.ndarray, u2: np.ndarray, rho: float) -> np.ndarray:
    x, y = stats.norm.ppf(u1), stats.norm.ppf(u2)
    one_m = 1.0 - rho**2
    return -0.5 * np.log(one_m) - (rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * one_m)


def logpdf_t(u1: np.ndarray, u2: np.ndarray, rho: float, nu: float) -> np.ndarray:
    x, y = stats.t.ppf(u1, nu), stats.t.ppf(u2, nu)
    one_m = 1.0 - rho**2
    quad = (x**2 + y**2 - 2 * rho * x * y) / (nu * one_m)
    log_joint = (
        gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi)
        - 0.5 * np.log(one_m) - (nu + 2) / 2 * np.log1p(quad)
    )
    return log_joint - stats.t.logpdf(x, nu) - stats.t.logpdf(y, nu)


def _summary(loglik: float, k: int, n: int, **params) -> dict:
    return {**params, "loglik": loglik, "aic": -2 * loglik + 2 * k,
            "bic": -2 * loglik + np.log(n) * k, "n": n}


def fit_gaussian(u1: np.ndarray, u2: np.ndarray) -> dict:
    res = optimize.minimize_scalar(
        lambda r: -logpdf_gaussian(u1, u2, r).sum(),
        bounds=(-RHO_BOUND, RHO_BOUND), method="bounded",
    )
    return _summary(-res.fun, k=1, n=len(u1), rho=res.x)


def fit_t(u1: np.ndarray, u2: np.ndarray) -> dict:
    """Joint MLE of (rho, nu), started from the Kendall's tau inversion and a nu grid."""
    tau = stats.kendalltau(u1, u2)[0]
    rho0 = np.clip(np.sin(np.pi * tau / 2), -0.95, 0.95)
    nu0 = max(NU_GRID, key=lambda nu: logpdf_t(u1, u2, rho0, nu).sum())

    res = optimize.minimize(
        lambda p: -logpdf_t(u1, u2, p[0], p[1]).sum(),
        x0=[rho0, nu0],
        bounds=[(-RHO_BOUND, RHO_BOUND), (NU_MIN, NU_MAX)],
        method="L-BFGS-B",
    )
    rho, nu = res.x
    return _summary(-res.fun, k=2, n=len(u1), rho=rho, nu=nu)


def h_gaussian(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """P(U <= u | V = v) for the Gaussian copula."""
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    return stats.norm.cdf((x - rho * y) / np.sqrt(1 - rho**2))


def h_t(u: np.ndarray, v: np.ndarray, rho: float, nu: float) -> np.ndarray:
    """P(U <= u | V = v) for the t-copula (Krauss & Stübinger 2015, eq. 4)."""
    x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    scale = np.sqrt((nu + y**2) * (1 - rho**2) / (nu + 1))
    return stats.t.cdf((x - rho * y) / scale, nu + 1)


def tail_dependence_t(rho: float, nu: float) -> float:
    """Upper (= lower) tail dependence coefficient of the t-copula. Gaussian is 0."""
    return 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)


def sample_t(n: int, rho: float, nu: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw (u1, u2) from a t-copula; used for tests and simulation checks."""
    rng = np.random.default_rng(seed)
    z = stats.multivariate_t(shape=[[1, rho], [rho, 1]], df=nu).rvs(size=n, random_state=rng)
    u = stats.t.cdf(z, nu)
    return u[:, 0], u[:, 1]
