"""Bivariate Gaussian and Student-t copulas (semiparametric fit).

To implement (Krauss & Stübinger 2015, eq. 3-4):
- pseudo_observations: ranks / (n + 1) from the estimation window only
- log-density, MLE fit (rho, nu), and h-functions h(u1|u2), h(u2|u1)
"""
import numpy as np


def pseudo_observations(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def ecdf_transform(train: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Map new observations through the empirical CDF of `train` (no leakage)."""
    raise NotImplementedError


def fit_gaussian(u1: np.ndarray, u2: np.ndarray) -> dict:
    raise NotImplementedError


def fit_t(u1: np.ndarray, u2: np.ndarray) -> dict:
    raise NotImplementedError


def h_t(u: np.ndarray, v: np.ndarray, rho: float, nu: float) -> np.ndarray:
    """P(U <= u | V = v) for the t-copula."""
    raise NotImplementedError


def h_gaussian(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """P(U <= u | V = v) for the Gaussian copula."""
    raise NotImplementedError
