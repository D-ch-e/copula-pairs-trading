import numpy as np
import pytest
from scipy import integrate

from cpt import copula


def test_pseudo_observations_inside_unit_interval():
    u = copula.pseudo_observations(np.array([3.0, -1.0, 2.0, 2.0, 10.0]))
    assert np.all((u > 0) & (u < 1))
    assert u.argmax() == 4 and u.argmin() == 1


def test_ecdf_transform_matches_pseudo_obs_and_clips():
    train = np.array([0.5, -0.2, 0.1, 0.9])
    np.testing.assert_allclose(copula.ecdf_transform(train, train), copula.pseudo_observations(train))
    out = copula.ecdf_transform(train, np.array([-5.0, 5.0]))
    assert 0 < out[0] < out[1] < 1


@pytest.mark.parametrize("family", ["gaussian", "t"])
def test_h_function_equals_integral_of_density(family):
    rho, nu, u, v = 0.6, 4.0, 0.3, 0.8
    if family == "gaussian":
        density = lambda s: np.exp(copula.logpdf_gaussian(s, v, rho))
        h = copula.h_gaussian(u, v, rho)
    else:
        density = lambda s: np.exp(copula.logpdf_t(s, v, rho, nu))
        h = copula.h_t(u, v, rho, nu)
    integral, _ = integrate.quad(density, 1e-12, u)
    assert h == pytest.approx(integral, abs=1e-6)


def test_t_converges_to_gaussian_for_large_nu():
    u, v = np.array([0.1, 0.5, 0.9]), np.array([0.7, 0.2, 0.95])
    np.testing.assert_allclose(copula.h_t(u, v, 0.5, 1e6), copula.h_gaussian(u, v, 0.5), atol=1e-4)
    np.testing.assert_allclose(
        copula.logpdf_t(u, v, 0.5, 1e6), copula.logpdf_gaussian(u, v, 0.5), atol=1e-4
    )


def test_fit_t_recovers_known_parameters():
    u1, u2 = copula.sample_t(4000, rho=0.7, nu=4, seed=1)
    fit = copula.fit_t(copula.pseudo_observations(u1), copula.pseudo_observations(u2))
    assert fit["rho"] == pytest.approx(0.7, abs=0.03)
    assert fit["nu"] == pytest.approx(4, abs=1.5)
    assert fit["converged"] and not fit["nu_at_bound"] and not fit["rho_at_bound"]


def test_fit_t_on_gaussian_data_goes_to_upper_nu():
    rng = np.random.default_rng(6)
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=3000)
    fit = copula.fit_t(copula.pseudo_observations(z[:, 0]), copula.pseudo_observations(z[:, 1]))
    assert fit["converged"]
    assert fit["nu"] > 30  # no tail dependence: nu drifts toward the Gaussian limit


def test_bound_flags():
    assert copula.bound_flags(0.5, copula.NU_MAX)["nu_at_bound"]
    assert copula.bound_flags(0.5, copula.NU_MIN)["nu_at_bound"]
    assert not copula.bound_flags(0.5, 8.0)["nu_at_bound"]
    assert copula.bound_flags(0.9995, 8.0)["rho_at_bound"]


def test_t_beats_gaussian_on_t_data():
    u1, u2 = copula.sample_t(2000, rho=0.5, nu=3, seed=2)
    p1, p2 = copula.pseudo_observations(u1), copula.pseudo_observations(u2)
    assert copula.fit_t(p1, p2)["aic"] < copula.fit_gaussian(p1, p2)["aic"]


def test_tail_dependence_decreases_with_nu():
    assert copula.tail_dependence_t(0.5, 3) > copula.tail_dependence_t(0.5, 30) > 0
