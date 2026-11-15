import math

import pytest

from gds_core.posterior import compute_posterior_1d
from gds_core.sinc_prior import SincModulatedPrior


def test_constant_likelihood_returns_truncated_prior_and_zero_log_evidence():
    prior = SincModulatedPrior(location=0.2, bandwidth=1.1, scale=0.8)
    posterior = compute_posterior_1d(
        prior, lambda theta: 0.0, bounds=(-8.0, 8.0), points=2048
    )
    assert posterior.log_marginal_likelihood == pytest.approx(0.0, abs=1e-12)
    assert posterior.mean == pytest.approx(0.2, abs=2e-3)
    assert posterior.variance > 0.0
    assert posterior.integrated_density == pytest.approx(1.0, abs=1e-12)
    assert sum(posterior.probability_masses) == pytest.approx(1.0)
    assert posterior.credible_interval[0] < posterior.mean
    assert posterior.mean < posterior.credible_interval[1]


def test_gaussian_log_likelihood_updates_prior_toward_observation():
    prior = SincModulatedPrior(location=-1.0, bandwidth=1.0, scale=1.0)
    observation = 1.25
    observation_sd = 0.35

    def log_likelihood(theta):
        z = (observation - theta) / observation_sd
        return -0.5 * z * z - math.log(observation_sd * math.sqrt(2.0 * math.pi))

    posterior = compute_posterior_1d(
        prior,
        log_likelihood,
        bounds=(-5.0, 5.0),
        points=8192,
        credible_level=0.90,
    )
    assert posterior.mean > prior.location
    assert abs(posterior.mean - observation) < abs(prior.location - observation)
    assert posterior.variance > 0.0
    assert posterior.credible_interval[0] < posterior.mean
    assert posterior.mean < posterior.credible_interval[1]
    assert posterior.credible_level == pytest.approx(0.90)


def test_constant_shift_in_log_likelihood_shifts_log_evidence_only():
    prior = SincModulatedPrior()
    base = compute_posterior_1d(
        prior, lambda theta: -0.5 * theta * theta,
        bounds=(-6.0, 6.0), points=1024,
    )
    shifted = compute_posterior_1d(
        prior, lambda theta: -0.5 * theta * theta + 7.3,
        bounds=(-6.0, 6.0), points=1024,
    )
    assert shifted.mean == pytest.approx(base.mean, abs=1e-12)
    assert shifted.variance == pytest.approx(base.variance, abs=1e-12)
    assert shifted.log_marginal_likelihood - base.log_marginal_likelihood == pytest.approx(7.3)


def test_posterior_refines_stably_for_smooth_likelihood():
    prior = SincModulatedPrior(location=0.0, bandwidth=1.2, scale=1.0)
    likelihood = lambda theta: -0.5 * ((theta - 0.6) / 0.7) ** 2
    coarse = compute_posterior_1d(
        prior, likelihood, bounds=(-7.0, 7.0), points=1024
    )
    fine = compute_posterior_1d(
        prior, likelihood, bounds=(-7.0, 7.0), points=4096
    )
    assert fine.mean == pytest.approx(coarse.mean, abs=2e-5)
    assert fine.variance == pytest.approx(coarse.variance, abs=2e-5)
    assert fine.log_marginal_likelihood == pytest.approx(
        coarse.log_marginal_likelihood, abs=2e-5
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": (1.0, 1.0)},
        {"bounds": (2.0, -2.0)},
        {"bounds": (float("-inf"), 2.0)},
        {"bounds": (0.0, float("nan"))},
        {"points": 0},
        {"points": 15},
        {"points": True},
        {"credible_level": 0.0},
        {"credible_level": 1.0},
        {"credible_level": float("nan")},
    ],
)
def test_invalid_posterior_arguments_are_rejected(kwargs):
    base = {"bounds": (-3.0, 3.0)}
    base.update(kwargs)
    with pytest.raises(ValueError):
        compute_posterior_1d(SincModulatedPrior(), lambda theta: 0.0, **base)


def test_non_callable_likelihood_is_rejected():
    with pytest.raises(TypeError):
        compute_posterior_1d(SincModulatedPrior(), 3, bounds=(-2.0, 2.0))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_likelihood_values_are_rejected(bad_value):
    with pytest.raises(ValueError):
        compute_posterior_1d(
            SincModulatedPrior(),
            lambda theta: bad_value,
            bounds=(-2.0, 2.0),
        )


def test_all_zero_likelihood_support_is_rejected():
    with pytest.raises(ValueError):
        compute_posterior_1d(
            SincModulatedPrior(),
            lambda theta: float("-inf"),
            bounds=(-3.0, 3.0),
        )


def test_log_likelihood_minus_infinity_is_valid_on_part_of_domain():
    posterior = compute_posterior_1d(
        SincModulatedPrior(),
        lambda theta: 0.0 if theta >= 0.0 else float("-inf"),
        bounds=(-4.0, 4.0),
        points=1024,
    )
    assert posterior.mean > 0.0
    assert posterior.credible_interval[0] >= 0.0
