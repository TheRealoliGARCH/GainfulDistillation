import math

import pytest

from gds_core.balanced_loss import balanced_loss_variance_1d
from gds_core.posterior import compute_posterior_1d
from gds_core.sinc_prior import SincModulatedPrior


def _posterior():
    return compute_posterior_1d(
        SincModulatedPrior(location=0.0, bandwidth=1.0, scale=1.0),
        lambda theta: -0.5 * theta * theta,
        bounds=(-5.0, 5.0),
        points=1024,
    )


def test_constant_score_information_correction_matches_scalar_formula():
    posterior = _posterior()
    result = balanced_loss_variance_1d(
        posterior,
        lambda theta: ((1.0, -2.0), 4.0),
    )
    expected_omega = ((1.0**2 + (-2.0)**2) / 2.0) / 4.0
    assert result.point_estimate == pytest.approx(posterior.mean)
    assert result.posterior_variance == pytest.approx(posterior.variance)
    assert result.omega == pytest.approx(expected_omega)
    assert result.robust_variance == pytest.approx(
        posterior.variance * expected_omega
    )
    assert result.robust_standard_error == pytest.approx(
        math.sqrt(result.robust_variance)
    )
    assert result.sample_size == 2
    assert result.posterior_points == posterior.points


def test_posterior_expectation_averages_theta_dependent_correction():
    posterior = _posterior()

    def score_information(theta):
        return ((1.0 + theta, 1.0 - theta), 2.0)

    result = balanced_loss_variance_1d(posterior, score_information)
    expected_values = tuple(
        (((1.0 + theta) ** 2 + (1.0 - theta) ** 2) / 2.0) / 2.0
        for theta in posterior.grid
    )
    expected = sum(
        mass * value
        for mass, value in zip(posterior.probability_masses, expected_values)
    )
    assert result.omega == pytest.approx(expected)
    assert result.robust_variance == pytest.approx(posterior.variance * expected)


def test_normal_mean_scores_use_average_empirical_information():
    observations = (0.2, 0.8, 1.1, 1.4)
    posterior = compute_posterior_1d(
        SincModulatedPrior(location=0.0, bandwidth=1.0, scale=1.0),
        lambda theta: -0.5 * sum((value - theta) ** 2 for value in observations),
        bounds=(-2.0, 3.0),
        points=4096,
    )

    def score_information(theta):
        scores = tuple(value - theta for value in observations)
        # For a unit-variance Normal location model, average negative
        # second derivative is exactly one.
        return scores, 1.0

    result = balanced_loss_variance_1d(posterior, score_information)
    assert result.sample_size == len(observations)
    assert result.omega > 0.0
    assert result.robust_variance >= 0.0


def test_nonconstant_score_count_is_rejected():
    posterior = _posterior()

    def callback(theta):
        return ((1.0,), 1.0) if theta < 0.0 else ((1.0, 2.0), 1.0)

    with pytest.raises(ValueError, match="length must be constant"):
        balanced_loss_variance_1d(posterior, callback)


@pytest.mark.parametrize(
    "callback",
    [
        lambda theta: ((), 1.0),
        lambda theta: ((1.0, float("nan")), 1.0),
        lambda theta: ((1.0,), 0.0),
        lambda theta: ((1.0,), -1.0),
        lambda theta: ((1.0,), float("inf")),
        lambda theta: ((1.0,), float("nan")),
        lambda theta: ((1.0,),),
        lambda theta: (1.0, 1.0),
    ],
)
def test_invalid_score_information_callback_results_are_rejected(callback):
    with pytest.raises(ValueError):
        balanced_loss_variance_1d(_posterior(), callback)


def test_zero_expected_correction_is_rejected():
    posterior = _posterior()
    with pytest.raises(ValueError, match="strictly positive"):
        balanced_loss_variance_1d(posterior, lambda theta: ((0.0, 0.0), 1.0))


def test_nonposterior_argument_is_rejected():
    with pytest.raises(TypeError):
        balanced_loss_variance_1d(object(), lambda theta: ((1.0,), 1.0))


def test_noncallable_callback_is_rejected():
    with pytest.raises(TypeError):
        balanced_loss_variance_1d(_posterior(), 4)
