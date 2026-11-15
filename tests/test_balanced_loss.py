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



def test_balanced_loss_tracks_sandwich_variance_under_scale_misspecification():
    # Deterministic repeated pattern with empirical variance near four.
    # The working likelihood assumes a unit observation variance.
    observations = tuple(
        value
        for _ in range(100)
        for value in (-3.0, -1.0, 0.0, 1.0, 3.0)
    )
    n = len(observations)
    sample_mean = sum(observations) / n
    sample_variance = sum((value - sample_mean) ** 2 for value in observations) / (n - 1)

    prior = SincModulatedPrior(
        location=0.0, bandwidth=1.0, scale=2.0, modulation_depth=0.0
    )
    posterior = compute_posterior_1d(
        prior,
        lambda theta: -0.5 * sum((value - theta) ** 2 for value in observations),
        bounds=(-5.0, 5.0),
        points=4096,
    )
    result = balanced_loss_variance_1d(
        posterior,
        lambda theta: (tuple(value - theta for value in observations), 1.0),
    )

    frequentist_sandwich_variance = sample_variance / n
    assert posterior.variance < 0.35 * frequentist_sandwich_variance
    assert result.robust_variance == pytest.approx(
        frequentist_sandwich_variance, rel=0.01
    )



def test_aggregate_score_summary_matches_individual_score_vector():
    from gds_core.balanced_loss import ScalarScoreSummary

    posterior = _posterior()
    individual = balanced_loss_variance_1d(
        posterior,
        lambda theta: ((0.5 + theta, -1.0 + theta, 2.0 - theta), 1.5),
    )
    aggregate = balanced_loss_variance_1d(
        posterior,
        lambda theta: (
            ScalarScoreSummary(
                3,
                ((0.5 + theta) ** 2 + (-1.0 + theta) ** 2 + (2.0 - theta) ** 2) / 3.0,
            ),
            1.5,
        ),
    )
    assert aggregate == individual


@pytest.mark.parametrize(
    "sample_size,mean_squared_score",
    [
        (0, 1.0),
        (-1, 1.0),
        (True, 1.0),
        (2, -1.0),
        (2, float("nan")),
        (2, float("inf")),
        (2, True),
    ],
)
def test_scalar_score_summary_rejects_invalid_values(sample_size, mean_squared_score):
    from gds_core.balanced_loss import ScalarScoreSummary

    with pytest.raises(ValueError):
        ScalarScoreSummary(sample_size, mean_squared_score)


def test_balanced_loss_accepts_aggregate_score_summary_with_constant_n():
    from gds_core.balanced_loss import ScalarScoreSummary

    posterior = _posterior()
    result = balanced_loss_variance_1d(
        posterior,
        lambda theta: (ScalarScoreSummary(7, (theta - 0.2) ** 2 + 1.0), 1.0),
    )
    assert result.sample_size == 7
    assert result.robust_variance > 0.0
