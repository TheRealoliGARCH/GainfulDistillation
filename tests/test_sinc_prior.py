import math

import pytest

from gds_core.sinc_prior import (
    IndependentSincPrior,
    SincModulatedPrior,
    sinc4_spectral_overlap,
)


def _simpson(function, lower, upper, intervals=20000):
    assert intervals % 2 == 0
    step = (upper - lower) / intervals
    total = function(lower) + function(upper)
    total += 4.0 * sum(function(lower + i * step) for i in range(1, intervals, 2))
    total += 2.0 * sum(function(lower + i * step) for i in range(2, intervals, 2))
    return total * step / 3.0


@pytest.mark.parametrize("frequency, expected", [
    (0.0, 2.0 / 3.0),
    (0.5, 23.0 / 48.0),
    (1.0, 1.0 / 6.0),
    (1.5, 1.0 / 48.0),
    (2.0, 0.0),
    (3.0, 0.0),
])
def test_sinc4_spectral_overlap(frequency, expected):
    assert sinc4_spectral_overlap(frequency) == pytest.approx(expected)


def test_spectral_overlap_is_even_and_nonnegative():
    for frequency in [0.0, 0.1, 0.8, 1.0, 1.4, 1.99, 2.0, 3.0]:
        assert sinc4_spectral_overlap(-frequency) == pytest.approx(
            sinc4_spectral_overlap(frequency)
        )
        assert sinc4_spectral_overlap(frequency) >= 0.0


def test_prior_is_nonnegative_symmetric_and_has_finite_log_density():
    prior = SincModulatedPrior(
        location=1.25,
        bandwidth=1.4,
        scale=0.8,
        center_frequency=0.35,
        modulation_depth=0.9,
    )
    for offset in [0.0, 0.1, 0.5, 1.0, 2.3, 5.0]:
        left = prior.density(prior.location - offset)
        right = prior.density(prior.location + offset)
        assert left >= 0.0
        assert right >= 0.0
        assert left == pytest.approx(right, rel=1e-12, abs=1e-15)
        assert math.isfinite(prior.log_density(prior.location + offset)) or (
            prior.log_density(prior.location + offset) == float("-inf")
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bandwidth": 0.0},
        {"bandwidth": -1.0},
        {"scale": 0.0},
        {"scale": -1.0},
        {"center_frequency": -0.1},
        {"modulation_depth": -0.01},
        {"modulation_depth": 1.01},
        {"location": float("nan")},
        {"center_frequency": float("inf")},
        {"scale": True},
    ],
)
def test_invalid_prior_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SincModulatedPrior(**kwargs)


def test_prior_has_exact_positive_normalizer_and_zero_frequency_case():
    prior = SincModulatedPrior(
        bandwidth=2.0,
        scale=3.0,
        center_frequency=0.0,
        modulation_depth=0.75,
    )
    # At zero frequency, the modulation is a constant multiplier 1+rho.
    expected = (3.0 / 2.0) * (2.0 / 3.0) * (1.0 + 0.75)
    assert prior.normalization_constant == pytest.approx(expected)
    assert prior.density(prior.location) == pytest.approx(
        (1.0 + 0.75) / expected
    )


def test_prior_integrates_to_one_numerically():
    prior = SincModulatedPrior(
        location=-0.4,
        bandwidth=1.3,
        scale=0.9,
        center_frequency=0.7,
        modulation_depth=0.8,
    )
    integral = _simpson(prior.density, -250.0, 250.0, intervals=20000)
    assert integral == pytest.approx(1.0, abs=2e-5)


def test_unmodulated_prior_is_proper_and_has_sinc_fourth_power_shape():
    prior = SincModulatedPrior(
        location=0.0,
        bandwidth=1.0,
        scale=2.0,
        center_frequency=0.0,
        modulation_depth=0.0,
    )
    assert prior.density(0.0) == pytest.approx(3.0 / 4.0)
    assert prior.density(1.0) >= 0.0
    assert prior.normalization_constant == pytest.approx(4.0 / 3.0)


def test_prior_log_density_agrees_with_density_away_from_zeros():
    prior = SincModulatedPrior(
        bandwidth=1.1,
        scale=0.7,
        center_frequency=0.3,
        modulation_depth=0.6,
    )
    for value in [0.13, 0.28, 0.67, 1.31]:
        assert math.exp(prior.log_density(value)) == pytest.approx(
            prior.density(value), rel=1e-12
        )


def test_independent_product_prior_and_log_density():
    prior = IndependentSincPrior((
        SincModulatedPrior(location=0.0, bandwidth=1.0, scale=1.0),
        SincModulatedPrior(location=2.0, bandwidth=0.8, scale=1.2),
    ))
    point = (0.2, 2.4)
    assert prior.density(point) == pytest.approx(
        prior.marginals[0].density(point[0])
        * prior.marginals[1].density(point[1])
    )
    assert prior.log_density(point) == pytest.approx(math.log(prior.density(point)))


def test_independent_product_prior_rejects_bad_dimensions_and_marginals():
    with pytest.raises(ValueError):
        IndependentSincPrior(())
    with pytest.raises(ValueError):
        IndependentSincPrior((object(),))
    prior = IndependentSincPrior((SincModulatedPrior(), SincModulatedPrior()))
    with pytest.raises(ValueError):
        prior.density((0.0,))
