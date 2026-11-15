import pytest

from gds_core.posterior import compute_posterior_1d
from gds_core.prior_diagnostics import (
    PriorBiasThresholds,
    assess_posterior_bias_1d,
    prior_predictive_check_1d,
    summarize_prior_1d,
)
from gds_core.sinc_prior import SincModulatedPrior


def test_prior_summary_reports_retained_mass_and_conditional_moments():
    prior = SincModulatedPrior(location=0.4, bandwidth=1.0, scale=1.5)
    summary = summarize_prior_1d(prior, bounds=(-10.0, 10.0), points=4096)
    assert 0.99 < summary.retained_prior_mass <= 1.0
    assert summary.conditional_mean == pytest.approx(0.4, abs=2e-3)
    assert summary.conditional_variance > 0.0
    assert summary.central_interval[0] < summary.conditional_mean
    assert summary.central_interval[1] > summary.conditional_mean
    assert summary.interval_width > 0.0
    assert summary.boundary_cell_mass >= 0.0


def test_prior_summary_detects_truncation_by_finite_bounds():
    prior = SincModulatedPrior(location=0.0, bandwidth=1.0, scale=3.0)
    wide = summarize_prior_1d(prior, bounds=(-30.0, 30.0), points=4096)
    narrow = summarize_prior_1d(prior, bounds=(-0.5, 0.5), points=4096)
    assert wide.retained_prior_mass > narrow.retained_prior_mass
    assert narrow.retained_prior_mass < 0.8


def test_prior_predictive_simulation_is_reproducible():
    prior = SincModulatedPrior(location=0.0, bandwidth=1.0, scale=1.5)

    def simulator(theta, rng):
        return rng.gauss(theta, 0.5)

    first = prior_predictive_check_1d(
        prior, simulator, bounds=(-8.0, 8.0), draws=1000,
        points=2048, seed=123, observed_statistic=0.0,
    )
    second = prior_predictive_check_1d(
        prior, simulator, bounds=(-8.0, 8.0), draws=1000,
        points=2048, seed=123, observed_statistic=0.0,
    )
    assert first == second
    assert first.draws == 1000
    assert first.predictive_variance > 0.0
    assert first.predictive_interval[0] < first.predictive_interval[1]
    assert first.two_sided_tail_probability is not None


def test_prior_predictive_conflict_is_flagged_for_extreme_observation():
    prior = SincModulatedPrior(location=0.0, bandwidth=1.0, scale=0.7)
    result = prior_predictive_check_1d(
        prior, lambda theta, rng: rng.gauss(theta, 0.1),
        bounds=(-8.0, 8.0), draws=2000, points=4096,
        seed=44, observed_statistic=5.0, surprise_threshold=0.01,
    )
    assert result.two_sided_tail_probability <= 0.01
    assert any("low-probability" in warning for warning in result.warnings)


def test_prior_predictive_flags_truncation_of_original_prior():
    prior = SincModulatedPrior(location=0.0, bandwidth=1.0, scale=3.0)
    result = prior_predictive_check_1d(
        prior, lambda theta, rng: theta, bounds=(-0.25, 0.25),
        draws=100, points=1024, seed=2, minimum_retained_prior_mass=0.99,
    )
    assert result.prior_summary.retained_prior_mass < 0.99
    assert any("excludes appreciable mass" in warning for warning in result.warnings)


def test_posterior_bias_diagnostic_identifies_prior_pull():
    prior = SincModulatedPrior(location=-1.0, bandwidth=1.0, scale=1.0)
    summary = summarize_prior_1d(prior, bounds=(-6.0, 6.0), points=2048)
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * ((theta - 1.0) / 0.5) ** 2,
        bounds=(-6.0, 6.0), points=4096,
    )
    diagnostic = assess_posterior_bias_1d(
        summary, posterior, likelihood_reference=1.0,
        reference_standard_error=0.5,
    )
    assert diagnostic.prior_mean < diagnostic.posterior_mean < 1.0
    assert diagnostic.prior_reference_gap < 0.0
    assert diagnostic.posterior_pull_fraction > 0.0
    assert any("materially displaced from the likelihood-only reference" in warning for warning in diagnostic.warnings)


def test_posterior_bias_diagnostic_is_quiet_when_posterior_tracks_reference():
    prior = SincModulatedPrior(location=0.0, bandwidth=1.0, scale=2.0)
    summary = summarize_prior_1d(prior, bounds=(-8.0, 8.0), points=4096)
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * ((theta - 0.0) / 0.15) ** 2,
        bounds=(-8.0, 8.0), points=4096,
    )
    diagnostic = assess_posterior_bias_1d(
        summary, posterior, likelihood_reference=0.0,
        reference_standard_error=0.15,
    )
    assert diagnostic.warnings == ()
    assert diagnostic.posterior_pull_fraction is None


def test_posterior_bias_thresholds_are_configurable():
    prior = SincModulatedPrior(location=-1.0, bandwidth=1.0, scale=1.0)
    summary = summarize_prior_1d(prior, bounds=(-6.0, 6.0), points=2048)
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * ((theta - 1.0) / 0.2) ** 2,
        bounds=(-6.0, 6.0), points=4096,
    )
    strict = assess_posterior_bias_1d(
        summary, posterior, likelihood_reference=1.0,
        reference_standard_error=0.2,
        thresholds=PriorBiasThresholds(
            max_prior_reference_gap_z=0.5,
            max_posterior_pull_fraction=0.01,
            max_boundary_cell_mass=1.0,
        ),
    )
    permissive = assess_posterior_bias_1d(
        summary, posterior, likelihood_reference=1.0,
        reference_standard_error=0.2,
        thresholds=PriorBiasThresholds(
            max_prior_reference_gap_z=100.0,
            max_posterior_pull_fraction=100.0,
            max_boundary_cell_mass=1.0,
        ),
    )
    assert len(strict.warnings) > len(permissive.warnings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": (1.0, 1.0)},
        {"bounds": (2.0, -2.0)},
        {"bounds": (float("-inf"), 1.0)},
        {"points": 63},
        {"points": True},
        {"credible_level": 0.0},
    ],
)
def test_prior_summary_rejects_invalid_arguments(kwargs):
    base = {"bounds": (-4.0, 4.0)}
    base.update(kwargs)
    with pytest.raises(ValueError):
        summarize_prior_1d(SincModulatedPrior(), **base)


def test_prior_predictive_rejects_invalid_arguments_and_outputs():
    with pytest.raises(TypeError):
        prior_predictive_check_1d(object(), lambda theta, rng: theta, bounds=(-2, 2))
    with pytest.raises(TypeError):
        prior_predictive_check_1d(SincModulatedPrior(), 5, bounds=(-2, 2))
    with pytest.raises(ValueError):
        prior_predictive_check_1d(
            SincModulatedPrior(), lambda theta, rng: theta,
            bounds=(-2, 2), draws=0,
        )
    with pytest.raises(ValueError):
        prior_predictive_check_1d(
            SincModulatedPrior(), lambda theta, rng: float("nan"),
            bounds=(-2, 2), draws=10,
        )
    with pytest.raises(ValueError):
        prior_predictive_check_1d(
            SincModulatedPrior(), lambda theta, rng: theta,
            bounds=(-2, 2), observed_statistic=float("inf"),
        )


def test_bias_diagnostic_validates_bounds_and_reference_standard_error():
    prior = SincModulatedPrior()
    summary = summarize_prior_1d(prior, bounds=(-4.0, 4.0), points=1024)
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * theta**2,
        bounds=(-3.0, 3.0), points=1024,
    )
    with pytest.raises(ValueError, match="identical bounds"):
        assess_posterior_bias_1d(summary, posterior, likelihood_reference=0.0)
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * theta**2,
        bounds=(-4.0, 4.0), points=1024,
    )
    with pytest.raises(ValueError):
        assess_posterior_bias_1d(
            summary, posterior, likelihood_reference=0.0,
            reference_standard_error=0.0,
        )


def test_prior_predictive_sampling_uses_finite_seed_and_preserves_statistic_shape():
    prior = SincModulatedPrior(location=0.5, bandwidth=1.0, scale=1.0)
    seen = []

    def simulator(theta, rng):
        seen.append(theta)
        return theta

    result = prior_predictive_check_1d(
        prior, simulator, bounds=(-4.0, 4.0),
        draws=200, points=1024, seed=77,
    )
    assert len(seen) == 200
    assert all(-4.0 <= value <= 4.0 for value in seen)
    assert result.observed_statistic is None
    assert result.two_sided_tail_probability is None


def test_prior_concentration_is_measured_relative_to_reference_uncertainty():
    prior = SincModulatedPrior(location=0.0, bandwidth=1.0, scale=0.5)
    summary = summarize_prior_1d(prior, bounds=(-4.0, 4.0), points=2048)
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * theta**2 / 25.0,
        bounds=(-4.0, 4.0), points=2048,
    )
    diagnostic = assess_posterior_bias_1d(
        summary, posterior, likelihood_reference=0.0,
        reference_standard_error=2.0,
        thresholds=PriorBiasThresholds(
            max_prior_reference_gap_z=100.0,
            max_posterior_pull_fraction=100.0,
            max_boundary_cell_mass=1.0,
            minimum_prior_to_reference_sd_ratio=0.5,
        ),
    )
    assert diagnostic.prior_to_reference_sd_ratio < 0.5
    assert any("Prior spread is concentrated" in warning for warning in diagnostic.warnings)


def test_prior_concentration_threshold_is_configurable():
    prior = SincModulatedPrior(location=0.0, bandwidth=1.0, scale=0.5)
    summary = summarize_prior_1d(prior, bounds=(-4.0, 4.0), points=2048)
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * theta**2 / 25.0,
        bounds=(-4.0, 4.0), points=2048,
    )
    diagnostic = assess_posterior_bias_1d(
        summary, posterior, likelihood_reference=0.0,
        reference_standard_error=2.0,
        thresholds=PriorBiasThresholds(
            max_prior_reference_gap_z=100.0,
            max_posterior_pull_fraction=100.0,
            max_boundary_cell_mass=1.0,
            minimum_prior_to_reference_sd_ratio=0.01,
        ),
    )
    assert not any("Prior spread is concentrated" in warning for warning in diagnostic.warnings)


def test_prior_bias_thresholds_reject_invalid_concentration_ratio():
    with pytest.raises(ValueError):
        PriorBiasThresholds(minimum_prior_to_reference_sd_ratio=-0.1)



def test_contamination_robust_prior_is_proper_and_more_diffuse():
    from math import log
    from gds_core.sinc_prior import SincPriorMixture, contamination_robust_prior

    nominal = SincModulatedPrior(
        location=-1.0, bandwidth=1.0, scale=0.8,
        center_frequency=0.2, modulation_depth=0.7,
    )
    robust = contamination_robust_prior(
        nominal, contamination_weight=0.15, scale_multiplier=6.0,
    )
    assert isinstance(robust, SincPriorMixture)
    assert sum(weight for weight, _ in robust.components) == pytest.approx(1.0)
    assert robust.density(0.0) > 0.0
    assert robust.log_density(0.0) == pytest.approx(log(robust.density(0.0)))
    nominal_summary = summarize_prior_1d(nominal, bounds=(-8.0, 8.0), points=4096)
    robust_summary = summarize_prior_1d(robust, bounds=(-8.0, 8.0), points=4096)
    assert robust_summary.conditional_standard_deviation > nominal_summary.conditional_standard_deviation
    assert robust_summary.conditional_mean == pytest.approx(nominal_summary.conditional_mean, abs=0.03)


def test_mixture_prior_can_be_used_in_posterior_and_predictive_check():
    from gds_core.sinc_prior import contamination_robust_prior

    prior = contamination_robust_prior(
        SincModulatedPrior(location=0.2, scale=0.6),
        contamination_weight=0.1,
        scale_multiplier=5.0,
    )
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * ((theta - 0.5) / 0.2) ** 2,
        bounds=(-6.0, 6.0), points=2048,
    )
    predictive = prior_predictive_check_1d(
        prior, lambda theta, rng: rng.gauss(theta, 0.1),
        bounds=(-6.0, 6.0), draws=100, points=1024, seed=9,
    )
    assert posterior.standard_deviation > 0.0
    assert predictive.predictive_variance > 0.0


def test_contamination_robust_prior_rejects_invalid_configuration():
    from gds_core.sinc_prior import contamination_robust_prior

    nominal = SincModulatedPrior()
    for kwargs in (
        {"contamination_weight": 0.0},
        {"contamination_weight": 1.0},
        {"contamination_weight": float("nan")},
        {"scale_multiplier": 1.0},
        {"scale_multiplier": float("inf")},
    ):
        with pytest.raises(ValueError):
            contamination_robust_prior(nominal, **kwargs)


def test_bias_diagnostic_warns_on_large_signed_overshoot():
    prior = SincModulatedPrior(location=1.0, bandwidth=1.0, scale=1.0)
    summary = summarize_prior_1d(prior, bounds=(-6.0, 6.0), points=2048)
    posterior = compute_posterior_1d(
        prior, lambda theta: -0.5 * ((theta + 1.0) / 0.15) ** 2,
        bounds=(-6.0, 6.0), points=4096,
    )
    diagnostic = assess_posterior_bias_1d(
        summary, posterior, likelihood_reference=0.0,
        reference_standard_error=0.15,
    )
    assert diagnostic.posterior_pull_fraction < 0.0
    assert abs(diagnostic.posterior_pull_fraction) > 0.5
    assert any("overshoot" in warning for warning in diagnostic.warnings)
