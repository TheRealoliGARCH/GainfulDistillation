import pytest

from experiments.prior_bias_control_stress import run_stress_grid


def test_bias_control_stress_grid_is_reproducible_and_paired():
    kwargs = {
        "sample_sizes": (24,),
        "repetitions": 4,
        "points": 96,
        "seed": 13579,
        "scenarios": ("normal_sd_1", "student_t5_var4"),
        "location_offsets": (-0.5, 0.5),
        "contamination_weights": (0.1, 0.2),
        "scale_multipliers": (3.0,),
    }
    first = run_stress_grid(**kwargs)
    second = run_stress_grid(**kwargs)
    assert first == second
    assert first["settings"]["common_random_numbers_across_prior_variants"] is True
    assert first["settings"]["cell_count"] == len(first["results"]) == 14
    assert {row["scenario"] for row in first["results"]} == {
        "normal_sd_1", "student_t5_var4",
    }
    assert {row["prior_variant"] for row in first["results"]} == {
        "nominal", "diffuse_contamination", "centred_diffuse_reference",
    }
    for row in first["results"]:
        assert row["rmse"] >= 0.0
        assert row["bias_mcse"] >= 0.0
        assert row["mean_balanced_loss_robust_se"] > 0.0
        assert row["mean_oracle_variance"] > 0.0
        assert row["robust_to_oracle_variance_ratio"] > 0.0
        assert 0.0 <= row["posterior_credible_interval_coverage"] <= 1.0
        assert 0.0 <= row["robust_wald_interval_coverage"] <= 1.0
        assert 0.0 <= row["prior_conflict_warning_rate"] <= 1.0


def test_heteroskedastic_generator_reports_oracle_mean_variance():
    from experiments.prior_bias_control_stress import _draw_sample
    import random

    _, variance = _draw_sample(
        random.Random(4),
        scenario="heteroskedastic_normal_var4",
        theta_true=0.5,
        sample_size=80,
    )
    assert variance == pytest.approx(4.0 / 80.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_sizes": ()},
        {"sample_sizes": (1,)},
        {"sample_sizes": (20, 20)},
        {"sample_sizes": (True,)},
        {"repetitions": 1},
        {"points": 32},
        {"scenarios": ()},
        {"scenarios": ("not-a-scenario",)},
        {"location_offsets": ()},
        {"location_offsets": (float("nan"),)},
        {"contamination_weights": (0.0,)},
        {"scale_multipliers": (1.0,)},
    ],
)
def test_bias_control_stress_grid_rejects_invalid_settings(kwargs):
    settings = {
        "sample_sizes": (24,),
        "repetitions": 2,
        "points": 64,
        "scenarios": ("normal_sd_1",),
        "location_offsets": (-0.5,),
        "contamination_weights": (0.1,),
        "scale_multipliers": (3.0,),
    }
    settings.update(kwargs)
    with pytest.raises(ValueError):
        run_stress_grid(**settings)


def test_cached_posterior_matches_reference_midpoint_engine():
    from gds_core.posterior import compute_posterior_1d
    from gds_core.sinc_prior import contamination_robust_prior
    from experiments.prior_bias_control_stress import (
        _build_prior_grid_cache,
        _compute_posterior_from_cache,
    )

    prior = contamination_robust_prior(
        SincModulatedPrior(location=-0.5, scale=2.0),
        contamination_weight=0.2,
        scale_multiplier=5.0,
    )
    log_likelihood = lambda theta: -0.5 * ((theta - 0.3) / 0.4) ** 2
    bounds = (-16.0, 16.0)
    points = 1024
    reference = compute_posterior_1d(
        prior, log_likelihood, bounds=bounds, points=points,
    )
    cache = _build_prior_grid_cache(prior, bounds=bounds, points=points)
    cached = _compute_posterior_from_cache(cache, log_likelihood)
    assert cached.mean == pytest.approx(reference.mean, abs=1e-12)
    assert cached.variance == pytest.approx(reference.variance, abs=1e-12)
    assert cached.credible_interval == pytest.approx(reference.credible_interval, abs=1e-12)
    assert cached.log_marginal_likelihood == pytest.approx(
        reference.log_marginal_likelihood, abs=1e-12,
    )
