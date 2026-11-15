import math

import pytest

from experiments.normal_mean_misspecification import run_experiment


def test_expanded_experiment_is_reproducible_and_prior_paired():
    kwargs = {
        "sample_sizes": (24,),
        "repetitions": 4,
        "points": 96,
        "seed": 12345,
        "scenarios": ("normal_sd_1", "student_t5_var4"),
        "priors": ("centered_diffuse", "shifted_diffuse"),
    }
    first = run_experiment(**kwargs)
    second = run_experiment(**kwargs)

    assert first == second
    assert first["settings"]["prior_comparisons_use_common_random_datasets"] is True
    assert len(first["results"]) == 4

    for result in first["results"]:
        assert math.isfinite(result["bias"])
        assert result["rmse"] >= 0.0
        assert result["mean_posterior_sd"] > 0.0
        assert result["mean_balanced_loss_robust_se"] > 0.0
        assert result["mean_frequentist_sandwich_se"] > 0.0
        assert result["mean_oracle_se"] > 0.0
        assert 0.0 <= result["posterior_credible_interval_coverage"] <= 1.0
        assert 0.0 <= result["robust_wald_interval_coverage"] <= 1.0
        assert result["robust_to_frequentist_sandwich_variance_ratio"] > 0.0
        assert result["robust_to_oracle_variance_ratio"] > 0.0


def test_heteroskedastic_generator_has_finite_oracle_variance():
    from experiments.normal_mean_misspecification import _draw_data
    import random

    observations, oracle_variance = _draw_data(
        scenario="heteroskedastic_normal_var4",
        theta_true=0.5,
        sample_size=80,
        rng=random.Random(7),
    )
    assert len(observations) == 80
    assert all(isinstance(value, float) for value in observations)
    assert oracle_variance == pytest.approx(4.0 / 80.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_sizes": ()},
        {"sample_sizes": (10, 10)},
        {"sample_sizes": (1,)},
        {"repetitions": 0},
        {"points": 15},
        {"scenarios": ("unknown",)},
        {"priors": ("unknown",)},
        {"theta_true": 8.0},
    ],
)
def test_expanded_experiment_rejects_invalid_settings(kwargs):
    settings = {
        "sample_sizes": (24,),
        "repetitions": 2,
        "points": 64,
        "scenarios": ("normal_sd_1",),
        "priors": ("centered_diffuse",),
    }
    settings.update(kwargs)
    with pytest.raises(ValueError):
        run_experiment(**settings)
