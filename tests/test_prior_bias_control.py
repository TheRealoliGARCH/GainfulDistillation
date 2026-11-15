import pytest

from experiments.prior_bias_control import run_experiment


def test_prior_bias_control_is_reproducible_and_reports_all_variants():
    kwargs = {
        "sample_sizes": (24,),
        "repetitions": 4,
        "points": 128,
        "seed": 2468,
    }
    first = run_experiment(**kwargs)
    second = run_experiment(**kwargs)
    assert first == second
    assert first["settings"]["common_random_numbers_across_priors"] is True
    rows = first["results"]
    assert len(rows) == 5
    assert {row["prior"] for row in rows} == {
        "shifted_nominal", "contamination_05", "contamination_10",
        "contamination_20", "centered_diffuse_reference",
    }
    for row in rows:
        assert row["rmse"] >= 0.0
        assert row["bias_mcse"] >= 0.0
        assert row["mean_balanced_loss_robust_se"] > 0.0
        assert 0.0 <= row["posterior_credible_interval_coverage"] <= 1.0
        assert 0.0 <= row["robust_wald_interval_coverage"] <= 1.0
        assert 0.0 <= row["prior_conflict_warning_rate"] <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_sizes": ()},
        {"sample_sizes": (1,)},
        {"sample_sizes": (20, 20)},
        {"sample_sizes": (True,)},
        {"repetitions": 1},
        {"points": 32},
        {"theta_true": 8.0},
    ],
)
def test_prior_bias_control_rejects_invalid_settings(kwargs):
    settings = {"sample_sizes": (24,), "repetitions": 2, "points": 64}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        run_experiment(**settings)
