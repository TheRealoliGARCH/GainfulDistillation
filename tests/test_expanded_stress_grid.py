import pytest

from experiments.expanded_stress_grid import (
    PRIORS,
    SCENARIOS,
    run_stress_grid,
)


def test_expanded_stress_grid_is_reproducible_and_returns_full_cell_grid():
    kwargs = {
        "sample_sizes": (20,),
        "repetitions": 3,
        "points": 128,
        "seed": 9876,
        "scenarios": SCENARIOS[:2],
        "priors": PRIORS[:2],
    }
    first = run_stress_grid(**kwargs)
    second = run_stress_grid(**kwargs)

    assert first == second
    assert first["settings"]["cell_count"] == 4
    assert len(first["results"]) == 4
    for result in first["results"]:
        assert result["rmse"] >= 0.0
        assert result["mean_posterior_sd"] > 0.0
        assert result["mean_balanced_loss_robust_se"] > 0.0
        assert result["mean_frequentist_sandwich_se"] > 0.0
        assert 0.0 <= result["posterior_credible_interval_coverage"] <= 1.0
        assert 0.0 <= result["robust_wald_interval_coverage"] <= 1.0
        assert result["robust_to_empirical_sandwich_variance_ratio"] > 0.0
        assert result["robust_to_theoretical_variance_ratio"] > 0.0


def test_stress_grid_uses_common_random_datasets_across_prior_variants():
    result = run_stress_grid(
        sample_sizes=(24,),
        repetitions=4,
        points=128,
        seed=456,
        scenarios=(SCENARIOS[0],),
        priors=PRIORS[:3],
    )
    rows = result["results"]
    assert len(rows) == 3
    # All priors see the same simulated data. With this relatively small n,
    # the posterior estimates may differ slightly, but their RMSE inputs
    # have been produced from the same underlying four datasets.
    assert {row["sample_size"] for row in rows} == {24}
    assert {row["repetitions"] for row in rows} == {4}


@pytest.mark.parametrize("sample_sizes", [(), (1,), (20, 0), (True,)])
def test_invalid_sample_size_grid_is_rejected(sample_sizes):
    with pytest.raises(ValueError):
        run_stress_grid(
            sample_sizes=sample_sizes,
            repetitions=2,
            points=64,
            scenarios=(SCENARIOS[0],),
            priors=(PRIORS[0],),
        )


def test_invalid_repetitions_points_and_empty_dimensions_are_rejected():
    for kwargs in [
        {"repetitions": 1},
        {"points": 15},
        {"scenarios": ()},
        {"priors": ()},
    ]:
        arguments = {
            "sample_sizes": (20,),
            "repetitions": 2,
            "points": 64,
            "scenarios": (SCENARIOS[0],),
            "priors": (PRIORS[0],),
        }
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            run_stress_grid(**arguments)
