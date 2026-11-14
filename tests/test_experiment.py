from experiments.normal_mean_misspecification import run_experiment


def test_normal_mean_misspecification_experiment_is_reproducible():
    kwargs = {
        "sample_size": 24,
        "repetitions": 5,
        "points": 256,
        "seed": 12345,
        "true_sds": (1.0, 1.8),
    }
    first = run_experiment(**kwargs)
    second = run_experiment(**kwargs)

    assert first == second
    assert len(first["results"]) == 2
    for result in first["results"]:
        assert result["rmse"] >= 0.0
        assert result["mean_posterior_sd"] > 0.0
        assert result["mean_balanced_loss_robust_se"] > 0.0
        assert result["mean_frequentist_sandwich_se"] > 0.0
        assert 0.0 <= result["posterior_credible_interval_coverage"] <= 1.0
        assert 0.0 <= result["robust_wald_interval_coverage"] <= 1.0
        assert result["robust_to_sandwich_variance_ratio"] > 0.0
