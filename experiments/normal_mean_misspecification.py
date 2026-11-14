"""Monte Carlo study of scalar Sinc-prior Bayesian sandwich variance.

This is a GDS experiment inspired by the Normal-mean misspecification example
in Li and Rice (2024), "A Bayesian 'Sandwich' for Variance Estimation",
Statistical Science 39(4), 589-600, DOI: 10.1214/24-STS935.

Unlike the paper's conjugate Normal-prior example, this experiment uses the
proper Sinc-modulated prior implemented by GDS. It is therefore an adapted
experiment, not an exact numerical replication of the published tables.

The working likelihood is Normal(theta, 1). Data are generated from
Normal(theta_true, true_sd**2). The model is correctly specified at true_sd=1
and variance-misspecified when true_sd differs from one.
"""

from __future__ import annotations

import argparse
import json
import random
from math import fsum, sqrt
from statistics import mean
from typing import Any

from gds_core.balanced_loss import balanced_loss_variance_1d
from gds_core.posterior import compute_posterior_1d
from gds_core.sinc_prior import SincModulatedPrior


Z_975 = 1.959963984540054


def run_scenario(
    *,
    true_sd: float,
    theta_true: float,
    sample_size: int,
    repetitions: int,
    points: int,
    rng: random.Random,
) -> dict[str, Any]:
    prior = SincModulatedPrior(
        location=0.0,
        bandwidth=1.0,
        scale=2.0,
        center_frequency=0.0,
        modulation_depth=0.0,
    )
    bounds = (-8.0, 8.0)

    estimates: list[float] = []
    posterior_ses: list[float] = []
    robust_ses: list[float] = []
    sandwich_ses: list[float] = []
    posterior_covered = 0
    robust_covered = 0
    robust_variances: list[float] = []
    sandwich_variances: list[float] = []

    for _ in range(repetitions):
        observations = tuple(
            rng.gauss(theta_true, true_sd) for _ in range(sample_size)
        )

        def log_likelihood(theta: float) -> float:
            squared_error = fsum(
                (value - theta) ** 2 for value in observations
            )
            return -0.5 * (
                sample_size * 1.8378770664093453 + squared_error
            )

        posterior = compute_posterior_1d(
            prior,
            log_likelihood,
            bounds=bounds,
            points=points,
            credible_level=0.95,
        )

        def score_information(theta: float) -> tuple[tuple[float, ...], float]:
            scores = tuple(value - theta for value in observations)
            # Under the working unit-variance Normal location model,
            # the average negative Hessian is I_n(theta) = 1.
            return scores, 1.0

        robust = balanced_loss_variance_1d(
            posterior, score_information
        )

        sample_mean = fsum(observations) / sample_size
        sample_variance = fsum(
            (value - sample_mean) ** 2 for value in observations
        ) / (sample_size - 1)
        sandwich_variance = sample_variance / sample_size

        estimates.append(posterior.mean)
        posterior_ses.append(posterior.standard_deviation)
        robust_ses.append(robust.robust_standard_error)
        sandwich_ses.append(sqrt(sandwich_variance))
        robust_variances.append(robust.robust_variance)
        sandwich_variances.append(sandwich_variance)

        lower_credible, upper_credible = posterior.credible_interval
        if lower_credible <= theta_true <= upper_credible:
            posterior_covered += 1

        robust_lower = posterior.mean - Z_975 * robust.robust_standard_error
        robust_upper = posterior.mean + Z_975 * robust.robust_standard_error
        if robust_lower <= theta_true <= robust_upper:
            robust_covered += 1

    errors = tuple(value - theta_true for value in estimates)
    bias = fsum(errors) / repetitions
    rmse = sqrt(fsum(value * value for value in errors) / repetitions)
    posterior_coverage = posterior_covered / repetitions
    robust_coverage = robust_covered / repetitions
    mean_robust_variance = mean(robust_variances)
    mean_sandwich_variance = mean(sandwich_variances)

    return {
        "true_sd": true_sd,
        "working_sd": 1.0,
        "model_correctly_specified": abs(true_sd - 1.0) < 1e-12,
        "bias": bias,
        "rmse": rmse,
        "mean_posterior_sd": mean(posterior_ses),
        "mean_balanced_loss_robust_se": mean(robust_ses),
        "mean_frequentist_sandwich_se": mean(sandwich_ses),
        "posterior_credible_interval_coverage": posterior_coverage,
        "posterior_coverage_mcse": sqrt(
            posterior_coverage * (1.0 - posterior_coverage) / repetitions
        ),
        "robust_wald_interval_coverage": robust_coverage,
        "robust_coverage_mcse": sqrt(
            robust_coverage * (1.0 - robust_coverage) / repetitions
        ),
        "mean_robust_variance": mean_robust_variance,
        "mean_frequentist_sandwich_variance": mean_sandwich_variance,
        "robust_to_sandwich_variance_ratio": (
            mean_robust_variance / mean_sandwich_variance
            if mean_sandwich_variance > 0.0 else None
        ),
    }


def run_experiment(
    *,
    sample_size: int = 80,
    repetitions: int = 200,
    points: int = 1024,
    seed: int = 20261114,
    theta_true: float = 0.5,
    true_sds: tuple[float, ...] = (1.0, 2.0),
) -> dict[str, Any]:
    if sample_size < 2:
        raise ValueError("sample_size must be at least two")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if points < 16:
        raise ValueError("points must be at least 16")
    if not true_sds or any(value <= 0.0 for value in true_sds):
        raise ValueError("true_sds must contain positive values")

    rng = random.Random(seed)
    results = [
        run_scenario(
            true_sd=true_sd,
            theta_true=theta_true,
            sample_size=sample_size,
            repetitions=repetitions,
            points=points,
            rng=rng,
        )
        for true_sd in true_sds
    ]
    return {
        "experiment": "GDS Sinc-prior Normal-mean misspecification study",
        "reference": "Li and Rice (2024), DOI 10.1214/24-STS935",
        "design_note": (
            "Adapted experiment using the GDS Sinc prior; not an exact "
            "replication of the paper's Normal-prior example."
        ),
        "settings": {
            "sample_size": sample_size,
            "repetitions_per_scenario": repetitions,
            "quadrature_points": points,
            "seed": seed,
            "true_parameter": theta_true,
            "working_likelihood": "Normal(theta, 1)",
            "prior": "SincModulatedPrior(location=0, bandwidth=1, scale=2, rho=0)",
            "parameter_bounds": [-8.0, 8.0],
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--theta-true", type=float, default=0.5)
    parser.add_argument("--true-sds", type=float, nargs="+", default=(1.0, 2.0))
    args = parser.parse_args()

    result = run_experiment(
        sample_size=args.sample_size,
        repetitions=args.repetitions,
        points=args.points,
        seed=args.seed,
        theta_true=args.theta_true,
        true_sds=tuple(args.true_sds),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
