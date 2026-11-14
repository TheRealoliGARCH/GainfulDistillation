"""Monte Carlo study of diffuse-contamination prior robustification.

The design isolates prior-induced location bias under a correctly specified
Normal likelihood. Each common dataset is evaluated with the shifted nominal
Sinc prior, same-centre diffuse-contamination mixtures, and a centred diffuse
benchmark. Robustification weakens concentration but does not recenter a
misspecified prior.
"""

from __future__ import annotations

import argparse
import json
import random
from math import fsum, sqrt
from statistics import mean
from typing import Any

from gds_core.balanced_loss import ScalarScoreSummary, balanced_loss_variance_1d
from gds_core.posterior import compute_posterior_1d
from gds_core.prior_diagnostics import (
    PriorBiasThresholds,
    assess_posterior_bias_1d,
    summarize_prior_1d,
)
from gds_core.sinc_prior import (
    SincModulatedPrior,
    SincPriorMixture,
    contamination_robust_prior,
)

Z_975 = 1.959963984540054
PARAMETER_BOUNDS = (-8.0, 8.0)


def _priors() -> dict[str, SincModulatedPrior | SincPriorMixture]:
    shifted = SincModulatedPrior(
        location=-1.0, bandwidth=1.0, scale=2.0,
        center_frequency=0.0, modulation_depth=0.0,
    )
    return {
        "shifted_nominal": shifted,
        "contamination_05": contamination_robust_prior(
            shifted, contamination_weight=0.05, scale_multiplier=5.0,
        ),
        "contamination_10": contamination_robust_prior(
            shifted, contamination_weight=0.10, scale_multiplier=5.0,
        ),
        "contamination_20": contamination_robust_prior(
            shifted, contamination_weight=0.20, scale_multiplier=5.0,
        ),
        "centered_diffuse_reference": SincModulatedPrior(
            location=0.0, bandwidth=1.0, scale=2.0,
            center_frequency=0.0, modulation_depth=0.0,
        ),
    }


def _mean_squared_score(
    observations: tuple[float, ...], theta: float, sample_mean: float,
    centred_sum_squares: float,
) -> float:
    n = len(observations)
    return centred_sum_squares / n + (sample_mean - theta) ** 2


def _fit_one(
    observations: tuple[float, ...],
    *,
    theta_true: float,
    prior: SincModulatedPrior | SincPriorMixture,
    prior_summary: Any,
    points: int,
) -> dict[str, float]:
    n = len(observations)
    sample_mean = fsum(observations) / n
    centred_sum_squares = fsum((value - sample_mean) ** 2 for value in observations)

    def log_likelihood(theta: float) -> float:
        squared_error = centred_sum_squares + n * (sample_mean - theta) ** 2
        return -0.5 * (n * 1.8378770664093453 + squared_error)

    posterior = compute_posterior_1d(
        prior, log_likelihood, bounds=PARAMETER_BOUNDS,
        points=points, credible_level=0.95,
    )

    def score_information(theta: float) -> tuple[ScalarScoreSummary, float]:
        return ScalarScoreSummary(
            n,
            _mean_squared_score(
                observations, theta, sample_mean, centred_sum_squares,
            ),
        ), 1.0

    robust = balanced_loss_variance_1d(posterior, score_information)
    diagnostic = assess_posterior_bias_1d(
        prior_summary,
        posterior,
        likelihood_reference=sample_mean,
        reference_standard_error=1.0 / sqrt(n),
        thresholds=PriorBiasThresholds(),
    )
    lower, upper = posterior.credible_interval
    robust_lower = posterior.mean - Z_975 * robust.robust_standard_error
    robust_upper = posterior.mean + Z_975 * robust.robust_standard_error
    error = posterior.mean - theta_true
    warning_text = " | ".join(diagnostic.warnings)

    return {
        "posterior_error": error,
        "squared_error": error * error,
        "posterior_sd": posterior.standard_deviation,
        "robust_se": robust.robust_standard_error,
        "posterior_covered": float(lower <= theta_true <= upper),
        "robust_wald_covered": float(robust_lower <= theta_true <= robust_upper),
        "absolute_posterior_pull": (
            abs(diagnostic.posterior_pull_fraction)
            if diagnostic.posterior_pull_fraction is not None else 0.0
        ),
        "posterior_pull": (
            diagnostic.posterior_pull_fraction
            if diagnostic.posterior_pull_fraction is not None else 0.0
        ),
        "prior_conflict_flag": float(
            any("Prior mean is materially separated" in warning for warning in diagnostic.warnings)
        ),
        "concentration_flag": float(
            any("Prior spread is concentrated" in warning for warning in diagnostic.warnings)
        ),
        "posterior_displacement_flag": float(
            any("materially displaced" in warning for warning in diagnostic.warnings)
        ),
        "warning_count": float(len(diagnostic.warnings)),
        "robust_variance": robust.robust_variance,
    }


def _aggregate(
    records: list[dict[str, float]],
    *,
    name: str,
    sample_size: int,
    repetitions: int,
    theta_true: float,
) -> dict[str, Any]:
    bias = mean(record["posterior_error"] for record in records)
    posterior_coverage = mean(record["posterior_covered"] for record in records)
    robust_coverage = mean(record["robust_wald_covered"] for record in records)
    return {
        "prior": name,
        "sample_size": sample_size,
        "repetitions": repetitions,
        "true_parameter": theta_true,
        "bias": bias,
        "absolute_bias": abs(bias),
        "bias_mcse": sqrt(
            fsum((record["posterior_error"] - bias) ** 2 for record in records)
            / (repetitions * (repetitions - 1))
        ),
        "rmse": sqrt(mean(record["squared_error"] for record in records)),
        "posterior_credible_interval_coverage": posterior_coverage,
        "posterior_coverage_mcse": sqrt(
            posterior_coverage * (1.0 - posterior_coverage) / repetitions
        ),
        "robust_wald_interval_coverage": robust_coverage,
        "robust_coverage_mcse": sqrt(
            robust_coverage * (1.0 - robust_coverage) / repetitions
        ),
        "mean_posterior_sd": mean(record["posterior_sd"] for record in records),
        "mean_balanced_loss_robust_se": mean(record["robust_se"] for record in records),
        "mean_absolute_posterior_pull": mean(
            record["absolute_posterior_pull"] for record in records
        ),
        "mean_signed_posterior_pull": mean(
            record["posterior_pull"] for record in records
        ),
        "prior_conflict_warning_rate": mean(
            record["prior_conflict_flag"] for record in records
        ),
        "prior_concentration_warning_rate": mean(
            record["concentration_flag"] for record in records
        ),
        "posterior_displacement_warning_rate": mean(
            record["posterior_displacement_flag"] for record in records
        ),
        "mean_diagnostic_warning_count": mean(
            record["warning_count"] for record in records
        ),
        "mean_balanced_loss_variance": mean(
            record["robust_variance"] for record in records
        ),
        "oracle_variance_of_sample_mean": 1.0 / sample_size,
    }


def run_experiment(
    *,
    sample_sizes: tuple[int, ...] = (40, 80, 160),
    repetitions: int = 200,
    points: int = 1024,
    seed: int = 20261114,
    theta_true: float = 0.5,
) -> dict[str, Any]:
    """Run paired prior-bias-control simulations."""
    if not sample_sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in sample_sizes
    ):
        raise ValueError("sample_sizes must contain integers of at least two")
    if len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("sample_sizes must be unique")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("repetitions must be an integer of at least two")
    if isinstance(points, bool) or not isinstance(points, int) or points < 64:
        raise ValueError("points must be an integer of at least 64")
    if isinstance(theta_true, bool) or not isinstance(theta_true, (int, float)):
        raise ValueError("theta_true must be finite")
    if not -8.0 < float(theta_true) < 8.0:
        raise ValueError("theta_true must lie strictly inside the parameter bounds")

    priors = _priors()
    summaries = {
        name: summarize_prior_1d(
            prior, bounds=PARAMETER_BOUNDS, points=max(points, 4096),
        )
        for name, prior in priors.items()
    }
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    for sample_size in sample_sizes:
        buckets = {name: [] for name in priors}
        for _ in range(repetitions):
            observations = tuple(
                rng.gauss(float(theta_true), 1.0) for _ in range(sample_size)
            )
            for name, prior in priors.items():
                buckets[name].append(
                    _fit_one(
                        observations,
                        theta_true=float(theta_true),
                        prior=prior,
                        prior_summary=summaries[name],
                        points=points,
                    )
                )
        for name, records in buckets.items():
            results.append(
                _aggregate(
                    records, name=name, sample_size=sample_size,
                    repetitions=repetitions, theta_true=float(theta_true),
                )
            )

    return {
        "experiment": "GDS diffuse-contamination prior bias-control study",
        "design_note": (
            "Independent Normal observations with SD 1 and a correctly specified "
            "Normal(theta,1) working likelihood. This isolates prior-induced "
            "finite-sample bias. Prior comparisons use common random datasets. "
            "The contamination mixtures share the shifted nominal prior's centre; "
            "they can weaken concentration but cannot repair a wrong prior centre."
        ),
        "settings": {
            "sample_sizes": list(sample_sizes),
            "repetitions_per_cell": repetitions,
            "quadrature_points": points,
            "seed": seed,
            "true_parameter": float(theta_true),
            "working_likelihood": "independent Normal(theta, 1)",
            "parameter_bounds": list(PARAMETER_BOUNDS),
            "common_random_numbers_across_priors": True,
        },
        "prior_definitions": {
            "shifted_nominal": {
                "location": -1.0, "scale": 2.0,
                "type": "single SincModulatedPrior",
            },
            "contamination_05": {
                "weight": 0.05, "diffuse_scale_multiplier": 5.0,
                "diffuse_component_location": -1.0,
            },
            "contamination_10": {
                "weight": 0.10, "diffuse_scale_multiplier": 5.0,
                "diffuse_component_location": -1.0,
            },
            "contamination_20": {
                "weight": 0.20, "diffuse_scale_multiplier": 5.0,
                "diffuse_component_location": -1.0,
            },
            "centered_diffuse_reference": {
                "location": 0.0, "scale": 2.0,
                "type": "single SincModulatedPrior",
            },
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=(40, 80, 160))
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--theta-true", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = run_experiment(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.repetitions,
        points=args.points,
        seed=args.seed,
        theta_true=args.theta_true,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
