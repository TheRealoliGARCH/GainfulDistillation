"""Expanded Monte Carlo stress grid for GDS Sinc-prior robust inference.

Design:
- Correctly specified Normal location model.
- Normal variance misspecification at multiple degrees.
- Standardised Student-t and Laplace heavy tails.
- Random-design heteroscedastic errors.
- Standardised Gaussian contamination.
- Standardised skewed Gamma errors.
- Multiple Sinc-prior scale and spectral-modulation choices.
- Multiple sample sizes.

All generated observation pairs are iid across i. The working likelihood is
Normal(theta, 1); the scalar balanced-loss score and average empirical
information are therefore s_i(theta)=y_i-theta and I_n(theta)=1.

This is a GDS adaptation, not an exact replication of Li and Rice (2024).
The experiment conditions each proper prior on the declared finite interval.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from math import fsum, log, sqrt
from statistics import mean
from typing import Any

from gds_core.balanced_loss import (
    ScalarScoreSummary,
    balanced_loss_variance_1d,
)
from gds_core.posterior import compute_posterior_1d
from gds_core.sinc_prior import SincModulatedPrior


Z_975 = 1.959963984540054
LOG_2PI = log(2.0 * 3.141592653589793)


@dataclass(frozen=True)
class DataScenario:
    name: str
    family: str
    parameter: float
    theoretical_variance: float
    correctly_specified: bool
    description: str


SCENARIOS = (
    DataScenario(
        "normal_sd_1",
        "normal",
        1.0,
        1.0,
        True,
        "Normal errors with unit standard deviation.",
    ),
    DataScenario(
        "normal_sd_1_5",
        "normal",
        1.5,
        2.25,
        False,
        "Normal errors with standard deviation 1.5; working model fixes it at 1.",
    ),
    DataScenario(
        "normal_sd_2",
        "normal",
        2.0,
        4.0,
        False,
        "Normal errors with standard deviation 2; working model fixes it at 1.",
    ),
    DataScenario(
        "student_t5_unit_variance",
        "student_t5",
        5.0,
        1.0,
        False,
        "Student-t(5) errors rescaled to unit variance.",
    ),
    DataScenario(
        "laplace_unit_variance",
        "laplace",
        1.0,
        1.0,
        False,
        "Symmetric Laplace errors rescaled to unit variance.",
    ),
    DataScenario(
        "heteroscedastic_mean_variance_1_5",
        "heteroscedastic",
        0.0,
        1.5,
        False,
        "Random-design conditional variance 0.5 + 3*X^2, X~Uniform(-1,1).",
    ),
    DataScenario(
        "contaminated_normal_unit_variance",
        "contaminated",
        4.0,
        1.0,
        False,
        "95% N(0,1), 5% N(0,16), rescaled to unit marginal variance.",
    ),
    DataScenario(
        "skewed_gamma_unit_variance",
        "gamma",
        2.0,
        1.0,
        False,
        "Gamma(shape=2, scale=1) errors centred and rescaled to unit variance.",
    ),
)


@dataclass(frozen=True)
class PriorSpec:
    name: str
    location: float
    bandwidth: float
    scale: float
    center_frequency: float
    modulation_depth: float

    def build(self) -> SincModulatedPrior:
        return SincModulatedPrior(
            location=self.location,
            bandwidth=self.bandwidth,
            scale=self.scale,
            center_frequency=self.center_frequency,
            modulation_depth=self.modulation_depth,
        )


PRIORS = (
    PriorSpec("baseline", 0.0, 1.0, 2.0, 0.0, 0.0),
    PriorSpec("concentrated", 0.0, 1.0, 0.75, 0.0, 0.0),
    PriorSpec("diffuse", 0.0, 0.8, 4.0, 0.0, 0.0),
    PriorSpec("spectrally_modulated", 0.0, 1.0, 2.0, 0.45, 0.8),
)


def _draw_one_error(rng: random.Random, scenario: DataScenario) -> float:
    family = scenario.family
    if family == "normal":
        return rng.gauss(0.0, scenario.parameter)
    if family == "student_t5":
        degrees = 5.0
        chi_square_over_df = rng.gammavariate(degrees / 2.0, 2.0 / degrees)
        standard_t = rng.gauss(0.0, 1.0) / sqrt(chi_square_over_df)
        return standard_t * sqrt((degrees - 2.0) / degrees)
    if family == "laplace":
        # Inverse-CDF Laplace with scale 1/sqrt(2), hence unit variance.
        u = rng.random() - 0.5
        scale = 1.0 / sqrt(2.0)
        if u == 0.0:
            return 0.0
        return -scale * (1.0 if u > 0.0 else -1.0) * log(1.0 - 2.0 * abs(u))
    if family == "heteroscedastic":
        x = rng.uniform(-1.0, 1.0)
        conditional_sd = sqrt(0.5 + 3.0 * x * x)
        return rng.gauss(0.0, conditional_sd)
    if family == "contaminated":
        component_sd = 4.0 if rng.random() < 0.05 else 1.0
        return rng.gauss(0.0, component_sd / sqrt(1.75))
    if family == "gamma":
        return (rng.gammavariate(2.0, 1.0) - 2.0) / sqrt(2.0)
    raise ValueError(f"unknown data-generating family: {family}")


def _draw_sample(
    rng: random.Random,
    scenario: DataScenario,
    *,
    theta_true: float,
    sample_size: int,
) -> tuple[float, ...]:
    return tuple(
        theta_true + _draw_one_error(rng, scenario)
        for _ in range(sample_size)
    )


def _summarise_fit(
    observations: tuple[float, ...],
    *,
    theta_true: float,
    prior: SincModulatedPrior,
    bounds: tuple[float, float],
    points: int,
) -> dict[str, float]:
    n = len(observations)
    sample_mean = fsum(observations) / n
    centred_sum_squares = fsum(
        (value - sample_mean) ** 2 for value in observations
    )
    sample_variance = centred_sum_squares / (n - 1)

    def log_likelihood(theta: float) -> float:
        squared_error = centred_sum_squares + n * (sample_mean - theta) ** 2
        return -0.5 * (n * LOG_2PI + squared_error)

    posterior = compute_posterior_1d(
        prior,
        log_likelihood,
        bounds=bounds,
        points=points,
        credible_level=0.95,
    )

    def score_information(theta: float):
        mean_squared_score = (
            centred_sum_squares / n + (sample_mean - theta) ** 2
        )
        return ScalarScoreSummary(n, mean_squared_score), 1.0

    robust = balanced_loss_variance_1d(posterior, score_information)
    sandwich_variance = sample_variance / n
    sandwich_se = sqrt(sandwich_variance)

    credible_lower, credible_upper = posterior.credible_interval
    robust_lower = posterior.mean - Z_975 * robust.robust_standard_error
    robust_upper = posterior.mean + Z_975 * robust.robust_standard_error
    sandwich_lower = sample_mean - Z_975 * sandwich_se
    sandwich_upper = sample_mean + Z_975 * sandwich_se

    return {
        "estimate": posterior.mean,
        "error": posterior.mean - theta_true,
        "mle_estimate": sample_mean,
        "mle_error": sample_mean - theta_true,
        "posterior_sd": posterior.standard_deviation,
        "robust_se": robust.robust_standard_error,
        "sandwich_se": sandwich_se,
        "posterior_covered": float(credible_lower <= theta_true <= credible_upper),
        "robust_covered": float(robust_lower <= theta_true <= robust_upper),
        "sandwich_wald_covered": float(sandwich_lower <= theta_true <= sandwich_upper),
        "posterior_interval_width": credible_upper - credible_lower,
        "robust_interval_width": robust_upper - robust_lower,
        "sandwich_wald_interval_width": sandwich_upper - sandwich_lower,
        "robust_variance": robust.robust_variance,
        "sandwich_variance": sandwich_variance,
        "boundary_cell_mass": (
            posterior.probability_masses[0] + posterior.probability_masses[-1]
        ),
    }


def _aggregate(
    records: list[dict[str, float]],
    *,
    scenario: DataScenario,
    prior_spec: PriorSpec,
    sample_size: int,
    repetitions: int,
) -> dict[str, Any]:
    errors = tuple(record["error"] for record in records)
    mle_errors = tuple(record["mle_error"] for record in records)
    bias = mean(errors)
    mle_bias = mean(mle_errors)
    error_sd = sqrt(
        fsum((value - bias) ** 2 for value in errors) / max(1, repetitions - 1)
    )
    mle_error_sd = sqrt(
        fsum((value - mle_bias) ** 2 for value in mle_errors) / max(1, repetitions - 1)
    )
    posterior_coverage = mean(record["posterior_covered"] for record in records)
    robust_coverage = mean(record["robust_covered"] for record in records)
    sandwich_coverage = mean(record["sandwich_wald_covered"] for record in records)
    avg_robust_variance = mean(record["robust_variance"] for record in records)
    avg_sandwich_variance = mean(record["sandwich_variance"] for record in records)
    theoretical_variance_of_mean = scenario.theoretical_variance / sample_size

    return {
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "correctly_specified_working_model": scenario.correctly_specified,
        "theoretical_observation_variance": scenario.theoretical_variance,
        "prior": {
            "name": prior_spec.name,
            "location": prior_spec.location,
            "bandwidth": prior_spec.bandwidth,
            "scale": prior_spec.scale,
            "center_frequency": prior_spec.center_frequency,
            "modulation_depth": prior_spec.modulation_depth,
        },
        "sample_size": sample_size,
        "repetitions": repetitions,
        "bias": bias,
        "bias_mcse": error_sd / sqrt(repetitions),
        "rmse": sqrt(mean(value * value for value in errors)),
        "mle_bias": mle_bias,
        "mle_bias_mcse": mle_error_sd / sqrt(repetitions),
        "mle_rmse": sqrt(mean(value * value for value in mle_errors)),
        "mean_posterior_sd": mean(record["posterior_sd"] for record in records),
        "mean_balanced_loss_robust_se": mean(record["robust_se"] for record in records),
        "mean_frequentist_sandwich_se": mean(record["sandwich_se"] for record in records),
        "posterior_credible_interval_coverage": posterior_coverage,
        "posterior_coverage_mcse": sqrt(
            posterior_coverage * (1.0 - posterior_coverage) / repetitions
        ),
        "robust_wald_interval_coverage": robust_coverage,
        "robust_coverage_mcse": sqrt(
            robust_coverage * (1.0 - robust_coverage) / repetitions
        ),
        "frequentist_sandwich_wald_interval_coverage": sandwich_coverage,
        "frequentist_sandwich_coverage_mcse": sqrt(
            sandwich_coverage * (1.0 - sandwich_coverage) / repetitions
        ),
        "mean_posterior_credible_interval_width": mean(
            record["posterior_interval_width"] for record in records
        ),
        "mean_robust_wald_interval_width": mean(
            record["robust_interval_width"] for record in records
        ),
        "mean_frequentist_sandwich_wald_interval_width": mean(
            record["sandwich_wald_interval_width"] for record in records
        ),
        "robust_to_empirical_sandwich_variance_ratio": (
            avg_robust_variance / avg_sandwich_variance
            if avg_sandwich_variance > 0.0 else None
        ),
        "robust_to_theoretical_variance_ratio": (
            avg_robust_variance / theoretical_variance_of_mean
        ),
        "mean_posterior_boundary_cell_mass": mean(
            record["boundary_cell_mass"] for record in records
        ),
    }


def run_stress_grid(
    *,
    sample_sizes: tuple[int, ...] = (40, 80, 160),
    repetitions: int = 200,
    points: int = 2048,
    seed: int = 20261115,
    theta_true: float = 0.5,
    bounds: tuple[float, float] = (-8.0, 8.0),
    scenarios: tuple[DataScenario, ...] = SCENARIOS,
    priors: tuple[PriorSpec, ...] = PRIORS,
) -> dict[str, Any]:
    if not sample_sizes or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in sample_sizes
    ):
        raise ValueError("sample_sizes must contain integers of at least two")
    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if points < 16:
        raise ValueError("points must be at least 16")
    if not scenarios or not priors:
        raise ValueError("at least one scenario and prior are required")
    if bounds[0] >= bounds[1]:
        raise ValueError("bounds must be strictly increasing")

    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    # Common random numbers: each generated dataset is evaluated under every
    # prior specification, improving the comparability of prior sensitivity.
    for scenario in scenarios:
        for sample_size in sample_sizes:
            buckets: dict[str, list[dict[str, float]]] = {
                prior_spec.name: [] for prior_spec in priors
            }
            prior_objects = {
                prior_spec.name: prior_spec.build() for prior_spec in priors
            }

            for _ in range(repetitions):
                observations = _draw_sample(
                    rng,
                    scenario,
                    theta_true=theta_true,
                    sample_size=sample_size,
                )
                for prior_spec in priors:
                    fit = _summarise_fit(
                        observations,
                        theta_true=theta_true,
                        prior=prior_objects[prior_spec.name],
                        bounds=bounds,
                        points=points,
                    )
                    buckets[prior_spec.name].append(fit)

            for prior_spec in priors:
                results.append(
                    _aggregate(
                        buckets[prior_spec.name],
                        scenario=scenario,
                        prior_spec=prior_spec,
                        sample_size=sample_size,
                        repetitions=repetitions,
                    )
                )

    return {
        "experiment": "GDS expanded Sinc-prior Bayesian robust-inference stress grid",
        "reference": "Li and Rice (2024), Statistical Science 39(4), 589-600, DOI 10.1214/24-STS935",
        "design_note": (
            "GDS adaptation using Sinc-modulated priors and a Normal(theta,1) "
            "working likelihood. It is not an exact replication of Li and Rice. "
            "Generated observations are independent across i; the heteroscedastic "
            "scenario uses iid random covariate-outcome pairs."
        ),
        "settings": {
            "sample_sizes": list(sample_sizes),
            "repetitions_per_cell": repetitions,
            "quadrature_points": points,
            "seed": seed,
            "true_parameter": theta_true,
            "working_likelihood": "Normal(theta, 1)",
            "bounds": list(bounds),
            "scenario_count": len(scenarios),
            "prior_count": len(priors),
            "cell_count": len(results),
            "common_random_numbers_across_priors": True,
        },
        "scenario_definitions": [
            {
                "name": scenario.name,
                "description": scenario.description,
                "theoretical_observation_variance": scenario.theoretical_variance,
                "correctly_specified_working_model": scenario.correctly_specified,
            }
            for scenario in scenarios
        ],
        "prior_definitions": [
            {
                "name": prior.name,
                "location": prior.location,
                "bandwidth": prior.bandwidth,
                "scale": prior.scale,
                "center_frequency": prior.center_frequency,
                "modulation_depth": prior.modulation_depth,
            }
            for prior in priors
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=(40, 80, 160))
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--theta-true", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = run_stress_grid(
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
