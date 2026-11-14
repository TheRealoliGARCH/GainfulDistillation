"""Expanded Monte Carlo study for scalar Sinc-prior Bayesian sandwich inference.

This adapted GDS experiment examines:
* correctly specified Normal observations;
* homoskedastic Normal variance misspecification;
* heavy-tailed Student-t observations with finite variance;
* independent heteroskedastic Normal observations;
* multiple sample sizes; and
* Sinc-prior location, scale, and spectral-modulation sensitivity.

The working likelihood is Normal(theta, 1) in every scenario. Each experiment
cell compares the ordinary posterior credible interval, the balanced-loss
robust Wald interval, and a conventional sample-variance sandwich benchmark.

This is an adaptation using the GDS Sinc prior, not an exact replication of
Li and Rice (2024), whose published worked example uses a conjugate Normal
prior. The reported robust Wald interval is not a Bayesian credible interval.
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
DEFAULT_SCENARIOS = (
    "normal_sd_1",
    "normal_sd_2",
    "student_t5_var4",
    "heteroskedastic_normal_var4",
)
DEFAULT_PRIORS = (
    "centered_diffuse",
    "shifted_diffuse",
    "shifted_concentrated",
    "modulated_diffuse",
)
DEFAULT_SAMPLE_SIZES = (40, 80, 160)
PARAMETER_BOUNDS = (-8.0, 8.0)


def _prior_from_name(name: str) -> SincModulatedPrior:
    """Return one prespecified prior-sensitivity configuration."""
    specifications = {
        "centered_diffuse": dict(
            location=0.0, bandwidth=1.0, scale=2.0,
            center_frequency=0.0, modulation_depth=0.0,
        ),
        "shifted_diffuse": dict(
            location=-1.0, bandwidth=1.0, scale=2.0,
            center_frequency=0.0, modulation_depth=0.0,
        ),
        "shifted_concentrated": dict(
            location=-0.25, bandwidth=1.0, scale=1.2,
            center_frequency=0.0, modulation_depth=0.0,
        ),
        "modulated_diffuse": dict(
            location=0.0, bandwidth=1.0, scale=2.0,
            center_frequency=0.35, modulation_depth=0.8,
        ),
    }
    try:
        return SincModulatedPrior(**specifications[name])
    except KeyError as exc:
        raise ValueError(f"unknown prior configuration: {name}") from exc


def _prior_description(name: str) -> dict[str, float]:
    prior = _prior_from_name(name)
    return {
        "location": prior.location,
        "bandwidth": prior.bandwidth,
        "scale": prior.scale,
        "center_frequency": prior.center_frequency,
        "modulation_depth": prior.modulation_depth,
    }


def _draw_data(
    *,
    scenario: str,
    theta_true: float,
    sample_size: int,
    rng: random.Random,
) -> tuple[tuple[float, ...], float]:
    """Generate independent observations and oracle Var(sample mean)."""
    if scenario == "normal_sd_1":
        observations = tuple(
            rng.gauss(theta_true, 1.0) for _ in range(sample_size)
        )
        oracle_variance = 1.0 / sample_size

    elif scenario == "normal_sd_2":
        observations = tuple(
            rng.gauss(theta_true, 2.0) for _ in range(sample_size)
        )
        oracle_variance = 4.0 / sample_size

    elif scenario == "student_t5_var4":
        # A t_5 variable has variance 5/3. This scale gives variance four.
        scale = sqrt(4.0 * 3.0 / 5.0)
        values = []
        for _ in range(sample_size):
            normal = rng.gauss(0.0, 1.0)
            chi_square = rng.gammavariate(2.5, 2.0)
            values.append(theta_true + scale * normal / sqrt(chi_square / 5.0))
        observations = tuple(values)
        oracle_variance = 4.0 / sample_size

    elif scenario == "heteroskedastic_normal_var4":
        raw_scales = tuple(
            0.5 + 1.5 * (index + 0.5) / sample_size
            for index in range(sample_size)
        )
        raw_mean_square = fsum(value * value for value in raw_scales) / sample_size
        normalizer = 2.0 / sqrt(raw_mean_square)
        scales = tuple(value * normalizer for value in raw_scales)
        observations = tuple(
            theta_true + rng.gauss(0.0, scale)
            for scale in scales
        )
        # The scale profile is normalised to average variance exactly four.
        oracle_variance = fsum(scale * scale for scale in scales) / (sample_size**2)

    else:
        raise ValueError(f"unknown data-generating scenario: {scenario}")

    return observations, oracle_variance


def _fit_one_prior(
    *,
    observations: tuple[float, ...],
    oracle_variance: float,
    theta_true: float,
    prior_name: str,
    points: int,
) -> dict[str, float]:
    """Fit one prior to one common dataset and calculate diagnostics."""
    prior = _prior_from_name(prior_name)
    n = len(observations)
    sum_y = fsum(observations)
    sum_y_squared = fsum(value * value for value in observations)
    normal_log_constant = n * 1.8378770664093453

    def log_likelihood(theta: float) -> float:
        squared_error = (
            sum_y_squared - 2.0 * theta * sum_y + n * theta * theta
        )
        # The exact sum of squares is non-negative; guard against cancellation.
        squared_error = max(0.0, squared_error)
        return -0.5 * (normal_log_constant + squared_error)

    posterior = compute_posterior_1d(
        prior,
        log_likelihood,
        bounds=PARAMETER_BOUNDS,
        points=points,
        credible_level=0.95,
    )

    def score_information(theta: float) -> tuple[tuple[float, ...], float]:
        # Working likelihood is independent Normal(theta, 1):
        # score_i(theta) = y_i - theta and I_n(theta) = 1.
        scores = tuple(value - theta for value in observations)
        return scores, 1.0

    robust = balanced_loss_variance_1d(posterior, score_information)

    sample_mean = sum_y / n
    sample_variance = fsum(
        (value - sample_mean) ** 2 for value in observations
    ) / (n - 1)
    frequentist_sandwich_variance = sample_variance / n

    lower_credible, upper_credible = posterior.credible_interval
    robust_lower = posterior.mean - Z_975 * robust.robust_standard_error
    robust_upper = posterior.mean + Z_975 * robust.robust_standard_error
    error = posterior.mean - theta_true

    return {
        "bias_contribution": error,
        "squared_error": error * error,
        "posterior_sd": posterior.standard_deviation,
        "robust_se": robust.robust_standard_error,
        "frequentist_sandwich_se": sqrt(frequentist_sandwich_variance),
        "oracle_se": sqrt(oracle_variance),
        "posterior_covered": float(
            lower_credible <= theta_true <= upper_credible
        ),
        "robust_wald_covered": float(
            robust_lower <= theta_true <= robust_upper
        ),
        "posterior_interval_width": upper_credible - lower_credible,
        "robust_wald_interval_width": robust_upper - robust_lower,
        "robust_variance": robust.robust_variance,
        "frequentist_sandwich_variance": frequentist_sandwich_variance,
        "oracle_variance": oracle_variance,
    }


def _summarise_cell(
    *,
    cell: dict[str, list[float]],
    scenario: str,
    prior_name: str,
    sample_size: int,
    repetitions: int,
    theta_true: float,
) -> dict[str, Any]:
    bias = mean(cell["bias_contribution"])
    rmse = sqrt(mean(cell["squared_error"]))
    posterior_coverage = mean(cell["posterior_covered"])
    robust_coverage = mean(cell["robust_wald_covered"])
    robust_variance = mean(cell["robust_variance"])
    frequentist_variance = mean(cell["frequentist_sandwich_variance"])
    oracle_variance = mean(cell["oracle_variance"])

    return {
        "scenario": scenario,
        "sample_size": sample_size,
        "prior": prior_name,
        "prior_parameters": _prior_description(prior_name),
        "repetitions": repetitions,
        "bias": bias,
        "rmse": rmse,
        "mean_posterior_sd": mean(cell["posterior_sd"]),
        "mean_balanced_loss_robust_se": mean(cell["robust_se"]),
        "mean_frequentist_sandwich_se": mean(cell["frequentist_sandwich_se"]),
        "mean_oracle_se": mean(cell["oracle_se"]),
        "posterior_credible_interval_coverage": posterior_coverage,
        "posterior_coverage_mcse": sqrt(
            posterior_coverage * (1.0 - posterior_coverage) / repetitions
        ),
        "robust_wald_interval_coverage": robust_coverage,
        "robust_coverage_mcse": sqrt(
            robust_coverage * (1.0 - robust_coverage) / repetitions
        ),
        "mean_posterior_interval_width": mean(cell["posterior_interval_width"]),
        "mean_robust_wald_interval_width": mean(cell["robust_wald_interval_width"]),
        "mean_robust_variance": robust_variance,
        "mean_frequentist_sandwich_variance": frequentist_variance,
        "mean_oracle_variance": oracle_variance,
        "robust_to_frequentist_sandwich_variance_ratio": (
            robust_variance / frequentist_variance
            if frequentist_variance > 0.0 else None
        ),
        "robust_to_oracle_variance_ratio": (
            robust_variance / oracle_variance if oracle_variance > 0.0 else None
        ),
    }


def run_experiment(
    *,
    sample_sizes: tuple[int, ...] = DEFAULT_SAMPLE_SIZES,
    repetitions: int = 100,
    points: int = 512,
    seed: int = 2026111401,
    theta_true: float = 0.5,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    priors: tuple[str, ...] = DEFAULT_PRIORS,
) -> dict[str, Any]:
    """Run the expanded design using common random datasets across priors."""
    if not sample_sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in sample_sizes
    ):
        raise ValueError("sample_sizes must contain integers of at least two")
    if len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("sample_sizes must be unique")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(points, bool) or not isinstance(points, int) or points < 16:
        raise ValueError("points must be an integer of at least 16")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be non-empty and unique")
    if not priors or len(set(priors)) != len(priors):
        raise ValueError("priors must be non-empty and unique")
    if isinstance(theta_true, bool) or not isinstance(theta_true, (int, float)):
        raise ValueError("theta_true must be a finite real number")
    if not (-8.0 < float(theta_true) < 8.0):
        raise ValueError("theta_true must lie strictly inside the fixed parameter bounds")

    for name in scenarios:
        if name not in DEFAULT_SCENARIOS:
            raise ValueError(f"unknown data-generating scenario: {name}")
    for name in priors:
        _prior_from_name(name)

    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    for scenario in scenarios:
        for sample_size in sample_sizes:
            # Generate each dataset once and evaluate every prior on the same
            # observations. This common-random-number design improves the
            # precision of pairwise prior-sensitivity comparisons.
            cells: dict[str, dict[str, list[float]]] = {
                name: {
                    key: [] for key in (
                        "bias_contribution", "squared_error", "posterior_sd",
                        "robust_se", "frequentist_sandwich_se", "oracle_se",
                        "posterior_covered", "robust_wald_covered",
                        "posterior_interval_width", "robust_wald_interval_width",
                        "robust_variance", "frequentist_sandwich_variance",
                        "oracle_variance",
                    )
                }
                for name in priors
            }

            for _ in range(repetitions):
                observations, oracle_variance = _draw_data(
                    scenario=scenario,
                    theta_true=float(theta_true),
                    sample_size=sample_size,
                    rng=rng,
                )
                for prior_name in priors:
                    metrics = _fit_one_prior(
                        observations=observations,
                        oracle_variance=oracle_variance,
                        theta_true=float(theta_true),
                        prior_name=prior_name,
                        points=points,
                    )
                    for key, value in metrics.items():
                        cells[prior_name][key].append(value)

            for prior_name in priors:
                results.append(
                    _summarise_cell(
                        cell=cells[prior_name],
                        scenario=scenario,
                        prior_name=prior_name,
                        sample_size=sample_size,
                        repetitions=repetitions,
                        theta_true=float(theta_true),
                    )
                )

    return {
        "experiment": "GDS expanded Sinc-prior Bayesian robustness study",
        "reference": "Li and Rice (2024), DOI 10.1214/24-STS935",
        "design_note": (
            "Adapted GDS experiment, not an exact replication of the paper's "
            "conjugate Normal-prior worked example. The robust Wald interval "
            "is a repeated-sampling interval, not a Bayesian credible interval."
        ),
        "settings": {
            "sample_sizes": list(sample_sizes),
            "repetitions_per_cell": repetitions,
            "quadrature_points": points,
            "seed": seed,
            "true_parameter": float(theta_true),
            "working_likelihood": "independent Normal(theta, 1)",
            "parameter_bounds": list(PARAMETER_BOUNDS),
            "scenarios": list(scenarios),
            "priors": {name: _prior_description(name) for name in priors},
            "prior_comparisons_use_common_random_datasets": True,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026111401)
    parser.add_argument("--theta-true", type=float, default=0.5)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--priors", nargs="+", default=DEFAULT_PRIORS)
    args = parser.parse_args()

    result = run_experiment(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.repetitions,
        points=args.points,
        seed=args.seed,
        theta_true=args.theta_true,
        scenarios=tuple(args.scenarios),
        priors=tuple(args.priors),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
